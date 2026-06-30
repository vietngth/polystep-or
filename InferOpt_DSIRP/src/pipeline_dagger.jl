using InferOpt: GeneralizedMaximizer, PerturbedAdditive, FenchelYoungLoss
using ProgressMeter
using Flux
using Gurobi
using HiGHS
using JuMP
using Distributions
using Statistics
using JSON
using Dates
using Parameters
using BSON: @save, @load
include("auxiliar.jl")
include("sirp_model.jl")
include("sirp_solver.jl")
include("stat_model.jl")
include("evaluation_pipeline.jl")
include("pctsp.jl")


@with_kw struct dagger_settings
    nb_outer_epochs::Int=150
    nb_inner_epochs::Int=50
    nb_iterations::Int=20
    alpha_0::Float64=1.0
    alpha_decay::Float64=0.05
    alpha_min::Float64=0.1
    nb_keep_epochs::Int=10
    portion_keep_epochs::Float64=0.5
    early_stopping::Int=5
    nb_scenarios::Int=5
    lr_start::Float64=0.01
    fyl_samples::Int=5
    fyl_epsilon::Float64=20.0
    look_ahead::Int=6
    demand_quantiles::Vector{Float64}=[0.01, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99]
    load_model::String="none" 
    create_model::Bool=true
    sys_cpus::Int=nothing
    num_threads::Int=nothing
    milp_solver::String="gurobi"
end


function run_pipeline(patterns::Vector{String}, penalties::Vector{Int}, instances::Vector{String}, settings::dagger_settings)
    
    (; nb_outer_epochs, nb_inner_epochs, nb_iterations,
        nb_keep_epochs, portion_keep_epochs, early_stopping, nb_scenarios,
        lr_start, fyl_samples, fyl_epsilon, look_ahead, demand_quantiles,
        load_model, create_model, alpha_0, alpha_decay, alpha_min) = settings
    
    α_min = alpha_min
    α_0 = alpha_0
    α_decay = alpha_decay
            
    co_problem = "pctsp"
    if patterns[1]=="contextual"
        nb_features = 8
    else
        nb_features = 0
    end
        
    # Create folder
    models_folder = "training/models/dagger/"
    solutions_folder = "training/solutions/dagger/"
    
    if length(patterns)==1
        models_folder *= patterns[1]*"/"
        solutions_folder *= patterns[1]*"/"
    end
    if length(penalties)==1
        models_folder *= "penalty_"*string(penalties[1])*"/"
        solutions_folder *= "penalty_"*string(penalties[1])*"/"
    end
    if length(instances)==1
        models_folder *= instances[1]*"/"
        solutions_folder *= instances[1]*"/"
    end
    if !ispath(models_folder)
        mkpath(models_folder)
    end
    if !ispath(solutions_folder)
        mkpath(solutions_folder)
    end
    # Create model_id and initalize solution log
    if create_model
        model_id = Dates.format(now(), "yymmdd_HHMMSSs")
        solution = Dict{String, Dict{String, Any}}()
        solution[co_problem] = Dict()
        solution[co_problem]["settings"] = settings
        solution[co_problem]["seconds"] = Dict("state transition" => 0., "sample generation" => 0., "policy update" => 0., "policy evaluation" => 0., "log" => 0.)
        if load_model != "none"
            solution[co_problem]["pretrained"] = load_model
        else
            solution[co_problem]["pretrained"] = nothing
        end
        solution[co_problem]["best_iteration"] = nothing
    else
        model_id = load_model
        solution = JSON.parsefile(realpath(solutions_folder*model_id*"_solutions.json"))
        solution[co_problem]["settings"] = settings
    end
    
    instance_dict = Dict{String, IRPInstance}()
    problem_dict = Dict{String, IRPProblem}()
    indices_dict = Dict{String, IRPIndices}()
    
    for instance_id in instances    
        for pattern in patterns
            for penalty_inv in penalties
                # Instance, problem and indices for learning pipeline
                idx = instance_id*"_"*pattern*"_"*string(penalty_inv)
                instance_dict[idx] = IRPInstance()
                problem_dict[idx] = IRPProblem()
                indices_dict[idx] = IRPIndices()

                readInstance("instances/"*instance_id*".json", pattern, instance_dict[idx]; penalty_inv=penalty_inv)
                createProblem(problem_dict[idx], instance_dict[idx]; horizon=look_ahead)
                createIndices(indices_dict[idx], problem_dict[idx])
            end
        end
    end
    
    generalized_maximizer = GeneralizedMaximizer(pctsp, single_g, single_h)
    loss = FenchelYoungLoss(PerturbedAdditive(generalized_maximizer; ε=fyl_epsilon, nb_samples=fyl_samples))
    flux_loss = sample -> loss(φ_w(sample.feature_array), sample.label[:,:,1]; sample=sample)

    # Initalize statistical model (during first iteration)       
    if load_model != "none"
        @load models_folder*model_id*"_model.bson" φ_w regression opt start_inventory_dict demands_hist_dict demand_samples_dict previous_samples
        load_weights = Dict(string(k) => deepcopy(weights) for (k, weights) in enumerate(Flux.params(φ_w, regression)))
        
        φ_w, regression = build_stat_model(demand_quantiles, look_ahead; nb_features=nb_features, weights=load_weights) 
    else 
        opt = Adam(lr_start)
        start_inventory_dict = Dict{String, Dict{Int, Float64}}()
        demands_hist_dict = Dict{String, Dict{Int, Vector{Float64}}}()
        demand_samples_dict = Dict{String, Dict{Int, Vector{Any}}}()
        for idx in keys(instance_dict)
            start_inventory_dict[idx] = deepcopy(problem_dict[idx].start_inventory)
            demands_hist_dict[idx] = deepcopy(instance_dict[idx].demands_hist)   
        end

        φ_w, regression = build_stat_model(demand_quantiles, look_ahead; nb_features=nb_features)
        previous_samples = []
    end

    global par = Flux.params(φ_w, regression)
    
    # Log results
    json_string = JSON.json(solution)
    open(solutions_folder*model_id*"_solutions.json", "w") do f
        write(f, json_string)
    end
    
    @showprogress "Iterations for $(solutions_folder*model_id) : " for e_outer in 1:nb_outer_epochs
        if string(e_outer) in keys(solution[co_problem])
            continue
        end
        
        current_samples = []
        previous_samples = previous_samples[max(1, end-(nb_scenarios*nb_iterations*nb_keep_epochs)+1):end]
        # Build the training set on a diverse set of states and scenarios (nb_scenarios different scenarios for each state)
        
        # Compute alpha with minimum threshold
        α = max(α_min, α_0 * exp(-α_decay * e_outer))
        println("Current α = $(round(α, digits=3))")
        
        for p in 1:nb_iterations
            # State transition
            state_transition_start = now()
            
            # Policy selection (mixture)
            α_mixture = rand()
              
            for idx in keys(instance_dict)
                if p==1
                    demand_samples_dict[idx] = Dict(i => rand(instance_dict[idx].sample_demands[i], problem_dict[idx].horizon) for i in indices_dict[idx].V_cus) 
                end

                # Policy selection (mixture)
                if α_mixture < α              
                    println("Anticipative policy in iteration $(p) with current α = $(round(α, digits=3)) and α_mixture $(α_mixture)")
                    
                    roll_problem = IRPProblem()
                    roll_indices = IRPIndices()
                    createProblem(roll_problem, instance_dict[idx]; horizon=look_ahead)
                    createIndices(roll_indices, roll_problem)
                    
                    if nb_features>0
                        roll_problem.demands = Dict(1 => Dict(i => [[y for (y,_) in sort([(sample["label"], sum((sample["features"] - demand_samples_dict[idx][i][p+k-1]["features"]).^2))
                                                    for sample in demands_hist_dict[idx][i]], by=x->x[2])][1] for k in 1:look_ahead] for i in roll_indices.V_cus))
                        
                    else
                        roll_problem.demands = Dict(1 => Dict(i => demands_hist_dict[idx][i][end-(look_ahead-1):end] for i in roll_indices.V_cus))
                    end

                    roll_problem.scenarios = 1
                    roll_indices.S = collect(1:roll_problem.scenarios)
                    x = sirp_solver(roll_problem, roll_indices)
                    y = single_g(x[:,:,1])
                    
                else
                    println("Current policy")
                    
                    if nb_features>0
                        contextual_features = Dict(k => Dict(i => vec(convert(Vector{Float64}, values(demand_samples_dict[idx][i][k]["features"]))) for i in indices_dict[idx].V_cus) for k in 1:look_ahead)
                    else
                        contextual_features = Dict{Int, Dict{Int, Vector{Float64}}}()
                    end
                    
                    sample = createSample(instance_dict[idx], indices_dict[idx], start_inventory_dict[idx], demands_hist_dict[idx], Int[];
                    contextual_features=contextual_features, demand_quantiles=demand_quantiles, look_ahead=look_ahead, nb_features=nb_features)

                    θ = φ_w(sample.feature_array)
                    x = pctsp(θ; sample)
                    y = single_g(x[:,:,1])
                end
                    

                for i in indices_dict[idx].V_cus
                    true_demand = if (nb_features>0) demand_samples_dict[idx][i][1]["label"] else demand_samples_dict[idx][i][1] end
                    start_inventory_dict[idx][i] = max(0, start_inventory_dict[idx][i] - true_demand + y[i-1] * (problem_dict[idx].max_inventory[i] - start_inventory_dict[idx][i]))

                    demands_hist_dict[idx][i][1:end-1] = demands_hist_dict[idx][i][2:end]
                    demands_hist_dict[idx][i][end] = true_demand

                    demand_samples_dict[idx][i][1:end-1] = demand_samples_dict[idx][i][2:end]
                    demand_samples_dict[idx][i][end] = rand(instance_dict[idx].sample_demands[i])  
                end 
            end                              
                
            solution[co_problem]["seconds"]["state transition"] += ((now() - state_transition_start)/Millisecond(1000))

            # Sample generation
            sample_generation_start = now()
            for idx in keys(instance_dict)
                for _ in 1:nb_scenarios
                    roll_demand_samples = Dict(i => rand(instance_dict[idx].sample_demands[i], problem_dict[idx].horizon) for i in indices_dict[idx].V_cus)
                    if nb_features>0
                        problem_dict[idx].demands = Dict(1 => Dict(i => [roll_demand_samples[i][k]["label"] for k in indices_dict[idx].H] for i in indices_dict[idx].V_cus))
                        contextual_features = Dict(k => Dict(i => vec(convert(Vector{Float64}, values(roll_demand_samples[i][k]["features"]))) for i in indices_dict[idx].V_cus) for k in 1:look_ahead)
                    else
                        problem_dict[idx].demands = Dict(1 => Dict(i => [roll_demand_samples[i][k] for k in indices_dict[idx].H] for i in indices_dict[idx].V_cus))
                        contextual_features=Dict{Int, Dict{Int, Vector{Float64}}}()
                    end
                    problem_dict[idx].start_inventory = start_inventory_dict[idx]
                    label = sirp_solver(problem_dict[idx], indices_dict[idx])
                    sample = createSample(instance_dict[idx], indices_dict[idx], start_inventory_dict[idx], demands_hist_dict[idx], label;
                        contextual_features=contextual_features, look_ahead=look_ahead, demand_quantiles=demand_quantiles, nb_features=nb_features)
                    push!(current_samples, sample)
                end
            end
            solution[co_problem]["seconds"]["sample generation"] += ((now() - sample_generation_start)/Millisecond(1000))
        end 
        
        # Policy update
        policy_update_start = now()
        if length(previous_samples) > 0
            train_samples = vcat(rand(previous_samples, Int(round(portion_keep_epochs*length(previous_samples)))), current_samples)
        else
            train_samples = current_samples
        end
        
        for e_inner in 1:max(1, nb_inner_epochs-Int(ceil(0.5*e_outer))+1)
            gs = gradient(par) do
                global l = mean(flux_loss(sample) for sample in train_samples)
            end
            Flux.update!(opt, par, gs)
        end
        solution[co_problem]["seconds"]["policy update"] += ((now() - policy_update_start)/Millisecond(1000))

        # Evaluate policy
        policy_evaluation_start = now()
        solution[co_problem][string(e_outer)] = Dict()
        solution[co_problem][string(e_outer)]["loss"] = l
        sum_total_cost = 0.
        for idx in keys(instance_dict)
            for scenario in 1:nb_scenarios
                _, inv_cost, penalty_cost, routing_cost, _ = evaluate_pctsp(φ_w, instance_dict[idx];
                demand="eval", scenario=scenario, nb_features=nb_features, demand_quantiles=demand_quantiles, look_ahead=look_ahead, evaluation_horizon=10)      
                sum_total_cost +=  inv_cost + penalty_cost + routing_cost
            end
            solution[co_problem][string(e_outer)][idx] = Dict(
                "states" => Dict(i => deepcopy([sample.start_inventory[i]/instance_dict[idx].max_inventory[i] for sample in current_samples]) for i in indices_dict[idx].V_cus))
        end
        solution[co_problem][string(e_outer)]["objective"] = sum_total_cost
        if isnothing(solution[co_problem]["best_iteration"])
            solution[co_problem]["best_iteration"] = e_outer
        elseif sum_total_cost < solution[co_problem][string(solution[co_problem]["best_iteration"])]["objective"]
            solution[co_problem]["best_iteration"] = e_outer
        end
        solution[co_problem]["seconds"]["policy evaluation"] += ((now() - policy_evaluation_start)/Millisecond(1000))
        
        
        # Log results and model
        log_start = now()
        solution[co_problem][string(e_outer)]["weights"] = Dict(k => deepcopy(weights) for (k, weights) in enumerate(par))
        @save models_folder*model_id*"_model.bson" φ_w regression opt start_inventory_dict demands_hist_dict demand_samples_dict previous_samples
        json_string = JSON.json(solution)
        open(solutions_folder*model_id*"_solutions.json", "w") do f
            write(f, json_string)
        end
        solution[co_problem]["seconds"]["log"] += ((now() - log_start)/Millisecond(1000))
        
        if solution[co_problem]["best_iteration"] + early_stopping == e_outer
            break
        end
        append!(previous_samples, current_samples)
    end
    
    return joinpath(splitpath(solutions_folder)[3:end])*"/"*model_id*"_solutions.json"
end