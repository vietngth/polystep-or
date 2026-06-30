using InferOpt
using ProgressMeter
using Flux
using Gurobi
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


@with_kw struct sampling_settings
    nb_outer_epochs::Int=150
    nb_inner_epochs::Int=50
    early_stopping::Int=5
    nb_scenarios::Int=600
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


function run_pipeline(patterns::Vector{String}, penalties::Vector{Int}, instances::Vector{String}, settings::sampling_settings)
    
    (; nb_outer_epochs, nb_inner_epochs, 
        early_stopping, nb_scenarios, lr_start, 
        fyl_samples, fyl_epsilon, look_ahead, demand_quantiles,
        load_model, create_model) = settings
    
    co_problem = "pctsp"
    if patterns[1]=="contextual"
        nb_features = 8
    else
        nb_features = 0
    end
            
    # Create folder
    models_folder = "training/models/sampling/"
    solutions_folder = "training/solutions/sampling/"
    
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
    
    generalized_maximizer=GeneralizedMaximizer(pctsp, single_g, single_h)
    loss=FenchelYoungLoss(PerturbedAdditive(generalized_maximizer; ε=fyl_epsilon, nb_samples=fyl_samples))
    flux_loss = sample -> loss(φ_w(sample.feature_array), sample.label[:,:,1]; sample=sample)

    # Initalize statistical model (during first iteration)       
    if load_model != "none"
        @load models_folder*model_id*"_model.bson" φ_w regression opt inventory_dict samples_dict
        load_weights = Dict(string(k) => deepcopy(weights) for (k, weights) in enumerate(Flux.params(φ_w, regression)))
        
        φ_w, regression = build_stat_model(demand_quantiles, look_ahead; nb_features=nb_features, weights=load_weights) 
    else 
        opt = Adam(lr_start)
        inventory_dict = Dict{String, Dict{Int, Dict{Int, Float64}}}()
        for idx in keys(instance_dict)
            inventory_dict[idx] = Dict{Int, Dict{Int, Float64}}()
            for s in 1:nb_scenarios
                inventory_dict[idx][s] =  Dict(i => rand(Uniform(0., 1.0)) * problem_dict[idx].max_inventory[i] for i in indices_dict[idx].V_cus)  
            end
        end

        φ_w, regression = build_stat_model(demand_quantiles, look_ahead; nb_features=nb_features)   
        samples_dict = Dict{String, Vector{IRPSample}}()
    end
    
    
    for idx in keys(instance_dict)
        if idx ∉ keys(samples_dict)
            samples_dict[idx] = []
        end
        
        for s in (length(samples_dict[idx])+1):nb_scenarios
            sample_generation_start = now()
            demands_hist_dict = deepcopy(instance_dict[idx].demands_hist)
            start_inventory_dict = deepcopy(inventory_dict[idx][s])
            roll_demand_samples = Dict(i => rand(instance_dict[idx].sample_demands[i], problem_dict[idx].horizon + look_ahead) for i in indices_dict[idx].V_cus)
            if nb_features>0
                problem_dict[idx].demands = Dict(1 => Dict(i => [roll_demand_samples[i][k]["label"] for k in indices_dict[idx].H] for i in indices_dict[idx].V_cus))
            else
                problem_dict[idx].demands = Dict(1 => Dict(i => [roll_demand_samples[i][k] for k in indices_dict[idx].H] for i in indices_dict[idx].V_cus))
            end
            problem_dict[idx].start_inventory = start_inventory_dict
            label = sirp_solver(problem_dict[idx], indices_dict[idx])
            y = single_g(label[:,:,1])

            for p in 1:1
                if nb_features>0
                    contextual_features = Dict(k => Dict(i => vec(convert(Vector{Float64}, values(roll_demand_samples[i][k]["features"]))) for i in indices_dict[idx].V_cus) for k in p:p+look_ahead-1)
                else
                    contextual_features = Dict{Int, Dict{Int, Vector{Float64}}}()
                end

                sample = createSample(instance_dict[idx], indices_dict[idx], start_inventory_dict, demands_hist_dict, label[:,:,p:end];
                    contextual_features=contextual_features, look_ahead=look_ahead, demand_quantiles=demand_quantiles, nb_features=nb_features)
                push!(samples_dict[idx], sample)

                for i in indices_dict[idx].V_cus
                    true_demand =  problem_dict[idx].demands[1][i][p]
                    start_inventory_dict[i] = max(0, start_inventory_dict[i] - true_demand + y[i-1] * (problem_dict[idx].max_inventory[i] - start_inventory_dict[i]))

                    demands_hist_dict[i][1:end-1] = demands_hist_dict[i][2:end]
                    demands_hist_dict[i][end] = true_demand
                end
            end
            solution[co_problem]["seconds"]["sample generation"] += ((now() - sample_generation_start)/Millisecond(1000))

            # Log results
            json_string = JSON.json(solution)
            open(solutions_folder*model_id*"_solutions.json", "w") do f
                write(f, json_string)
            end
            @save models_folder*model_id*"_model.bson" φ_w regression opt inventory_dict samples_dict

        end
    end

    
    train_samples = reduce(vcat, values(samples_dict))
    global par = Flux.params(φ_w, regression)
    
    solution[co_problem]["states"] = Dict()
    for idx in keys(instance_dict)
        solution[co_problem]["states"][idx] = Dict(i => deepcopy([sample.start_inventory[i]/sample.max_inventory[i] for sample in train_samples]) for i in indices_dict[idx].V_cus)
    end
    
    @showprogress "Iterations for $(solutions_folder*model_id) : " for e_outer in 1:nb_outer_epochs
        if string(e_outer) in keys(solution[co_problem])
            continue
        end
        
        # Policy update
        policy_update_start = now()
       
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
        solution[co_problem][string(e_outer)]["weights"] = Dict{Int, Vector{Float64}}()
        sum_total_cost = 0.

        for idx in keys(instance_dict)
            for scenario in 1:nb_scenarios
                _, inv_cost, penalty_cost, routing_cost, _ = evaluate_pctsp(φ_w, instance_dict[idx];
                demand="eval", scenario=scenario, nb_features=nb_features, demand_quantiles=demand_quantiles, look_ahead=look_ahead, evaluation_horizon=10)      
                sum_total_cost +=  inv_cost + penalty_cost + routing_cost
            end
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
        @save models_folder*model_id*"_model.bson" φ_w regression opt inventory_dict samples_dict
        json_string = JSON.json(solution)
        open(solutions_folder*model_id*"_solutions.json", "w") do f
            write(f, json_string)
        end
        solution[co_problem]["seconds"]["log"] += ((now() - log_start)/Millisecond(1000))
        
        if solution[co_problem]["best_iteration"] + early_stopping == e_outer
            break
        end
    end

    return joinpath(splitpath(solutions_folder)[3:end])*"/"*model_id*"_solutions.json"
end