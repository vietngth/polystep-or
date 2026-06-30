using ProgressMeter

include("subtours.jl")
include("tsp.jl")

function evaluate_saa(instance::IRPInstance; penalty_inv=200, demand="test", nb_scenarios=3, horizon=10, roll_horizon=6, model_id=nothing)
    
    (demand in ["test"]) || error("Selected demand type not implemented for SAA-3.")
    
    if (instance.nb_features>0)
        nearest = true
    else 
        nearest = false
    end
    pattern = convert(String, split(instance.name, "-")[1])
    
    # Problem and indices for benchmarks
    eval_problem = IRPProblem()
    eval_indices = IRPIndices()
    createProblem(eval_problem, instance; horizon=horizon)
    createIndices(eval_indices, eval_problem)
    
    # Create folder
    folder = "benchmarks/"*pattern*"/penalty_"*string(penalty_inv)*"/"*instance.name*"/"
    if !ispath(folder)
        mkpath(folder)
    end
    # Create model_id and initalize solution log
    if isnothing(model_id)
        model_id = model_id = Dates.format(now(), "yymmdd_HHMMSSs")
        solution = Dict(["periods_done" => 0, "nb_scenarios" => nb_scenarios, "nearest" => nearest,
                "inventory" => 0., "penalty" => 0., "routing" => 0., "total" => 0.,
                "x" => convert(AbstractArray{Int}, zeros(instance.n+1, instance.n+1, horizon)),
                "state" => deepcopy(eval_problem.start_inventory),
                "seconds" => 0.])
    else
        solution = JSON.parsefile(realpath(folder*model_id*"_solutions.json"))
        solution["x"] = cat(FloatFromAny.(reduce.(hcat, solution["x"]))...,dims=3)
        eval_problem.start_inventory =  keys_to_int(solution["state"])
    end
    
    
    demands_hist = deepcopy(instance.demands_hist)
    if nearest
        samples_hist = deepcopy(instance.samples_hist)
    end
    @showprogress for p in eval_indices.H
        
        benchmark_start = now()
        
        if p > 1
            for i in eval_indices.V_cus
                push!(demands_hist[i],instance.demands_test[i][p-1])
                if nearest
                    push!(samples_hist[i], instance.samples_test[i][p-1])
                end
            end 
        end
        
        if p > solution["periods_done"]
            delivery_combinations = hcat(bit_to_array.(string.(0:(2^(eval_problem.n-1)) - 1, base=2), (eval_problem.n-1))...)
            ou_delivery_quantities = [eval_problem.max_inventory[i] - eval_problem.start_inventory[i] for i in eval_indices.V_cus]
            feasible_delivery_combinations = delivery_combinations[:, (sum(eachrow(delivery_combinations .* ou_delivery_quantities))  .<= eval_problem.v_cap)]
            quantites = ou_delivery_quantities .* feasible_delivery_combinations
            if nearest
                demands = Dict(s => Dict(i => [[y for (y,_) in sort([(sample["label"], sum((sample["features"] - instance.samples_test[i][p+k-1]["features"]).^2))
                                            for sample in samples_hist[i]], by=x->x[2])][s] for k in 1:roll_horizon] for i in eval_indices.V_cus) for s in 1:nb_scenarios)
            else
                demands = Dict(s => Dict(i => demands_hist[i][end-s*(roll_horizon-1):end-(s-1)*(roll_horizon-1)] for i in eval_indices.V_cus) for s in 1:nb_scenarios)
            end
            states = Dict((k, s) => Dict([i => 0.0 for i in eval_indices.V_cus]) for k in 1:size(feasible_delivery_combinations,2) for s in 1:nb_scenarios)
            roll_solution = Dict((k, s) => Dict(["inventory" => 0.0, "penalty" => 0.0, "routing" => 0.0, "x" => zeros(eval_problem.n+1,eval_problem.n+1)])
                for k in 1:size(feasible_delivery_combinations,2) for s in 1:nb_scenarios)

            for k in 1:size(feasible_delivery_combinations,2)
                x, routing_cost = tsp(feasible_delivery_combinations[:,k]; instance.distances)    
                for s in 1:nb_scenarios
                    roll_solution[k,s]["routing"] = routing_cost
                    roll_solution[k,s]["x"] = x
                    for i in eval_indices.V_cus
                        inventory_tmp = eval_problem.start_inventory[i] - demands[s][i][1] + quantites[i-1,k]
                        if inventory_tmp < 0
                            roll_solution[k,s]["penalty"] -= inventory_tmp * instance.penalty_cost[i]
                            states[k,s][i] = 0
                        else
                            roll_solution[k,s]["inventory"] += inventory_tmp * instance.holding_cost[i]
                            states[k,s][i] = inventory_tmp
                        end
                    end
                    roll_solution[k,s]["total"] = roll_solution[k,s]["routing"] + roll_solution[k,s]["inventory"] +roll_solution[k,s]["penalty"]
                end  
            end

            roll_problem = IRPProblem()
            roll_indices = IRPIndices()
            createProblem(roll_problem, instance; horizon=roll_horizon-1)
            createIndices(roll_indices, roll_problem)
            for iter in 1:length(states)
                k = Int(ceil(iter/nb_scenarios))
                s = iter-(k-1)*nb_scenarios
                state = states[k,s]
                roll_problem.start_inventory = state
                roll_problem.demands =  Dict(1 => Dict(i => mean(reduce(hcat, [demands[s_tmp][i][2:end] for s_tmp in 1:nb_scenarios])) * ones(roll_horizon-1) for i in roll_indices.V_cus))
                _, penalty_cost, inv_cost, routing_cost = sirp_solver(roll_problem, roll_indices, obj=true)
                roll_solution[k,s]["penalty"] += penalty_cost
                roll_solution[k,s]["inventory"] += inv_cost
                roll_solution[k,s]["routing"] += routing_cost
                roll_solution[k,s]["total"] += (penalty_cost + inv_cost + routing_cost)  
            end

            k_min = argmin([sum(roll_solution[k,s]["total"] for s in 1:nb_scenarios) for k in 1:size(feasible_delivery_combinations,2)])
            x = roll_solution[k_min,1]["x"]
            solution["x"][:,:,p] = x

            y = single_g(x)

            q = [(eval_problem.max_inventory[i] - eval_problem.start_inventory[i]) * y[i-1] for i in eval_indices.V_cus]

            for i in eval_indices.V_cus
                inventory_tmp = eval_problem.start_inventory[i] + q[i-1] - instance.demands_test[i][p]
                if inventory_tmp < 0
                    solution["penalty"] -= inventory_tmp * instance.penalty_cost[i]
                    eval_problem.start_inventory[i] = 0
                else
                    solution["inventory"] += inventory_tmp * instance.holding_cost[i]
                    eval_problem.start_inventory[i] = inventory_tmp
                end
            end
            solution["routing"] -= single_routing(x; instance.distances)
            solution["total"] = solution["inventory"] + solution["penalty"] + solution["routing"]
            solution["state"] = deepcopy(eval_problem.start_inventory)
            solution["periods_done"] = p
            solution["seconds"] += (now() - benchmark_start)/Millisecond(1000)  
            
            json_string = JSON.json(solution)
            open(folder*model_id*"_solutions.json", "w") do f
                write(f, json_string)
            end
        end  
    end
    
    return solution["x"], solution["inventory"], solution["penalty"], solution["routing"]
end