using Gurobi
using HiGHS
using JuMP
using Distributions
using Statistics
using JSON
using Dates
using BSON: @save, @load

include("auxiliar.jl")
include("sirp_model.jl")
include("sirp_solver.jl")
include("saa_approx.jl")

function evaluate_rolling_horizon(instance::IRPInstance; demand="test", scenario=1, method="mean", horizon=10, roll_horizon=6)
    
    if (instance.nb_features>0) & (method=="saa_1")
        method = "saa_1_nearest"
    end
    
    x_val = convert(AbstractArray{Int}, zeros(instance.n+1, instance.n+1, horizon))
    eval_problem = IRPProblem()
    eval_indices = IRPIndices()
    createProblem(eval_problem, instance; horizon=horizon)
    createIndices(eval_indices, eval_problem)
    
    roll_problem = IRPProblem()
    roll_indices = IRPIndices()
    createProblem(roll_problem, instance; horizon=roll_horizon)
    createIndices(roll_indices, roll_problem)
    
    total_inv_cost = 0.
    total_penalty_cost = 0.
    total_routing_cost = 0.
    total_demand =  Dict(i => 0. for i in eval_indices.V_cus)
    total_lost_demand =  Dict(i => 0. for i in eval_indices.V_cus)
    
    demands_hist = deepcopy(instance.demands_hist)
    if method in ["saa_1_nearest"]
        samples_hist = deepcopy(instance.samples_hist)
    end
    
    for p in eval_indices.H
        
        if p > 1
            for i in eval_indices.V_cus
                if demand=="test"
                    if method in ["saa_1_nearest"]
                        push!(samples_hist[i],instance.samples_test[i][p-1])
                    end
                    push!(demands_hist[i],instance.demands_test[i][p-1])
                elseif demand=="eval"
                    if method in ["saa_1_nearest"]
                        push!(samples_hist[i],instance.samples_eval[i][scenario][p-1])
                    end
                    push!(demands_hist[i],instance.demands_eval[i][scenario][p-1])
                end
            end 
        end

        if method == "mean"
            roll_problem.demands = Dict(1 => Dict(i => mean(demands_hist[i]) * ones(roll_horizon) for i in roll_indices.V_cus))
            roll_problem.scenarios = 1
        elseif method == "saa_1"
            roll_problem.demands = Dict(1 => Dict(i => demands_hist[i][end-(roll_horizon-1):end] for i in roll_indices.V_cus))
            roll_problem.scenarios = 1
        elseif method == "saa_1_nearest"
            if demand=="test"
                roll_problem.demands = Dict(1 => Dict(i => [[y for (y,_) in sort([(sample["label"], sum((sample["features"] - instance.samples_test[i][p+k-1]["features"]).^2))
                                        for sample in samples_hist[i]], by=x->x[2])][1] for k in 1:roll_horizon] for i in roll_indices.V_cus))
            elseif demand=="eval"
                roll_problem.demands = Dict(1 => Dict(i => [[y for (y,_) in sort([(sample["label"], sum((sample["features"] - instance.samples_eval[i][scenario][p+k-1]["features"]).^2))
                                        for sample in samples_hist[i]], by=x->x[2])][1] for k in 1:roll_horizon] for i in roll_indices.V_cus))
            end
            roll_problem.scenarios = 1
        else
            error("Selected method not implemented.")
        end

        
        roll_indices.S = collect(1:roll_problem.scenarios)
        x = sirp_solver(roll_problem, roll_indices)
        x = x[:,:,1]
        x_val[:,:,p] = x
        
        y = single_g(x)

        q = [(eval_problem.max_inventory[i] - eval_problem.start_inventory[i]) * y[i-1] for i in eval_indices.V_cus]

        for i in eval_indices.V_cus
            if demand=="test"
                inventory_tmp = eval_problem.start_inventory[i] + q[i-1] - instance.demands_test[i][p]
                total_demand[i] += instance.demands_test[i][p]
            elseif demand=="eval"
                inventory_tmp = eval_problem.start_inventory[i] + q[i-1] - instance.demands_eval[i][scenario][p]
                total_demand[i] += instance.demands_eval[i][scenario][p]
            end
            if inventory_tmp < 0
                total_lost_demand[i] -= inventory_tmp
                total_penalty_cost -= inventory_tmp * instance.penalty_cost[i]
                eval_problem.start_inventory[i] = 0
                roll_problem.start_inventory[i] = 0
            else
                total_inv_cost += inventory_tmp * instance.holding_cost[i]
                eval_problem.start_inventory[i] = inventory_tmp
                roll_problem.start_inventory[i] = inventory_tmp
            end
        end

        total_routing_cost += sum([x[i,j] * 1/2 * instance.distances[fix_index(i,instance.n+1)][fix_index(j,instance.n+1)] for i in 1:instance.n+1 for j in 1:instance.n+1]) 

    end
    
    return x_val, total_inv_cost, total_penalty_cost, total_routing_cost, Dict(i => 1-(total_lost_demand[i]/total_demand[i]) for i in eval_indices.V_cus)
end


function run_benchmark(penalty_inv::Int, instance_id::String; demand="test", policy="mean", evaluation_horizon=10, look_ahead=6)
        
    @info "Starting $(policy) policy evaluation with $(look_ahead) periods look-ahead..."
    @info "Instance ID: $(instance_id), using $(demand) demand samples and a shortage penalty of $(penalty_inv)"
    @info "Evaluation horizon: $(evaluation_horizon) periods"

    pattern = convert(String, split(instance_id, "-")[1])
    (policy in ["anticipative", "mean", "saa_1", "saa_3"]) || error("Selected benchmark not implemented.")
    
    instance = IRPInstance()
    readInstance("instances/"*instance_id*".json", pattern, instance; penalty_inv=penalty_inv)
    
    benchmark_start = now()
  
    if policy=="anticipative"
        problem = IRPProblem()
        indices = IRPIndices()
        createProblem(problem, instance; horizon=evaluation_horizon)
        createIndices(indices, problem)
        if demand=="test"
            problem.demands = Dict(1 => deepcopy(instance.demands_test))
        elseif demand=="eval"
            problem.demands = Dict(1 => deepcopy(instance.demands_eval))
        end 
        _, stockout_costs, holding_costs, routing_costs, _ = sirp_solver(problem, indices; obj=true);
    elseif policy=="saa_3"
         _, holding_costs, stockout_costs, routing_costs = evaluate_saa(instance;
            penalty_inv=penalty_inv, demand=demand, nb_scenarios=3, horizon=evaluation_horizon, roll_horizon=look_ahead)     
    else
        _, holding_costs, stockout_costs, routing_costs, _ = evaluate_rolling_horizon(instance;
            demand=demand, method=policy, horizon=evaluation_horizon, roll_horizon=look_ahead) 
    end
    
    total_costs = routing_costs + stockout_costs + holding_costs
    
    inference_time = (now() - benchmark_start) / Millisecond(1000)

    # Round costs to two decimal places for printing
    routing_costs_rounded = round(routing_costs, digits=2)
    stockout_costs_rounded = round(stockout_costs, digits=2)
    holding_costs_rounded = round(holding_costs, digits=2)
    total_costs_rounded = round(total_costs, digits=2)
    println("-----")
    @info "Routing costs: $routing_costs_rounded"
    @info "Stock-out costs: $stockout_costs_rounded"
    @info "Holding costs: $holding_costs_rounded"
    println("-----")
    @info "Total costs: $total_costs_rounded"
    @info "Inference time: $inference_time seconds"   
end