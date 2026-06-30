function evaluate_pctsp(φ_w, instance::IRPInstance; demand="test", scenario=1, evaluation_horizon=10, look_ahead=5, kwargs...)
     
    x_val = convert(AbstractArray{Int}, zeros(instance.n+1, instance.n+1, evaluation_horizon))
    eval_problem = IRPProblem()
    eval_indices = IRPIndices()
    createProblem(eval_problem, instance; horizon=evaluation_horizon)
    createIndices(eval_indices, eval_problem)
    
    total_inv_cost = 0.
    total_penalty_cost = 0.
    total_routing_cost = 0.
    total_prices = 0.
    total_demand =  Dict(i => 0. for i in eval_indices.V_cus)
    total_lost_demand =  Dict(i => 0. for i in eval_indices.V_cus)
    
    demands_hist = deepcopy(instance.demands_hist)
    
    for p in eval_indices.H
        if p > 1
            for i in eval_indices.V_cus
                if demand=="test"
                    push!(demands_hist[i],instance.demands_test[i][p-1])
                elseif demand=="eval"
                    push!(demands_hist[i],instance.demands_eval[i][scenario][p-1])
                end
            end 
        end        
        
        if instance.nb_features > 0
            if demand=="test"
                demand_samples = Dict(i => instance.samples_test[i][p:p+look_ahead-1] for i in eval_indices.V_cus)
            elseif demand=="eval"
                demand_samples = Dict(i => values(instance.samples_eval[i][scenario])[p:p+look_ahead-1] for i in eval_indices.V_cus)
            end
            contextual_features = Dict(k => Dict(i => vec(convert(Vector{Float64}, values(demand_samples[i][k]["features"]))) for i in eval_indices.V_cus) for k in 1:look_ahead)
        else
            contextual_features = contextual_features=Dict{Int, Dict{Int, Vector{Float64}}}()
        end
        
        sample = createSample(instance, eval_indices, eval_problem.start_inventory, demands_hist, Int[];
            contextual_features=contextual_features, look_ahead=look_ahead, nb_features=instance.nb_features, kwargs...)
        θ = φ_w(sample.feature_array)
        x = pctsp(θ; sample)
        x_val[:,:,p] = x
        y = single_g(x)

        q = [(eval_problem.max_inventory[i] - eval_problem.start_inventory[i]) * y[i-1] for i in eval_indices.V_cus]
    
        total_prices += sum([θ[i-1] * y[i-1] for i in eval_indices.V_cus])
        
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
            else
                total_inv_cost += inventory_tmp * instance.holding_cost[i]
                eval_problem.start_inventory[i] = inventory_tmp
            end
        end
        total_routing_cost -= single_h(x; sample)

    end
    
    return x_val, total_inv_cost, total_penalty_cost, total_routing_cost, Dict(i => 1-(total_lost_demand[i]/total_demand[i]) for i in eval_indices.V_cus)
end