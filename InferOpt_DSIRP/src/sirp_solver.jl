function fix_index(i::Int, indices::IRPIndices)
    idx = max(1, mod(i, length(indices.V_aug)))
    return idx
end


function sirp_solver(problem::IRPProblem, indices::IRPIndices; model_builder=milp_builder, obj=false)
    
    # 1. MODEL
    m = model_builder()

    # 2. VARIABLES
    @variable(m, z[s=indices.S, i=indices.V_aug, t=indices.H], Bin)
    @variable(m, x[s=indices.S, i=indices.V_aug, j=indices.V_aug, t=indices.H; j > i], Bin)
    @variable(m, 0 <= q[s=indices.S, i=indices.V_cus, t=indices.H] <= problem.v_cap)
    @variable(m, l[s=indices.S, i=indices.V_cus, t=indices.H] >= 0)
    @variable(m, 0 <= y[s=indices.S, i=indices.V_aug, j=indices.V_aug, t=indices.H; i != j] <= problem.v_cap)
    @variable(m, I[s=indices.S, i=indices.V_cus, t=indices.T] >= 0)
    
    # 3. BASIC CONSTRAINTS
    @constraint(m, inv_max[s=indices.S, i=indices.V_cus, t=indices.T],
        I[s,i,t] - problem.max_inventory[i] <= 0)
    
    # XX. First period decision
    s_tmp = [s for s in indices.S if s < maximum(indices.S)]
    @constraint(m, first_period_routing[s=s_tmp, i=indices.V_aug, j=indices.V_aug; j > i],
        x[s,i,j,1] ==  x[s+1,i,j,1])
    @constraint(m, first_period_inventory[s=s_tmp, i=indices.V_cus],
        q[s,i,1] ==  q[s+1,i,1])

    # XX. Node degree
    @constraint(m, degree[s=indices.S, i=indices.V_cus, t=indices.H],
        sum(x[s,j,i,t] for j=indices.V_aug if j < i) + sum(x[s,i,j,t] for j=indices.V_aug if j > i) - 2*z[s,i,t] == 0)

    # XX. Depot outbound edges
    @constraint(m, depot_out_edges[s=indices.S, t=indices.H],
        sum(x[s,1,j,t] for j=indices.V_cus) - length(indices.K) <= 0)

    # XX. Fix vehicles number
    @constraint(m, veh_num[s=indices.S, t=indices.H],
        sum(x[s,1,j,t] for j=indices.V_cus) - sum(x[s,i,length(indices.V_aug),t] for i=indices.V_cus) == 0)

    # XX. Commodity flow variables and arc passage variables coupling
    @constraint(m, xy_coupling[s=indices.S, i=indices.V_aug, j=indices.V_aug, t=indices.H; j > i],
        y[s,i,j,t] + y[s,j,i,t] - problem.v_cap*x[s,i,j,t] == 0)

    # XX. Flow and Demand
    @constraint(m, demand_flow[s=indices.S, i=indices.V_cus, t=indices.H],
        sum(y[s,i,j,t] for j=indices.V_aug if i != j) - problem.v_cap*z[s,i,t] + q[s,i,t] == 0)

    # XX. Depot Outbound flow
    @constraint(m, depot_out_flow[s=indices.S, t=indices.H],
        sum(y[s,1,j,t] for j=indices.V_cus) - sum(q[s,i,t] for i=indices.V_cus) == 0)

    # XX. Depot Inbound  flow
    @constraint(m, copy_depot_in_flow[s=indices.S, t=indices.H],
        sum(y[s,i,length(indices.V_aug),t] for i=indices.V_cus) == 0)

    #NOTE: not necessary for optimality, but aaded to exclude 1 --> 1 and n+1 --> n+1 closed paths
    
    # XX. Clone depot outbound flow
    @constraint(m, copy_depot_out_flow[s=indices.S, t=indices.H],
        sum(y[s,length(indices.V_aug),j,t] for j=indices.V_cus) - problem.v_cap*sum(x[s,i,length(indices.V_aug),t] for i=indices.V_cus) == 0)

    # XX. Depot inbound flow
    @constraint(m, depot_in_flow[s=indices.S, t=indices.H],
        sum(y[s,i,1,t] for i=indices.V_cus) - problem.v_cap*sum(x[s,1,i,t] for i=indices.V_cus) + sum(q[s,i,t] for i in indices.V_cus) == 0)

    # Inventory Routing Constraints (Coelho, Laporte)
    # XX. Inventory conservation at customers
    @constraint(m, inv_cons[s=indices.S, i=indices.V_cus, t=indices.H],
        I[s, i, t] - I[s, i,t-1] - q[s,i,t] - l[s,i,t] + problem.demands[s][i][t] == 0)      

    # XX. Total quantity delivered to customer per time period
    @constraint(m, ml_policy[s=indices.S, i=indices.V_cus, t=indices.H],
        q[s,i,t] - problem.max_inventory[i] + I[s, i, t-1] <= 0)

    if problem.OU_policy
        @constraint(m, ou_policy1[s=indices.S, i=indices.V_cus, t=indices.H],
            q[s,i,t] - problem.max_inventory[i]*z[s,i,t] + I[s, i, t-1] >= 0)
    end

    # 4. OBJECTIVE FUNCTION
    @objective(m, Min,
        sum((1/problem.scenarios) * problem.penalty_cost[i] * l[s,i,t] for s=indices.S, i=indices.V_cus, t=indices.H) +
        sum((1/problem.scenarios) * problem.holding_cost[i] * I[s,i,t] for s=indices.S, i=indices.V_cus, t=indices.H) + # (without initial inventory)
        sum((1/problem.scenarios) * problem.distances[fix_index(i, indices)][fix_index(j, indices)] * x[s,i,j,t] for s=indices.S, i=indices.V_aug, j=indices.V_aug, t=indices.H if j > i))

    # 5. SET INITIAL INVENTORY (with bound)
    for s in indices.S, i in indices.V_cus
        fix(I[s,i,0], problem.start_inventory[i], force=true)
    end
    
    JuMP.optimize!(m)
    
    x_val = convert(AbstractArray{Int}, zeros(problem.n+1, problem.n+1, problem.horizon))
    
    for t in indices.H
        for i in indices.V_aug, j in indices.V_aug
            j > i || continue
            x_val[i,j,t] = convert(Int, round(value(x[1,i,j,t])))
        end
    end
    
    label = convert(AbstractArray{Int},zeros(size(x_val)))
    for t in 1:size(x_val, 3)
        label[:,:,t] = (transpose(x_val[:,:,t]) .+ x_val[:,:,t]) .> 0.5
    end
    
    if obj
        return label,
        sum((1/problem.scenarios) * problem.penalty_cost[i] * value(l[s,i,t]) for s=indices.S, i=indices.V_cus, t=indices.H), 
        sum((1/problem.scenarios) * problem.holding_cost[i] * value(I[s,i,t]) for s=indices.S, i=indices.V_cus, t=indices.H), 
        sum((1/problem.scenarios) * problem.distances[fix_index(i, indices)][fix_index(j, indices)] * value(x[s,i,j,t]) for s=indices.S, i=indices.V_aug, j=indices.V_aug, t=indices.H if j > i),
        Dict(i => sum((1/problem.scenarios) * value(l[s,i,t]) for s=indices.S, t=indices.H) for i=indices.V_cus)
    else 
        return label
    end
end