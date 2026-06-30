include("subtours.jl")

function remove_cycles!(model, x, y)
    
    cycles = get_cycles(value.(x), value.(y))
    
    length(cycles) == round(Int, min(sum(value.(y)), 1))  && return false
    for cycle in cycles
        constr =  2 * length(cycle) - 2
        @constraint(model, sum(x[cycle, cycle]) <= constr)
    end
    return true
end


function pctsp(θ::Vector{Float64}; sample::IRPSample, model_builder=milp_builder)

    (; n, distances, max_inventory, start_inventory, v_cap) = sample
    
    nb_nodes = n+1
    
    model = model_builder()

    @variable(model, y[1:nb_nodes], binary = true)
    @variable(model, x[i=1:nb_nodes, j=1:nb_nodes], binary = true)
    
    @objective(
        model,
        Max,
        (sum(θ[i-1] * (y[i]) for i = 2:nb_nodes-1) + sum(x[i, j] * -1/2 * distances[fix_index(i, nb_nodes)][fix_index(j, nb_nodes)] for i=1:nb_nodes for j=1:nb_nodes))
    )

    @constraint(model, sum((max_inventory[i] - start_inventory[i]) * y[i] for i = 2:nb_nodes-1) <= v_cap)
    
    @constraint(model, nb_nodes * y[1] >= sum(y[i] for i = 2:nb_nodes-1))
    @constraint(model, y[1] <= sum(y[i] for i = 2:nb_nodes-1))
    
    @constraint(model, nb_nodes * y[nb_nodes] >= sum(y[i] for i = 2:nb_nodes-1))
    @constraint(model, y[nb_nodes] <= sum(y[i] for i = 2:nb_nodes-1))
    
    @constraint(model, [i in 1:nb_nodes], sum(x[i, :]) == 2 * y[i])
    @constraint(model, [i in 1:nb_nodes], x[i, i] == 0)
    @constraint(model, [i=1:nb_nodes, j=1:nb_nodes], x[i,j]==x[j,i])
    
    iter = Ref(0) 
    all_time = Ref(0.0)
    
    condition = true
    while condition
        t = @elapsed JuMP.optimize!(model)
        all_time[] += t
        status = termination_status(model)
        status == MOI.OPTIMAL || @warn("Problem status not optimal; got status $status")
        condition = remove_cycles!(model, x, y)
        iter[] += 1
    end
    
    return round.(Int, value.(x))
end

export pctsp