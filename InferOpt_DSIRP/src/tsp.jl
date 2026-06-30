function remove_cycles!(model, x, y_val)
    
    cycles = get_cycles(value.(x), y_val)
    
    length(cycles) == round(Int, min(sum(y_val), 1))  && return false
    for cycle in cycles
        constr =  2 * length(cycle) - 2
        @constraint(model, sum(x[cycle, cycle]) <= constr)
    end
    return true
end


function tsp(θ::Vector{Int64}; distances::Dict{Int, Dict{Int, Float64}}, model_builder=grb_model, verbose=false)

    nb_nodes = length(distances)+1
    
    y_val = zeros(nb_nodes)
    y_val[2:end-1] = θ
    y_val[1] = y_val[end] = maximum(θ)
    
    model = model_builder()

    @variable(model, x[i=1:nb_nodes, j=1:nb_nodes], binary = true)
    
    @objective(
        model,
        Min,
        (sum(x[i, j] * 1/2 * distances[fix_index(i, nb_nodes)][fix_index(j, nb_nodes)] for i=1:nb_nodes for j=1:nb_nodes))
    )

    @constraint(model, [i in 1:nb_nodes], sum(x[i,:]) == 2 * y_val[i])
    @constraint(model, [i in 1:nb_nodes], x[i,i] == 0)
    @constraint(model, [i=1:nb_nodes, j=1:nb_nodes], x[i,j]==x[j,i])
    
    iter = Ref(0) 
    all_time = Ref(0.0)
    
    condition = true
    while condition
        t = @elapsed JuMP.optimize!(model)
        all_time[] += t
        status = termination_status(model)
        status == MOI.OPTIMAL || @warn("Problem status not optimal; got status $status")
        condition = remove_cycles!(model, x, y_val)
        iter[] += 1
    end
    
    return round.(Int, value.(x)), sum(value(x[i, j]) * 1/2 * distances[fix_index(i, nb_nodes)][fix_index(j, nb_nodes)] for i=1:nb_nodes for j=1:nb_nodes)
end

export tsp