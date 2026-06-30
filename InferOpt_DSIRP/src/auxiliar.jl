FloatFromAny(x) = Float64[i for i ∈ x]

function bit_to_array(delivery_combination, nb_cst)
    return [parse(Int, x) for x in split.(lpad(delivery_combination, nb_cst, "0"), "")]
end


function fix_index(i::Int, nb_nodes::Int)
    idx = max(1, mod(i, nb_nodes))
    return idx
end


const GRB_ENV = Ref{Gurobi.Env}()

function grb_model(; silent::Bool = true)
    if GRB_ENV[] === nothing
        GRB_ENV[] = Gurobi.Env()
    end
    model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
    silent && set_optimizer_attribute(model, "OutputFlag", 0)


    return model
end

function highs_model(; silent::Bool = true)
    
    model = Model(HiGHS.Optimizer)
    silent && set_optimizer_attribute(model, "output_flag", false)

    return model
end


function model_definition(; milp_solver::String = "gurobi", silent::Bool = true)
    if milp_solver == "gurobi"
        return () -> grb_model(silent = silent)

    elseif milp_solver == "highs"
        return () -> highs_model(silent = silent)

    else
        error("Unknown solver: $milp_solver. Use gurobi or highs")
    end
end


function keys_to_int(dict; levels=1)
    if levels == 1
        return Dict([parse(Int, string(key)) => val for (key, val) in pairs(dict)])
    else
        return Dict([parse(Int, string(key1)) => Dict([parse(Int, string(key2)) => val2 for (key2, val2) in pairs(val1)]) for (key1, val1) in pairs(dict)])
    end
end


single_g(y; kwargs...) = vec(sum(y; dims=2)[2:end-1] .> 1.5)
single_h(y; sample) = -sum([y[i,j] * 1/2 * sample.distances[fix_index(i,sample.n+1)][fix_index(j,sample.n+1)] for i in 1:sample.n+1 for j in 1:sample.n+1])
single_routing(y; distances) = -sum([y[i,j] * 1/2 * distances[fix_index(i,length(distances)+1)][fix_index(j,length(distances)+1)] for i in 1:length(distances)+1 for j in 1:length(distances)+1])