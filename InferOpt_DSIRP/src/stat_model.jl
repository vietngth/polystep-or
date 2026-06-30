function build_stat_model(demand_quantiles::Vector{Float64}, look_ahead::Int; nb_features=0, weights=Dict{String, Any}()) 
    
    nb_observations = length(demand_quantiles) * look_ahead
    
    if (nb_features==0) | (weights==Dict{String, Any}())
        regression = Dense(zeros(1, nb_features + Int(((nb_features-1)^2+(nb_features-1))/2)), false)
    else
        regression = Dense(reduce(hcat, FloatFromAny.(weights["3"])), false)
    end
    
    cumulative = Chain(
        if nb_features > 0
            Chain(x -> x[2,:,:] + regression(x[5:end,:,:])[1,:,:], relu)
        else
            Chain(x -> x[2,:,:], relu)
        end, 
        Chain(x -> reshape(x, look_ahead, length(demand_quantiles), :)),
        Chain(x -> (ones(look_ahead) .* x[1:1,:,:]) + ((cumsum(ones(look_ahead)).-1) .* mean(x[2:end,:,:], dims=1))),
        Chain(x -> reshape(x, nb_observations, :))
    )

    single_inventory = Chain(x -> x[3,:,:] .* (x[1,:,:] - cumulative(x)), relu)
    single_penalty = Chain(x -> x[4,:,:] .* (cumulative(x) - x[1,:,:]), relu)
    
    if weights==Dict{String, Any}()
        weighted_inventory = Chain(single_inventory, Dense((-1*ones(1, nb_observations)/nb_observations), false), vec)
        weighted_penalty = Chain(single_penalty, Dense((+1*ones(1, nb_observations)/nb_observations), false), vec)
    else
        weighted_inventory = Chain(single_inventory, Dense(reduce(hcat, FloatFromAny.(weights["1"])), false), vec)
        weighted_penalty = Chain(single_penalty, Dense(reduce(hcat, FloatFromAny.(weights["2"])), false), vec)
    end

    return Parallel(+, α=weighted_inventory, β=weighted_penalty), regression
end
