function find_cycle(perm_matrix, starting_ind=1)
    cycle = [starting_ind]
    prev_ind = ind = starting_ind
    while true
        # the comparisons `x > (0.5)` should mean `x == 1`. Due to floating point results returned
        # by the solvers, instead we sometimes have `x ≈ 1.0` instead. Since these are binary
        # values, we might as well just compare to 1/2.
        next_ind = findfirst(>(0.5), @views(perm_matrix[ind, 1:prev_ind-1]))
        if isnothing(next_ind)
            next_ind = findfirst(>(0.5), @views(perm_matrix[ind, prev_ind+1:end]))
            if isnothing(next_ind) 
                next_ind = 0
            end
            next_ind += prev_ind
        end
        next_ind == starting_ind && break
        push!(cycle, next_ind)
        prev_ind, ind = ind, next_ind
    end
    return cycle
end


function get_cycles(perm_matrix, perm_y)
    remaining_inds = Set(findall(>(0.5), convert(Array{Float64}, perm_y)))
    cycles = Vector{Int}[]
    while length(remaining_inds) > 0
        cycle = find_cycle(perm_matrix, first(remaining_inds))
        push!(cycles, cycle)
        setdiff!(remaining_inds, cycle)
    end
    return cycles
end


export get_cycles