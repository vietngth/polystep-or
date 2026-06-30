"""
    is_valid_move(instance::Instance, solution::Solution, node::Int, current_district::District, target_district::District)::Bool

Checks if moving a node from its current district to a target district is a valid move according to the constraints.

# Arguments
- `instance::Instance`: The problem instance.
- `solution::Solution`: The current solution.
- `node::Int`: The node to be moved.
- `current_district::District`: The district from which the node is being moved.
- `target_district::District`: The district to which the node is being moved.

# Returns
- `Bool`: `true` if the move is valid, `false` otherwise.
"""

function is_valid_move(
    instance::Instance,
    solution::Solution,
    node::Int,
    current_district::District,
    target_district::District,
)::Bool
    if node in target_district.nodes
        return false
    end

    if length(target_district.nodes) == instance.max_district_size
        return false
    end

    if !is_subgraph_connected(instance.graph, [node; target_district.nodes])
        return false
    end

    if length(current_district.nodes) > 1 &&
       !is_subgraph_connected(instance.graph, setdiff(current_district.nodes, [node]))
        return false
    end

    desired_number =
        floor(solution.instance.num_blocks / solution.instance.target_district_size)

    if solution.nb_districts <= desired_number &&
       length(current_district.nodes) == instance.min_district_size
        return false
    end

    if length(current_district.nodes) == 1 &&
       length(solution.districts) - 1 < desired_number
        return false
    end

    return true
end

"""
    apply_move!(solution::Solution, node::Int, current_district::District, target_district::District, costloader::Costloader, mode::String, model = nothing)

Applies a move by transferring a node from its current district to a target district, updating the districts and solution costs accordingly.

# Arguments
- `solution::Solution`: The current solution.
- `node::Int`: The node to be moved.
- `current_district::District`: The district from which the node is being moved.
- `target_district::District`: The district to which the node is being moved.
- `costloader::Costloader`: The loader containing precomputed district costs.
- `mode::String`: The mode of operation.
- `model`: An optional model used for cost computation.

# Effects
Modifies the solution by applying the move and updating costs.
"""

function apply_move!(
    solution::Solution,
    node::Int,
    current_district::District,
    target_district::District,
    costloader::Costloader,
    mode::String,
    model = nothing,
)
    deleteat!(current_district.nodes, findfirst(x -> x == node, current_district.nodes))
    push!(target_district.nodes, node)
    current_district.cost =
        compute_district_cost(solution.instance, current_district, costloader, mode, model)
    target_district.cost =
        compute_district_cost(solution.instance, target_district, costloader, mode, model)
    solution.cost = compute_solution_cost(solution)
end

"""
    revert_move!(solution::Solution, node::Int, current_district::District, target_district::District, idCurrentDistrict::Int, costloader::Costloader, mode::String, model = nothing)

Reverts a previously applied move, restoring the node to its original district.

# Arguments
- `solution::Solution`: The current solution.
- `node::Int`: The node to be reverted.
- `current_district::District`: The district to which the node is being reverted.
- `target_district::District`: The district from which the node is being moved back.
- `idCurrentDistrict::Int`: The ID of the current district.
- `costloader::Costloader`: The loader containing precomputed district costs.
- `mode::String`: The mode of operation.
- `model`: An optional model used for cost computation.

# Effects
Modifies the solution by reverting the move and updating costs.
"""

function revert_move!(
    solution::Solution,
    node::Int,
    current_district::District,
    target_district::District,
    idCurrentDistrict::Int,
    costloader::Costloader,
    mode::String,
    model = nothing,
)
    deleteat!(target_district.nodes, findfirst(x -> x == node, target_district.nodes))
    push!(current_district.nodes, node)

    current_district.cost =
        compute_district_cost(solution.instance, current_district, costloader, mode, model)
    target_district.cost =
        compute_district_cost(solution.instance, target_district, costloader, mode, model)
    solution.cost = compute_solution_cost(solution)
    solution.blocks_district_ids[node] = idCurrentDistrict
end

"""
    relocate(solution::Solution, node::Int, idTargetDistrict::Int, costloader::Costloader, mode::String="CMST", PERTURBATION_PROBABILITY::Bool=false, model = nothing)

Relocates a node from its current district to a target district if the move is valid.

# Arguments
- `solution::Solution`: The current solution.
- `node::Int`: The node to be relocated.
- `idTargetDistrict::Int`: The ID of the target district.
- `costloader::Costloader`: The loader containing precomputed district costs.
- `mode::String`: The mode of operation.
- `PERTURBATION_PROBABILITY::Bool`: The probability of applying the perturbation.
- `model`: An optional model used for cost computation.

# Returns
- `Bool`: `true` if the relocation is successful, `false` otherwise.
"""

function relocate(
    solution::Solution,
    node::Int,
    idTargetDistrict::Int,
    costloader::Costloader,
    mode::String = "CMST",
    PERTURBATION_PROBABILITY::Bool = false,
    model = nothing,
)
    idCurrentDistrict = solution.blocks_district_ids[node]
    current_district = solution.districts[idCurrentDistrict]
    target_district = solution.districts[idTargetDistrict]

    return if !is_valid_move(
        solution.instance,
        solution,
        node,
        current_district,
        target_district,
    )
        false
    else
        old_cost = solution.cost
        apply_move!(
            solution,
            node,
            current_district,
            target_district,
            costloader,
            mode,
            model,
        )
        if solution.cost >= old_cost && !PERTURBATION_PROBABILITY
            revert_move!(
                solution,
                node,
                current_district,
                target_district,
                idCurrentDistrict,
                costloader,
                mode,
                model,
            )
            false
        else
            solution.blocks_district_ids[node] = idTargetDistrict
            if isempty(current_district.nodes)
                deleteat!(solution.districts, idCurrentDistrict)
                for i in eachindex(solution.blocks_district_ids)
                    if solution.blocks_district_ids[i] > idCurrentDistrict
                        solution.blocks_district_ids[i] -= 1
                    end
                end
            end
            solution.nb_districts = length(solution.districts)
            solution.is_feasible = is_solution_feasible(solution)
            true
        end
    end
end

"""
    swap(solution::Solution, node1::Int, node2::Int, costloader::Costloader, mode::String="CMST", PERTURBATION_PROBABILITY::Bool=false, model = nothing)

Swaps two nodes between their respective districts if the move is valid.

# Arguments
- `solution::Solution`: The current solution.
- `node1::Int`: The first node to be swapped.
- `node2::Int`: The second node to be swapped.
- `costloader::Costloader`: The loader containing precomputed district costs.
- `mode::String`: The mode of operation.
- `PERTURBATION_PROBABILITY::Bool`: The probability of applying the perturbation.
- `model`: An optional model used for cost computation.

# Returns
- `Bool`: `true` if the swap is successful, `false` otherwise.
"""

function swap(
    solution::Solution,
    node1::Int,
    node2::Int,
    costloader::Costloader,
    mode::String = "CMST",
    PERTURBATION_PROBABILITY::Bool = false,
    model = nothing,
)
    # get the instance
    instance = solution.instance
    # get the current district of the node
    idCurrentDistrict1 = solution.blocks_district_ids[node1]
    idCurrentDistrict2 = solution.blocks_district_ids[node2]
    # get the current district
    current_district1 = solution.districts[idCurrentDistrict1]
    current_district2 = solution.districts[idCurrentDistrict2]
    # check if the nodes are in the same district
    if idCurrentDistrict1 == idCurrentDistrict2
        return false
    end
    # Check if the districts remain connected after the swap
    if !(
        is_subgraph_connected(
            instance.graph,
            [node2; setdiff(current_district1.nodes, [node1])],
        ) && is_subgraph_connected(
            instance.graph,
            [node1; setdiff(current_district2.nodes, [node2])],
        )
    )
        #println("Districts are not connected after swap.")
        return false
    end
    # get a copy of the cost of the solution before the relocation
    old_cost = deepcopy(solution.cost)
    # remove the node from the current district
    deleteat!(current_district1.nodes, findall(x -> x == node1, current_district1.nodes))
    deleteat!(current_district2.nodes, findall(x -> x == node2, current_district2.nodes))

    # add the node to the target district
    push!(current_district1.nodes, node2)
    push!(current_district2.nodes, node1)
    # update the feasibility of the districts
    current_district1.is_feasible = is_district_feasible(instance, current_district1)
    current_district2.is_feasible = is_district_feasible(instance, current_district2)
    # update the cost of the districts
    current_district1.cost =
        compute_district_cost(instance, current_district1, costloader, mode, model)
    current_district2.cost =
        compute_district_cost(instance, current_district2, costloader, mode, model)

    # update the cost of the solution
    solution.cost = compute_solution_cost(solution)
    #check if the cost of the solution has improved
    if (solution.cost >= old_cost && !(PERTURBATION_PROBABILITY))
        # revert the relocation
        # remove the node from the target district
        deleteat!(
            current_district1.nodes,
            findall(x -> x == node2, current_district1.nodes),
        )
        deleteat!(
            current_district2.nodes,
            findall(x -> x == node1, current_district2.nodes),
        )

        # add the node to the current district
        push!(current_district1.nodes, node1)
        push!(current_district2.nodes, node2)

        # update the feasibility of the districts
        current_district1.is_feasible = is_district_feasible(instance, current_district1)
        current_district2.is_feasible = is_district_feasible(instance, current_district2)
        # update the cost of the districts
        current_district1.cost =
            compute_district_cost(instance, current_district1, costloader, mode, model)
        current_district2.cost =
            compute_district_cost(instance, current_district2, costloader, mode, model)
        # update the cost of the solution
        solution.cost = compute_solution_cost(solution)
        # update the district id of the node
        solution.blocks_district_ids[node1] = idCurrentDistrict1
        solution.blocks_district_ids[node2] = idCurrentDistrict2
        # update the number of districts
        solution.nb_districts = length(solution.districts)
        # update the feasibility of the solution
        solution.is_feasible = is_solution_feasible(solution)
        return false

    else
        # update the district id of the node
        solution.blocks_district_ids[node1] = idCurrentDistrict2
        solution.blocks_district_ids[node2] = idCurrentDistrict1
        # update the number of districts
        solution.nb_districts = length(solution.districts)
        # update the feasibility of the solution
        solution.is_feasible = is_solution_feasible(solution)
        return true
    end

end


"""
    localSearch(init_solution::Solution, costloader::Costloader, start_time, mode::String="CMST", model = nothing)

Conducts local search  on a solution, applying relocation and swap moves to improve the solution.

# Arguments
- `init_solution::Solution`: The initial solution.
- `costloader::Costloader`: The loader containing precomputed district costs.
- `start_time`: The start time of the search for time tracking.
- `mode::String`: The mode of operation.
- `model`: An optional model used for cost computation.

# Returns
- The improved solution after a single local search iteration.
"""

function localSearch(
    init_solution::Solution,
    costloader::Costloader,
    start_time::Float64,
    mode::String = "CMST",
    model = nothing,
)

    change = true
    currentSolution = deepcopy(init_solution)
    while change
        if (time() - start_time) > MAX_TIME
            return currentSolution
        end
        change = false
        random_district_order = randperm(currentSolution.nb_districts)
        for i in random_district_order
            for j in random_district_order
                if (time() - start_time) > MAX_TIME
                    return currentSolution
                end
                if (i <= j)
                    continue
                end
                best_move = Dict()
                best_cost = currentSolution.cost
                border_nodes = get_border_nodes(currentSolution, i, j)
                if length(border_nodes) > 0
                    border1, border2 = border_nodes
                    for node1 in border1
                        if (time() - start_time) > MAX_TIME
                            break
                        end
                        solution = deepcopy(currentSolution)
                        relocate(solution, node1, j, costloader, mode, false, model)
                        if solution.cost < best_cost
                            best_move =
                                Dict(:node => node1, :district => j, :type => "relocate")
                            best_cost = solution.cost
                        end
                        for node2 in border2
                            if (time() - start_time) > MAX_TIME
                                break
                            end
                            solution = deepcopy(currentSolution)
                            swap(solution, node1, node2, costloader, mode, false, model)
                            if solution.cost < best_cost
                                best_move =
                                    Dict(:node1 => node1, :node2 => node2, :type => "swap")
                                best_cost = solution.cost
                            end
                        end
                    end
                    for node2 in border2
                        if (time() - start_time) > MAX_TIME
                            break
                        end
                        solution = deepcopy(currentSolution)
                        relocate(solution, node2, i, costloader, mode, false, model)
                        if solution.cost < best_cost
                            best_move =
                                Dict(:node => node2, :district => i, :type => "relocate")
                            best_cost = solution.cost
                        end
                    end
                end
                if best_cost < currentSolution.cost
                    if best_move[:type] == "relocate"
                        relocate(
                            currentSolution,
                            best_move[:node],
                            best_move[:district],
                            costloader,
                            mode,
                            false,
                            model,
                        )
                    elseif best_move[:type] == "swap"
                        swap(
                            currentSolution,
                            best_move[:node1],
                            best_move[:node2],
                            costloader,
                            mode,
                            false,
                            model,
                        )
                    end
                    change = true
                end
            end
        end
    end
    return currentSolution
end

"""
    perturbation(solution::Solution, costloader::Costloader, mode::String="CMST", model = nothing)

Applies perturbation moves to a solution to escape local optima. It randomly relocates and swaps nodes between districts.

# Arguments
- `solution::Solution`: The current solution.
- `costloader::Costloader`: The loader containing precomputed district costs.
- `mode::String`: The mode of operation.
- `model`: An optional model used for cost computation.

# Effects
Randomly modifies the solution to explore new configurations.
"""

function perturbation(
    solution::Solution,
    costloader::Costloader,
    mode::String = "CMST",
    model = nothing,
)
    # Create a random permutation of district indices
    random_district_order = randperm(solution.nb_districts)
    for i in random_district_order
        # Create a random permutation of district indices for the inner loop
        random_district_order_inner = randperm(solution.nb_districts)
        for j in random_district_order_inner
            if (i <= j)
                continue
            end
            border_nodes = get_border_nodes(solution, i, j)
            if length(border_nodes) > 0
                border1, border2 = border_nodes
                @views for node1 in border1
                    if (rand() > PERTURBATION_PROBABILITY)
                        relocate(solution, node1, j, costloader, mode, true, model)
                    end
                    @views for node2 in border2
                        if (rand() > PERTURBATION_PROBABILITY)
                            swap(solution, node1, node2, costloader, mode, true, model)
                        end
                    end
                end
                @views for node2 in border2
                    if (rand() > PERTURBATION_PROBABILITY)
                        relocate(solution, node2, i, costloader, mode, true, model)
                    end
                end
            end
        end
    end
end

"""
    Iterated_local_search(solution::Solution, costloader::Costloader, mode::String="CMST", model = nothing)

Implements the Iterated Local Search (ILS) algorithm to improve a given solution. It combines local search with perturbations.

# Arguments
- `solution::Solution`: The initial solution.
- `costloader::Costloader`: The loader containing precomputed district costs.
- `mode::String`: The mode of operation.
- `model`: An optional model used for cost computation.

# Returns
- The improved solution after applying the ILS algorithm.
"""

function Iterated_local_search(
    solution::Solution,
    costloader::Costloader,
    mode::String = "CMST",
    model = nothing,
)
    best_solution = deepcopy(solution)


    start_time = time()
    iters = 0
    solution = localSearch(solution, costloader, start_time, mode, model)
    while (time() - start_time) < MAX_TIME
        perturbation(solution, costloader, mode, model)
        solution = localSearch(solution, costloader, start_time, mode, model)
        if solution.cost < best_solution.cost
            best_solution = deepcopy(solution)
            iters = 0
        else
            iters += 1
        end
    end
    return best_solution
end

"""
    ILS_solve_instance(instance, costloader, solver_type, params)

Solves an instance of the districting problem using an Iterated Local Search (ILS) algorithm. It repeatedly tries to find a feasible solution by updating edge weights and applying ILS.

# Arguments
- `instance`: The problem instance.
- `costloader`: The costloader containing precomputed costs.
- `solver_type`: The type of solver to use (e.g., "CMST").
- `params`: Parameters for the solver.

# Returns
- `solution`: The solution found by the ILS algorithm.
"""

function ILS_solve_instance(instance, costloader, solver_type, params)
    update_edge_weights!(
        instance.graph,
        solver_type == "CMST" ? params : rand(ne(instance.graph)),
    )
    solution = initialize_solution(instance, costloader, solver_type, params)
    println("Initial solution feasibility before while loop: ", solution.is_feasible)
    max_try = 10
    while !solution.is_feasible && solver_type != "CMST" && max_try > 0
        W = rand(ne(instance.graph))
        update_edge_weights!(instance.graph, W)
        solution = initialize_solution(instance, costloader, solver_type, params)
    end
    println(
        "Initial solution feasibility after while loop: ",
        solution.is_feasible,
        " with cost: ",
        solution.cost,
    )
    println("max try: ", max_try)
    solution = Iterated_local_search(solution, costloader, solver_type, params)
    return solution
end
