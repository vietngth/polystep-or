"""
    enumerate_connected_subgraphs(G::AbstractGraph, k::Int, m::Int, remove_depots::Bool=true)

Generates all connected subgraphs of a graph G within a specified size range.

# Arguments
- `G::AbstractGraph`: The graph to generate subgraphs from.
- `k::Int`: The minimum number of nodes in the subgraphs.
- `m::Int`: The maximum number of nodes in the subgraphs.
- `remove_depots::Bool`: Whether to remove depot nodes from consideration (default is true).

# Returns
- A list of subgraphs, each represented as a set of node indices.
"""

function enumerate_connected_subgraphs(
    G::AbstractGraph,
    k::Int,
    m::Int,
    remove_depots::Bool = true,
)
    subgraphs = []
    non_root_nodes = collect(vertices(G))
    if remove_depots
        non_root_nodes = setdiff(vertices(G), nv(G))
    end
    for nb_nodes = k:m
        for selected_nodes in combinations(non_root_nodes, nb_nodes)
            SG, _ = induced_subgraph(G, selected_nodes)
            if is_connected(SG)
                push!(subgraphs, selected_nodes)
            end
        end
    end
    return subgraphs
end

"""
    compute_node_matrix(subgraphs, num_vertices)

Computes a binary matrix representing the presence of nodes in each subgraph.

# Arguments
- `subgraphs`: A list of subgraphs.
- `num_vertices`: The total number of vertices in the original graph.

# Returns
- A binary matrix where rows correspond to subgraphs and columns to nodes.
"""
function compute_node_matrix(subgraphs, num_vertices)
    matrix = zeros(Int, length(subgraphs), num_vertices)
    for (i, subgraph) in enumerate(subgraphs)
        for node in subgraph
            matrix[i, node] = 1
        end
    end
    return matrix
end

"""
    compute_subgraph_mst_cost(subgraphs, graph)

Calculates the minimum spanning tree (MST) cost for each subgraph.

# Arguments
- `subgraphs`: A list of subgraphs.
- `graph`: The original graph.

# Returns
- A list of MST costs for each subgraph.
"""

function compute_subgraph_mst_cost(subgraphs, graph)
    mst_costs = []
    for subgraph_nodes in subgraphs
        subgraph, _ = induced_subgraph(graph, subgraph_nodes)
        mst_edges = kruskal_mst(subgraph)
        cost = sum([get_prop(subgraph, e, :weight) for e in mst_edges])
        root_weights = [get_prop(graph, nv(graph), node, :weight) for node in subgraph_nodes]
        cost += minimum(root_weights)
        push!(mst_costs, cost)
    end
    return Float64.(mst_costs)
end

"""
    set_partitioning_optimizer(node_matrix, mst_costs, num_vertices, num_subgraphs)

Solves an optimization problem to select subgraphs that collectively cover all nodes and minimize the total MST cost.

# Arguments
- `node_matrix`: A binary matrix representing the presence of nodes in subgraphs.
- `mst_costs`: The MST costs for each subgraph.
- `num_vertices`: The total number of vertices in the original graph.
- `num_subgraphs`: The desired number of subgraphs to select.

# Returns
- The solution of the optimization problem as a binary array.
"""

# [TIMING] CMST = exact set-partitioning ILP (GLPK) over pre-enumerated subgraphs. Counters let the
# trainers report total CMST solves / time / avg (reset via SPO_CALLS[]=0; SPO_TIME[]=0.0).
const SPO_CALLS = Ref(0)
const SPO_TIME = Ref(0.0)
function set_partitioning_optimizer(node_matrix, mst_costs, num_vertices, num_subgraphs)
    SPO_CALLS[] += 1
    _t0 = time()
    try
        model = Model(GLPK.Optimizer)
        @variable(model, y[1:size(node_matrix, 1)], Bin)
        @objective(model, Min, sum(mst_costs[i] * y[i] for i = 1:size(node_matrix, 1)))

        for j = 1:num_vertices
            @constraint(
                model,
                sum(node_matrix[i, j] * y[i] for i = 1:size(node_matrix, 1)) == 1
            )
        end
        @constraint(model, sum(y) == num_subgraphs)
        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            return value.(y)
        else
            return 0
        end
    finally
        SPO_TIME[] += time() - _t0
    end
end

"""
    cmst_exact_solver(theta; kwargs...)

Solves the capacitated minimum spanning tree (CMST) problem using a given set of edge weights (theta).

# Arguments
- `theta`: The weights to assign to the edges.
- `kwargs...`: Keyword arguments including `instance` and `subgraphs`.

# Returns
- An array indicating the selected subgraphs in the solution.
"""

function cmst_exact_solver(theta; kwargs...)
    instance = kwargs[:instance]
    subgraphs = kwargs[:subgraphs]
    # Update edge weights
    weights = reshape(-theta, :)
    update_edge_weights!(instance.graph, weights)
    graph = instance.graph
    k, m, r = instance.min_district_size,
    instance.max_district_size,
    instance.num_blocks / instance.target_district_size

    # Compute matrices and costs
    node_matrix = compute_node_matrix(subgraphs, nv(graph) - 1)
    mst_costs = compute_subgraph_mst_cost(subgraphs, graph)

    # Optimization
    y_opt = set_partitioning_optimizer(node_matrix, mst_costs, nv(graph) - 1, r)
    districts = [Vector{Int64}(subgraphs[i]) for i = 1:length(y_opt) if y_opt[i] == 1]

    solution = create_solution(instance, districts, Costloader([], []))
    y = get_solution_edges(solution)
    return y
end

"""
    districting_exact_solver(instance, costloader)

Solves the districting problem for a given instance using exact methods and precomputed costs.

# Arguments
- `instance`: The problem instance.
- `costloader`: A loader for precomputed district costs.

# Returns
- A solution to the districting problem and a list of unique subgraphs, or `nothing` if no solution is found.
"""

function districting_exact_solver(instance, costloader)
    lower_bound, upper_bound, num_district = instance.min_district_size,
    instance.max_district_size,
    instance.num_blocks / instance.target_district_size
    # Compute unique subgraphs
    unique_subgraphs = enumerate_connected_subgraphs(instance.graph, lower_bound, upper_bound)
    node_matrix = compute_node_matrix(unique_subgraphs, nv(instance.graph) - 1)
    mapping = [
        [get_prop(instance.graph, j, :id) for j in unique_subgraphs[i]] for
        i = 1:length(unique_subgraphs)
    ]
    mapping_idx = [check_1d_in_2d(i, costloader.DistrictIds) for i in mapping]

    # Compute MST costs
    mst_costs = [
        mapping_idx[i] == nothing ? 1000 : costloader.Cost[mapping_idx[i]] for
        i = 1:length(mapping_idx)
    ]

    y_opt = set_partitioning_optimizer(node_matrix, mst_costs, nv(instance.graph) - 1, num_district)

    if y_opt == 0
        println("No solution found ",instance.city_name)
        return nothing, nothing
    else
        districts =[Vector{Int64}(unique_subgraphs[i]) for i = 1:length(y_opt) if y_opt[i] == 1]
        solution = create_solution(instance, districts, costloader, "Districting")
        return solution, unique_subgraphs
    end
end

function cmst_exact_solver_twostage(instance, subgraphs)
    graph = instance.graph
    k, m, r = instance.min_district_size,
    instance.max_district_size,
    instance.num_blocks / instance.target_district_size

    # Compute matrices and costs
    node_matrix = compute_node_matrix(subgraphs, nv(graph) - 1)
    mst_costs = compute_subgraph_mst_cost(subgraphs, graph)

    # Optimization
    y_opt = set_partitioning_optimizer(node_matrix, mst_costs, nv(graph) - 1, r)
    districts = [Vector{Int64}(subgraphs[i]) for i = 1:length(y_opt) if y_opt[i] == 1]

    solution = create_solution(instance, districts, Costloader([], []))
    y = get_solution_edges(solution)
    return y, solution
end


"""
    Exact_solve_instance(instance, costloader, solver_type, params)

Solves an instance of the districting problem using an exact method. It generates all connected subgraphs, computes costs, and solves an optimization problem to select the best subgraphs.

# Arguments
- `instance`: The problem instance.
- `costloader`: The costloader containing precomputed costs.
- `solver_type`: The type of solver to use (e.g., "CMST" or "GNN").
- `params`: Parameters for the solver.

# Returns
- `solution`: The solution found by the exact method.
"""

function Exact_solve_instance(instance, costloader, solver_type, params)
    update_edge_weights!(
        instance.graph,
        solver_type == "CMST" ? params : rand(ne(instance.graph)),
    )
    graph = instance.graph
    lb, ub, k = instance.min_district_size,
    instance.max_district_size,
    instance.num_blocks / instance.target_district_size
    subgraphs = enumerate_connected_subgraphs(graph, lb, ub, solver_type == "GNN" ? false : true)
    node_matrix = compute_node_matrix(subgraphs, nv(graph) - (solver_type == "GNN" ? 0 : 1))


    districts = []
    for i = 1:length(subgraphs)
        d = District(i, subgraphs[i], 0.0, true)
        push!(districts, d)
    end
    district_cost = [
        compute_district_cost(instance, d, costloader, solver_type, params) for
        d in districts
    ]

    y_opt = set_partitioning_optimizer(
        node_matrix,
        district_cost,
        nv(graph) - (solver_type == "GNN" ? 0 : 1),
        k,
    )
    districts = [Vector{Int64}(subgraphs[i]) for i = 1:length(y_opt) if y_opt[i] == 1]
    solution = create_solution(instance, districts, costloader, solver_type, params)

    return solution
end
