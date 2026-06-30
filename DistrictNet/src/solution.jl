"""
Solution represents a solution to a districting problem.

Fields
- `instance::Instance`: The original problem instance
- `districts::Array{District}`: Array of districts
- `cost::Float64`: Total cost of the districts
- `nb_districts::Int`: Total number of districts
- `is_feasible::Bool`: Whether the solution is feasible or not
- `blocks_district_ids :: Array{Int}`: Array of district ids for each block
"""


"""
    create_solution(instance::Instance, districts::Vector{Vector{Int64}}, costloader::Costloader, mode::String="CMST", model = nothing)::Solution

Creates a Solution object given a set of districts, each represented by a vector of node IDs.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int64}}`: A vector where each element is a vector of node IDs representing a district.
- `costloader::Costloader`: A loader for precomputed district costs.
- `mode::String`: The mode used for computing district costs (default is "CMST").
- `model`: The model used in certain cost computation modes.

# Returns
- `Solution`: The created solution object.
"""

function create_solution(
    instance::Instance,
    districts::Vector{Vector{Int64}},
    costloader::Costloader,
    mode::String = "CMST",
    model = nothing,
)::Solution
    new_districts = [
        create_district(i, nodes, instance, costloader, mode, model) for
        (i, nodes) in enumerate(districts)
    ]
    cost = compute_solution_cost(new_districts)
    is_feasible = all(district.is_feasible for district in new_districts)
    if mode == "GNN"
        blocks_district_ids =
            [get_district_id(new_districts, node) for node = 1:nv(instance.graph)]
    else
        blocks_district_ids =
            [get_district_id(new_districts, node) for node = 1:nv(instance.graph)-1]
    end
    return Solution(
        instance,
        new_districts,
        cost,
        length(districts),
        is_feasible,
        blocks_district_ids,
    )
end

"""
    compute_solution_cost(districts::Array{District})::Float64

Computes the total cost of a solution given an array of districts.

# Arguments
- `districts::Array{District}`: Array of districts in the solution.

# Returns
- `Float64`: The total cost of the solution.
"""

function compute_solution_cost(districts::Array{District})::Float64
    return sum(district.cost for district in districts)
end

"""
    compute_solution_cost(solution::Solution)::Float64

Computes the total cost of a solution from a Solution object.

# Arguments
- `solution::Solution`: The solution object.

# Returns
- `Float64`: The total cost of the solution.
"""

function compute_solution_cost(solution::Solution)::Float64
    return sum(district.cost for district in solution.districts)
end

"""
    is_solution_feasible(solution::Solution)::Bool

Determines if a given solution is feasible.

# Arguments
- `solution::Solution`: The solution to check.

# Returns
- `Bool`: `true` if the solution is feasible, `false` otherwise.
"""

function is_solution_feasible(solution::Solution)::Bool
    return all(district.is_feasible for district in solution.districts)
end

"""
    get_district_id(districts::Array{District}, node::Int)::Int

Returns the district ID for a given node.

# Arguments
- `districts::Array{District}`: Array of districts.
- `node::Int`: The node for which the district ID is sought.

# Returns
- `Int`: The ID of the district containing the node.
"""

function get_district_id(districts::Array{District}, node::Int)::Int
    for district in districts
        if node in district.nodes
            return district.id
        end
    end
    error("The node $node is not in any district")
end

"""
    get_border_nodes(solution::Solution, id1::Int, id2::Int)

Identifies border nodes between two districts in a solution. If the districts are not adjacent, an empty tuple is returned.

# Arguments
- `solution::Solution`: The solution containing the districts.
- `id1::Int`: The ID of the first district.
- `id2::Int`: The ID of the second district.

# Returns
- A tuple of sets containing border nodes of each district.
"""

function get_border_nodes(solution::Solution, id1::Int, id2::Int)
    district1, district2 = solution.districts[id1], solution.districts[id2]
    border1, border2 = Set{Int}(), Set{Int}()

    for node1 in district1.nodes
        for node2 in district2.nodes
            if has_edge(solution.instance.graph, node1, node2)
                push!(border1, node1)
                push!(border2, node2)
            end
        end
    end

    return isempty(border1) ? () : (border1, border2)
end

"""
    get_solution_edges(solution::Solution)

Retrieves edges that are part of the districts in a solution.

# Arguments
- `solution::Solution`: The solution of interest.

# Returns
- An array indicating whether each edge in the solution's graph is part of any district.
"""

function get_solution_edges(solution::Solution)
    instance, graph = solution.instance, solution.instance.graph
    g_edges = [(src(edge), dst(edge)) for edge in edges(graph)]
    edge_in_solution = zeros(Bool, length(edges(graph)))

    for district in solution.districts
        nodes = deepcopy(district.nodes)
        sub_graph, original = induced_subgraph(graph, nodes)

        for edge in prim_mst(sub_graph)
            idx = find_edge_index((original[src(edge)], original[dst(edge)]), g_edges)
            edge_in_solution[idx] = true
        end

        idx_min = argmin([
            get_prop(graph, instance.num_blocks + 1, node, :weight) for node in nodes
        ])
        idx = find_edge_index((nodes[idx_min], instance.num_blocks + 1), g_edges)
        edge_in_solution[idx] = true
    end

    return Int.(edge_in_solution)
end


find_edge_index(edge, edges) = findfirst(e -> e == edge || reverse(e) == edge, edges)

"""
    build_graph_from_edges(graph::AbstractGraph, edge_in_solution::AbstractVector{Int})

Constructs a graph from a set of edges identified as part of a solution.

# Arguments
- `graph::AbstractGraph`: The original graph.
- `edge_in_solution::AbstractVector{Int}`: Indicator array for edges in the solution.

# Returns
- A graph built from the edges in the solution.
"""

function build_graph_from_edges(graph::AbstractGraph, edge_in_solution::AbstractVector{Int})
    g = SimpleGraph(nv(graph))

    for (idx, edge) in enumerate(edges(graph))
        if edge_in_solution[idx] == 1
            add_edge!(g, edge)
        end
    end

    return g
end

"""
    get_connected_subgraphs(graph::AbstractGraph)

Identifies and returns all connected subgraphs in a graph.

# Arguments
- `graph::AbstractGraph`: The graph to analyze.

# Returns
- An array of vectors, each representing a connected subgraph.
"""

function get_connected_subgraphs(graph::AbstractGraph)
    subgraphs = []
    visited = Set()

    for node in vertices(graph)
        if node == nv(graph)
            continue
        end
        if !(node in visited)
            subgraph = Set()
            queue = [node]
            while !isempty(queue)
                current_node = popfirst!(queue)
                if !(current_node in visited)
                    visited = union(visited, [current_node])
                    subgraph = union(subgraph, [current_node])
                    neighbor = neighbors(graph, current_node)
                    neighbor = setdiff(neighbor, [nv(graph)])
                    queue = vcat(queue, neighbor)
                end
            end
            subgraphs = push!(subgraphs, collect(subgraph))
        end
    end
    return [Vector{Int64}(subgraph) for subgraph in subgraphs]
end

"""
    build_solution_from_edges(y, instance, costloader)

Constructs a solution from a binary vector representing edges in the solution.

# Arguments
- `y`: A binary vector indicating edges in the solution.
- `instance::Instance`: The problem instance.
- `costloader`: Loader for precomputed costs.

# Returns
- A `Solution` object constructed from the given edges.
"""

function build_solution_from_edges(y, instance, costloader)
    graph = build_graph_from_edges(instance.graph, y)
    subgraphs = get_connected_subgraphs(graph)
    return create_solution(instance, subgraphs, costloader, "Districting")
end
