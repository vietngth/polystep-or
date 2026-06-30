"""
Kruskal computes a minimum spanning tree using Kruskal's algorithm.

Parameters:
- `instance::Instance`: The problem instance

Returns:
- `uf::UnionFind`: A UnionFind object representing the minimum spanning tree
"""
function run_kruskal(instance::Instance)
    g = instance.graph
    G = deepcopy(g)
    nb = instance.num_blocks
    g_edges =
        [Edge(src(i), dst(i), get_prop(G, i, :weight)) for i in edges(G) if src(i) != nb +1 &&
                                                                                         dst(
            i,
        ) != nb +1]
    sorted_edges = sort(g_edges, by = x -> x.weight)
    uf = UnionFind.UnionFinder(nb)
    for e in sorted_edges
        u, v, w = e.src, e.dst, e.weight
        if (find!(uf, u) != find!(uf, v)) &&
           (size!(uf, u) + size!(uf, v) <= instance.target_district_size)
            UnionFind.union!(uf, u, v)
        end
    end

    return uf
end


"""
findNeighborDistricts finds the neighboring districts of each district.

Parameters:
- `instance::Instance`: The problem instance
- `districts::Vector{Vector{Int}}`: A vector of vectors of nodes representing the districts

Returns:
- `neighborPairs::Dict{Tuple{Int, Int}, Int}`: A dictionary mapping a pair of neighboring districts to the number of nodes in the pair
"""
function findNeighborDistricts(instance::Instance, districts::Vector{Vector{Int}})
    neighborPairs = Dict{Tuple{Int,Int},Int}()
    g = instance.graph
    for i = 1:length(districts)
        for u in districts[i]
            for v in neighbors(g, u)
                if (v == instance.num_blocks + 1)
                    continue
                end
                found_v = false
                for j = (i+1):length(districts)
                    if v in districts[j]
                        node_sum = length(districts[i]) + length(districts[j])
                        neighborPairs[(i, j)] = get(neighborPairs, (i, j), node_sum)
                        found_v = true
                        break
                    end
                end
                if found_v
                    break
                end
            end
        end
    end

    return neighborPairs
end


"""
mergeDistricts merges two districts.

Parameters:
- `uf::UnionFinder`: A UnionFind object representing the minimum spanning tree
- `d1::Vector{Int}`: A vector of nodes representing the first district
- `d2::Vector{Int}`: A vector of nodes representing the second district
"""
function mergeDistricts(uf::UnionFinder, d1::Vector{Int}, d2::Vector{Int})
    for u in d1
        for v in d2
            UnionFind.union!(uf, u, v)
        end
    end
end


"""
groupBlocks groups the nodes in the minimum spanning tree into districts.

Parameters:
- `uf::UnionFind`: A UnionFind object representing the minimum spanning tree

Returns:
- `districts::Vector{Vector{Int}}`: A vector of vectors of nodes representing the districts
"""
function groupBlocks(uf::UnionFinder)
    cf = CompressedFinder(uf)
    groups = cf.groups
    district = Vector{Vector{Int}}(undef, groups)
    for i = 1:groups
        district[i] = Int[]
    end
    for i = 1:length(uf)
        k = find(cf, i)
        push!(district[k], i)
    end
    return district
end


"""
GreedyMerging merges districts greedily to repair the solution.

Parameters:
- `instance::Instance`: The problem instance
- `uf::UnionFind`: A UnionFind object representing the minimum spanning tree

Returns:
- `districts::Vector{Vector{Int}}`: A vector of vectors of nodes representing the districts
"""
function GreedyMerging(instance::Instance, uf::UnionFinder)
    targetNbDistricts = Int(floor(instance.num_blocks / instance.target_district_size))
    cf = CompressedFinder(uf)

    while cf.groups > targetNbDistricts
        districts = groupBlocks(uf)
        neighborPairs = findNeighborDistricts(instance, districts)
        minWeight = Inf
        mergePair = (0, 0)
        for (pair, weight) in neighborPairs
            if weight < minWeight
                minWeight = weight
                mergePair = pair
            end
        end

        mergeDistricts(uf, districts[mergePair[1]], districts[mergePair[2]])
        cf = CompressedFinder(uf)
    end

    districts = groupBlocks(uf)
    return [Vector{Int64}(subgraph) for subgraph in districts]
end

"""
    find_id_neighbors(i::Int, neighborPairs::Dict{Tuple{Int, Int}, Int})

Finds the IDs of neighboring districts.

# Arguments
- `i::Int`: The district ID.
- `neighborPairs::Dict{Tuple{Int, Int}, Int}`: A dictionary mapping pairs of neighboring districts to their ID.

# Returns
- A list of IDs for the neighboring districts.
"""

function find_id_neighbors(i::Int, neighborPairs::Dict{Tuple{Int,Int},Int})
    neighborIds = []
    for (pair, _) in neighborPairs
        if pair[1] == i
            push!(neighborIds, pair[2])
        elseif pair[2] == i
            push!(neighborIds, pair[1])
        end
    end
    return neighborIds
end

"""
    get_nodes_neighbor(instance::Instance, districts::Vector{Vector{Int}}, i::Int)

Retrieves nodes that are neighbors to a specific district.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts.
- `i::Int`: The index of the district to find neighbors for.

# Returns
- A list of nodes that are neighbors to the specified district.
"""

function get_nodes_neighbor(instance::Instance, districts::Vector{Vector{Int}}, i::Int)
    nodes = []
    for node in districts[i]
        for neighbor in neighbors(instance.graph, node)
            if neighbor in districts[i] ||
               neighbor == instance.num_blocks + 1 ||
               neighbor in nodes
                continue
            end
            push!(nodes, neighbor)
        end
    end
    return nodes
end

"""
    findSuitableNodeToAdd(instance::Instance, districts::Vector{Vector{Int}}, district_index::Int, neighborPairs::Dict{Tuple{Int, Int}, Int})

Finds a suitable node to add to a district that is under the minimum size limit.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts.
- `district_index::Int`: The index of the district needing an additional node.
- `neighborPairs::Dict{Tuple{Int, Int}, Int}`: A dictionary mapping pairs of neighboring districts to their ID.

# Returns
- The node to add and the district it belongs to, or (0, 0) if no suitable node is found.
"""

function findSuitableNodeToAdd(
    instance::Instance,
    districts::Vector{Vector{Int}},
    district_index::Int,
    neighborPairs::Dict{Tuple{Int,Int},Int},
)
    neighborIds = find_id_neighbors(district_index, neighborPairs)
    #order the neighbor districts by decreasing size
    neighborIds = sort(neighborIds, by = x -> length(districts[x]), rev = true)
    neighbor_nodes = get_nodes_neighbor(instance, districts, district_index)
    # if no suitable node is found, try with the next district
    for idx = 1:length(neighborIds)
        selected_district = neighborIds[idx]
        for node in districts[selected_district]
            if node in neighbor_nodes
                sub_graph, _ = induced_subgraph(
                    instance.graph,
                    setdiff(districts[selected_district], [node]),
                )
                if is_connected(sub_graph) &&
                   length(districts[selected_district]) > instance.min_district_size
                    return node, selected_district
                end
            end
        end
    end
    return 0, 0
end

"""
    findSuitableNodeToRemove(instance::Instance, districts::Vector{Vector{Int}}, district_index::Int, neighborPairs::Dict{Tuple{Int, Int}, Int})

Identifies a suitable node to remove from a district that is over the maximum size limit.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts.
- `district_index::Int`: The index of the district needing to shed a node.
- `neighborPairs::Dict{Tuple{Int, Int}, Int}`: A dictionary mapping pairs of neighboring districts to their ID.

# Returns
- The node to remove and the new district it should belong to, or (0, 0) if no suitable node is found.
"""

function findSuitableNodeToRemove(
    instance::Instance,
    districts::Vector{Vector{Int}},
    district_index::Int,
    neighborPairs::Dict{Tuple{Int,Int},Int},
)
    neighborIds = find_id_neighbors(district_index, neighborPairs)
    neighborIds = sort(neighborIds, by = x -> length(districts[x]))

    for idx = 1:length(neighborIds)
        selected_district = neighborIds[idx]
        neighbor_nodes = get_nodes_neighbor(instance, districts, selected_district)
        for node in districts[district_index]
            if node in neighbor_nodes
                sub_graph, _ = induced_subgraph(
                    instance.graph,
                    setdiff(districts[district_index], [node]),
                )
                if is_connected(sub_graph) &&
                   length(districts[selected_district]) < instance.max_district_size
                    return node, selected_district
                end
            end
        end
    end
    return 0, 0
end

"""
    repair_min_districts(instance::Instance, districts::Vector{Vector{Int}})

Repairs districts that do not meet the minimum size requirement by adding nodes from neighboring districts.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts to be repaired.

# Returns
- The repaired list of districts.
"""

function repair_min_districts(instance::Instance, districts::Vector{Vector{Int}})
    districts = sort(districts, by = x -> length(x))
    neighborPairs = findNeighborDistricts(instance, districts)
    for i = 1:length(districts)
        if length(districts[i]) < instance.min_district_size
            while length(districts[i]) < instance.min_district_size
                node, neighbor_district =
                    findSuitableNodeToAdd(instance, districts, i, neighborPairs)
                if node == 0
                    break
                end
                push!(districts[i], node)
                districts[neighbor_district] = setdiff(districts[neighbor_district], [node])
                neighborPairs = findNeighborDistricts(instance, districts)
            end
        end
    end
    return districts
end

"""
    repair_max_districts(instance::Instance, districts::Vector{Vector{Int}})

Repairs districts that exceed the maximum size limit by removing nodes to neighboring districts.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts to be repaired.

# Returns
- The repaired list of districts.
"""

function repair_max_districts(instance::Instance, districts::Vector{Vector{Int}})
    districts = sort(districts, by = x -> length(x), rev = true)
    neighborPairs = findNeighborDistricts(instance, districts)
    for i = 1:length(districts)
        if length(districts[i]) > instance.max_district_size
            while length(districts[i]) > instance.max_district_size
                node, new_district =
                    findSuitableNodeToRemove(instance, districts, i, neighborPairs)
                if node == 0
                    break
                end
                districts[i] = setdiff(districts[i], [node])
                push!(districts[new_district], node)
                neighborPairs = findNeighborDistricts(instance, districts)
            end
        end
    end
    return districts
end

"""
    is_node_adjacent_to_district(instance::Instance, node::Int, dnodes::Vector{Int})

True if `node` shares a graph edge with any block in district `dnodes` (depot excluded by construction).
"""
function is_node_adjacent_to_district(instance::Instance, node::Int, dnodes::Vector{Int})
    for nb in neighbors(instance.graph, node)
        if nb in dnodes
            return true
        end
    end
    return false
end

"""
    dissolve_district!(instance::Instance, districts::Vector{Vector{Int}}, i::Int)

Empties an undersized district `i` by reassigning each of its blocks into a connected neighbouring
district that still has spare capacity (`size < max_district_size`). Peeling boundary blocks first
keeps every receiving district connected. Returns `true` if district `i` was fully emptied.
"""
function dissolve_district!(instance::Instance, districts::Vector{Vector{Int}}, i::Int)
    progress = true
    while !isempty(districts[i]) && progress
        progress = false
        neighborPairs = findNeighborDistricts(instance, districts)
        nbr_ids = sort(find_id_neighbors(i, neighborPairs), by = x -> length(districts[x]))
        for node in copy(districts[i])
            placed = false
            for j in nbr_ids
                j == i && continue
                length(districts[j]) >= instance.max_district_size && continue
                if is_node_adjacent_to_district(instance, node, districts[j])
                    push!(districts[j], node)
                    districts[i] = setdiff(districts[i], [node])
                    placed = true
                    progress = true
                    break
                end
            end
            placed && break   # neighbour sizes changed -> recompute before next move
        end
    end
    return isempty(districts[i])
end

"""
    dissolve_min_districts(instance::Instance, districts::Vector{Vector{Int}})

Last-resort repair for orphan (undersized) districts that grow/shrink cannot fix — e.g. when
`2*min_district_size > max_district_size` makes merging two undersized districts impossible. Each
orphan is dissolved into connected neighbours with room, reducing the district count by one. Empty
districts are dropped. Terminates when no remaining orphan can be dissolved.
"""
function dissolve_min_districts(instance::Instance, districts::Vector{Vector{Int}})
    districts = filter(d -> !isempty(d), districts)
    changed = true
    while changed
        changed = false
        order = sort(collect(1:length(districts)), by = x -> length(districts[x]))
        for i in order
            i > length(districts) && continue
            if length(districts[i]) < instance.min_district_size
                if dissolve_district!(instance, districts, i)
                    districts = filter(d -> !isempty(d), districts)
                    changed = true
                    break
                end
            end
        end
    end
    return districts
end

"""
    repair_districts(instance::Instance, districts::Vector{Vector{Int}})

Repairs districting solutions that do not respect the district size constraints. It adjusts districts to meet both minimum and maximum size requirements.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts to be repaired.

# Returns
- The repaired list of districts.
"""

# --- Cascading chain-repair (constraint-aware): grow an undersized "orphan" district up to the
# minimum size WITHOUT dropping a district, by relocating one node along a path of adjacent
# districts from an oversized donor (size > min) down to the orphan. Each intermediate district
# gives one node and receives one (net zero, stays valid); only the donor loses one (still >= min).
# This fixes orphans even under the no-split window (2*min > max), where merging two undersized
# districts is impossible, so the deployed solution keeps the FULL target district count instead of
# being dissolved to count-1. ---

function _district_adjacency(instance::Instance, districts::Vector{Vector{Int}})
    nd = length(districts)
    adj = [Int[] for _ = 1:nd]
    neighborPairs = findNeighborDistricts(instance, districts)
    for (pair, _) in neighborPairs
        push!(adj[pair[1]], pair[2])
        push!(adj[pair[2]], pair[1])
    end
    return adj
end

# Move one boundary node from district `from` into adjacent district `to`, keeping `from` connected.
function _move_boundary_node!(instance::Instance, districts::Vector{Vector{Int}}, from::Int, to::Int)
    to_boundary = get_nodes_neighbor(instance, districts, to)   # nodes adjacent to district `to`
    for node in districts[from]
        if node in to_boundary
            sub, _ = induced_subgraph(instance.graph, setdiff(districts[from], [node]))
            if is_connected(sub)
                districts[from] = setdiff(districts[from], [node])
                push!(districts[to], node)
                return true
            end
        end
    end
    return false
end

# Grow `orphan` by one net node: BFS over the district-adjacency graph to the nearest donor with
# spare capacity (size > min), then cascade one node back along that path to the orphan.
function _cascade_grow_once!(instance::Instance, districts::Vector{Vector{Int}}, orphan::Int)
    adj = _district_adjacency(instance, districts)
    nd = length(districts)
    parent = fill(0, nd)
    visited = falses(nd)
    visited[orphan] = true
    queue = [orphan]
    donor = 0
    while !isempty(queue)
        u = popfirst!(queue)
        if u != orphan && length(districts[u]) > instance.min_district_size
            donor = u
            break
        end
        for v in adj[u]
            if !visited[v]
                visited[v] = true
                parent[v] = u
                push!(queue, v)
            end
        end
    end
    donor == 0 && return false                      # no reachable district with spare capacity
    path = Int[]                                     # donor -> ... -> orphan (via parent pointers)
    cur = donor
    while cur != 0
        push!(path, cur)
        cur = parent[cur]
    end
    for j = 1:(length(path) - 1)                     # relocate one node down the path toward orphan
        _move_boundary_node!(instance, districts, path[j], path[j + 1]) || return false
    end
    return true
end

# Repair all undersized districts via cascading chain moves (preserves district count).
function repair_min_districts_cascade(instance::Instance, districts::Vector{Vector{Int}})
    MAX_PASSES = 100 * length(districts)
    passes = 0
    changed = true
    while changed && passes < MAX_PASSES
        changed = false
        for i = 1:length(districts)
            while length(districts[i]) < instance.min_district_size && passes < MAX_PASSES
                _cascade_grow_once!(instance, districts, i) || break
                changed = true
                passes += 1
            end
        end
    end
    return districts
end


# Mirror for the MAX side: shed one node from an over-max district along a path to the nearest
# district with spare room (size < max). Fixes over-max districts whose immediate neighbours are all
# at the ceiling, where plain repair_max_districts stalls.
function _cascade_shrink_once!(instance::Instance, districts::Vector{Vector{Int}}, over::Int)
    adj = _district_adjacency(instance, districts)
    nd = length(districts)
    parent = fill(0, nd)
    visited = falses(nd)
    visited[over] = true
    queue = [over]
    receiver = 0
    while !isempty(queue)
        u = popfirst!(queue)
        if u != over && length(districts[u]) < instance.max_district_size
            receiver = u
            break
        end
        for v in adj[u]
            if !visited[v]
                visited[v] = true
                parent[v] = u
                push!(queue, v)
            end
        end
    end
    receiver == 0 && return false                    # no reachable district with spare room
    path = Int[]                                      # receiver -> ... -> over (via parent pointers)
    cur = receiver
    while cur != 0
        push!(path, cur)
        cur = parent[cur]
    end
    for j = length(path):-1:2                         # relocate one node from over toward receiver
        _move_boundary_node!(instance, districts, path[j], path[j - 1]) || return false
    end
    return true
end

function repair_max_districts_cascade(instance::Instance, districts::Vector{Vector{Int}})
    MAX_PASSES = 100 * length(districts)
    passes = 0
    changed = true
    while changed && passes < MAX_PASSES
        changed = false
        for i = 1:length(districts)
            while length(districts[i]) > instance.max_district_size && passes < MAX_PASSES
                _cascade_shrink_once!(instance, districts, i) || break
                changed = true
                passes += 1
            end
        end
    end
    return districts
end


"""
    repair_districts(instance::Instance, districts::Vector{Vector{Int}})

Repairs districting solutions that do not respect the size constraints, in escalating stages: plain
grow/shrink, then constraint-aware cascading chain-repair on BOTH sides (grow undersized / shed
oversized by relocating nodes along a path to a district with spare capacity, keeping the full
district count and working under the 2*min>max no-split window), then last-resort dissolve only if a
violation has no reachable donor/receiver.
"""
function repair_districts(instance::Instance, districts::Vector{Vector{Int}})
    get(ENV, "DN_NO_REPAIR", "0") == "1" && return districts   # [FIG] expose the raw (pre-repair) orphan
    MAX_ITER = 10
    iter = 1
    is_valid_solution = is_valid_districting_solution(instance, districts)
    while !is_valid_solution && iter <= MAX_ITER
        districts = repair_max_districts(instance, districts)
        districts = repair_min_districts(instance, districts)
        is_valid_solution = is_valid_districting_solution(instance, districts)
        iter += 1
    end
    # Constraint-aware cascading chain-repair (both sides): grow undersized and shed oversized
    # districts by relocating nodes along a path to/from a district with spare capacity, keeping the
    # FULL district count even under the 2*min>max no-split window where plain grow/shrink stalls.
    cpass = 0
    while !is_valid_districting_solution(instance, districts) && cpass < 3
        districts = repair_min_districts_cascade(instance, districts)
        districts = repair_max_districts_cascade(instance, districts)
        districts = repair_max_districts(instance, districts)
        districts = repair_min_districts(instance, districts)
        cpass += 1
    end
    # Last resort: dissolve an orphan into neighbours (drops a district) only if the cascades cannot
    # fix it (no reachable donor/receiver). Should now rarely trigger.
    if !is_valid_districting_solution(instance, districts)
        districts = dissolve_min_districts(instance, districts)
        districts = repair_max_districts(instance, districts)
    end
    return districts
end


"""
    is_valid_districting_solution(instance::Instance, districts::Vector{Vector{Int}})

Checks whether all districts are valid, i.e., they are within the maximum and minimum district size constraints.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts to be checked.

# Returns
- `True` if all districts are valid, and `False` if any district is invalid.
"""

function is_valid_districting_solution(instance::Instance, districts::Vector{Vector{Int}})
    for i = 1:length(districts)
        districtSize = length(districts[i])
        isTooLarge = districtSize > instance.max_district_size
        isTooSmall = districtSize < instance.min_district_size
        if isTooLarge || isTooSmall
            return false
        end
    end
    return true
end


"""
    initialize_solution(instance::Instance, costloader::Costloader, mode::String="CMST", model = nothing)

Creates an initial solution for the problem instance. The solution respects district size constraints and minimizes the total cost.

# Arguments
- `instance::Instance`: The problem instance.
- `costloader::Costloader`: The costloader object.
- `mode::String`: The mode of the solution (e.g., "CMST" or "Districting", etc.)

# Returns
- `solution::Solution`: The initial solution.
"""

function initialize_solution(
    instance::Instance,
    costloader::Costloader,
    mode::String = "CMST",
    model = nothing,
)
    tree = run_kruskal(instance)
    districts = GreedyMerging(instance, tree)
    districts = repair_districts(instance, districts)
    return create_solution(instance, districts, costloader, mode, model)
end
