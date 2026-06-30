"""
    extract_edge_feature(g::MetaGraph)

Extracts edge features from a MetaGraph `g`, filtering out certain properties and calculating additional features.

# Arguments
- `g::MetaGraph`: The graph to extract features from.

# Returns
- A list of vectors, each representing the features of an edge in the graph.
"""

function extract_edge_feature(g::MetaGraph)
    function get_filtered_properties(dict)
        return [val for (key, val) in dict if key ∉ [:id, :centre]]
    end
    edge_data = [
        begin
            prop_src, prop_dst = props(g, src(e)), props(g, dst(e))
            vector_src, vector_dst =
                get_filtered_properties(prop_src), get_filtered_properties(prop_dst)
            dist = euclidean_distance(prop_src[:centre], prop_dst[:centre])
            vcat(0.5 * (vector_src + vector_dst), dist)
        end for e in edges(g)
    ]

    return [Vector{Float32}(feature) for feature in edge_data]
end



"""
    get_instance_features(instance)

Extracts edge features from the graph of an `Instance` object.

# Arguments
- `instance`: The problem instance.

# Returns
- A matrix of edge features.
"""

function get_instance_features(instance)
    g = instance.graph
    edges_data = extract_edge_feature(g)
    return hcat(edges_data...)
end

"""
    get_edge_id(g, u, v)

Retrieves the edge ID for an edge in a graph `g` between nodes `u` and `v`.

# Arguments
- `g`: The graph.
- `u`: The source node.
- `v`: The destination node.

# Returns
- The edge ID if found, otherwise returns `nothing`.
"""

function get_edge_id(g, u, v)
    g_edges = [(src(i), dst(i)) for i in edges(g)]
    idx = findfirst(x -> x == (u, v) || x == (v, u), g_edges)
    return idx
end

"""
    create_edge_graph(g, node_features)

Creates an edge graph for a given graph `g` with node features. This is used for preparing data for GNN models.

# Arguments
- `g`: The original graph.
- `node_features`: Node features for the graph.

# Returns
- A `GNNGraph` object representing the edge graph with node features.
"""

function create_edge_graph(g, node_features)
    # Convert node features to Float32
    node_features_float32 = Float32.(node_features)
    # Compute edge graph adjacency matrix
    nb_edges = ne(g)
    adj = falses(nb_edges, nb_edges)

    for edge in edges(g)
        u1, u2 = src(edge), dst(edge)
        idx = get_edge_id(g, u1, u2)

        for u in (u1, u2)
            neighbors_u = neighbors(g, u)
            idx2 = filter(x -> x != idx, [get_edge_id(g, u, v) for v in neighbors_u]) # Filter out self-loops
            adj[idx, idx2] .= true
            adj[idx2, idx] .= true
        end
    end
    # Create edge graph and GNN graph objects
    edge_graph = SimpleGraph(adj)
    return GNNGraph(edge_graph; ndata = node_features_float32)
end

"""
    normalize_features(data, MEAN=nothing, STD=nothing)

Normalizes the features in a dataset of graph instances. Optionally takes mean and standard deviation for normalization.

# Arguments
- `data`: The data to normalize.
- `MEAN`: Optional; the mean value to use for normalization.
- `STD`: Optional; the standard deviation to use for normalization.

# Returns
- The normalized data, along with the mean and standard deviation used for normalization.
"""

function normalize_features(data, MEAN = nothing, STD = nothing)
    data_features = [x[1].feature for x in data]
    data_features = hcat(data_features...)
    if MEAN == nothing && STD == nothing
        MEAN = mean(data_features, dims = 2)*0
        STD = std(data_features, dims = 2)
    end

    for i = 1:length(data)
        data[i][1].feature = (data[i][1].feature .- MEAN) ./ STD
        data[i][1].gnn_graph.ndata.x = Float32.(data[i][1].feature)
    end
    data_features = nothing
    GC.gc()
    return data, MEAN, STD
end



"""
    load_precomputed_costs(path::String)

Loads precomputed cost data from a file.

# Arguments
- `path::String`: The file path to load cost data from.

# Returns
- `Costloader`: An object containing lists of districts and their associated costs.
"""

function load_precomputed_costs(path::String)
    data = open(path) do file
        JSON.parse(file)
    end
    raw_districts = data["districts"]
    districts_lists = []
    district_costs = []
    for i = 1:length(raw_districts)
        block_list = raw_districts[i]["list-blocks"] .+ 1
        push!(districts_lists, block_list)
        push!(district_costs, raw_districts[i]["average-cost"])
    end
    districts_lists = [Vector{Int64}(d) for d in districts_lists]
    district_costs = Float64.(district_costs)
    costloader = Costloader(districts_lists, district_costs)
    #clear memory
    data = nothing
    raw_districts = nothing
    districts_lists = nothing
    district_costs = nothing

    return costloader
end

"""
    are_vectors_equal(vec1::Vector{Int}, vec2::Vector{Int})::Bool

Checks if two vectors are equal (having the same elements regardless of order).

# Arguments
- `vec1`: The first vector.
- `vec2`: The second vector.

# Returns
- `true` if vectors are equal, otherwise `false`.
"""

function are_vectors_equal(vec1::Vector{Int}, vec2::Vector{Int})::Bool
    return length(vec1) == length(vec2) && all(sort(vec1) .== sort(vec2))
end

"""
    check_1d_in_2d(lst1d, lst2d)

Checks if a 1D list is present in a 2D list.

# Arguments
- `lst1d`: The 1D list to check.
- `lst2d`: The 2D list to check against.

# Returns
- The first index at which `lst1d` appears in `lst2d`, or `nothing` if not found.
"""

function check_1d_in_2d(lst1d, lst2d)
    return findfirst(vec -> are_vectors_equal(lst1d, vec), lst2d)
end

"""
    map_subvector(vect1, vect2, subvect1)

Maps elements from `subvect1` (subset of `vect1`) to their corresponding elements in `vect2`.

# Arguments
- `vect1`: The reference vector.
- `vect2`: The target vector with corresponding elements to `vect1`.
- `subvect1`: A subset of `vect1`.

# Returns
- A subset of `vect2` corresponding to the elements in `subvect1`.
"""

function map_subvector(vect1, vect2, subvect1)
    if length(vect1) != length(vect2)
        error("The lengths of vect1 and vect2 must be equal.")
    end

    indices = findall(x -> x in subvect1, vect1)
    subvect2 = vect2[indices]

    return subvect2
end

"""
    euclidean_distance(p1, p2)

Computes the Euclidean distance between two points.

# Arguments
- `p1`: The first point.
- `p2`: The second point.

# Returns
- The Euclidean distance between `p1` and `p2`.
"""

function euclidean_distance(p1, p2)
    return sqrt(sum((p1[i] - p2[i])^2 for i = 1:length(p1))) * 1000
end

"""
    remove_districts(instance, costloader)

Removes districts from the costloader that do not meet the instance's minimum district size.

# Arguments
- `instance`: The problem instance.
- `costloader`: The costloader containing district IDs and costs.

# Returns
- The updated costloader with districts smaller than the minimum size removed.
"""

function remove_districts(instance, costloader)
    districts = []
    cost = []
    for (idx, d) in enumerate(costloader.DistrictIds)
        if (length(d) < instance.min_district_size)
            continue
        end
        push!(districts, d)
        push!(cost, costloader.Cost[idx])
    end
    costloader.DistrictIds = districts
    costloader.Cost = cost
    return costloader
end

"""
    truncate_dataset(costloader, max_size)

Truncates the dataset in the costloader to a specified maximum size.

# Arguments
- `costloader`: The costloader containing district IDs and costs.
- `max_size`: The maximum size of the dataset.

# Returns
- The truncated costloader.
"""

function truncate_dataset(costloader, max_size)
    Districtids = costloader.DistrictIds
    Costs = costloader.Cost
    paired_data = shuffle!(collect(zip(Districtids, Costs)))
    Districtids, Costs = unzip(paired_data)
    Districtids = Districtids[1:max_size]
    Costs = Costs[1:max_size]
    costloader.DistrictIds = Districtids
    costloader.Cost = Costs
    return costloader
end
