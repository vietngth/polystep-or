"""
    Instance

Representation of the instance problem with fields for the city's metadata and graph.
"""

"""
    parse_json(file_path::String)::Dict

Parse a JSON file from the specified `file_path`. Returns a dictionary containing the parsed data.

# Arguments
- `file_path::String`: The path to the JSON file to be parsed.

# Returns
- `Dict`: A dictionary containing the parsed JSON data.

# Errors
Throws an error if the file cannot be parsed, with details about the file path and the error encountered.
"""

function parse_json(file_path::String)::Dict
    try
        return JSON.parsefile(file_path)
    catch e
        error("Failed to parse file at $file_path. Error: $e")
    end
end

"""
    set_block_properties!(G::MetaGraph, props::Dict, idx::Int, jsondata::Dict)

Set properties for a block in the graph `G`. Calculates additional properties like distance from a reference point and compactness score.

# Arguments
- `G::MetaGraph`: The graph to which the properties are added.
- `props::Dict`: A dictionary of properties for the block.
- `idx::Int`: The index of the block in the graph.
- `jsondata::Dict`: Additional data required for property calculation.

# Notes
Adds the calculated properties directly to the graph `G` at the specified index `idx`.
"""

function set_block_properties!(G::MetaGraph, props::Dict, idx::Int, jsondata::Dict, depot_corr::Array{Float64,1})
    #longlat_ref = jsondata["metadata"]["REFERENCE_LONGLAT"]
    centre = props["centre"]
    distance = norm(depot_corr - centre) * 1000
    compactness = calculate_polsby_popper_score(props["AREA"], props["PERIMETER"])
    sw_compact = calculate_schwartzberg_compactness(props["AREA"], props["PERIMETER"])

    block_properties = Dict(
        :id => props["ID"] + 1,
        :population => props["POPULATION"],
        :area => props["AREA"],
        :density => props["DENSITY"],
        :perimeter => props["PERIMETER"],
        :compactness => compactness,
        :centre => centre,
        :distDepot => distance,
    )

    set_props!(G, idx, block_properties)
end

"""
    add_edges!(G::MetaGraph, props::Dict, idx::Int, num_blocks::Int)

Adds edges to the graph `G` based on adjacency information in `props`.

# Arguments
- `G::MetaGraph`: The graph to which edges are added.
- `props::Dict`: A dictionary containing adjacency information.
- `idx::Int`: The index of the current block.
- `num_blocks::Int`: Total number of blocks in the graph.

# Notes
Only adds edges between valid blocks within the graph `G`.
"""

function add_edges!(G::MetaGraph, props::Dict, idx::Int, num_blocks::Int)
    for adjacent in props["LIST_ADJACENT"]
        adjacent <= num_blocks && add_edge!(G, idx, adjacent + 1)
    end
end

"""
    calculate_polsby_popper_score(area::Float64, perimeter::Float64)::Float64

Calculate the Polsby-Popper score, a measure of compactness, for a given area and perimeter.

# Arguments
- `area::Float64`: The area of the block or district.
- `perimeter::Float64`: The perimeter of the block or district.

# Returns
- `Float64`: The Polsby-Popper score.
"""

function calculate_polsby_popper_score(area::Float64, perimeter::Float64)::Float64
    return 4 * π * area / perimeter^2
end
function calculate_schwartzberg_compactness(area::Float64, perimeter::Float64)::Float64
    return perimeter / sqrt(4 * π * area)
end

"""
    calculate_bounding_box(features)

Calculate the bounding box for all coordinates in the given features.

# Arguments
- `features`: The features for which the bounding box is calculated.

# Returns
- A tuple containing the minimum and maximum latitude and longitude values.

"""


function calculate_bounding_box(features)
    min_lat, min_lon = Inf, Inf
    max_lat, max_lon = -Inf, -Inf

    for feature in features
        for polygon in feature["geometry"]["coordinates"]
            # check if polygon is multi-polygon
            if length(polygon) == 1
                polygon = polygon[1]
            end
            for point in polygon
                lon, lat = point
                min_lat = min(min_lat, lat)
                min_lon = min(min_lon, lon)
                max_lat = max(max_lat, lat)
                max_lon = max(max_lon, lon)
            end
        end
    end

    return (min_lat, min_lon, max_lat, max_lon)
end

"""
    get_depot_location(corners::Tuple{Float64, Float64, Float64, Float64}, depot_location::String)::Tuple{Float64, Float64}

Get the location of the depot based on the corners of the bounding box and the specified location.

# Arguments
- `corners::Tuple{Float64, Float64, Float64, Float64}`: The corners of the bounding box.
- `depot_location::String`: The location of the depot.

# Returns
- A tuple containing the latitude and longitude of the depot.

"""

function get_depot_location(
    corners::Tuple{Float64,Float64,Float64,Float64},
    depot_location::String,
)::Tuple{Float64,Float64}
    min_lat, min_lon, max_lat, max_lon = corners
    if depot_location == "C"
        return ((min_lon + max_lon) / 2, (min_lat + max_lat) / 2)
    elseif depot_location == "NW"
        return (min_lon, max_lat)
    elseif depot_location == "NE"
        return (max_lon, max_lat)
    elseif depot_location == "SW"
        return (min_lon, min_lat)
    elseif depot_location == "SE"
        return (max_lon, min_lat)
    else
        error("Invalid depot location: $depot_location")
    end
end



"""
    create_city_graph(city_name::String, num_blocks::Int)::MetaGraph

Create a MetaGraph representing a city layout. Reads city data, sets block properties, and adds edges and a depot to the graph.

# Arguments
- `city_name::String`: The name of the city.
- `num_blocks::Int`: The number of blocks in the city.

# Returns
- `MetaGraph`: The constructed city graph.
"""

function create_city_graph(city_name::String, num_blocks::Int, depot_location::String)::MetaGraph
    G = MetaGraph(num_blocks + 1)
    file_path = joinpath("data/geojson", "$city_name.geojson")
    city_data = parse_json(file_path)
    blocks_features = city_data["features"][1:num_blocks]
    corners = calculate_bounding_box(blocks_features)
    depot_corr = collect(get_depot_location(corners, depot_location))    
    for i = 1:num_blocks
        block_props = city_data["features"][i]["properties"]
        set_block_properties!(G, block_props, i, city_data, depot_corr)
        add_edges!(G, block_props, i, num_blocks)
    end

    depot_properties = Dict(
        :id => num_blocks + 1,
        :population => 0,
        :area => 0,
        :density => 0,
        :perimeter => 0,
        :compactness => 0,
        :centre => depot_corr,
        :distDepot => 0,
    )
    set_props!(G, num_blocks + 1, depot_properties)
    foreach(i -> add_edge!(G, num_blocks + 1, i), 1:num_blocks)

    return G
end

"""
    update_edge_weights!(graph::MetaGraph, weights::Vector{T}) where T <: Real

Update the weights of the edges in a MetaGraph.

# Arguments
- `graph::MetaGraph`: The graph whose edge weights are to be updated.
- `weights::Vector{T}`: A vector of weights corresponding to each edge in the graph.

# Errors
Throws an error if the number of provided weights does not match the number of edges in the graph.
"""

function update_edge_weights!(graph::MetaGraph, weights::Vector{T}) where {T<:Real}
    if length(weights) != ne(graph)
        error("Mismatch: Provided $(length(weights)) weights for $(ne(graph)) edges.")
    end
    for (i, e) in enumerate(edges(graph))
        set_prop!(graph, src(e), dst(e), :weight, weights[i])
    end
end

function get_edge_weights(graph::MetaGraph)::Vector{Float64}
    return [get_prop(graph, src(e), dst(e), :weight) for e in edges(graph)]
end

"""
    is_subgraph_connected(graph::MetaGraph, nodes::Vector{Int})::Bool

Check if a subgraph, formed by the specified nodes in a MetaGraph, is connected.

# Arguments
- `graph::MetaGraph`: The graph from which the subgraph is formed.
- `nodes::Vector{Int}`: The nodes that form the subgraph.

# Returns
- `Bool`: `true` if the subgraph is connected, `false` otherwise.
"""

function is_subgraph_connected(graph::MetaGraph, nodes::Vector{Int})::Bool
    subgraph, _ = induced_subgraph(graph, nodes)
    return is_connected(subgraph)
end

"""
    build_instance(city_name::String, num_blocks::Int, target_district_size::Int)::Instance

Build an `Instance` object representing a specific problem instance with a city graph and district sizes.

# Arguments
- `city_name::String`: Name of the city.
- `num_blocks::Int`: Number of blocks in the city.
- `target_district_size::Int`: Target size for the districts.

# Returns
- `Instance`: The constructed problem instance.
"""

function build_instance(
    city_name::String,
    num_blocks::Int,
    target_district_size::Int,
    depot_location::String,
)::Instance
    graph = create_city_graph(city_name, num_blocks, depot_location)
    min_size = floor(Int, 0.8 * target_district_size)
    max_size = ceil(Int, 1.2 * target_district_size)
    return Instance(
        city_name,
        num_blocks,
        target_district_size,
        min_size,
        max_size,
        graph,
        depot_location,
    )
end
