"""
    mutable struct Instance

Represents an instance of the districting problem.

# Fields
- `city_name::String`: The name of the city.
- `num_blocks::Int`: The number of blocks in the city.
- `target_district_size::Int`: The target size for each district.
- `min_district_size::Int`: The minimum size for a district.
- `max_district_size::Int`: The maximum size for a district.
- `graph::MetaGraph`: The graph representing the city's layout.
- `depot::String`: The depot location in the city.
"""

mutable struct Instance
    city_name::String
    num_blocks::Int
    target_district_size::Int
    min_district_size::Int
    max_district_size::Int
    graph::MetaGraph
    depot::String
end
"""
    mutable struct District

Represents a single district in the districting problem.

# Fields
- `id::Int`: The identifier for the district.
- `nodes::Array{Int}`: An array of nodes (blocks) that form the district.
- `cost::Float64`: The cost associated with the district.
- `is_feasible::Bool`: Indicates whether the district is feasible based on constraints.
"""

mutable struct District
    id::Int
    nodes::Array{Int}
    cost::Float64
    is_feasible::Bool
end
"""
    mutable struct Costloader

Stores precomputed cost data for districts.

# Fields
- `DistrictIds::Vector{Vector{Int}}`: A vector of district IDs.
- `Cost::Array{Float64}`: An array of costs corresponding to each district.
"""

mutable struct Costloader
    DistrictIds::Vector{Vector{Int}}
    Cost::Array{Float64}
end

"""
Edges represents an edge in a graph.

Fields
- `src::Int`: The source node of the edge
- `dst::Int`: The destination node of the edge
- `weight::Float64`: The weight associated with the edge
"""

struct Edge
    src::Int
    dst::Int
    weight::Float64
end

"""
    mutable struct Solution

Represents a solution to the districting problem.

# Fields
- `instance::Instance`: The original problem instance.
- `districts::Array{District}`: An array of districts forming the solution.
- `cost::Float64`: The total cost of the solution.
- `nb_districts::Int`: The total number of districts in the solution.
- `is_feasible::Bool`: Indicates whether the solution is feasible.
- `blocks_district_ids::Array{Int}`: An array mapping each block to its district ID.
"""

mutable struct Solution
    instance::Instance
    districts::Array{District}
    cost::Float64
    nb_districts::Int
    is_feasible::Bool
    blocks_district_ids::Array{Int}
end

"""
    mutable struct GraphFeaturesInstance

Represents a graph features instance for use in machine learning models.

# Fields
- `feature::Array{Real}`: An array of features associated with the graph.
- `instance::Instance`: The districting problem instance.
- `cost::Float64`: The cost associated with the instance.
- `gnn_graph::GNNGraph`: The graph neural network graph.
- `subgraphs::Vector{Vector{Int}}`: A vector of subgraphs.
- `costloader::Costloader`: The costloader for precomputed district costs.
"""

mutable struct GraphFeaturesInstance
    feature::Array{Real}
    instance::Instance
    cost::Float64
    gnn_graph::GNNGraph
    subgraphs::Vector{Vector{Int}}
    costloader::Costloader
    solution::Solution
end
"""
    struct Hyperparams

Represents hyperparameters for machine learning models.

# Fields
- `batch_size::Int`: The batch size.
- `hidden_size::Int`: The size of hidden layers.
- `nb_samples::Int`: The number of samples.
- `epsilon::Float64`: The epsilon parameter.
- `weight_decay::Float64`: The weight decay rate.
- `initial_lr::Float64`: The initial learning rate.
- `nb_epochs::Int`: The number of epochs.
"""

struct Hyperparams
    batch_size::Int
    hidden_size::Int
    nb_samples::Int
    epsilon::Float64
    weight_decay::Float64
    initial_lr::Float64
    nb_epochs::Int
    mc_samples::Int
end

"""
    module CostEvaluator

A module for evaluating costs using a C++ library.

# Functions
- `get_libCostEvaluator_path()`: Returns the path to the `libCostEvaluator.so` library.
"""

module CostEvaluator
using CxxWrap
function get_libCostEvaluator_path()
    script_dir = @__DIR__
    return joinpath(script_dir, "../deps/Evaluator/build/libCostEvaluator.so")
end
@wrapmodule(get_libCostEvaluator_path)
end

"""
    module GenerateScenario

A module for generating scenarios using a C++ library.

# Functions
- `get_libGenSC_path()`: Returns the path to the `libGenerateScenario.so` library.
"""

module GenerateScenario
using CxxWrap
function get_libGenSC_path()
    script_dir = @__DIR__
    return joinpath(script_dir, "../deps/Scenario/build/libGenerateScenario.so")
end
@wrapmodule(get_libGenSC_path)
end
