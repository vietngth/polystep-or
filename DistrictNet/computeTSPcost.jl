using GraphNeuralNetworks, Graphs, MetaGraphs
using JSON, LinearAlgebra, Combinatorics, Random

Random.seed!(1234)
include("src/struct.jl")
include("src/instance.jl")

using .CostEvaluator: EVmain
using .GenerateScenario: SCmain


"""
    createScenario(city::String)

Generates a scenario file for a specified city. The scenario includes various parameters like the number of blocks, the district size, and the depot location.

# Arguments
- `city`: Name of the city for which the scenario is being created.

The scenario data is saved in a JSON file specified by `pathScenario`.
"""

function createScenario(city)
    nb_scenario = 100
    depot_location = "C"
    nb_bu_large = 30
    district_size = 3
    instance = build_instance(city, nb_bu_large, district_size, depot_location)
    pathScenario = "deps/Scenario/output/$(city)_$(depot_location)_$(nb_bu_large)_$(district_size).json"

    if isfile(pathScenario)
        rm(pathScenario)
    end

    StringInstance = city * "_"* depot_location * "_" * string(nb_bu_large) * "_" * string(district_size) * "_" * string(nb_scenario)
    SCmain(StringInstance)
end

"""
    createScenario(city::String)

Generates a scenario file for a specified city. The scenario includes various parameters like the number of blocks, the district size, and the depot location.

# Arguments
- `city`: Name of the city for which the scenario is being created.

The scenario data is saved in a JSON file specified by `pathScenario`.
"""

function enumerate_connected_subgraphs(G::AbstractGraph, k::Int, m::Int)
    subgraphs = []
    non_root_nodes = setdiff(vertices(G), nv(G))
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
    compute__SAA(instance::Instance, nodes::Array{Int})

Computes the cost for a subset of nodes in a district using the Sample Average Approximation (SAA) method.

# Arguments
- `instance`: The instance of the problem.
- `nodes`: The array of node indices for which the cost is to be computed.

# Returns
The cost calculated for the given nodes.
"""

function compute__SAA(instance::Instance, nodes::Array{Int}, depot_location::String = "C")
    g = instance.graph
    NB_SCENARIO = 100
    formatted_nodes = "{" * join(nodes, ", ") * "}"
    target = instance.city_name * "_"* "C" * "_" * string(instance.num_blocks) * "_" * string(instance.target_district_size) * "_" * string(NB_SCENARIO)
    return EVmain(formatted_nodes, target)
end

"""
    computeCostLoader(city::String)

Computes and saves the cost data for different district configurations in a given city.

# Arguments
- `city`: The name of the city for which cost data is computed.

The function generates scenarios for the city, computes costs for different district configurations, and saves this data in a JSON file.
"""

function computeCostLoader(city)
    depot_location = "C"
    nb_bu_large = 30
    district_size = 3
    output_path = "data/tspCosts/$(city)_$(depot_location)_$(nb_bu_large)_$(district_size)_tsp.train_and_test.json"
    createScenario(city)
    instance = build_instance(city, nb_bu_large, district_size, depot_location)
    districts = enumerate_connected_subgraphs(instance.graph, 2, 4)
    districtsCost = []
    for district in districts
        district = district .- 1
        cost = compute__SAA(instance, district)
        push!(districtsCost, cost)
    end
    # make dict like {"average-cost": 39.89568, "list-blocks": [32, 43, 51]} and add them to one list
    data = []
    for (district, cost) in zip(districts, districtsCost)
        dict = Dict("average-cost" => cost, "list-blocks" => district .- 1)
        push!(data, dict)
    end
    # save data to json file
    final_dict = Dict("districts" => data)
    open(output_path, "w") do f
        JSON.print(f, final_dict)
    end
end
"""
    main()

The main function to compute and load the cost loader data for a specified city.

It reads the city name from command line arguments and calls `computeCostLoader` to generate cost data.
"""

function main()
    city = ARGS[1]
    computeCostLoader(city)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
