module BD
export find_city_beta, train_city_model, fitGeneralModel, find_solution_small_city
using InferOpt, Plots
using MLUtils, GraphNeuralNetworks, Graphs, MetaGraphs, GraphPlot
using JSON, FilePaths, Random, LinearAlgebra
using Distributions, Statistics, UnionFind, Flux
using DataStructures, Serialization, FileIO
using Base.Filesystem: isfile
using Combinatorics, Optim
using JuMP, GLPK, CxxWrap, DistributedArrays


include("../utils.jl")
include("../struct.jl")
include("../instance.jl")
include("../district.jl")
include("../solution.jl")
include("../learning.jl")
include("../Solver/Kruskal.jl")
include("../Solver/localsearch.jl")
include("../Solver/exactsolver.jl")


using .CostEvaluator: EVmain
using .GenerateScenario: SCmain
# Constants

const NB_BU_LARGE = 120
const NB_BU_SMALL = 30
const DEPOT_LOCATION = "C"
const STRATEGY = "BD"
const NB_SCENARIO = 100

# Solver Hyperparameters
const MAX_TIME = 120
const PERTURBATION_PROBABILITY = 0.985
const PENALITY = 10000

"""
load_scenarios(pathScenario)

Loads scenario data from a given file path.

# Arguments
- `pathScenario`: The file path to load scenario data from.

# Returns
- A tuple containing block scenarios and depot coordinates.
"""

function load_scenarios(pathScenario)
    sc = JSON.parsefile(pathScenario)
    blocks_scenarios = sc["blocks"]
    depot_corr = sc["metadata"]["DEPOT_XY"]
    return blocks_scenarios, depot_corr
end


"""
    get_districts_scenarios(district, scenarios)

Retrieves scenarios for a specific district.

# Arguments
- `district`: The district for which scenarios are to be retrieved.
- `scenarios`: The overall scenarios data.

# Returns
- A list of scenarios specific to the given district.
"""
function get_districts_scenarios(district, scenarios)
    scenarios_data = []
    nb_sc = length(scenarios[1]["Scenarios"])
    for i = 1:nb_sc
        scenario = []
        for block in district
            push!(scenario, scenarios[block]["Scenarios"][i])
        end
        push!(scenarios_data, vcat(scenario...))
    end
    return scenarios_data
end
"""
    average_distance_to_depot(points, depot)

Calculates the average distance from a set of points to the depot.

# Arguments
- `points`: A list of points (coordinates).
- `depot`: The depot coordinates.

# Returns
- The average distance to the depot.
"""
function average_distance_to_depot(points, depot)
    total_distance = 0.0
    depot_x, depot_y = depot
    for point in points
        x, y = point
        distance = sqrt((x - depot_x)^2 + (y - depot_y)^2)
        total_distance += distance
    end
    return total_distance / length(points)
end
"""
    MC_average_distance_to_depot(scenarios, depot)

Computes the Monte Carlo average distance to the depot for a set of scenarios.

# Arguments
- `scenarios`: A list of scenarios, each containing multiple points.
- `depot`: The depot coordinates.

# Returns
- The Monte Carlo average distance to the depot across all scenarios.
"""

function MC_average_distance_to_depot(scenarios, depot)
    avg = 0.0
    for points in scenarios
        avg += average_distance_to_depot(points, depot)
    end
    return avg / length(scenarios)
end


"""
    get_total_area(instance, district)

Calculates the total area of a district.

# Arguments
- `instance`: The problem instance.
- `district`: The district (list of blocks).

# Returns
- The total area of the given district.
"""
function get_total_area(instance, district)
    total_area = 0.0
    for block in district
        d = props(instance.graph, block)
        total_area += d[:area]
    end
    return total_area
end
"""
    get_expected_number_of_clients(instance, district)

Estimates the expected number of clients in a district.

# Arguments
- `instance`: The problem instance.
- `district`: The district (list of blocks).

# Returns
- The expected number of clients in the district.
"""

function get_expected_number_of_clients(instance, district)
    requests_prob = 96 / (8000 * instance.target_district_size)
    total_clients = 0.0
    for block in district
        d = props(instance.graph, block)
        total_clients += d[:population]
    end
    return total_clients * requests_prob
end
"""
    predictBD_district(metrics, beta)

Predicts the cost of a district based on its metrics.

# Arguments
- `metrics`: The metrics of the district.
- `beta`: The beta parameter for the model.

# Returns
- The predicted cost of the district.
"""

function predictBD_district(metrics, beta)
    total_area, expected_clients, avg_distance = metrics
    cost = beta * sqrt(total_area * expected_clients) + 2 * avg_distance
    return cost
end
"""
    precompute_district_metrics(instance, districts, blocks_scenarios, depot_corr)

Precomputes metrics for each district.

# Arguments
- `instance`: The problem instance.
- `districts`: A list of districts.
- `blocks_scenarios`: Block scenarios data.
- `depot_corr`: Depot coordinates.

# Returns
- A list of precomputed metrics for each district.
"""

function precompute_district_metrics(instance, districts, blocks_scenarios, depot_corr)
    district_metrics = []
    for district in districts
        total_area = get_total_area(instance, district)
        expected_clients = get_expected_number_of_clients(instance, district)
        scenarios = get_districts_scenarios(district, blocks_scenarios)
        avg_distance = MC_average_distance_to_depot(scenarios, depot_corr)
        push!(district_metrics, (total_area, expected_clients, avg_distance))
    end
    return district_metrics
end

"""
    fit_with_optim(instance, district_metrics, y_train)

Fits the Beta using optimization techniques.

# Arguments
- `instance`: The problem instance.
- `district_metrics`: The precomputed district metrics.
- `y_train`: The training data.

# Returns
- The best beta parameter and its associated mean squared error.
"""
function fit_with_optim(instance, district_metrics, y_train)
    function cost_function(beta)
        predictions = [predictBD_district(metrics, beta[1]) for metrics in district_metrics]
        return Flux.mse(y_train, predictions)
    end
    initial_beta = [0.0]

    res = optimize(
        cost_function,
        initial_beta,
        LBFGS(),
        Optim.Options(g_tol = 1e-12);
        autodiff = :forward,
    )

    best_beta = Optim.minimizer(res)
    best_mse = Optim.minimum(res)

    return best_beta[1], best_mse
end

"""
    fit_beta(district_metrics, y_train)

Fits the Beta Distribution model for district cost prediction.

# Arguments
- `district_metrics`: The precomputed district metrics.
- `y_train`: The training data.

# Returns
- The best beta parameter and its associated mean squared error.
"""

function fit_beta(district_metrics, y_train)
    function cost_function(beta)
        predictions = [predictBD_district(metrics, beta[1]) for metrics in district_metrics]
        return Flux.mse(y_train, predictions)
    end
    initial_beta = [0.0]

    res = optimize(
        cost_function,
        initial_beta,
        LBFGS(),
        Optim.Options(g_tol = 1e-12);
        autodiff = :forward,
    )

    best_beta = Optim.minimizer(res)
    best_mse = Optim.minimum(res)

    return best_beta[1], best_mse
end

"""
    create_training_data(costloader)

Creates training data from a costloader.

# Arguments
- `costloader`: The costloader containing district IDs and costs.

# Returns
- The training data (X_train) and labels (y_train).
"""

function create_training_data(costloader)
    X_train = []
    y_train = []
    for (idx, d) in enumerate(costloader.DistrictIds)
        push!(X_train, d)
        push!(y_train, costloader.Cost[idx])
    end
    return X_train, y_train
end


"""
    train_city_model(city::String, target_district_size::Int, experience::Int)

Trains a model for a specific city and target district size using the Beta Distribution approach.

# Arguments
- `city::String`: The name of the city.
- `target_district_size::Int`: The target size of districts.
- `experience::Int`: Identifier for the training experience.

# Returns
- The beta parameter for the trained model.
"""

function train_city_model(city::String, target_district_size::Int, experience::Int)
    instance = build_instance(city, NB_BU_LARGE, target_district_size, DEPOT_LOCATION)
    path = "data/tspCosts/$(city)_$(DEPOT_LOCATION)_$(NB_BU_LARGE)_$(target_district_size)_tsp.train_and_test.json"
    costloader = load_precomputed_costs(path)
    costloader = remove_districts(instance, costloader)
    model_name = "models/$(experience)$(STRATEGY)$(city)$(NB_BU_LARGE).json"
    if (experience == 2)
        model_name = "models/$(experience)$(STRATEGY)$(city)$(NB_BU_LARGE)$(target_district_size).json"
    end
    if isfile(model_name)
        @info "Loading model $(model_name)"
        # read beta from file
        beta = JSON.parsefile(model_name)["beta"]
        return beta
    end
    # Truncate the dataset if it's too large
    if length(costloader.DistrictIds) > 10000
        costloader = truncate_dataset(costloader, 10000)
    end
    println("Loading data ", length(costloader.DistrictIds))
    X_train, y_train = create_training_data(costloader)
    pathScenario = "deps/Scenario/output/$(city)_$(DEPOT_LOCATION)_$(NB_BU_LARGE)_$(target_district_size).json"
    if !isfile(pathScenario)
        StringInstance =
            city *
            "_C_" *
            string(NB_BU_LARGE) *
            "_" *
            string(target_district_size) *
            "_" *
            string(NB_SCENARIO)
        SCmain(StringInstance)
    end
    blocks_scenarios, depot_corr = load_scenarios(pathScenario)
    district_metrics =
        precompute_district_metrics(instance, X_train, blocks_scenarios, depot_corr)
    beta, mse = fit_with_optim(instance, district_metrics, y_train)
    @info "Best beta: ", beta, " with mse: ", mse
    if !isdir("models")
        mkpath("models")
    end
    open(model_name, "w") do f
        JSON.print(f, Dict("beta" => beta, "mse" => mse))
    end
    return beta
end


"""
    find_solution_city(city::String, target_district_size::Int, NB_BU::Int, beta::Float64)

Finds a districting solution for a city using the BD model.

# Arguments
- `city::String`: The name of the city.
- `target_district_size::Int`: The target district size.
- `NB_BU::Int`: The number of blocks in the city.
- `beta::Float64`: The beta parameter of the model.

# Returns
- A `Solution` object representing the districting solution for the city.
"""

function find_solution_city(
    city::String,
    target_district_size::Int,
    NB_BU::Int,
    depot_location::String,
    beta::Float64,
)
    instance = build_instance(city, NB_BU, target_district_size, depot_location)
    costloader = Costloader([], [])
    pathScenario = "deps/Scenario/output/$(city)_$(depot_location)_$(NB_BU)_$(target_district_size).json"
    if !isfile(pathScenario)
        StringInstance =
            city *
            "_" *
            depot_location *
            "_" *
            string(NB_BU) *
            "_" *
            string(target_district_size) *
            "_" *
            string(NB_SCENARIO)
        SCmain(StringInstance)
    end
    blocks_scenarios, depot_corr = load_scenarios(pathScenario)
    params = (blocks_scenarios, depot_corr, beta)
    solution = ILS_solve_instance(instance, costloader, "BD", params)
    return solution
end


function getCityTrainingData(city::String, target_district_size::Int)
    instance = build_instance(city, NB_BU_LARGE, target_district_size, DEPOT_LOCATION)
    path = "data/tspCosts/$(city)_$(DEPOT_LOCATION)_$(NB_BU_LARGE)_$(target_district_size)_tsp.train_and_test.json"
    costloader = load_precomputed_costs(path)
    costloader = remove_districts(instance, costloader)
    # Truncate the dataset if it's too large
    if length(costloader.DistrictIds) > 10000
        costloader = truncate_dataset(costloader, 10000)
    end
    X_train, y_train = create_training_data(costloader)
    pathScenario = "deps/Scenario/output/$(city)_$(DEPOT_LOCATION)_$(NB_BU_LARGE)_$(target_district_size).json"
    if !isfile(pathScenario)
        StringInstance =
            city *
            "_C_" *
            string(NB_BU_LARGE) *
            "_" *
            string(target_district_size) *
            "_" *
            string(NB_SCENARIO)
        SCmain(StringInstance)
    end
    blocks_scenarios, depot_corr = load_scenarios(pathScenario)
    district_metrics =
        precompute_district_metrics(instance, X_train, blocks_scenarios, depot_corr)
    return district_metrics, y_train
end
function aggregateCityTrainingData(cities, target_district_size)
    all_district_metrics = []
    all_y_train = []

    for city in cities
        instance = build_instance(city, NB_BU_SMALL, target_district_size, DEPOT_LOCATION)
        path = "data/tspCosts/$(city)_$(DEPOT_LOCATION)_$(NB_BU_SMALL)_$(target_district_size)_tsp.train_and_test.json"
        costloader = load_precomputed_costs(path)

        X_train, y_train = create_training_data(costloader)
        all_y_train = vcat(all_y_train, y_train)

        pathScenario = "deps/Scenario/output/$(city)_$(DEPOT_LOCATION)_$(NB_BU_SMALL)_$(target_district_size).json"
        if !isfile(pathScenario)
            StringInstance =
                city *
                "_C_" *
                string(NB_BU_SMALL) *
                "_" *
                string(target_district_size) *
                "_" *
                string(NB_SCENARIO)
            SCmain(StringInstance)
        end
        blocks_scenarios, depot_corr = load_scenarios(pathScenario)

        for district in X_train
            scenarios = get_districts_scenarios(district, blocks_scenarios)
            avg_distance = MC_average_distance_to_depot(scenarios, depot_corr)
            total_area = get_total_area(instance, district)
            expected_clients = get_expected_number_of_clients(instance, district)
            push!(all_district_metrics, (total_area, expected_clients, avg_distance))
        end
    end
    return all_district_metrics, all_y_train
end


"""
    fitGeneralModel(nb_data)

Trains a general model using aggregated data from multiple cities with the BD approach.

# Arguments
- `nb_data`: The number of cities to use in training.

# Returns
- The beta parameter for the general model.
"""

function fitGeneralModel(nb_data)
    cities = ["city" * string(i) for i = 1:nb_data]
    target_district_size = 3
    model_name = "models/GeneralBD.json"
    if isfile(model_name)
        @info "Loading model $(model_name)"
        # read beta from file
        beta = JSON.parsefile(model_name)["beta"]
        return beta
    end
    all_district_metrics, all_y_train =
        aggregateCityTrainingData(cities, target_district_size)
    combined_data = shuffle!(collect(zip(all_district_metrics, all_y_train)))
    if length(combined_data) > 10000
        combined_data = combined_data[1:10000]
    end
    all_district_metrics, all_y_train =
        [x[1] for x in combined_data], [x[2] for x in combined_data]
    start_time = time()
    beta, mse = fit_beta(all_district_metrics, all_y_train)
    @info "Best beta: ", beta, " with mse: ", mse
    if !isdir("models")
        mkpath("models")
    end
    open(model_name, "w") do f
        JSON.print(f, Dict("beta" => beta, "mse" => mse))
    end
    end_time = time()
    println("Training time: ", end_time - start_time)
    return beta
end

"""
    find_solution_small_city(city::String, target_district_size::Int, beta::Float64)

Finds a districting solution for a small city using the BD model.

# Arguments
- `city::String`: The name of the city.
- `target_district_size::Int`: The target district size.
- `beta::Float64`: The beta parameter of the model.

# Returns
- A `Solution` object representing the districting solution for the small city.
"""

function find_solution_small_city(city::String, target_district_size::Int, beta::Float64)
    instance = build_instance(city, NB_BU_SMALL, target_district_size, DEPOT_LOCATION)
    path = "data/tspCosts/$(city)_$(DEPOT_LOCATION)_$(NB_BU_SMALL)_$(target_district_size)_tsp.train_and_test.json"
    costloader = load_precomputed_costs(path)
    pathScenario = "deps/Scenario/output/$(city)_$(DEPOT_LOCATION)_$(NB_BU_SMALL)_$(target_district_size).json"
    if !isfile(pathScenario)
        StringInstance =
            city *
            "_C_" *
            string(NB_BU_SMALL) *
            "_" *
            string(target_district_size) *
            "_" *
            string(NB_SCENARIO)
        SCmain(StringInstance)
    end
    blocks_scenarios, depot_corr = load_scenarios(pathScenario)
    params = (blocks_scenarios, depot_corr, beta)
    solution = Exact_solve_instance(instance, costloader, "BD", params)
    pred_cost = 0
    for d in solution.districts
        pred_cost += compute_cost_with_precomputed_data(instance, d, costloader)
    end
    solution.cost = pred_cost
    return solution

end

end
