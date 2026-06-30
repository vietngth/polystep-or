module districtNet
export train_city_model, find_solution_city, fitGeneralModel, find_solution_small_city, find_districting_solution, get_districting_solution
using InferOpt, Plots
using MLUtils, GraphNeuralNetworks, Graphs, MetaGraphs, GraphPlot
using JSON, FilePaths, Random, LinearAlgebra
using Distributions, Statistics, UnionFind, Flux
using DataStructures, Serialization, FileIO
using Base.Filesystem: isfile
using Combinatorics, Optim
using JuMP, GLPK, CxxWrap, DistributedArrays, JLD2


Random.seed!(1234)
include("utils.jl")
include("struct.jl")
include("instance.jl")
include("district.jl")
include("solution.jl")
include("learning.jl")
include("Solver/Kruskal.jl")
include("Solver/localsearch.jl")
include("Solver/exactsolver.jl")


using .CostEvaluator: EVmain
using .GenerateScenario: SCmain
# Constants definition
const NB_BU_LARGE = 120
const DEPOT_LOCATION = "C"
const NB_DATA = 200
const NB_BU_SMALL = 30
const STRATEGY = "districtNet"
const NB_SCENARIO = 100
# Training Hyperparameters
const BATCH_SIZE = 1
const HIDDEN_SIZE = 64
const NB_SAMPLES = 20
const EPSILON = 1.0
const WEIGHT_DECAY = 0.0
const INITIAL_LR = 0.001
const NB_EPOCHS = parse(Int, get(ENV, "DN_EPOCHS", "40"))   # override for timing runs (e.g. DN_EPOCHS=5)
const MC_SAMPLES = 20

# Solver Hyperparameters
const MAX_TIME = 120
const PERTURBATION_PROBABILITY = 0.985
const PENALITY = 10000
# Helper functions

"""
    find_solution_city(city::String, target_district_size::Int, NB_BU::Int, model, mean_train, std_train)

Finds a districting solution for a city using a trained GNN model.

# Arguments
- `city::String`: The name of the city.
- `target_district_size::Int`: The target district size.
- `NB_BU::Int`: The number of blocks in the city.
- `model`: The trained GNN model.
- `mean_train`: Mean used for feature normalization during training.
- `std_train`: Standard deviation used for feature normalization during training.

# Returns
- A `Solution` object representing the districting solution for the city.
"""
function find_solution_city(
    city::String,
    target_district_size::Int,
    NB_BU::Int,
    depot_location::String,
    model,
    mean_train,
    std_train,
)
    instance = build_instance(city, NB_BU, target_district_size, depot_location)
    W = rand(ne(instance.graph))
    update_edge_weights!(instance.graph, W)
    ϴ = predict_theta(instance, STRATEGY, model, mean_train, std_train)
    update_edge_weights!(instance.graph, ϴ)
    costloader = Costloader([], [])
    solution = ILS_solve_instance(instance, costloader, "CMST", ϴ)
    return solution
end
"""
    aggregateCityTrainingData(nb_data)

Aggregates training data from multiple cities.

# Arguments
- `nb_data`: The number of cities to include in the training data.

# Returns
- Training data aggregated from multiple cities.
"""
function aggregateCityTrainingData(nb_data, start=1)
    TARGET_DISTRICT_SIZE = 3
    data_train = []
    cities = ["city" * string(i) for i = start:nb_data+start-1]
    t_inst = 0.0; t_cost = 0.0; t_exact = 0.0; t_feat = 0.0   # [TIMING] per-component accumulators
    for CITY in cities
        local instance, costloader, solution, unique_subgraphs
        t_inst += @elapsed (instance = build_instance(CITY, NB_BU_SMALL, TARGET_DISTRICT_SIZE, DEPOT_LOCATION))
        W = W = rand(ne(instance.graph))
        update_edge_weights!(instance.graph, W)
        path = string("data/tspCosts/", CITY, "_", DEPOT_LOCATION, "_", NB_BU_SMALL, "_", TARGET_DISTRICT_SIZE, "_tsp.train_and_test.json")
        t_cost += @elapsed (costloader = load_precomputed_costs(path))
        t_exact += @elapsed ((solution, unique_subgraphs) = districting_exact_solver(instance, costloader))
        if (solution == nothing)
            continue
        end
        cost = solution.cost
        tf = @elapsed begin
        y = randomized_constructor(solution, MC_SAMPLES)
        g = deepcopy(instance.graph)
        edges_data = extract_edge_feature(g)
        data = hcat(edges_data...)
        gnn_graph = create_edge_graph(g, data)
        end
        t_feat += tf
        push!(
            data_train,
            (
                GraphFeaturesInstance(
                    data,
                    instance,
                    cost,
                    gnn_graph,
                    unique_subgraphs,
                    costloader,
                    solution,
                ),
                y,
            ),
        )
    end
    println("[TIMING]   aggregate breakdown: build_instance=$(round(t_inst,digits=2))s  load_TSP_costs=$(round(t_cost,digits=2))s  exact_districting_solver=$(round(t_exact,digits=2))s  features=$(round(t_feat,digits=2))s")
    return data_train
end

"""
    fitGeneralModel(nb_data)

Trains a general GNN model using aggregated data from multiple cities.

# Arguments
- `nb_data`: The number of cities to use in training.

# Returns
- A trained general GNN model, along with mean and standard deviation for feature normalization.
"""

function fitGeneralModel(nb_data)
    TARGET_DISTRICT_SIZE = 3
    println("[TIMING] === DistrictNet fitGeneralModel(nb_data=$(nb_data)) ===")
    t_agg = @elapsed (data_train = aggregateCityTrainingData(nb_data))
    println("[TIMING] STAGE data-aggregation TOTAL = $(round(t_agg, digits=2)) s")
    data_train, data_test = splitobs(shuffleobs(data_train), at = 1.0)
    println("Training data size: ", length(data_train))
    data_train, mean_train, std_train = normalize_features(data_train)
    data_test, _, _ = normalize_features(data_test, mean_train, std_train)
    hyperparams = Hyperparams(
        BATCH_SIZE,
        HIDDEN_SIZE,
        NB_SAMPLES,
        EPSILON,
        WEIGHT_DECAY,
        INITIAL_LR,
        NB_EPOCHS,
        MC_SAMPLES,
    )
    if get(ENV, "DN_UNTRAINED", "0") == "1"   # [ABLATION] random GNN, no load/train (data-learns-something baseline)
        Random.seed!(parse(Int, get(ENV, "DN_SEED", "1234")))
        model = build_gnn_model(data_train[1][1].instance.graph, STRATEGY, HIDDEN_SIZE)
        println("[ABLATION] UNTRAINED random GNN (districtNet), seed=", get(ENV, "DN_SEED", "1234"))
        return model, mean_train, std_train
    end
    model_name = "models/GeneralDistrictNet_$(nb_data).jld2"
    if isfile(model_name)
        @info "Loading model from file"
        model_state = JLD2.load(model_name, "model_state")
        model = build_gnn_model(data_train[1][1].instance.graph, STRATEGY, HIDDEN_SIZE)
        Flux.loadmodel!(model, model_state)
        return model, mean_train, std_train
    end
    model = build_gnn_model(data_train[1][1].instance.graph, STRATEGY, HIDDEN_SIZE)
    t_train = @elapsed (model = train_GNN(STRATEGY, model, data_train, data_test, hyperparams))
    println("[TIMING] STAGE GNN-training TOTAL = $(round(t_train, digits=2)) s")
    model_state = Flux.state(model)
    if !isdir("models")
        mkpath("models")
    end
    jldsave(model_name; model_state)
    return model, mean_train, std_train
end

"""
    find_solution_small_city(city::String, target_district_size::Int, model, mean_train, std_train)

Finds a districting solution for a small city using a trained GNN model.

# Arguments
- `city::String`: The name of the city.
- `target_district_size::Int`: The target district size.
- `model`: The trained GNN model.
- `mean_train`: Mean used for feature normalization during training.
- `std_train`: Standard deviation used for feature normalization during training.

# Returns
- A `Solution` object representing the districting solution for the small city.
"""

function find_solution_small_city(
    city::String,
    target_district_size::Int,
    model,
    mean_train,
    std_train,
)
    path = "data/tspCosts/$(city)_$(DEPOT_LOCATION)_$(NB_BU_SMALL)_$(target_district_size)_tsp.train_and_test.json"
    costloader = load_precomputed_costs(path)
    instance = build_instance(city, NB_BU_SMALL, target_district_size, DEPOT_LOCATION)
    W = rand(ne(instance.graph))
    update_edge_weights!(instance.graph, W)
    ϴ = predict_theta(instance, STRATEGY, model, mean_train, std_train)
    solution = Exact_solve_instance(instance, costloader, "CMST", ϴ)
    pred_cost = 0
    for d in solution.districts
        pred_cost += compute_cost_with_precomputed_data(instance, d, costloader)
    end
    solution.cost = pred_cost
    return solution
end
"""
    find_districting_solution(city::String, target_district_size::Int)

Finds a districting solution for a city using exact methods.

# Arguments
- `city::String`: The name of the city.
- `target_district_size::Int`: The target district size.

# Returns
- A `Solution` object representing the districting solution for the city.
"""

function find_districting_solution(city::String, target_district_size::Int)
    instance = build_instance(city, NB_BU_SMALL, target_district_size, DEPOT_LOCATION)
    path = "data/tspCosts/$(city)_$(DEPOT_LOCATION)_$(NB_BU_SMALL)_$(target_district_size)_tsp.train_and_test.json"
    costloader = load_precomputed_costs(path)
    solution, _ = districting_exact_solver(instance, costloader)
    return solution
end

end
