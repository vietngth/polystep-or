module predictGnn
export train_city_model, find_solution_city, fitGeneralModel, find_solution_small_city
using InferOpt, Plots
using MLUtils, GraphNeuralNetworks, Graphs, MetaGraphs, GraphPlot
using JSON, FilePaths, Random, LinearAlgebra
using Distributions, Statistics, UnionFind, Flux
using DataStructures, Serialization, FileIO
using Base.Filesystem: isfile
using Combinatorics, Optim
using JuMP, GLPK, CxxWrap, DistributedArrays, JLD2

Random.seed!(1234)
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


const NB_BU_LARGE = 120
const NB_BU_SMALL = 30
const DEPOT_LOCATION = "C"
const STRATEGY = "PredictGnn"
const NB_SCENARIO = 100

# Training Hyperparameters
const BATCH_SIZE = 64
const HIDDEN_SIZE = 64
const INITIAL_LR = 0.0001
const NB_EPOCHS = 40
const SIGNIFICANT_LOSS_CHANGE_THRESHOLD = 0.0001

# Solver Hyperparameters
const MAX_TIME = 120
const PERTURBATION_PROBABILITY = 0.985
const PENALITY = 10000

"""
    load_instance(city::String, nb_bu::Int, target_district_size::Int)

Loads a specific instance of the problem for a given city, number of blocks, and target district size.

# Arguments
- `city::String`: The name of the city.
- `nb_bu::Int`: The number of blocks (building units) in the city.
- `target_district_size::Int`: The target size of each district.

# Returns
- An `Instance` object representing the specific problem instance.
"""

function load_instance(city::String, nb_bu::Int, target_district_size::Int, depot_location::String)
    instance = build_instance(city, nb_bu, target_district_size, depot_location)
    W = rand(ne(instance.graph))
    update_edge_weights!(instance.graph, W)
    non_roots = collect(1:nv(instance.graph)-1)
    new_graph, _ = induced_subgraph(instance.graph, non_roots)
    instance.graph = new_graph
    return instance
end

"""
    load_instance_feat(instance::Instance)

Loads and normalizes the features for a given instance.

# Arguments
- `instance::Instance`: The problem instance.

# Returns
- A `GNNGraph` object with loaded and normalized features.
"""

function load_instance_feat(instance::Instance)
    g = instance.graph
    nb = nv(g)
    data = Array{Array{Float32}}(undef, nv(g))
    for i = 1:nv(g)
        d = props(g, i)
        data[i] = Float32[
            d[:density],
            d[:population],
            d[:area],
            d[:perimeter],
            d[:distDepot],
            d[:district],
            sqrt(d[:area]),
            sqrt(d[:population]),
        ]
    end
    data = hcat(data...)
    gn = GNNGraph(g, ndata = data)
    return gn
end

"""
    load_and_prepare_data(city::String, target_district_size::Int, instance)

Loads and prepares training data for a given city and target district size.

# Arguments
- `city::String`: The name of the city.
- `target_district_size::Int`: The target size of each district.
- `instance`: The problem instance.

# Returns
- Prepared training data.
"""

function load_and_prepare_data(city::String, target_district_size::Int, instance)
    path = "data/tspCosts/$(city)_$(DEPOT_LOCATION)_$(NB_BU_LARGE)_$(target_district_size)_tsp.train_and_test.json"
    costloader = load_precomputed_costs(path)
    costloader = remove_districts(instance, costloader)
    # Truncate the dataset if it's too large
    if length(costloader.DistrictIds) > 10000
        costloader = truncate_dataset(costloader, 10000)
    end
    data_train = collect_data_train(costloader, instance)
    return data_train
end
"""
    collect_data_train(costloader, instance)

Collects training data based on the costloader and instance.

# Arguments
- `costloader`: The costloader containing district IDs and costs.
- `instance`: The problem instance.

# Returns
- Data loaders for training and testing.
"""

function collect_data_train(costloader, instance)
    data_train = getDatatrain(costloader, instance)
    data_train, data_test = splitobs(shuffleobs(data_train), at = 0.8)
    train_loader =
        DataLoader(data_train, batchsize = BATCH_SIZE, shuffle = true, collate = true)
    test_loader =
        DataLoader(data_test, batchsize = BATCH_SIZE, shuffle = true, collate = true)
    return train_loader, test_loader
end

"""
    getDatatrain(costloader, instance)

Gathers training data for GNN training.

# Arguments
- `costloader`: The costloader containing district IDs and costs.
- `instance`: The problem instance.

# Returns
- Training data including graphs and costs.
"""

function getDatatrain(costloader, instance)
    data_train = []
    for (idx, d) in enumerate(costloader.DistrictIds)
        for i = 1:nv(instance.graph)
            set_prop!(instance.graph, i, :district, 0)
        end
        for i in d
            set_prop!(instance.graph, i, :district, 1)
        end
        gn = load_instance_feat(instance)
        push!(data_train, (gn, costloader.Cost[idx]))
    end
    graphs = [x[1] for x in data_train]
    costs = [x[2] for x in data_train]
    return graphs, costs
end

"""
    build_predictGnn_model(hidden_size)

Builds a predictive GNN model.

# Arguments
- `hidden_size`: The size of the hidden layers.

# Returns
- A predictive GNN model.
"""

function build_predictGnn_model(hidden_size)
    model = GNNChain(
        GraphConv(8 => hidden_size, relu),
        GraphConv(hidden_size => hidden_size, relu),
        GraphConv(hidden_size => hidden_size, relu),
        GraphConv(hidden_size => 16 * hidden_size, relu),
        GlobalPool(+),
        Dense(16 * hidden_size, 100, relu),
        Dense(100, 1),
    )
    return model
end
"""
    train_predictGnn(model, train_loader, test_loader)

Trains the predictive GNN model.

# Arguments
- `model`: The predictive GNN model to be trained.
- `train_loader`: The data loader for training data.
- `test_loader`: The data loader for test data.

# Returns
- The trained predictive GNN model.
"""

function train_predictGnn(model, train_loader, test_loader)
    println("Training predictive GNN model...")
    start_time = time()
    mae_loss(y_pred, y) = Flux.mae(y_pred, y)
    optimizer = Flux.Optimise.ADAM(INITIAL_LR)
    LinearAlgebra.BLAS.set_num_threads(10)

    previous_loss = Inf
    insignificant_loss_change_count = 0
    max_insignificant_epochs = 1000
    max_duration_seconds = 24 * 60 * 60 # 24 hours

    for epoch = 1:NB_EPOCHS
        if time() - start_time > max_duration_seconds
            println("Training stopped early: Exceeded 24-hour time limit.")
            break
        end

        epoch_losses = []
        for (batch_graphs, batch_costs) in train_loader
            loss = 0.0
            gradients = gradient(Flux.params(model)) do
                y_pred = model(batch_graphs, batch_graphs.ndata.x)
                loss = mae_loss(vec(y_pred), batch_costs)
            end
            push!(epoch_losses, loss)
            Flux.update!(optimizer, Flux.params(model), gradients)
        end
        epoch_loss = mean(epoch_losses)
        println("Epoch: ", epoch, " Loss: ", epoch_loss)

        # Check for insignificant loss change
        if abs(epoch_loss - previous_loss) < SIGNIFICANT_LOSS_CHANGE_THRESHOLD
            insignificant_loss_change_count += 1
        else
            insignificant_loss_change_count = 0
        end
        if insignificant_loss_change_count >= max_insignificant_epochs
            println(
                "Training stopped early: Loss change insignificant for 1000 consecutive epochs.",
            )
            break
        end

        previous_loss = epoch_loss
    end

    end_time = time()
    println("Training time: ", end_time - start_time)
    return model
end

"""
    find_solution_city(city::String, target_district_size::Int, NB_BU::Int, model)

Finds a districting solution for a city using a trained GNN model.

# Arguments
- `city::String`: The name of the city.
- `target_district_size::Int`: The target district size.
- `NB_BU::Int`: The number of blocks in the city.
- `model`: The trained GNN model.

# Returns
- A `Solution` object representing the districting solution for the city.
"""

function find_solution_city(
    city::String,
    target_district_size::Int,
    NB_BU::Int,
    depot_location::String,
    model,
    mean_train, std_train
)
    model_params = (model, mean_train, std_train)
    instance = load_instance(city, NB_BU, target_district_size, depot_location)
    costloader = Costloader([], [])
    solution = ILS_solve_instance(instance, costloader, "GNN", model_params)
    return solution
end
"""
    aggregateCityTrainingData(cities, target_district_size)

Aggregates training data from multiple cities for GNN model training.

# Arguments
- `cities`: A list of city names.
- `target_district_size`: The target district size.

# Returns
- A data loader with aggregated training data from multiple cities.
"""

function aggregateCityTrainingData(cities, target_district_size)
    all_graphs = []
    all_costs = []

    for city in cities
        instance = load_instance(city, NB_BU_SMALL, target_district_size, DEPOT_LOCATION)
        path = "data/tspCosts/$(city)_$(DEPOT_LOCATION)_$(NB_BU_SMALL)_$(target_district_size)_tsp.train_and_test.json"
        costloader = load_precomputed_costs(path)
        data_train = getDatatrain(costloader, instance)
        all_graphs = vcat(all_graphs, data_train[1])
        all_costs = vcat(all_costs, data_train[2])
    end
    shuffled_data_train = shuffle!(collect(zip(all_graphs, all_costs)))
    # Shuffle and truncate the dataset if it's too large
    truncated_data_train = shuffled_data_train[1:min(10000, length(shuffled_data_train))]
    graphs = [x[1] for x in truncated_data_train]
    costs = [x[2] for x in truncated_data_train]
    graphs, train_mean, train_std = normalization(graphs)
    all_data_train = (graphs, costs)
    data_train, data_test = splitobs(shuffleobs(all_data_train), at = 1.0)
    # Create data loaders
    train_loader = DataLoader(data_train, batchsize = BATCH_SIZE, shuffle = true, collate = true)
    return train_loader, train_mean, train_std
end

function normalization(data)
    data_features = [data[i].ndata.x for i = 1:length(data)]
    data_features = hcat(data_features...)
    # Identify binary columns
    is_binary_feature = [length(unique(data_features[i, :])) == 2 for i in 1:size(data_features, 1)]
    # Initialize mean and std arrays
    MEAN = zeros(size(data_features, 1))
    STD = ones(size(data_features, 1))

    # Compute mean and std for non-binary features
    for i in 1:length(is_binary_feature)
        if !is_binary_feature[i]
            MEAN[i] = mean(data_features[i, :])
            STD[i] = std(data_features[i, :])
        end
    end

    for i in 1:length(data)
        data[i].ndata.x = (data[i].ndata.x .- MEAN)./ STD
        #Convert to Float32
        data[i].ndata.x = Float32.(data[i].ndata.x)
    end
    return data, MEAN, STD

end
"""
    fitGeneralModel(nb_data)

Trains a general GNN model using data from multiple cities.

# Arguments
- `nb_data`: The number of cities to use in training.

# Returns
- A general GNN model trained on data from multiple cities.
"""

function fitGeneralModel(nb_data)
    cities = ["city" * string(i) for i = 1:nb_data]
    target_district_size = 3
    train_loader, train_mean, train_std = aggregateCityTrainingData(cities, target_district_size)
    model_name = "models/GeneralPredictGnn.jld2"
    state_name = "models/GeneralPredictGnn_state.json"
    if isfile(model_name)
        @info "Loading model from file"
        model_state = JLD2.load(model_name, "model_state")
        trained_model = build_predictGnn_model(HIDDEN_SIZE)
        Flux.loadmodel!(trained_model, model_state)
        train_mean = JSON.parsefile(state_name)["mean"]
        train_std = JSON.parsefile(state_name)["std"]
        return trained_model, train_mean, train_std
    end
    model = build_predictGnn_model(HIDDEN_SIZE)
    trained_model = train_predictGnn(model, train_loader, nothing)
    model_state = Flux.state(trained_model)
    if !isdir("models")
        mkpath("models")
    end
    jldsave(model_name; model_state)
    #save the mean and std for normalization in form of dictionary
    open(state_name, "w") do io
       JSON.print(io, Dict("mean" => train_mean, "std" => train_std))
    end
    return trained_model, train_mean, train_std
end

"""
    find_solution_small_city(city::String, target_district_size::Int, model)

Finds a districting solution for a small city using a trained GNN model.

# Arguments
- `city::String`: The name of the city.
- `target_district_size::Int`: The target district size.
- `model`: The trained GNN model.

# Returns
- A `Solution` object representing the districting solution for the small city.
"""

function find_solution_small_city(city::String, target_district_size::Int,  model, mean_train, std_train)
    model_params = (model, mean_train, std_train)
    instance = load_instance(city, NB_BU_SMALL, target_district_size, DEPOT_LOCATION)
    path = "data/tspCosts/$(city)_$(DEPOT_LOCATION)_$(NB_BU_SMALL)_$(target_district_size)_tsp.train_and_test.json"
    costloader = load_precomputed_costs(path)
    solution = Exact_solve_instance(instance, costloader, "GNN", model_params)
    pred_cost = 0
    for d in solution.districts
        pred_cost += compute_cost_with_precomputed_data(instance, d, costloader)
    end
    solution.cost = pred_cost
    return solution
end
end
