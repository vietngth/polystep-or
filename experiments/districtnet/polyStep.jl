module polyStep
export find_solution_city, fitGeneralModel, find_solution_small_city, find_districting_solution
using InferOpt, Plots
using MLUtils, GraphNeuralNetworks, Graphs, MetaGraphs, GraphPlot
using JSON, FilePaths, Random, LinearAlgebra
using Distributions, Statistics, UnionFind, Flux
using DataStructures, Serialization, FileIO
using Base.Filesystem: isfile
using Combinatorics, Optim
using JuMP, GLPK, CxxWrap, DistributedArrays, JLD2
using Distributed
import PolyStep

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
const DEPOT_LOCATION = "C"
const NB_BU_SMALL = 30
const STRATEGY = "districtNet"
const NB_SCENARIO = 100
const HIDDEN_SIZE = 64
const MAX_TIME = parse(Int, get(ENV, "DN_MAX_TIME", "1200"))
const PERTURBATION_PROBABILITY = 0.985
const PENALITY = 10000

"""Return the PolyStep configuration of the run, read from the file named by POLYSTEP_CONFIG."""
function run_config()
    path = get(ENV, "POLYSTEP_CONFIG", "")
    isempty(path) && error("set POLYSTEP_CONFIG to the JSON configuration of the run")
    return JSON.parsefile(path)
end

model_path(nb_data, cfg) = "models/GeneralPolyStep_$(nb_data)_s$(cfg["seed"])_$(cfg["tag"]).jld2"

cost_path(city, target) =
    "data/tspCosts/$(city)_$(DEPOT_LOCATION)_$(NB_BU_SMALL)_$(target)_tsp.train_and_test.json"

schedule(pair, steps) = pair[1] == pair[2] ? Float64(pair[1]) :
    PolyStep.CosineEpsilon(; init = Float64(pair[1]), target = Float64(pair[2]),
                           total_steps = max(steps - 1, 1))

function flat_layout(model)
    shapes = Tuple[]
    marked = Flux.fmap(model) do x
        x isa AbstractArray{<:Number} || return x
        push!(shapes, size(x))
        fill(Float32(length(shapes)), size(x))
    end
    v, _ = Flux.destructure(marked)
    blocks = Tuple{Int,Int,Tuple}[]
    i, n = 1, length(v)
    while i <= n
        id = Int(v[i])
        j = i
        while j < n && Int(v[j + 1]) == id
            j += 1
        end
        push!(blocks, (i, j - i + 1, shapes[id]))
        i = j + 1
    end
    return blocks
end

function hybrid_subspace(layout, n, rank, max_dim, seed)
    offset = 0
    for (start, count, _) in layout
        start == offset + 1 || error("non-contiguous parameter layout at offset $start")
        offset += count
    end
    offset == n || error("layout covers $offset parameters, theta has $n")
    params = PolyStep.ParamLayout([("p$(k)" => shape) for (k, (_, _, shape)) in enumerate(layout)])
    return PolyStep.HybridSubspace(params; rank = rank, seed = seed,
                                   max_subspace_dim = max_dim > 0 ? max_dim : nothing)
end

"""Realized districting cost of a parameter vector, averaged over the training cities."""
function realized_cost(vec, re, data_train, mean_train, std_train)
    model = re(vec)
    total = 0.0
    for (gfi, _) in data_train
        phi = predict_theta(gfi.instance, STRATEGY, model, mean_train, std_train)
        solution = try
            Exact_solve_instance(gfi.instance, gfi.costloader, "CMST", phi)
        catch e
            println("candidate solve failed, penalty cost used: ", sprint(showerror, e))
            nothing
        end
        if solution === nothing
            total += Float64(PENALITY)
            continue
        end
        total += sum(compute_cost_with_precomputed_data(gfi.instance, d, gfi.costloader)
                     for d in solution.districts; init = 0.0)
    end
    return total / length(data_train)
end

const WORKER_CONTEXT = Ref{Any}(nothing)

function set_worker_context!(data_train, mean_train, std_train, re, subspace, theta0)
    WORKER_CONTEXT[] = (data_train = data_train, mean_train = mean_train, std_train = std_train,
                        re = re, subspace = subspace, theta0 = collect(Float64, theta0))
    return myid()
end

function worker_cost(z::Vector{Float64})
    c = WORKER_CONTEXT[]
    theta = c.theta0 .+ PolyStep.expand(c.subspace, z)
    return realized_cost(theta, c.re, c.data_train, c.mean_train, c.std_train)
end

"""Train the districting GNN with PolyStep on the realized cost of its districtings."""
function train_polystep(model, data_train, mean_train, std_train, cfg)
    theta, re = Flux.destructure(model)
    theta0 = collect(Float64, theta)
    n = length(theta0)
    steps, seed = Int(cfg["steps"]), Int(cfg["seed"])
    subspace = hybrid_subspace(flat_layout(model), n, Int(cfg["rank"]),
                               Int(something(cfg["max_subspace_dim"], 0)), seed)
    q = PolyStep.subspace_dim(subspace)
    println("PolyStep: $(n) parameters, subspace dimension $(q), $(steps) steps, seed $(seed)")
    pool = nworkers() > 1 ? WorkerPool(workers()) : nothing
    if pool !== nothing
        for w in workers()
            remotecall_wait(set_worker_context!, w, data_train, mean_train, std_train, re,
                            subspace, theta0)
        end
    end
    tmp = Vector{Float64}(undef, n)
    function batch_cost(Z)
        columns = [Vector{Float64}(view(Z, :, j)) for j in 1:size(Z, 2)]
        pool === nothing || return Vector{Float64}(pmap(worker_cost, pool, columns))
        return [realized_cost(theta0 .+ PolyStep.expand!(tmp, subspace, z), re, data_train,
                              mean_train, std_train) for z in columns]
    end
    momentum = Float64(cfg["momentum"])
    config = PolyStep.PolyStepConfig(
        dim = q,
        polytope = Symbol(cfg["polytope"]),
        solver = PolyStep.SoftmaxSolver(),
        epsilon = schedule(cfg["epsilon"], steps),
        step_radius = schedule(cfg["step_radius"], steps),
        probe_radius = schedule(cfg["probe_radius"], steps),
        scale_cost = :mean,
        num_probe = Int(cfg["num_probe"]),
        max_iterations = steps,
        min_iterations = steps,
        use_momentum = momentum > 0.0,
        momentum_init = momentum,
        momentum_final = momentum,
        velocity_lr = 1.0,
    )
    state = PolyStep.init_state(config, reshape(zeros(Float64, q), q, 1))
    PolyStep.solve!(batch_cost, config, state; rng = Xoshiro(seed + 1))
    best = round(state.best_f, digits = 4)
    println("PolyStep: best training cost $(best) after $(state.iteration) steps")
    return re(collect(Float32, theta0 .+ PolyStep.expand(subspace, state.best_x)))
end

function aggregateCityTrainingData(nb_data, start = 1)
    target_district_size = 3
    data_train = []
    for city in ["city" * string(i) for i = start:nb_data+start-1]
        instance = build_instance(city, NB_BU_SMALL, target_district_size, DEPOT_LOCATION)
        update_edge_weights!(instance.graph, rand(ne(instance.graph)))
        costloader = load_precomputed_costs(cost_path(city, target_district_size))
        solution, unique_subgraphs = districting_exact_solver(instance, costloader)
        solution === nothing && continue
        y = randomized_constructor(solution, 20)
        g = deepcopy(instance.graph)
        data = hcat(extract_edge_feature(g)...)
        gnn_graph = create_edge_graph(g, data)
        push!(data_train, (GraphFeaturesInstance(data, instance, solution.cost, gnn_graph,
                                                 unique_subgraphs, costloader, solution), y))
    end
    return data_train
end

"""Train the PolyStep model on nb_data training cities, or load it when it exists."""
function fitGeneralModel(nb_data)
    cfg = run_config()
    Random.seed!(Int(cfg["seed"]))
    data_train = aggregateCityTrainingData(nb_data)
    data_train, _ = splitobs(shuffleobs(data_train), at = 1.0)
    println("Training data size: ", length(data_train))
    data_train, mean_train, std_train = normalize_features(data_train)
    model = build_gnn_model(data_train[1][1].instance.graph, STRATEGY, HIDDEN_SIZE)
    path = model_path(nb_data, cfg)
    if isfile(path)
        @info "Loading model from file" path
        Flux.loadmodel!(model, JLD2.load(path, "model_state"))
        return model, mean_train, std_train
    end
    model = train_polystep(model, data_train, mean_train, std_train, cfg)
    isdir("models") || mkpath("models")
    model_state = Flux.state(model)
    jldsave(path; model_state)
    return model, mean_train, std_train
end

function find_solution_city(city::String, target_district_size::Int, NB_BU::Int,
                            depot_location::String, model, mean_train, std_train)
    instance = build_instance(city, NB_BU, target_district_size, depot_location)
    update_edge_weights!(instance.graph, rand(ne(instance.graph)))
    phi = predict_theta(instance, STRATEGY, model, mean_train, std_train)
    update_edge_weights!(instance.graph, phi)
    return ILS_solve_instance(instance, Costloader([], []), "CMST", phi)
end

function find_solution_small_city(city::String, target_district_size::Int, model, mean_train,
                                  std_train)
    costloader = load_precomputed_costs(cost_path(city, target_district_size))
    instance = build_instance(city, NB_BU_SMALL, target_district_size, DEPOT_LOCATION)
    update_edge_weights!(instance.graph, rand(ne(instance.graph)))
    phi = predict_theta(instance, STRATEGY, model, mean_train, std_train)
    solution = Exact_solve_instance(instance, costloader, "CMST", phi)
    solution.cost = sum(compute_cost_with_precomputed_data(instance, d, costloader)
                        for d in solution.districts; init = 0.0)
    return solution
end

function find_districting_solution(city::String, target_district_size::Int)
    instance = build_instance(city, NB_BU_SMALL, target_district_size, DEPOT_LOCATION)
    costloader = load_precomputed_costs(cost_path(city, target_district_size))
    solution, _ = districting_exact_solver(instance, costloader)
    return solution
end

end
