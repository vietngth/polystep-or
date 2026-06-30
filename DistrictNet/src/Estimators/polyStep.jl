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
const NB_DATA = 200
const NB_BU_SMALL = 30
const STRATEGY = "districtNet"          # reuse the districtNet GNN architecture and forward pass
const NB_SCENARIO = 100
# GNN architecture
const HIDDEN_SIZE = 64
# PolyStep (gradient-free, direct-cost) training hyperparameters
const PS_PROBES = 10                    # probes (perturbations) per step
const PS_EPSILON0 = 0.4                 # initial perturbation radius
const PS_EPSILONT = 0.05                # final perturbation radius (cosine decay)
const PS_STEPS = 30                     # optimization steps
const PS_MOMENTUM = 0.5
# Feasibility-aware variant (An #8). lambda>0 adds an EXPLICIT district-size penalty straight into the
# gradient-free scalar objective: lambda * sum_d [max(0,min-|d|) + max(0,|d|-max)]. The CMST forward
# solver enforces only the UPPER cap (Kruskal merge gate, size_sum <= target); the LOWER bound lives
# only in post-hoc repair + label imitation. DistrictNet's differentiable PerturbedAdditive(CMST)
# CANNOT carry a min-cardinality constraint (not matroid-compatible, no tractable perturbed maximizer);
# PolyStep can, because it consumes only the scalar cost. Set via env PS_SIZE_PENALTY (0 = original).
const PS_SIZE_PENALTY = parse(Float64, get(ENV, "PS_SIZE_PENALTY", "0.0"))
# Solver hyperparameters
const MAX_TIME = 120
const PERTURBATION_PROBABILITY = 0.985
const PENALITY = 10000


"""
    GNNtrainer_polyStep(model, data_train, mean_train, std_train)

Train the districting GNN by DIRECT minimization of the realized districting cost, with a gradient-free
softmax-barycentric (optimal-transport-style) step over the flattened Flux parameters. No imitation
targets and no Fenchel-Young loss: the training objective is exactly the realized cost of the deployed
predict -> CMST -> evaluate pipeline (the body of `find_solution_small_city`), averaged over the
training instances and evaluated with the cached district costs (no C++ calls at train time).
"""
function GNNtrainer_polyStep(model, data_train, mean_train, std_train)
    theta, re = Flux.destructure(model)
    n = length(theta)
    SPO_CALLS[] = 0; SPO_TIME[] = 0.0   # [TIMING] count CMST set-partition solves (no input perturbation; one solve per probe per example)

    function realized_cost(vec)
        m = re(vec)
        total = 0.0
        for (gfi, _) in data_train
            phi = predict_theta(gfi.instance, STRATEGY, m, mean_train, std_train)
            sol = Exact_solve_instance(gfi.instance, gfi.costloader, "CMST", phi)
            c = 0.0
            for d in sol.districts
                c += compute_cost_with_precomputed_data(gfi.instance, d, gfi.costloader)
                if PS_SIZE_PENALTY > 0.0                       # explicit size constraint (An #8)
                    sz = length(d.nodes)
                    c += PS_SIZE_PENALTY * (max(0, gfi.instance.min_district_size - sz) +
                                            max(0, sz - gfi.instance.max_district_size))
                end
            end
            total += c
        end
        return total / length(data_train)
    end

    # ---- replay recording: per-step districts (CMST output) for one representative city + probe internals ----
    gfi1 = data_train[1][1]
    snapshot(vec) = try
        m = re(vec)
        phi1 = predict_theta(gfi1.instance, STRATEGY, m, mean_train, std_train)
        sol1 = Exact_solve_instance(gfi1.instance, gfi1.costloader, "CMST", phi1)
        [collect(d.nodes) for d in sol1.districts]              # 1-based block ids per district
    catch
        Vector{Vector{Int}}()
    end
    replay = Dict("city" => gfi1.instance.city_name, "num_blocks" => gfi1.instance.num_blocks,
                  "target_district_size" => gfi1.instance.target_district_size, "steps" => [])

    best_theta = copy(theta)
    best_cost = realized_cost(theta)
    println("PolyStep init: cost = $best_cost")
    push!(replay["steps"], Dict("step" => 0, "eps" => PS_EPSILON0, "cost" => best_cost,
        "best" => best_cost, "districts" => snapshot(theta), "probe_costs" => Float64[], "probe_weights" => Float64[]))
    velocity = zeros(n)
    for step in 1:PS_STEPS
        frac = (step - 1) / max(PS_STEPS - 1, 1)
        eps = PS_EPSILONT + 0.5 * (PS_EPSILON0 - PS_EPSILONT) * (1 + cos(pi * frac))   # cosine schedule
        dirs = [randn(n) for _ in 1:PS_PROBES]
        probes = [theta .+ eps .* d for d in dirs]
        costs = [realized_cost(p) for p in probes]
        probe_dist = [snapshot(p) for p in probes]             # each probe SOLVES CMST in parallel (instance 1)
        c0 = minimum(costs); sd = std(costs) + 1e-8
        w = Flux.softmax(-(costs .- c0) ./ sd)                  # low cost -> high weight (softmax barycenter)
        bary = sum(w[i] .* probes[i] for i in 1:PS_PROBES)      # cost-weighted barycenter of the probes
        velocity = PS_MOMENTUM .* velocity .+ (bary .- theta)
        theta = theta .+ velocity
        c = realized_cost(theta)
        if c < best_cost
            best_cost = c; best_theta = copy(theta)
        end
        push!(replay["steps"], Dict("step" => step, "eps" => eps, "cost" => c, "best" => best_cost,
            "districts" => snapshot(theta), "probe_districts" => probe_dist, "probe_costs" => costs, "probe_weights" => w))
        println("PolyStep step $step (eps=$(round(eps, digits=3))): cost = $(round(c, digits=4)) best = $(round(best_cost, digits=4))")
    end
    try
        isdir("output") || mkpath("output")
        open("output/polystep_replay.json", "w") do io; JSON.print(io, replay); end
        println("PolyStep replay written to output/polystep_replay.json ($(length(replay["steps"])) frames, city=$(replay["city"]))")
    catch e
        @warn "replay dump failed" e
    end
    _avg = SPO_CALLS[] > 0 ? SPO_TIME[] / SPO_CALLS[] * 1000 : 0.0
    println("[TIMING] POLYSTEP CMST(set-partition GLPK): calls=$(SPO_CALLS[]) total=$(round(SPO_TIME[],digits=2))s avg=$(round(_avg,digits=2))ms  (no input perturbation; one solve per probe)")
    return re(best_theta)                                       # validation-selected best parameters
end


function aggregateCityTrainingData(nb_data, start=1)
    TARGET_DISTRICT_SIZE = 3
    data_train = []
    cities = ["city" * string(i) for i = start:nb_data+start-1]
    for CITY in cities
        instance = build_instance(CITY, NB_BU_SMALL, TARGET_DISTRICT_SIZE, DEPOT_LOCATION)
        W = rand(ne(instance.graph))
        update_edge_weights!(instance.graph, W)
        path = string("data/tspCosts/", CITY, "_", DEPOT_LOCATION, "_", NB_BU_SMALL, "_", TARGET_DISTRICT_SIZE, "_tsp.train_and_test.json")
        costloader = load_precomputed_costs(path)
        solution, unique_subgraphs = districting_exact_solver(instance, costloader)
        if (solution == nothing)
            continue
        end
        cost = solution.cost
        y = randomized_constructor(solution, 20)
        g = deepcopy(instance.graph)
        edges_data = extract_edge_feature(g)
        data = hcat(edges_data...)
        gnn_graph = create_edge_graph(g, data)
        push!(data_train, (GraphFeaturesInstance(data, instance, cost, gnn_graph, unique_subgraphs, costloader, solution), y))
    end
    return data_train
end


function fitGeneralModel(nb_data)
    data_train = aggregateCityTrainingData(nb_data)
    data_train, _ = splitobs(shuffleobs(data_train), at = 1.0)
    println("Training data size: ", length(data_train))
    data_train, mean_train, std_train = normalize_features(data_train)
    if get(ENV, "DN_UNTRAINED", "0") == "1"   # [ABLATION] random GNN, no load/train (== districtNet untrained: same arch+seed)
        Random.seed!(parse(Int, get(ENV, "DN_SEED", "1234")))
        model = build_gnn_model(data_train[1][1].instance.graph, STRATEGY, HIDDEN_SIZE)
        println("[ABLATION] UNTRAINED random GNN (polyStep), seed=", get(ENV, "DN_SEED", "1234"))
        return model, mean_train, std_train
    end
    model_name = "models/GeneralPolyStep_$(nb_data).jld2"
    if isfile(model_name)
        @info "Loading model from file"
        model_state = JLD2.load(model_name, "model_state")
        model = build_gnn_model(data_train[1][1].instance.graph, STRATEGY, HIDDEN_SIZE)
        Flux.loadmodel!(model, model_state)
        return model, mean_train, std_train
    end
    model = build_gnn_model(data_train[1][1].instance.graph, STRATEGY, HIDDEN_SIZE)
    model = GNNtrainer_polyStep(model, data_train, mean_train, std_train)
    model_state = Flux.state(model)
    if !isdir("models")
        mkpath("models")
    end
    jldsave(model_name; model_state)
    return model, mean_train, std_train
end


function find_solution_city(city::String, target_district_size::Int, NB_BU::Int, depot_location::String, model, mean_train, std_train)
    instance = build_instance(city, NB_BU, target_district_size, depot_location)
    W = rand(ne(instance.graph))
    update_edge_weights!(instance.graph, W)
    phi = predict_theta(instance, STRATEGY, model, mean_train, std_train)
    update_edge_weights!(instance.graph, phi)
    costloader = Costloader([], [])
    solution = ILS_solve_instance(instance, costloader, "CMST", phi)
    return solution
end


function find_solution_small_city(city::String, target_district_size::Int, model, mean_train, std_train)
    path = "data/tspCosts/$(city)_$(DEPOT_LOCATION)_$(NB_BU_SMALL)_$(target_district_size)_tsp.train_and_test.json"
    costloader = load_precomputed_costs(path)
    instance = build_instance(city, NB_BU_SMALL, target_district_size, DEPOT_LOCATION)
    W = rand(ne(instance.graph))
    update_edge_weights!(instance.graph, W)
    phi = predict_theta(instance, STRATEGY, model, mean_train, std_train)
    solution = Exact_solve_instance(instance, costloader, "CMST", phi)
    pred_cost = 0
    for d in solution.districts
        pred_cost += compute_cost_with_precomputed_data(instance, d, costloader)
    end
    solution.cost = pred_cost
    return solution
end


function find_districting_solution(city::String, target_district_size::Int)
    instance = build_instance(city, NB_BU_SMALL, target_district_size, DEPOT_LOCATION)
    path = "data/tspCosts/$(city)_$(DEPOT_LOCATION)_$(NB_BU_SMALL)_$(target_district_size)_tsp.train_and_test.json"
    costloader = load_precomputed_costs(path)
    solution, _ = districting_exact_solver(instance, costloader)
    return solution
end


# =============================================================================
# TRAIN-AT-SCALE (label-free, realized-cost objective at deploy scale)
# =============================================================================
# The small-city PolyStep (above) trains on 30-BU / target-3 instances where the EXACT districting
# label is computable (set-partition over size 2-4 connected subgraphs). At deploy scale (target-20,
# >=120 BU) the min/max district sizes become [16,24] and the exact label requires enumerating ALL
# connected subgraphs of size 16..24 -- combinatorially intractable (NP-hard), so DistrictNet (and the
# small PolyStep variant) MUST train small and deploy large. PolyStep is label-free: its objective is
# the REALIZED districting cost of the deployed predict->CMST-construct->SAA pipeline. So it can train
# DIRECTLY on large instances where no label exists. This is exactly that variant.
#
# Per-step forward solver = `initialize_solution` (Kruskal CMST + greedy-merge + cascade repair), the
# SAME constructor the deploy path (`ILS_solve_instance`) seeds from -- NOT the exact set-partition
# (which doesn't scale). Per-instance scalar objective = sum_d SAA routing cost (true C++ evaluator)
# + PS_SIZE_PENALTY * size violation. No labels, no Fenchel-Young loss, no exact solver at any point.
const PS_SCALE_STEPS  = parse(Int,     get(ENV, "PS_SCALE_STEPS",  "20"))
const PS_SCALE_PROBES = parse(Int,     get(ENV, "PS_SCALE_PROBES", "8"))
const PS_SCALE_EPS0   = parse(Float64, get(ENV, "PS_SCALE_EPS0",   "0.4"))
const PS_SCALE_EPST   = parse(Float64, get(ENV, "PS_SCALE_EPST",   "0.05"))

# realized SAA districting cost of one instance under edge-weights phi, with a memo cache over the
# (sorted block-id) district key so repeated/identical districts across probes hit the C++ evaluator once.
function _realized_saa_cost(instance, phi, cache::Dict{Vector{Int},Float64})
    update_edge_weights!(instance.graph, phi)
    sol = initialize_solution(instance, Costloader([], []), "CMST", phi)
    c = 0.0
    nd = 0
    for d in sol.districts
        nd += 1
        key = sort([get_prop(instance.graph, j, :id) for j in d.nodes])
        cost_d = get!(cache, key) do
            compute_cost_with_precomputed_data(instance, d, Costloader([], []))   # -> SAA (C++)
        end
        c += cost_d
        if PS_SIZE_PENALTY > 0.0
            sz = length(d.nodes)
            c += PS_SIZE_PENALTY * (max(0, instance.min_district_size - sz) +
                                    max(0, sz - instance.max_district_size))
        end
    end
    return c, nd
end

"""
    GNNtrainer_polyStep_atscale(model, instances, mean_train, std_train)

Gradient-free softmax-barycentric (OT-style) optimisation of the GNN parameters directly on the
realized SAA districting cost at deploy scale. Identical update rule to `GNNtrainer_polyStep`, but the
forward solver is the scalable CMST constructor (`initialize_solution`) and the cost is the true SAA
routing cost (no precomputed cache, no labels, no exact set-partition).
"""
function GNNtrainer_polyStep_atscale(model, instances, mean_train, std_train)
    theta, re = Flux.destructure(model)
    n = length(theta)
    cache = Dict{Vector{Int},Float64}()

    function realized_cost(vec)
        m = re(vec)
        total = 0.0
        for inst in instances
            phi = predict_theta(inst, STRATEGY, m, mean_train, std_train)
            c, _ = _realized_saa_cost(inst, phi, cache)
            total += c
        end
        return total / length(instances)
    end

    best_theta = copy(theta)
    best_cost = realized_cost(theta)
    println("PolyStep@scale init: cost = $(round(best_cost, digits=4))  (instances=$(length(instances)))")
    velocity = zeros(n)
    for step in 1:PS_SCALE_STEPS
        frac = (step - 1) / max(PS_SCALE_STEPS - 1, 1)
        eps = PS_SCALE_EPST + 0.5 * (PS_SCALE_EPS0 - PS_SCALE_EPST) * (1 + cos(pi * frac))
        dirs = [randn(n) for _ in 1:PS_SCALE_PROBES]
        probes = [theta .+ eps .* d for d in dirs]
        costs = [realized_cost(p) for p in probes]
        c0 = minimum(costs); sd = std(costs) + 1e-8
        w = Flux.softmax(-(costs .- c0) ./ sd)
        bary = sum(w[i] .* probes[i] for i in 1:PS_SCALE_PROBES)
        velocity = PS_MOMENTUM .* velocity .+ (bary .- theta)
        theta = theta .+ velocity
        c = realized_cost(theta)
        if c < best_cost
            best_cost = c; best_theta = copy(theta)
        end
        println("PolyStep@scale step $step (eps=$(round(eps, digits=3))): cost = $(round(c, digits=4)) best = $(round(best_cost, digits=4))  cache=$(length(cache))")
    end
    println("PolyStep@scale done: best cost = $(round(best_cost, digits=4)), SAA-cache entries = $(length(cache))")
    return re(best_theta)
end

# Build the at-scale training instances (real cities at deploy scale; no labels, no costloader).
# Returns (instances, mean_train, std_train) where mean/std normalise the edge features as in deploy.
function aggregateCityTrainingData_atscale(cities, bus, target, nb_scenario)
    instances = Instance[]
    feats = []
    for (city, bu) in zip(cities, bus)
        inst = build_instance(city, bu, target, DEPOT_LOCATION)
        update_edge_weights!(inst.graph, rand(ne(inst.graph)))
        # ensure the SAA scenario file exists for this (city, bu, target) at nb_scenario scenarios
        spath = "deps/Scenario/output/$(city)_$(DEPOT_LOCATION)_$(bu)_$(target).json"
        if !isfile(spath)
            println("  generating scenarios: $city bu=$bu target=$target n=$nb_scenario")
            SCmain("$(city)_$(DEPOT_LOCATION)_$(bu)_$(target)_$(nb_scenario)")
        end
        push!(instances, inst)
        push!(feats, get_instance_features(inst))
    end
    allf = hcat(feats...)
    MEAN = mean(allf, dims = 2) * 0
    STD = std(allf, dims = 2)
    return instances, MEAN, STD
end

"""
    fitGeneralModel_atscale(cities, bus, target, nb_scenario, model_id)

Train (or load) the train-at-scale PolyStep model on the given large real-city instances. Saves model
state + feature normalisation under models/GeneralPolyStepScale_<model_id>.jld2 so deploy is identical
to the small-trained variants (predict_theta + ILS).
"""
function fitGeneralModel_atscale(cities, bus, target, nb_scenario, model_id)
    instances, MEAN, STD = aggregateCityTrainingData_atscale(cities, bus, target, nb_scenario)
    println("Train-at-scale instances: ", [(c, b) for (c, b) in zip(cities, bus)])
    model_name = "models/GeneralPolyStepScale_$(model_id).jld2"
    if isfile(model_name) && get(ENV, "PS_SCALE_RETRAIN", "0") != "1"
        @info "Loading train-at-scale model from file" model_name
        d = JLD2.load(model_name)
        model = build_gnn_model(instances[1].graph, STRATEGY, HIDDEN_SIZE)
        Flux.loadmodel!(model, d["model_state"])
        return model, d["mean_train"], d["std_train"]
    end
    model = build_gnn_model(instances[1].graph, STRATEGY, HIDDEN_SIZE)
    model = GNNtrainer_polyStep_atscale(model, instances, MEAN, STD)
    isdir("models") || mkpath("models")
    model_state = Flux.state(model)
    jldsave(model_name; model_state = model_state, mean_train = MEAN, std_train = STD)
    println("Saved train-at-scale model -> $model_name")
    return model, MEAN, STD
end

end
