# Feasibility side-by-side on the large instance (ilsdefrance, 2000 BU, target-20 -> sizes [16,24]).
#
# THE POINT (user's framing, code-verified): the deploy path is predict -> CMST -> ILS+repair for
# EVERYONE. `find_solution_city` calls `ILS_solve_instance(..., "CMST", phi)` (CMST init + ILS +
# repair_districts); `Exact_solve_instance` is the RAW CMST (no repair) that exposes the orphan. The
# CMST greedy gates only the UPPER cap (Kruskal.jl: size_sum <= target); the LOWER bound lives only
# in repair + label imitation. So the fair comparison applies the SAME repair to original PolyStep.
#
# Four cases on the SAME instance, same edge-weight RNG seed (so geometry is identical):
#   1. DistrictNet (orig)            : FY model        -> ILS+repair        (feasible via repair)
#   2. PolyStep, raw                 : cost-only model -> RAW CMST, no repair (shows the orphan)
#   3. PolyStep + same repair  <NEW> : cost-only model -> ILS+repair        (feasible via repair)
#   4. PolyStep + size penalty       : penalty model   -> ILS+repair        (feasible from training)
#
# Emits output/feasibility_sidebyside.json {case -> {districts:[[block ids]], sizes, cost,
# violated:[idx], feasible, n_districts}} + per-case saved solutions for generate_city_mapplot, and
# (case 2) the violated district id to CIRCLE. Run on a COMPUTE node after the 3 models are trained.
#
#   julia +1.10 --project=. feasibility_sidebyside.jl
using JSON, JLD2
include("src/struct.jl"); include("src/instance.jl"); include("src/district.jl")
include("src/solution.jl"); include("src/utils.jl")
include("src/Solver/Kruskal.jl"); include("src/Solver/exactsolver.jl"); include("src/Solver/localsearch.jl")
include("src/Estimators/polyStep.jl"); include("src/Estimators/predictGnn.jl")

const CITY   = get(ENV, "DN_CITY", "ilsdefrance")
const NB_BU  = parse(Int, get(ENV, "DN_NBBU", "2000"))
const TARGET = parse(Int, get(ENV, "DN_TARGET", "20"))
const DEPOT  = get(ENV, "DN_DEPOT", "C")
const SEED   = parse(Int, get(ENV, "DN_SEED", "1"))

# model paths (trained beforehand; PolyStep cost-only = PS_SIZE_PENALTY=0, penalty = >0)
const M_DNET    = get(ENV, "DN_MODEL_DNET",    "models/$(CITY)_districtnet.jld2")
const M_PS_COST = get(ENV, "DN_MODEL_PSCOST",  "models/$(CITY)_polystep_cost.jld2")
const M_PS_PEN  = get(ENV, "DN_MODEL_PSPEN",   "models/$(CITY)_polystep_penalty.jld2")

load_model(p) = (d = JLD2.load(p); (d["model"], d["mean_train"], d["std_train"]))

# districts (Vector of block-id Vectors) from a districting solution
districts_of(sol) = [collect(d.nodes) for d in sol.districts]

function summarize(instance, districts)
    lo, hi = instance.min_district_size, instance.max_district_size
    sizes = [length(d) for d in districts]
    violated = [i for (i, s) in enumerate(sizes) if s < lo || s > hi]
    Dict("districts" => districts, "sizes" => sizes,
         "violated" => violated, "feasible" => isempty(violated),
         "n_districts" => length(districts),
         "min" => lo, "max" => hi)
end

# nominal districting cost = sum of per-district routing cost (SAA/precomputed). Empty costloader ->
# computed on the fly inside compute_cost_with_precomputed_data as in find_solution_city.
function nominal_cost(instance, sol, costloader)
    c = 0.0
    for d in sol.districts
        c += compute_cost_with_precomputed_data(instance, d, costloader)
    end
    c
end

function build()
    inst = build_instance(CITY, NB_BU, TARGET, DEPOT)
    Random.seed!(SEED)                      # identical edge-weight init across cases
    W = rand(ne(inst.graph)); update_edge_weights!(inst.graph, W)
    inst
end

function run_case(inst, model_tuple, mode)
    model, mean_train, std_train = model_tuple
    phi = predict_theta(inst, STRATEGY, model, mean_train, std_train)
    update_edge_weights!(inst.graph, phi)
    costloader = Costloader([], [])
    sol = mode == :raw ? Exact_solve_instance(inst, costloader, "CMST", phi) :
                         ILS_solve_instance(inst, costloader, "CMST", phi)   # CMST + ILS + repair
    ds = districts_of(sol)
    info = summarize(inst, ds)
    info["cost"] = nominal_cost(inst, sol, costloader)
    info
end

function main()
    inst = build()
    out = Dict{String,Any}("city" => CITY, "nb_bu" => NB_BU, "target" => TARGET,
                            "min" => inst.min_district_size, "max" => inst.max_district_size)
    cases = Dict{String,Any}()
    # VERIFICATION (the crux): does DistrictNet's OWN raw CMST also violate? The Kruskal merge gate
    # (size_sum <= target) is identical for any theta; theta only reorders edges, not the cap -> at
    # the [16,24]/no-split-window scale the greedy strands undersized leftovers regardless of model.
    # If case 0 violates like case 2, the repair is the SHARED feasibility mechanism for everyone.
    @info "Case 0: DistrictNet (FY) -> RAW CMST (no repair) [does THEIR raw output violate too?]"
    cases["0_districtnet_raw"]  = run_case(inst, load_model(M_DNET),    :raw)
    @info "Case 1: DistrictNet (FY) -> ILS+repair"
    cases["1_districtnet"]      = run_case(inst, load_model(M_DNET),    :ils)
    @info "Case 2: PolyStep cost-only -> RAW CMST (no repair) [exposes orphan]"
    cases["2_polystep_raw"]     = run_case(inst, load_model(M_PS_COST), :raw)
    @info "Case 3: PolyStep cost-only -> SAME ILS+repair  [the fair baseline]"
    cases["3_polystep_repair"]  = run_case(inst, load_model(M_PS_COST), :ils)
    @info "Case 4: PolyStep + size penalty -> ILS+repair"
    cases["4_polystep_penalty"] = run_case(inst, load_model(M_PS_PEN),  :ils)
    out["cases"] = cases
    mkpath("output")
    open("output/feasibility_sidebyside.json", "w") do f; JSON.print(f, out, 2) end
    # console summary
    for k in sort(collect(keys(cases)))
        c = cases[k]
        @info "$k : n=$(c["n_districts"]) cost=$(round(c["cost"],digits=1)) " *
              "feasible=$(c["feasible"]) violated=$(length(c["violated"]))"
    end
    @info "wrote output/feasibility_sidebyside.json — render maps with src/Export/plots.jl (circle the case-2 orphan)"
end

main()
