# =============================================================================
# DistrictNet "train-at-scale" experiment.
#
# THESIS: DistrictNet (and the small-trained PolyStep variant) train on 30-BU / target-3 instances
# where the EXACT optimal districting LABEL is computable, then deploy on large (target-20, >=120 BU)
# instances where the label is NP-hard / intractable. PolyStep is label-free (minimises the realised
# SAA routing cost of the deployed predict->CMST-construct->ILS pipeline), so it can train DIRECTLY at
# (or near) deploy scale. This driver compares, on held-out large instances, three deploy pipelines:
#   (a) DistrictNet  train-small (30BU/t3, FY label imitation)      -> deploy large  [existing model]
#   (b) PolyStep     train-small (30BU/t3, realised cost)           -> deploy large  [existing model]
#   (c) PolyStep     TRAIN-AT-SCALE (large real cities, target-20)  -> deploy large  [NEW]
# Metric: SAA routing cost (shared C++ evaluator, identical for all three) + feasibility (full district
# count, no size-violated/orphan districts). Success = (c) matches/beats (a),(b) on cost AND feasible.
#
#   julia +1.10 -t 1 --project=. trainatscale_driver.jl
# ENV: PS_SCALE_RETRAIN=1 forces retrain; PS_SCALE_ID names the model; PS_SIZE_PENALTY>0 adds the
#      explicit size-constraint term to the at-scale objective; PS_SCALE_STEPS/PS_SCALE_PROBES tune it.
# =============================================================================
using JSON, Random, Statistics, JLD2
using Dates: now
include("experiments.jl")
using .districtNet, .polyStep

const TARGET    = parse(Int, get(ENV, "TAS_TARGET", "20"))
const NB_SCEN   = parse(Int, get(ENV, "TAS_NB_SCENARIO", "100"))   # must match polyStep.NB_SCENARIO (SAA target string)
const SCALE_ID  = get(ENV, "PS_SCALE_ID", "trainatscale")
const SEED      = parse(Int, get(ENV, "TAS_SEED", "1"))
const SMALL_ID  = parse(Int, get(ENV, "TAS_SMALL_ID", "10"))       # existing small-trained model ids (nb_data)

# --- train / test split over real cities at deploy scale (target-20). Train and test cities/sizes are
#     disjoint where possible (Marseille/Birmingham fully held-out cities; London@500 a scale-up
#     extrapolation beyond the largest training size). Scenario files exist on disk for all of these. ---
parse_pairs(s, dc, db) = begin
    if isempty(s); return (dc, db); end
    cs = String[]; bs = Int[]
    for tok in split(s, ',')
        c, b = split(strip(tok), ':'); push!(cs, String(c)); push!(bs, parse(Int, b))
    end
    (cs, bs)
end
TRAIN_CITIES, TRAIN_BUS = parse_pairs(get(ENV, "TAS_TRAIN", ""),
    ["London", "Leeds",  "Bristol", "Manchester", "Lyon", "Paris"],
    [ 200,      190,      160,       120,          120,    120])
TEST_CITIES, TEST_BUS = parse_pairs(get(ENV, "TAS_TEST", ""),
    ["Marseille", "Birmingham", "London", "Leeds"],
    [ 120,         120,          300,      240])

# SCmain (C++ scenario generator) is wrapped inside each estimator module.
ensure_scenarios(city, bu) = begin
    inst_str = "$(city)_C_$(bu)_$(TARGET)_$(NB_SCEN)"
    spath = "deps/Scenario/output/$(city)_C_$(bu)_$(TARGET).json"
    if get(ENV, "TAS_REGEN_SCEN", "1") == "1" || !isfile(spath)
        println("  scenarios: $city bu=$bu t=$TARGET n=$NB_SCEN"); flush(stdout)
        try; polyStep.SCmain(inst_str); catch e; @warn "SCmain failed" city bu e; end
    end
end

# Evaluate with the SAME module that produced the solution: each estimator module defines its own
# District/Instance/Costloader types (identical layout, distinct types), so cross-module cost calls
# MethodError. The C++ SAA evaluator is module-independent (same node ids + scenario), so this is a fair
# common-evaluator comparison.
function saa_eval(modmod, city, bu, sol)
    instance = modmod.build_instance(city, bu, TARGET, "C")
    lo, hi = instance.min_district_size, instance.max_district_size
    sizes = [length(d.nodes) for d in sol.districts]
    violated = count(s -> s < lo || s > hi, sizes)
    cost = 0.0
    for d in sol.districts
        cost += modmod.compute_cost_with_precomputed_data(instance, d, modmod.Costloader([], []))  # -> SAA (C++)
    end
    target_count = Int(floor(instance.num_blocks / instance.target_district_size))
    return Dict("cost" => cost, "n_districts" => length(sol.districts),
                "target_count" => target_count, "min_size" => lo, "max_size" => hi,
                "sizes" => sizes, "n_violated" => violated,
                "feasible" => (violated == 0))
end

function deploy(modmod, city, bu, mtuple)
    Random.seed!(SEED)
    sol = modmod.find_solution_city(city, TARGET, bu, "C", mtuple...)
    return sol
end

println("=== TRAIN-AT-SCALE EXPERIMENT @ $(now()) ===")
println("train = ", collect(zip(TRAIN_CITIES, TRAIN_BUS)))
println("test  = ", collect(zip(TEST_CITIES, TEST_BUS)))
println("target=$TARGET  nb_scenario=$NB_SCEN  size_penalty=$(polyStep.PS_SIZE_PENALTY)")
flush(stdout)

# scenarios for every train+test instance (consistent NB_SCEN for the SAA evaluator)
for (c, b) in vcat(collect(zip(TRAIN_CITIES, TRAIN_BUS)), collect(zip(TEST_CITIES, TEST_BUS)))
    ensure_scenarios(c, b)
end

# --- models ---
println("\n--- loading DistrictNet train-small model (GeneralDistrictNet_$(SMALL_ID)) ---"); flush(stdout)
dnet = districtNet.fitGeneralModel(SMALL_ID)
println("--- loading PolyStep train-small model (GeneralPolyStep_$(SMALL_ID)) ---"); flush(stdout)
ps_small = polyStep.fitGeneralModel(SMALL_ID)
println("--- training/loading PolyStep TRAIN-AT-SCALE model (GeneralPolyStepScale_$(SCALE_ID)) ---"); flush(stdout)
ps_scale = polyStep.fitGeneralModel_atscale(TRAIN_CITIES, TRAIN_BUS, TARGET, NB_SCEN, SCALE_ID)

methods = [("DistrictNet_trainsmall", districtNet, dnet),
           ("PolyStep_trainsmall",    polyStep,    ps_small),
           ("PolyStep_trainatscale",  polyStep,    ps_scale)]

results = Dict{String,Any}()
for (city, bu) in zip(TEST_CITIES, TEST_BUS)
    key = "$(city)_$(bu)_$(TARGET)"
    results[key] = Dict{String,Any}()
    println("\n========== TEST INSTANCE $key =========="); flush(stdout)
    for (name, modmod, mtuple) in methods
        try
            sol = deploy(modmod, city, bu, mtuple)
            m = saa_eval(modmod, city, bu, sol)
            results[key][name] = m
            println(">>> $name : SAA cost=$(round(m["cost"],digits=3))  districts=$(m["n_districts"])/$(m["target_count"])  violated=$(m["n_violated"])  feasible=$(m["feasible"])"); flush(stdout)
        catch e
            results[key][name] = Dict("error" => string(e))
            @warn "deploy failed" key name e
        end
    end
end

# --- write results ---
isdir("exp_results") || mkpath("exp_results")
payload = Dict("meta" => Dict("target" => TARGET, "nb_scenario" => NB_SCEN, "seed" => SEED,
                              "train" => collect(zip(TRAIN_CITIES, TRAIN_BUS)),
                              "test" => collect(zip(TEST_CITIES, TEST_BUS)),
                              "size_penalty" => polyStep.PS_SIZE_PENALTY,
                              "scale_steps" => polyStep.PS_SCALE_STEPS,
                              "scale_probes" => polyStep.PS_SCALE_PROBES,
                              "sfge_julia" => "not wired in Julia/InferOpt (only PerturbedAdditive+FenchelYoung); SFGE-vs-PolyStep scaling done separately in Python"),
               "results" => results)
open("exp_results/districtnet_trainatscale_$(SCALE_ID).json", "w") do io; JSON.print(io, payload, 2); end
println("\nwrote exp_results/districtnet_trainatscale_$(SCALE_ID).json")

# markdown summary
open("exp_results/districtnet_trainatscale_$(SCALE_ID).md", "w") do io
    println(io, "# DistrictNet train-at-scale (label-free PolyStep) -- target-$TARGET, $NB_SCEN scenarios [$(SCALE_ID)]\n")
    println(io, "Train (at scale, no labels): ", join(["$(c)@$(b)" for (c,b) in zip(TRAIN_CITIES,TRAIN_BUS)], ", "))
    println(io, "\n| test instance | method | SAA cost | districts/target | violated | feasible |")
    println(io, "|---|---|---|---|---|---|")
    for (city, bu) in zip(TEST_CITIES, TEST_BUS)
        key = "$(city)_$(bu)_$(TARGET)"
        for (name, _, _) in methods
            r = get(results[key], name, Dict())
            if haskey(r, "error")
                println(io, "| $key | $name | ERROR | | | |")
            else
                println(io, "| $key | $name | $(round(r["cost"],digits=2)) | $(r["n_districts"])/$(r["target_count"]) | $(r["n_violated"]) | $(r["feasible"]) |")
            end
        end
    end
    println(io, "\nSFGE at train-at-scale: not attempted in Julia -- the Julia/InferOpt stack only wires")
    println(io, "PerturbedAdditive+FenchelYoung (DistrictNet's differentiable surrogate); no score-function")
    println(io, "estimator is implemented. The SFGE-vs-PolyStep scaling comparison is done in the Python track.")
end
println("wrote exp_results/districtnet_trainatscale.md")
