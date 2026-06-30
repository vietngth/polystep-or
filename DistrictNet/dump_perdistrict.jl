# Dump per-district SAA costs for each method's solution (for Figure 4: district-cost distributions).
# Reuses the same SAA evaluator as eval_driver; writes one file per (city,t,bu) with lines
# "<method> <c1> <c2> ... <ck>" (one district cost per token after the method name).
# Usage: julia +1.10 --project=. dump_perdistrict.jl <city> <t> <bu>
include("experiment_evaluator.jl")

function dump_perdistrict()
    city = ARGS[1]
    t = parse(Int, ARGS[2])
    bu = parse(Int, ARGS[3])
    exp = "Experiment_General_multisize_cities"
    println("Generating scenarios for $(city) (bu=$bu, t=$t) ...")
    createScenario(city, DEPOT_LOCATION, bu, t, NB_SCENARIO)
    instance = build_instance(city, bu, t, DEPOT_LOCATION)
    outdir = "output/solution/PerDistrict/$(exp)"
    mkpath(outdir)
    outpath = "$(outdir)/$(city)_$(DEPOT_LOCATION)_$(bu)_$(t).txt"
    open(outpath, "w") do f
        for m in solution_types
            sp = "output/solution/$(exp)/$(city)_$(DEPOT_LOCATION)_$(bu)_$(t).$(m).txt"
            if !isfile(sp)
                @info "missing solution $sp"; continue
            end
            districts = readSolution(sp)
            costs = [compute_cost_via_SAA(instance, d) for d in districts]
            write(f, m * " " * join(costs, " ") * "\n")
            println("$m: $(length(costs)) districts, total=$(round(sum(costs),digits=2))")
        end
    end
    println("wrote $outpath")
end

dump_perdistrict()
