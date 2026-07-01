# Measure from-scratch, small-scale (target-3) training wall time for both learners:
#   (1) DistrictNet original (Fenchel-Young imitation of the exact optimal-partition label)
#   (2) PolyStep (Julia estimator, realized-cost, label-free)
# Both are pure-CPU Flux. Any cached model is moved aside and restored, so the canonical
# models on disk are left untouched. Result lands in exp_results/timing_smallscale.md.
using Dates, Statistics
include("experiments.jl")
using .districtNet, .polyStep

N = parse(Int, get(ENV, "TIMING_NB_DATA", "10"))   # number of small training cities (canonical setup)
cached = ["models/GeneralDistrictNet_$(N).jld2", "models/GeneralPolyStep_$(N).jld2"]
for f in cached; isfile(f) && mv(f, f * ".timingbak"; force = true); end

t_dn = NaN; t_ps = NaN
try
    println("=== [TIMING] DistrictNet ORIGINAL (target-3, FY label imitation) from scratch, nb_data=$N @ $(now()) ==="); flush(stdout)
    t_dn = @elapsed districtNet.fitGeneralModel(N)
    println("[RESULT] DistrictNet-original small from-scratch wall = $(round(t_dn, digits = 1)) s"); flush(stdout)

    println("=== [TIMING] PolyStep (Julia, realized cost) target-3 from scratch, nb_data=$N @ $(now()) ==="); flush(stdout)
    t_ps = @elapsed polyStep.fitGeneralModel(N)
    println("[RESULT] PolyStep-Julia small from-scratch wall = $(round(t_ps, digits = 1)) s"); flush(stdout)

    isdir("exp_results") || mkpath("exp_results")
    open("exp_results/timing_smallscale.md", "w") do io
        println(io, "# Small-scale (target-3) from-scratch training wall time (CPU Flux, nb_data=$N)\n")
        println(io, "| method | training | wall (s) |")
        println(io, "|---|---|---|")
        println(io, "| DistrictNet original | Fenchel-Young label imitation | $(round(t_dn, digits = 1)) |")
        println(io, "| PolyStep (Julia)      | realized cost, label-free     | $(round(t_ps, digits = 1)) |")
    end
    println("wrote exp_results/timing_smallscale.md")
finally
    for f in cached
        isfile(f * ".timingbak") && mv(f * ".timingbak", f; force = true)
    end
    println("restored canonical models @ $(now())")
end
