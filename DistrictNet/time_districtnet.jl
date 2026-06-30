# Timing driver: ORIGINAL DistrictNet (FY-GNN) vs PolyStep+DistrictNet, with [TIMING] per stage + CMST counts.
# include() loads modules + @wrapmodule(C++); experiments.jl main() is PROGRAM_FILE-guarded (no auto-run).
include("experiments.jl")
nb = parse(Int, get(ENV, "DN_NBDATA", "30"))
isdir("models") && for f in readdir("models"); occursin("_$(nb).jld2", f) && rm("models/$f"); end   # force real training
println("[TIMING] ===== DistrictNet timing: nb_data=$nb epochs=$(get(ENV,"DN_EPOCHS","40")) nb_samples(FY)=20 probes(PolyStep)=10 =====")
println("[TIMING] >>> DISTRICTNET-FY (PerturbedAdditive CMST + FenchelYoung) <<<")
t_fy = @elapsed districtNet.fitGeneralModel(nb)
println("[TIMING] DISTRICTNET-FY whole fitGeneralModel = $(round(t_fy, digits=2)) s")
isdir("models") && for f in readdir("models"); occursin("_$(nb).jld2", f) && rm("models/$f"); end
println("[TIMING] >>> POLYSTEP (gradient-free, realized cost) <<<")
t_ps = @elapsed polyStep.fitGeneralModel(nb)
println("[TIMING] POLYSTEP whole fitGeneralModel = $(round(t_ps, digits=2)) s")
println("[TIMING] ===== SUMMARY  DistrictNet-FY=$(round(t_fy,digits=2))s  PolyStep=$(round(t_ps,digits=2))s =====")
