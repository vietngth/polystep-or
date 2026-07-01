# At-scale PolyStep training that COMBINES the big deploy-scale cities (target-20) with the
# small origin instances (target-3), rather than big only. Pure-CPU Flux. Times the training
# and saves the model so it can be deployed and compared against the big-only at-scale model.
using Dates, Statistics, Flux, JLD2, Graphs
include("experiments.jl")
using .polyStep

NB = parse(Int, get(ENV, "TAS_NB_SCENARIO", "100"))
big_c   = ["London", "Leeds", "Bristol", "Manchester", "Lyon", "Paris"]; big_b = [200, 190, 160, 120, 120, 120]
small_c = ["London", "Leeds", "Bristol"];                                small_b = [30, 30, 30]

println("=== at-scale COMBINED (big target-20 + small target-3) @ $(now()) ==="); flush(stdout)
inst_big, _, _   = polyStep.aggregateCityTrainingData_atscale(big_c, big_b, 20, NB)
inst_small, _, _ = polyStep.aggregateCityTrainingData_atscale(small_c, small_b, 3, NB)
instances = vcat(inst_big, inst_small)
println("combined instances: big=$(length(inst_big)) small=$(length(inst_small)) total=$(length(instances))"); flush(stdout)

feats = hcat([polyStep.get_instance_features(i) for i in instances]...)
MEAN = mean(feats, dims = 2) * 0; STD = std(feats, dims = 2)
model = polyStep.build_gnn_model(instances[1].graph, polyStep.STRATEGY, polyStep.HIDDEN_SIZE)

t = @elapsed (trained = polyStep.GNNtrainer_polyStep_atscale(model, instances, MEAN, STD))
println("[RESULT] at-scale BIG+SMALL combined training wall = $(round(t, digits = 1)) s, n_instances=$(length(instances))"); flush(stdout)

isdir("models") || mkpath("models")
jldsave("models/GeneralPolyStepScale_combined.jld2"; model_state = Flux.state(trained), mean_train = MEAN, std_train = STD)
isdir("exp_results") || mkpath("exp_results")
open("exp_results/trainatscale_combined.md", "w") do io
    println(io, "# At-scale PolyStep trained on BIG (target-20) + SMALL (target-3), combined (CPU Flux)\n")
    println(io, "big (target-20): ", join(["$(c)@$(b)" for (c, b) in zip(big_c, big_b)], ", "))
    println(io, "small (target-3): ", join(["$(c)@$(b)" for (c, b) in zip(small_c, small_b)], ", "))
    println(io, "\nn_instances = $(length(instances)) (big $(length(inst_big)) + small $(length(inst_small))), training wall = $(round(t, digits = 1)) s")
end
println("wrote exp_results/trainatscale_combined.md @ $(now())")
