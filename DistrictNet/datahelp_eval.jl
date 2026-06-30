# "Does data help?" SAA evaluator: same as eval_driver.jl but restricts solution_types to
# the two methods produced by the ablation (districtNet, polyStep) so it never errors on the
# missing BD/FIG/AvgTSP/predictGnn solution files. Computes TRUE SAA districting cost (C++
# evaluator) of whatever solve wrote into Experiment_General_multisize_cities.
#   julia +1.10 --project=. datahelp_eval.jl <city> <target_size> <num_bu> <nb_data>
include("experiment_evaluator.jl")

function run_eval()
    city = ARGS[1]
    t = parse(Int, ARGS[2])
    bu = parse(Int, ARGS[3])
    nbdata = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 10
    exp = "Experiment_General_multisize_cities"
    types = ["districtNet", "polyStep"]
    println("Generating scenarios for $(city) (bu=$bu, t=$t) ...")
    createScenario(city, DEPOT_LOCATION, bu, t, NB_SCENARIO)
    println("Evaluating SAA cost for methods: ", types)
    evaluate_experiment(exp, [city], [t], types, bu, DEPOT_LOCATION, nbdata)
    cmp = "output/solution/Comparaison/$(exp)/$(city)_$(DEPOT_LOCATION)_$(bu)_$(t).txt"
    println("=== comparison ($cmp) ===")
    isfile(cmp) ? print(read(cmp, String)) : println("(no comparison file written)")
end

run_eval()
