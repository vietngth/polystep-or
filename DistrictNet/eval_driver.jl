# Generic solution evaluator for arbitrary (city, target size, #BU): computes the TRUE districting cost
# (C_TSP via the C++ SAA evaluator) of each method's solution file and writes a comparison file.
# Reuses experiment_evaluator.jl (generic evaluate_experiment), bypassing its hardcoded per-experiment
# city sweeps. Solutions are read from the folder where `experiments.jl 2 ... solve ...` writes
# (Experiment_General_multisize_cities). Usage:
#   julia +1.10 --project=. eval_driver.jl <city> <target_size> <num_bu> <nb_data>
include("experiment_evaluator.jl")

function run_eval()
    city = ARGS[1]
    t = parse(Int, ARGS[2])
    bu = parse(Int, ARGS[3])
    nbdata = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 10
    exp = "Experiment_General_multisize_cities"
    println("Generating scenarios for $(city) (bu=$bu, t=$t) ...")
    createScenario(city, DEPOT_LOCATION, bu, t, NB_SCENARIO)
    println("Evaluating solutions for methods: ", solution_types)
    evaluate_experiment(exp, [city], [t], solution_types, bu, DEPOT_LOCATION, nbdata)
    cmp = "output/solution/Comparaison/$(exp)/$(city)_$(DEPOT_LOCATION)_$(bu)_$(t).txt"
    println("=== comparison ($cmp) ===")
    if isfile(cmp)
        print(read(cmp, String))
    else
        println("(no comparison file written)")
    end
end

run_eval()
