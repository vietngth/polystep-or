using JSON, Graphs, MetaGraphs, DataStructures, Statistics, Combinatorics, Random, LinearAlgebra
using UnionFind
include(joinpath(pwd(), "experiment_evaluator.jl"))
include(joinpath(pwd(), "src/Solver/Kruskal.jl"))

const EXPERIMENT = "Experiment_General_multisize_cities"

"""Score saved districtings of an instance under the scenario file of the working copy."""
function score(city, bu, t, methods, out)
    instance = build_instance(city, bu, t, DEPOT_LOCATION)
    costs, feasible, districts = Dict{String,Any}(), Dict{String,Any}(), Dict{String,Any}()
    for method in methods
        path = buildSolutionPath(EXPERIMENT, city, bu, t, DEPOT_LOCATION, method)
        isfile(path) || continue
        solution = Vector{Vector{Int}}([Vector{Int}(d) for d in readSolution(path)])
        costs[method] = sum(compute_cost_via_SAA(instance, d) for d in solution)
        feasible[method] = is_valid_districting_solution(instance, solution)
        districts[method] = length(solution)
        println("$(city) $(bu) t$(t) $(method) cost $(costs[method])")
    end
    record = Dict("city" => city, "size" => bu, "target" => t, "nb_scenario" => NB_SCENARIO,
                  "costs" => costs, "feasible" => feasible, "ndistricts" => districts)
    open(out, "w") do io
        JSON.print(io, record, 1)
    end
end

function main(args)
    mode, city = args[1], args[2]
    bu, t = parse(Int, args[3]), parse(Int, args[4])
    if mode in ("scenario", "instance")
        createScenario(city, DEPOT_LOCATION, bu, t, NB_SCENARIO)
    end
    mode == "scenario" && return
    methods = length(args) >= 6 ? split(args[6], ",") : solution_types
    score(city, bu, t, methods, args[5])
end

main(ARGS)
