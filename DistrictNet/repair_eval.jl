# Post-hoc feasibility repair + re-evaluation of the PolyStep large-instance districting.
# Loads the saved ilsdefrance PolyStep solution (one size-1 orphan district), applies the
# existing repair_districts (grows undersized districts by pulling adjacent BUs from spare
# neighbors, preserving connectivity and the 100-district count), then re-evaluates the true
# SAA routing cost of both the original and repaired solutions.
# Run: julia +1.10 --project=. repair_eval.jl
using Graphs, MetaGraphs, DataStructures, Statistics, Combinatorics, Random, LinearAlgebra
try
    using UnionFind
catch
end

include("experiment_evaluator.jl")          # build_instance, readSolution, createScenario, compute_cost_via_SAA, NB_SCENARIO, DEPOT_LOCATION
include("src/Solver/Kruskal.jl")            # repair_districts, is_valid_districting_solution

function run_repair()
    city = "ilsdefrance"; t = 20; bu = 2000
    instance = build_instance(city, bu, t, DEPOT_LOCATION)
    path = "output/solution/Experiment_General_multisize_cities/$(city)_C_$(bu)_$(t).polyStep.txt"
    districts = Vector{Vector{Int}}([Vector{Int}(d) for d in readSolution(path)])
    println("LOADED ndistricts=", length(districts), " sizes=", sort(length.(districts)))
    println("VALID_BEFORE ", is_valid_districting_solution(instance, districts))

    rep = repair_districts(instance, districts)
    println("VALID_AFTER ", is_valid_districting_solution(instance, rep),
            " ndistricts=", length(rep), " sizes=", sort(length.(rep)))

    println("Generating scenarios ...")
    createScenario(city, DEPOT_LOCATION, bu, t, NB_SCENARIO)
    cost_orig = sum([compute_cost_via_SAA(instance, d) for d in districts])
    cost_rep  = sum([compute_cost_via_SAA(instance, d) for d in rep])
    println("ORIG_COST ", cost_orig)
    println("REPAIRED_COST ", cost_rep)
    println("REPAIRED_FEASIBLE ", is_valid_districting_solution(instance, rep))

    outp = "output/solution/Experiment_General_multisize_cities/$(city)_C_$(bu)_$(t).polyStepRepaired.txt"
    open(outp, "w") do f
        println(f, city); println(f, bu); println(f, length(rep))
        println(f, "COST $cost_rep"); println(f, is_valid_districting_solution(instance, rep))
        for d in rep
            println(f, join(d, " "))
        end
    end
    println("WROTE ", outp)
end

run_repair()
