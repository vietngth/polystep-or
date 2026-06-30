using Graphs, MetaGraphs, DataStructures, Statistics, Combinatorics, Random, LinearAlgebra
try; using UnionFind; catch; end
include("experiment_evaluator.jl")
include("src/Solver/Kruskal.jl")
city="ilsdefrance"; t=20; bu=2000
instance = build_instance(city, bu, t, DEPOT_LOCATION)
path = "output/solution/Experiment_General_multisize_cities/$(city)_C_$(bu)_$(t).polyStep.txt"
districts = Vector{Vector{Int}}([Vector{Int}(d) for d in readSolution(path)])
println("BEFORE ndist=", length(districts), " valid=", is_valid_districting_solution(instance, districts),
        " min=", minimum(length.(districts)), " max=", maximum(length.(districts)))
rep = repair_districts(instance, districts)
println("AFTER  ndist=", length(rep), " valid=", is_valid_districting_solution(instance, rep),
        " min=", minimum(length.(rep)), " max=", maximum(length.(rep)),
        " total_blocks=", sum(length.(rep)))
flush(stdout)
