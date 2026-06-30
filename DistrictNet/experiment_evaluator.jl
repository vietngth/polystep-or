using GraphNeuralNetworks, Graphs, MetaGraphs
using JSON, Random, LinearAlgebra
using Base.Filesystem: isfile
using CxxWrap

Random.seed!(1234)
include("src/utils.jl")
include("src/struct.jl")
include("src/instance.jl")
include("src/district.jl")

using .CostEvaluator: EVmain
using .GenerateScenario: SCmain

const EXPERIMENTS = Dict(
    2 => "Experiment_General_cities",
    3 => "Experiment_General_multisize_cities",
    4 => "Experiment_General_multisize_data",
)

const CITIES = ["London", "Leeds", "Bristol", "Manchester", "Paris", "Lyon", "Marseille"]
const DISTRICT_SIZES = [3, 6, 12, 20, 30]
solution_types = ["districtNet", "predictGnn", "BD", "FIG", "AvgTSP", "polyStep"]


const NB_BU_LARGE = 120
const DEPOT_LOCATION = "C"
const NB_SCENARIO = 50
"""
    readSolution(solution_path)

Reads a solution from a file and returns it as a list of districts.

# Arguments
- `solution_path`: Path to the solution file.

# Returns
A list of districts, where each district is a list of node numbers.
"""

function readSolution(solution_path)
    districts = []
    open(solution_path, "r") do file
        for _ = 1:5
            readline(file)
        end
        # Read each district line
        while !eof(file)
            line = readline(file)
            district = parse.(Int, split(line))
            push!(districts, district)
        end
    end
    return districts
end

"""
    createScenario(city, depot_location, nb_bu_large, district_size, nb_scenario)

Creates a scenario file for a given city, depot location, block size, district size, and number of scenarios.

# Arguments
- `city`: The city for which the scenario is created.
- `depot_location`: The location of the depot in the city.
- `nb_bu_large`: The number of large blocks in the city.
- `district_size`: The size of the districts.
- `nb_scenario`: The number of scenarios to generate.

The function builds a scenario and saves it in JSON format.
"""
function createScenario(city, depot_location, nb_bu_large, district_size, nb_scenario)
    instance = build_instance(city, nb_bu_large, district_size, depot_location)
    pathScenario = "deps/Scenario/output/$(city)_$(depot_location)_$(nb_bu_large)_$(district_size).json"

    if isfile(pathScenario)
        rm(pathScenario)
    end

    StringInstance = city * "_"* depot_location * "_" * string(nb_bu_large) * "_" * string(district_size) * "_" * string(nb_scenario)
    SCmain(StringInstance)
end


function buildSolutionPath(
    experience_type::String,
    city::String,
    NB_BU_LARGE,
    district_size,
    depot_location,
    solution_type,
    nb_data = 100,
)
    solution_path = ""
    if experience_type == "Experiment_General_multisize_data"
        solution_path = "output/solution/$(experience_type)/$(city)_$(depot_location)_$(NB_BU_LARGE)_$(district_size)_$(nb_data).$(solution_type).txt"
    else
        solution_path = "output/solution/$(experience_type)/$(city)_$(depot_location)_$(NB_BU_LARGE)_$(district_size).$(solution_type).txt"
    end
    return solution_path
end

function evaluateSolution(solution_path, instance)
    if isfile(solution_path)
        districts_solution = readSolution(solution_path)
        true_cost = sum([
            compute_cost_via_SAA(instance, district) for district in districts_solution
        ])
        return true_cost
    end
    return nothing
end

function writeComparisonResults(
    output_path,
    experience_type,
    city,
    NB_BU_LARGE,
    district_size,
    depot_location,
    solutions_cost,
    nb_data = 100,
)
    if !isdir(output_path)
        mkdir(output_path)
    end
    comparison_file = "output/solution/Comparaison/$(experience_type)/$(city)_$(depot_location)_$(NB_BU_LARGE)_$(district_size).txt"
    if experience_type == "Experiment_General_multisize_data"
        comparison_file = "output/solution/Comparaison/$(experience_type)/$(city)_$(depot_location)_$(NB_BU_LARGE)_$(district_size)_$(nb_data).txt"
    end
    writeComparison(comparison_file, solutions_cost)
end

"""
writeComparisonwriteComparisonResults(filename, solutions_cost)

Writes the comparison results of various solutions to a file.

# Arguments
- `filename`: Name of the file to write the results to.
- `solutions_cost`: Dictionary containing solution types and their costs.

Each line in the output file contains a solution type and its associated cost.
"""

function writeComparison(filename, solutions_cost)
    open(filename, "w") do file
        for (solution_type, cost) in solutions_cost
            write(file, "$solution_type $cost\n")
        end
    end
    @info "Comparison results written to $filename"
end
#


function evaluate_experiment(
    experience_type::String,
    cities,
    district_sizes,
    solution_types,
    NB_BU_LARGE,
    depot_location,
    nb_data = 100,
)
    for city in cities
        for district_size in district_sizes
            instance = build_instance(city, NB_BU_LARGE, district_size, depot_location)
            solutions_cost = Dict()

            for solution_type in solution_types
                solution_path = buildSolutionPath(
                    experience_type,
                    city,
                    NB_BU_LARGE,
                    district_size,
                    depot_location,
                    solution_type,
                    nb_data,
                )
                @info "Evaluating solution for $solution_type"
                cost = evaluateSolution(solution_path, instance)
                solutions_cost[solution_type] = cost
            end

            output_path = "output/solution/Comparaison/$(experience_type)/"
            writeComparisonResults(
                output_path,
                experience_type,
                city,
                NB_BU_LARGE,
                district_size,
                depot_location,
                solutions_cost,
                nb_data,
            )
        end
    end
end


function evaluate_Experiment_General_cities(
    cities,
    district_sizes,
    solution_types,
    NB_BU_LARGE,
    depot_location,
    nb_data = 100,
)
    experience_type = "Experiment_General_cities"
    evaluate_experiment(
        experience_type,
        cities,
        district_sizes,
        solution_types,
        NB_BU_LARGE,
        depot_location,
        nb_data,
    )
end

function evaluate_Experiment_General_multisize_cities(
    cities,
    district_sizes,
    solution_types,
    NB_BU_LARGE,
    depot_location,
    nb_data = 100,
)
    experience_type = "Experiment_General_multisize_cities"
    NB_BU_LARGES = Dict(
        "London" =>  [100, 200, 300, 400, 500, 600, 700, 800, 900],
        "Leeds" =>   [90, 150, 190, 240, 290],
        "Bristol" => [110, 160, 210, 260],
    )
    for city in keys(NB_BU_LARGES)
        for nb_bu in NB_BU_LARGES[city]
            createScenario(city, depot_location, nb_bu, 20, NB_SCENARIO)
            evaluate_experiment(
                experience_type,
                [city],
                [20],
                solution_types,
                nb_bu,
                depot_location,
                nb_data,
            )
        end
    end

end


function evaluate_Experiment_General_multisize_data(
    cities,
    district_sizes,
    solution_types,
    NB_BU_LARGE,
    depot_location,
    nb_data = 100,
)
    solution_types = ["districtNet"]
    experience_type = "Experiment_General_multisize_data"
    NB_DATAS= [20, 50, 100, 150, 200]
    for nb_data in NB_DATAS
        evaluate_experiment(
            experience_type,
            cities,
            district_sizes,
            solution_types,
            NB_BU_LARGE,
            depot_location,
            nb_data,
        )
    end
end

"""
EvaluateAllExp()

The EvaluateAllExp function to run evaluations for different scenarios and experiments.

It evaluates solutions for a set of cities, district sizes, and solution types under various experiment types. The scenarios are first generated and then used for evaluating solutions. Results are saved in specified output directories.
"""

function EvaluateAllExp()
    folder_path = "output/solution/Comparaison/"
    if !isdir(folder_path)
        mkdir(folder_path)
    end
    for city in cities
        for district_size in district_sizes
            createScenario(city, DEPOT_LOCATION, NB_BU_LARGE, district_size, NB_SCENARIO)
        end
    end

    evaluate_Experiment_General_cities(cities, district_sizes, solution_types, NB_BU_LARGE, DEPOT_LOCATION)
    evaluate_Experiment_General_multisize_cities(cities, district_sizes, solution_types, NB_BU_LARGE, DEPOT_LOCATION)
    evaluate_Experiment_General_multisize_data(cities, district_sizes, solution_types, NB_BU_LARGE, DEPOT_LOCATION)

end

function OneExpEvaluation(experience_type, cities, district_sizes, NB_BU_LARGE, depot_location, nb_data = 100)
    folder_path = "output/solution/Comparaison/"
    if !isdir(folder_path)
        mkdir(folder_path)
    end
    for city in cities
        for district_size in district_sizes
            createScenario(city, depot_location, NB_BU_LARGE, district_size, NB_SCENARIO)
        end
    end
    if experience_type == 2
        evaluate_Experiment_General_cities(cities, district_sizes, solution_types, NB_BU_LARGE, depot_location, nb_data)
    elseif experience_type == 3
        evaluate_Experiment_General_multisize_cities(cities, district_sizes, solution_types, NB_BU_LARGE, depot_location, nb_data)
    elseif experience_type == 4
        evaluate_Experiment_General_multisize_data(cities, district_sizes, solution_types, NB_BU_LARGE, depot_location, nb_data)
    end
   
end

function OneCityEvaluation(experience_type, city, district_size, NB_BU_LARGE, nb_data = 100)
    if experience_type == 2
        evaluate_Experiment_General_cities([city], [district_size], solution_types, NB_BU_LARGE, DEPOT_LOCATION, nb_data)
    elseif experience_type == 3
        evaluate_Experiment_General_multisize_cities([city], [district_size], solution_types, NB_BU_LARGE, DEPOT_LOCATION, nb_data)
    elseif experience_type == 4
        evaluate_Experiment_General_multisize_data([city], [district_size], solution_types, NB_BU_LARGE, DEPOT_LOCATION, nb_data)
    end
end

function main()
    evale_action = ARGS[1]
    if evale_action == "runCity"
        experience_type = parse(Int, ARGS[2])
        city = ARGS[3]
        district_size = parse(Int, ARGS[4])
        NB_BU = parse(Int, ARGS[5])
        depot_location = ARGS[6]
        nb_data = parse(Int, ARGS[7])
        OneCityEvaluation(experience_type, city, district_size, NB_BU, nb_data)
    elseif evale_action == "runExp"
        experience_type = parse(Int, ARGS[2])
        cities = CITIES
        district_sizes = DISTRICT_SIZES
        NB_BU = 120
        depot_location = ARGS[3]
        nb_data = 100
        OneExpEvaluation(experience_type, cities, district_sizes, NB_BU, depot_location , nb_data)
    elseif evale_action == "runAll"
        EvaluateAllExp()
    else 
        println("Invalid action")
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
