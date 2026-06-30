include("src/Estimators/BD.jl")
include("src/Estimators/FIG.jl")
include("src/Estimators/predictGnn.jl")
include("src/Estimators/AvgTsp.jl")
include("src/districtNet.jl")
include("src/Estimators/polyStep.jl")

using .BD
using .FIG
using .predictGnn
using .districtNet
using .AvgTSP
using .polyStep
using Statistics

"""
    export_solution(solution, filename)

Exports a districting solution to a file.

# Arguments
- `solution`: The solution object to be exported.
- `filename`: The name of the file where the solution will be saved.

The function sorts the nodes in each district before exporting and writes the solution details to the specified file.
"""
function export_solution(solution, filename)
    for district in solution.districts
        district.nodes = sort(district.nodes)
    end

    file = open(filename, "w")
    write(file, "$(solution.instance.city_name)\n")
    write(file, "$(solution.instance.num_blocks)\n")
    write(file, "$(solution.nb_districts)\n")
    write(file, "COST $(solution.cost)\n")
    write(file, "$(solution.is_feasible)\n")
    for district in solution.districts
        for node in district.nodes
            write(file, "$node ")
        end
        write(file, "\n")
    end
    close(file)
    println("Solution exported to $filename")
end

"""
    train_model(city::String, district_size::Int, modelModule, experiment_type::Int, nb_data::Int=100)

Trains a model based on the specified parameters.

# Arguments
- `city`: The city for which the model is trained.
- `district_size`: The size of districts for the model.
- `modelModule`: The module that contains the model training function.
- `experiment_type`: The type of experiment to conduct.
- `nb_data`: The number of data points to use for training (default is 100).

# Returns
The result of the model training process.
"""

# Define constants for clarity in identifying experiments
const GENERAL_MODEL_EXPERIMENTS = [2, 3, 4, 5]
const GENERAL_EXPERIMENTS = [
    "General_cities",
    "General_multisize_cities",
    "General_multisize_data",
    "General_small_cities",
]

function train_model(
    city::String,
    district_size::Int,
    modelModule,
    experiment_type::Int,
    nb_data::Int = 100,
)
    if experiment_type in GENERAL_MODEL_EXPERIMENTS
        # General model for Experiments 2 to 5
        return modelModule.fitGeneralModel(nb_data)
    else
        error("Invalid experiment type")
    end
end


"""
    find_solution(city::String, district_size::Int, bu::Int, modelModule, training_result)

Finds a districting solution using the specified model and training result.

# Arguments
- `city`: The city for which the solution is found.
- `district_size`: The size of districts for the solution.
- `bu`: The number of blocks in the city.
- `modelModule`: The module that contains the solution finding function.
- `training_result`: The result of the model training process.

# Returns
A `Solution` object representing the districting solution.
"""

function find_solution(
    city::String,
    district_size::Int,
    bu::Int,
    depot_location::String,
    modelModule,
    training_result,
)
    if modelModule == districtNet || modelModule == predictGnn || modelModule == polyStep
        return modelModule.find_solution_city(
            city,
            district_size,
            bu,
            depot_location,
            training_result...,
        )
    else
        return modelModule.find_solution_city(
            city,
            district_size,
            bu,
            depot_location,
            training_result,
        )
    end
end

"""
    find_solution_small(city::String, district_size::Int, modelModule, training_result)

Finds a districting solution for a small city using the specified model and training result.

# Arguments
- `city`: The name of the small city.
- `district_size`: The target size of districts.
- `modelModule`: The module that contains the solution finding function.
- `training_result`: The result of the model training process.

# Returns
A `Solution` object representing the districting solution for the small city.
"""

function find_solution_small(city::String, district_size::Int, modelModule, training_result)
    if modelModule == districtNet|| modelModule == predictGnn || modelModule == polyStep
        return modelModule.find_solution_small_city(city, district_size, training_result...)
    else
        return modelModule.find_solution_small_city(city, district_size, training_result)
    end
end


function perform_solving(
    city::String,
    district_size::Int,
    bu::Int,
    modelModule,
    experiment_type::Int,
    nb_data::Int,
    depot_location::String,
)
    training_result =
        train_model(city, district_size, modelModule, experiment_type, nb_data)
    solution =
        find_solution(city, district_size, bu, depot_location, modelModule, training_result)
    experiment_name = GENERAL_EXPERIMENTS[experiment_type]
    folder_path = "output/solution/Experiment_$(experiment_name)/"
    if !isdir(folder_path)
        mkpath(folder_path)
    end
    filename = "output/solution/Experiment_$(experiment_name)/$(city)_$(depot_location)_$(bu)_$(district_size).$(nameof(modelModule)).txt"
    if experiment_type == 4
        filename = "output/solution/Experiment_$(experiment_name)/$(city)_$(depot_location)_$(bu)_$(district_size)_$(nb_data).$(nameof(modelModule)).txt"
    end
    if experiment_type == 7
        filename = "output/solution/Experiment_$(experiment_name)/$(city)_$(depot_location)_$(bu)_$(district_size)_5.0.$(nameof(modelModule)).txt"
    end
    export_solution(solution, filename)
    println("Solution finding completed for $(city) with district size $(district_size)")
end

function perform_gap_computation(
    city::String,
    district_size::Int,
    modelModule,
    experiment_type::Int,
    nb_data::Int,
)
    training_result =
        train_model(city, district_size, modelModule, experiment_type, nb_data)
    train_cities = ["city" * string(i) for i = 1:100]
    test_cities = ["city" * string(i) for i = 101:200]
    experiment_name = GENERAL_EXPERIMENTS[experiment_type]
    trainGaps = []
    testGaps = []
    for city in test_cities
        pred_solution =
            find_solution_small(city, district_size, modelModule, training_result)
        districting_solution = districtNet.find_districting_solution(city, district_size)
        gap =
            100 * (pred_solution.cost - districting_solution.cost) /
            districting_solution.cost
        folder_path = "output/solution/Experiment_$(experiment_name)/"
        if !isdir(folder_path)
            mkpath(folder_path)
        end
        filename = "output/solution/Experiment_$(experiment_name)/$(nameof(modelModule)).txt"
        file = open(filename, "a")
        write(file, "$(gap)\n")
        close(file)
        push!(testGaps, gap)
    end
    folder_path = "output/solution/Comparaison/Experiment_$(experiment_name)/"
    if !isdir(folder_path)
        mkpath(folder_path)
    end
    filename_compared = "output/solution/Comparaison/Experiment_$(experiment_name)/meantestcompared.txt"
    file = open(filename_compared, "a")
    write(file, "$(nameof(modelModule)) : $(mean(testGaps))\n")
    close(file)
    filename_compared = "output/solution/Comparaison/Experiment_$(experiment_name)/maxtestcompared.txt"
    file = open(filename_compared, "a")
    write(file, "$(nameof(modelModule)) : $(maximum(testGaps))\n")
    close(file)
    filename_compared = "output/solution/Comparaison/Experiment_$(experiment_name)/mintestcompared.txt"
    file = open(filename_compared, "a")
    write(file, "$(nameof(modelModule)) : $(minimum(testGaps))\n")
    close(file)
    for city in train_cities
        pred_solution =
            find_solution_small(city, district_size, modelModule, training_result)
        districting_solution = districtNet.find_districting_solution(city, district_size)
        gap =
            100 * (pred_solution.cost - districting_solution.cost) /
            districting_solution.cost
        push!(trainGaps, gap)
    end
    filename_compared = "output/solution/Comparaison/Experiment_$(experiment_name)/meantraincompared.txt"
    file = open(filename_compared, "a")
    write(file, "$(nameof(modelModule)) : $(mean(trainGaps))\n")
    close(file)
    filename_compared = "output/solution/Comparaison/Experiment_$(experiment_name)/maxtraincompared.txt"
    file = open(filename_compared, "a")
    write(file, "$(nameof(modelModule)) : $(maximum(trainGaps))\n")
    close(file)
    filename_compared = "output/solution/Comparaison/Experiment_$(experiment_name)/mintraincompared.txt"
    file = open(filename_compared, "a")
    write(file, "$(nameof(modelModule)) : $(minimum(trainGaps))\n")
    close(file)
end

"""
    run_experiment(experiment_type::Int, city::String, district_size::Int, bu::Int, modelModule, action::String, nb_data::Int=100)

Runs a districting experiment based on specified parameters.

# Arguments
- `experiment_type`: The type of experiment.
- `city`: The city for the experiment.
- `district_size`: The size of districts for the experiment.
- `bu`: The number of blocks in the city.
- `modelModule`: The module containing modeling functions.
- `action`: The action to be performed ('train', 'solve', or 'compute_gap').
- `nb_data`: The number of data points to use (default is 100).

The function performs actions like training, solving, or computing the gap based on the specified experiment type and action.
"""

function run_experiment(
    experiment_type::Int,
    city::String,
    district_size::Int,
    bu::Int,
    modelModule,
    action::String,
    nb_data::Int = 100,
    depot_location::String = "C",
)
    if action == "train"
        train_model(city, district_size, modelModule, experiment_type, nb_data)

    elseif action == "solve"
        perform_solving(
            city,
            district_size,
            bu,
            modelModule,
            experiment_type,
            nb_data,
            depot_location,
        )

    elseif action == "compute_gap"
        perform_gap_computation(city, district_size, modelModule, experiment_type, nb_data)

    else
        error("Invalid action")
    end
end



"""
    parse_arg(args, index, default, type=nothing)

Parses an argument from a list, returning a default value if the argument is not present.

# Arguments
- `args`: The list of arguments.
- `index`: The index of the argument to parse.
- `default`: The default value to return if the argument is not present.
- `type`: The type to which the argument should be parsed (optional).

# Returns
The parsed argument or the default value if the argument is not present.
"""

function parse_arg(args, index, default, type = nothing)
    if length(args) >= index
        return type === nothing ? args[index] : parse(type, args[index])
    else
        return default
    end
end


"""
    main()

The main function for running experiments based on command line arguments.

Parses arguments for experiment type, city, district size, model module, action, and other parameters. Runs the specified experiment using these parameters.
"""

function main()
    # Defaults for all arguments
    default_values = (2, "general", "districtNet", "train", 3, 120, "C", 100)

    # Parsing arguments with defaults
    experiment_type = parse_arg(ARGS, 1, default_values[1], Int)
    city = parse_arg(ARGS, 2, default_values[2])
    model_module = getfield(Main, Symbol(parse_arg(ARGS, 3, default_values[3])))
    action = parse_arg(ARGS, 4, default_values[4])
    district_size = parse_arg(ARGS, 5, default_values[5], Int)
    bu = parse_arg(ARGS, 6, default_values[6], Int)
    depot_location = parse_arg(ARGS, 7, default_values[7])
    number_of_data = parse_arg(ARGS, 8, default_values[8], Int)

    # Logging the experiment details
    println(
        "Running experiment $(experiment_type) for $(city) with district size $(district_size), BU $(bu) and model $(nameof(model_module))",
    )

    # Running the experiment
    run_experiment(
        experiment_type,
        city,
        district_size,
        bu,
        model_module,
        action,
        number_of_data,
        depot_location,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
