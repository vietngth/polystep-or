using ArgParse
using Base.Threads

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--instance_id"
            arg_type = String 
            required = true 
            help = "Instance ID"
        "--paradigm"
            arg_type = String
            help = "Learning paradigm (choose from: 'baty', 'sampling', 'dagger')" 
            default = "dagger"
            range_tester = in(["baty", "sampling", "dagger"])
        "--pipeline_epochs"
            arg_type = Int 
            help = "Number of epochs for the entire pipeline; the pipeline is evaluated each epoch"
            default = 150
        "--update_epochs"
            arg_type = Int
            help = "Number of epochs for updating the statistical model"
            default = 50
        "--dagger_iterations"
            arg_type = Int 
            help = "Number of dagger iterations (only applicable if --paradigm is set to 'dagger')"
            default = 20
        "--alpha_0"
            arg_type = Float64
            help = "Initial alpha value for expert mixing (only applicable if --paradigm is set to 'dagger')"
            default = 1.0
        "--alpha_decay"
            arg_type = Float64
            help = "Alpha decay rate per iteration (only applicable if --paradigm is set to 'dagger')"
            default = 0.05
        "--alpha_min"
            arg_type = Float64
            help = "Minimum alpha value to retain expert guidance (only applicable if --paradigm is set to 'dagger')"
            default = 0.1
        "--retain_epochs"
            arg_type = Int 
            help = "Number of past pipeline epochs from which to retain samples (only applicable if --paradigm is set to 'dagger')"
            default = 10
        "--subsampling_proportion"
            arg_type = Float64 
            help = "Proportion for subsampling samples from each retained pipeline epoch (only applicable if --paradigm is set to 'dagger')"
            default = 0.5
        "--early_stopping"
            arg_type = Int 
            help = "Number of unsuccessful pipeline epochs before stopping training"
            default = 5
        "--num_scenarios"
            arg_type = Int 
            help = "Number of demand scenarios to sample for generating samples of the training set" 
            default = 5
        "--lr_start"
            arg_type = Float64 
            help = "Learning rate at the start" 
            default = 0.01
        "--fyl_samples"
            arg_type = Int 
            help = "Number of samples to calculate Fenchel-Young-Loss (FYL)" 
            default = 5
        "--fyl_epsilon"
            arg_type = Float64 
            help = "Epsilon to calculate Fenchel-Young-Loss (FYL)" 
            default = 20.0
        "--look_ahead"
            arg_type = Int 
            help = "Periods to look ahead" 
            default = 6
        "--demand_quantiles"
            arg_type = Vector{Float64} 
            help = "Demand quantiles for the statistical model" 
            default = [0.01, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99]
        "--shortage_penalty"
            arg_type = Int 
            help = "Shortage penalty (default: 200). Multiplier for calculating stock-out costs per unit from the holding cost"
            default = 200
        "--load_model"
            arg_type = String 
            default = "none"
            help = "Model ID to load a pre-trained model (default: 'none')"
        "--create_model"
            arg_type = Bool
            default = true
            help = "Indicates whether to create a new model or resume training with the loaded model (if --load_model is set to 'none', this option is always true)"
        "--milp_solver"
            arg_type = String
            help = "Solver for solving MILP problems"
            default = "gurobi"
            range_tester = in(["gurobi", "highs"])
        "--silent_solver"
            arg_type = Bool
            help = "Run solver in silent mode"
            default = true
    end

    try
        return parse_args(s)
    catch err
        error("Error parsing command-line arguments: $err")
    end
end


@info "Starting pipeline training..."
args = parse_commandline()
sys_cpus = length(Sys.cpu_info())
num_threads = nthreads()

if args["paradigm"]=="dagger" 
    include("src/pipeline_dagger.jl");
    settings = dagger_settings(
        nb_outer_epochs = args["pipeline_epochs"],
        nb_inner_epochs = args["update_epochs"],
        nb_iterations = args["dagger_iterations"],
        alpha_0 = args["alpha_0"],
        alpha_decay = args["alpha_decay"],
        alpha_min = args["alpha_min"],
        nb_keep_epochs = args["retain_epochs"],
        portion_keep_epochs = args["subsampling_proportion"],
        early_stopping = args["early_stopping"],
        nb_scenarios = args["num_scenarios"],
        lr_start = args["lr_start"],
        fyl_samples = args["fyl_samples"],
        fyl_epsilon = args["fyl_epsilon"],
        look_ahead = args["look_ahead"],
        demand_quantiles = args["demand_quantiles"],
        load_model = args["load_model"],
        create_model = args["create_model"],
        sys_cpus = sys_cpus,
        num_threads = num_threads,
        milp_solver = args["milp_solver"]
    )
elseif args["paradigm"]=="sampling" 
    include("src/pipeline_sampling.jl");
    settings = sampling_settings(
        nb_outer_epochs = args["pipeline_epochs"],
        nb_inner_epochs = args["update_epochs"],
        early_stopping = args["early_stopping"],
        nb_scenarios = args["num_scenarios"],
        lr_start = args["lr_start"],
        fyl_samples = args["fyl_samples"],
        fyl_epsilon = args["fyl_epsilon"],
        look_ahead = args["look_ahead"],
        demand_quantiles = args["demand_quantiles"],
        load_model = args["load_model"],
        create_model = args["create_model"],
        sys_cpus = sys_cpus,
        num_threads = num_threads,
        milp_solver = args["milp_solver"]
    )
elseif args["paradigm"]=="baty" 
    include("src/pipeline_baty.jl");
    settings = baty_settings(
        nb_outer_epochs = args["pipeline_epochs"],
        nb_inner_epochs = args["update_epochs"],
        early_stopping = args["early_stopping"],
        nb_scenarios = args["num_scenarios"],
        lr_start = args["lr_start"],
        fyl_samples = args["fyl_samples"],
        fyl_epsilon = args["fyl_epsilon"],
        look_ahead = args["look_ahead"],
        demand_quantiles = args["demand_quantiles"],
        load_model = args["load_model"],
        create_model = args["create_model"],
        sys_cpus = sys_cpus,
        num_threads = num_threads,
        milp_solver = args["milp_solver"]
    )
end

println(settings)   
patterns::Vector{String} = [split(args["instance_id"], "-")[1]]
penalties::Vector{Int} = [args["shortage_penalty"]]
instances::Vector{String} = [args["instance_id"]]

milp_builder = model_definition(;milp_solver=args["milp_solver"], silent=args["silent_solver"])

solution_path = run_pipeline(patterns, penalties, instances, settings)
@info "Solution path: $(solution_path)"
@info "Continue to pipeline evaluation? Please enter 'yes' or 'no'."
while true
    evaluate_pipeline = readline()

    if evaluate_pipeline == "yes"
        println("Number of periods to evaluate?")
        evaluation_horizon = readline()
        run(`julia evaluate_pipeline.jl --solution_path=$(solution_path) --evaluation_horizon=$(evaluation_horizon) --milp_solver=$(args["milp_solver"]) --silent_solver=$(args["silent_solver"])`)
        break
    elseif evaluate_pipeline == "no"
        println("Evaluation cancelled.")
        break
    else
        println("Invalid input. Please enter 'yes' or 'no'.")
    end
end