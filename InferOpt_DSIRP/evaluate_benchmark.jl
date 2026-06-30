using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--instance_id"
            arg_type = String 
            required = true 
            help = "Instance ID"
        "--shortage_penalty"
            arg_type = Int 
            help = "Shortage penalty (default: 200). Multiplier for calculating stock-out costs per unit from the holding cost"
            default = 200
        "--demand_type"
            arg_type = String
            help = "Type of demand samples to use (choose from: 'eval', 'test')." 
            default = "test"
            range_tester = in(["eval", "test"])
        "--policy"
            arg_type = String
            help = "Policy (choose from: 'anticipative', 'mean', 'saa_1', 'saa_3')" 
            default = "mean"
            range_tester = in(["anticipative", "mean", "saa_1", "saa_3"])
        "--evaluation_horizon"
            arg_type = Int 
            default = 10 
            help = "Number of periods to evaluate"
        "--look_ahead"
            arg_type = Int 
            help = "Periods to look ahead" 
            default = 6
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

include("src/evaluation_benchmark.jl");

args = parse_commandline()

milp_builder = model_definition(;milp_solver=args["milp_solver"], silent=args["silent_solver"])

run_benchmark(args["shortage_penalty"], args["instance_id"]; policy=args["policy"],
demand=args["demand_type"], evaluation_horizon=args["evaluation_horizon"], look_ahead=args["look_ahead"])