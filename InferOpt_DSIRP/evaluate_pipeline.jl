using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--solution_path"
            arg_type = String 
            required = true 
            help = "Path to the solution file of the trained model"
        "--evaluation_horizon"
            arg_type = Int 
            default = 10 
            help = "Number of periods to evaluate"
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

using Flux
using Gurobi
using HiGHS
using JuMP
using Distributions
using Statistics
using JSON
using Dates
using BSON
include("src/auxiliar.jl")
include("src/sirp_model.jl")
include("src/sirp_solver.jl")
include("src/stat_model.jl")
include("src/evaluation_pipeline.jl")
include("src/pctsp.jl")


@info "Starting pipeline evaluation..."
args = parse_commandline()

milp_builder = model_definition(;milp_solver=args["milp_solver"], silent=args["silent_solver"])

_, pattern, penalty, instance_id, _ = split(args["solution_path"], "/")
pattern = convert(String, pattern)
penalty_inv = parse(Int, split(penalty, "_")[2])
instance_id = convert(String, instance_id)

instance = IRPInstance()
readInstance("instances/"*instance_id*".json", pattern, instance; penalty_inv=penalty_inv);

max_evaluation_horizon = maximum([length(evaluation_demand) for evaluation_demand in values(instance.demands_test)])
args["evaluation_horizon"] <= max_evaluation_horizon || error("Evaluation horizon exceeds the number of samples in the evaluation demand of the instance.")

@info "Solution path: $(args["solution_path"])"
@info "Evaluation horizon: $(args["evaluation_horizon"]) periods"  

horizon = args["evaluation_horizon"]
data = JSON.parsefile(realpath("training/solutions/"*args["solution_path"]))

demand_quantiles = FloatFromAny(data["pctsp"]["settings"]["demand_quantiles"])
look_ahead = data["pctsp"]["settings"]["look_ahead"]

weights = data["pctsp"][string(data["pctsp"]["best_iteration"])]["weights"]

φ_w, _ = build_stat_model(demand_quantiles, look_ahead; nb_features=instance.nb_features, weights=weights);

benchmark_start = now()
x_val, holding_costs, stockout_costs, routing_costs, _ = evaluate_pctsp(φ_w, instance; demand="test", 
    demand_quantiles=demand_quantiles, look_ahead=look_ahead, evaluation_horizon=horizon)

total_costs = routing_costs + stockout_costs + holding_costs

inference_time = (now() - benchmark_start) / Millisecond(1000)

# Round costs to two decimal places for printing
routing_costs_rounded = round(routing_costs, digits=2)
stockout_costs_rounded = round(stockout_costs, digits=2)
holding_costs_rounded = round(holding_costs, digits=2)
total_costs_rounded = round(total_costs, digits=2)
println("-----")
@info "Routing costs: $routing_costs_rounded"
@info "Stock-out costs: $stockout_costs_rounded"
@info "Holding costs: $holding_costs_rounded"
println("-----")
@info "Total costs: $total_costs_rounded"
@info "Inference time: $inference_time seconds"