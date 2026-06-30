using DelimitedFiles
using Statistics
using HypothesisTests
using Printf

"""
Generate filenames for all experiments.
"""

function generate_filenames(folder_path, cities, depots, sizes)
    filenames = []
    for city in cities
        for depot in depots
            for t in sizes
                file_name = join([city, "_", depot, "_", 120, "_", t,".txt"], "")
                file_path = joinpath(folder_path, file_name)
                push!(filenames, file_path)
            end
        end
    end
    return filenames
end

"""
Read results for a single experiment from its .txt file.
"""
function read_experiment_file(file_path)
    file_name = basename(file_path)
    # Extract city and t from the file name
    parts = split(file_name, '_')
    city = parts[1]
    t = parse(Int, split(parts[end], '.')[1])

    # Read the file and extract costs
    costDict = Dict{String,Float64}()
    for line in eachline(file_path)
        method, cost = split(line)
        costDict[method] = round(parse(Float64, cost), digits = 2)
    end
    println("Result of experiment ($city, $t): $costDict.")
    return city, t, costDict
end

""" 
Compute relative cost of each method compared to DistrictNet.
"""
function Compute_relative_cost(costDict)
    relative_cost = Dict{String,Float64}()
    for (method, cost) in costDict
        if method == "districtNet"
            relative_cost[method] = 0.0
        else
            relative_cost[method] = (cost - costDict["districtNet"]) / costDict["districtNet"]*100
        end
    end
    return relative_cost
end

"""
Return costs for a specific method.
"""
function get_method_cost(data, method)
    costs = []
    for relative_cost in data
        push!(costs, relative_cost[method])
    end
    return costs
end

"""
Return list of all unique methods across all files.
"""
function get_unique_methods(data)
    unique_methods = Set{String}()
    for (_, _, costDict) in data
        for method in keys(costDict)
            push!(unique_methods, method)
        end
    end
    return sort(collect(unique_methods))
end


"""
Find indices of best and second-best methods.
"""
function indices_top_two_methods(costs)
    costs = collect(costs)
    sorted_costs = sort(costs, by = x -> x[2])
    min_cost = sorted_costs[1][2]
    second_min_cost = length(sorted_costs) > 1 ? sorted_costs[2][2] : NaN
    return sorted_costs, min_cost, second_min_cost
end


"""
Format cost: min in blue, second best in orange.
"""
function format_cost(cost, min_cost, second_min_cost)
    if cost isa Number
        cost = round(cost, digits = 2)
    end
    formatted_cost =
        cost == min_cost ? "\\textcolor{blue}{\$$cost\$}" :
        cost == second_min_cost ? "\\textcolor{orange}{\$$cost\$}" : "\$ $cost \$"
    return formatted_cost
end


"""
Add relative cost to the table's row.
"""
function add_relative_cost_to_row(result_table, costDict, sorted_costs)
    districtCost = get(costDict, "districtNet", NaN)
    if isnan(districtCost)
        result_table *= " & "
    else
        if sorted_costs[1][1] == "districtNet"
            bestNonDistrictCost = sorted_costs[2][2]
        else
            bestNonDistrictCost = sorted_costs[1][2]
        end
        relativeCost = 100 * (districtCost - bestNonDistrictCost) / bestNonDistrictCost
        if relativeCost > 0
            result_table *= " & (\$" * string(round(relativeCost, digits = 1)) * "\$\\%)"
        else
            result_table *=
                (" & (\$\\mathbf{" * string(round(relativeCost, digits = 1)) * "\\%}\$)")
        end
    end
    return result_table
end


"""
Initialize table: put in environment, add label, etc.
"""
function initialize_table(experiment, nb_columns)
    if experiment == "Experiment3"
        # This table will be in figure* LaTeX environment
        init_table = """
        \\captionof{table}{"""
    else
        init_table = """
        \\begin{table}[t]
            \\caption{"""
    end
    init_table *= """Districting costs across different cities and target district sizes for our method and benchmarks. The best result is shown in \\textcolor{blue}{blue} and the second best in \\textcolor{orange}{orange}. The last column shows the relative difference between \\model{DistrictNet} and the second-best method.}
            \\label{table:$experiment}
            \\resizebox{\\linewidth}{!}{
            \\begin{sc}
            \\begin{tabular}{$(repeat("c", nb_columns))}
                \\toprule
                City & \$t\$ & \\model{BD} & \\model{FIG} & \\model{PredGNN} & \\model{AvgTSP} &\\model{DistrictNet} & (Rel.) \\\\
                \\midrule
        """
    return init_table
end


"""
Format all results in LaTeX table and process them.
"""
function create_latex_result_table(data, methods, experiment)
    nb_columns = length(methods) + 3
    # Sort the data by city and then by t
    sort!(data, by = x -> (x[1], x[2]))

    # - Init -
    init_table = initialize_table(experiment, nb_columns)

    # - Results -
    current_city = ""
    city_row_span = 0
    result_table = ""
    #   Initialize dict to store result vectors
    allCostsDict = Dict{String,Vector{Float64}}()
    for method in methods
        allCostsDict[method] = []
    end


    for (city, t, costDict) in data
        # Read costs and identify best and second-best methods
        sorted_costs, min_cost, second_min_cost = indices_top_two_methods(costDict)

        # Format city
        if city != current_city
            # Add midrule if new city except for first
            if current_city != ""
                result_table *= "        \\cmidrule(lr){2-$nb_columns}\n"
            end
            # Print city name in multirow if first time
            current_city = city
            city_row_span = sum([(d[1] == city) for d in data])
            if city == "Manchester"
                result_table *= "        \\multirow{$city_row_span}{*}{Manch.}\n"
            else
                result_table *= "        \\multirow{$city_row_span}{*}{$city}\n"
            end
        end

        # Add cost to row and format according to value
        result_table *= "            & $t"
        for method in methods
            cost = get(costDict, method, "-")
            formatted_cost = format_cost(cost, min_cost, second_min_cost)
            result_table *= " & $formatted_cost"
            # Add cost to dict of all costs
            append!(allCostsDict[method], cost)
        end
        # Add percentage improvement of DistrictNet compared to otherwise best method
        result_table = add_relative_cost_to_row(result_table, costDict, sorted_costs)
        result_table *= " \\\\\n"
    end

    end_table = """
            \\bottomrule
        \\end{tabular}
        \\end{sc}
        }
    """
    if experiment !== "Experiment3"
        end_table *= """\\end{table}
        """
    end

    return init_table * result_table * end_table
end


"""
Read experiments results from file,
process them, format them in LaTeX tables,
and export to .tex file.
"""
function export_experiment_results_to_table(experiment, input_path, output_path)
    # Read experiment data from files
    folder_path = joinpath(input_path, experiment)
    data =
        [read_experiment_file(joinpath(folder_path, file)) for file in readdir(folder_path)]

    # Get unique methods
    methods = get_unique_methods(data)

    # Create the LaTeX table
    expected_methods = ["BD", "FIG", "predictGnn", "AvgTSP", "districtNet"]
    for method in methods
        @assert method in expected_methods
    end
    latex_table = create_latex_result_table(data, expected_methods, experiment)

    # Save the table to .tex
    file_name = "result_table_" * experiment * ".tex"
    output_file_name = joinpath(output_path, file_name)
    open(output_file_name, "w") do file
        write(file, latex_table)
    end

    return output_file_name
end


function get_Statistical_Analysis(data)
    method_names = ["districtNet", "predictGnn", "BD", "FIG", "AvgTSP"]
    mean_results = []
    p_values = []
    
    for method in method_names
        costs = Float64.(get_method_cost(data, method))
        p_value = pvalue(SignedRankTest(costs), tail=:right)
        mean_value = round(mean(costs), digits = 2)
        p_value_str = @sprintf("%.1e", p_value)
        
        push!(mean_results, mean_value)
        push!(p_values, p_value_str)
    end
    
    return mean_results, p_values
end
"""
Read experiments results from file,
process them, format them in LaTeX tables,
and export the ablation study to .tex file.
"""

function export_ablation_study_to_table(methods, input_path, output_path)
    cities =  ["London", "Leeds", "Bristol", "Manchester", "Lyon", "Paris", "Marseille"]
    depots = ["C"]
    sizes = [3, 6, 12, 20, 30]
    filenames = generate_filenames(input_path, cities, depots, sizes)
    data = [Compute_relative_cost(read_experiment_file(file)[3]) for file in filenames]
    mean_results, p_values = get_Statistical_Analysis(data)
    mean_districtNet, mean_predictGnn, mean_BD, mean_FIG , mean_AvgTSP = mean_results
    p_districtNet, p_predictGnn, p_BD, p_FIG, p_AvgTSP = p_values
    latex_table = """
    \\begin{table}[ht]
        \\centering
        \\caption{Ablation study showing the value of combining GNN and structured learning.}
        \\label{tab:ablation_study}
        \\begin{tabular}{lcc}
            \\toprule
                & Average relative cost& \$p\$-value \\\\
            \\midrule
                Benchmark 1: \\model{BD}, linear regression& $mean_BD \\% & $p_BD \\\\
                Benchmark 2: \\model{FIG}, linear regression & $mean_FIG \\% & $p_FIG  \\\\
                Benchmark 3: \\model{PredGnn}, Unstructured learning with GNN & $mean_predictGnn \\% & $p_predictGnn  \\\\
                Benchmark 4:  \\model{AvgTSP}, no learning  & $mean_AvgTSP \\% & $p_AvgTSP  \\\\
                \\model{DistrictNet}: structured learning with CMST and GNN & $mean_districtNet \\% & $p_districtNet  \\\\
            \\bottomrule
        \\end{tabular}
    \\end{table}
    """

    # Export to .tex file
    open(output_path, "w") do f
        write(f, latex_table)
    end
end




function read_experiment4_file(file_path, delimiter=":")
    costDict = Dict{String,Float64}()
    for line in eachline(file_path)
        println(line)
        method, cost = split(line, delimiter)
        method = strip(method)
        costDict[method] = round(parse(Float64, cost), digits = 2)
    end
    return costDict
end

"""
Wrapper function to load all necessary experiment data files.
"""
function load_experiment_data(base_path)
    files = ["meantestcompared.txt", "meantraincompared.txt", "maxtestcompared.txt",
             "maxtraincompared.txt", "mintestcompared.txt", "mintraincompared.txt"]
    data = Dict{String,Dict{String,Float64}}()
    
    for file in files
        file_name = joinpath(base_path, file)
        data[file] = read_experiment4_file(file_name, ":")
    end
    return data
end

"""
Helper function to extract statistical results for a specific method.
"""
function get_statistical_results(data, method, train=true)
    meant, maxt, mint = 0.0, 0.0, 0.0
    if train
        meant = data["meantraincompared.txt"][method]
        maxt = data["maxtraincompared.txt"][method]
        mint = data["mintraincompared.txt"][method]
    else
        meant = data["meantestcompared.txt"][method]
        maxt = data["maxtestcompared.txt"][method]
        mint = data["mintestcompared.txt"][method]
    end
    return meant, maxt, mint
end

"""
Helper function to generate LaTeX table.
"""
function generate_latex_table(file_path, table_caption, methods, stats)
    latex_table = """
    \\begin{table}[ht]
        \\centering
        \\caption{$table_caption}
        \\begin{tabular}{lccc}
            \\toprule
                & Average & Maximum & Minimum\\\\
            \\midrule
    """
    
    for method in methods
        mean, max, min = stats[method]
        latex_table *= """
            \\model{$method} & $mean \\% & $max \\% & $min \\%  \\\\
        """
    end
    
    latex_table *= """
            \\bottomrule
        \\end{tabular}
    \\end{table}
    """
    
    # Export to .tex file
    open(file_path, "w") do f
        write(f, latex_table)
    end
end

"""
Main function to handle the statistical analysis and LaTeX table export.
"""
function export_statistical_results(base_path, output_path)
    methods = ["districtNet", "predictGnn", "BD", "FIG"]
    
    # Load experiment data
    data = load_experiment_data(base_path)

    # Prepare results for each method
    train_stats = Dict{String, Tuple{Float64, Float64, Float64}}()
    test_stats = Dict{String, Tuple{Float64, Float64, Float64}}()
    for method in methods
        train_stats[method] = get_statistical_results(data, method, true)
        test_stats[method] = get_statistical_results(data, method, false)
    end

    # Generate LaTeX tables
    generate_latex_table(joinpath(output_path, "TestGap.tex"), "Evaluation of each model on the test set", methods, train_stats)
    generate_latex_table(joinpath(output_path, "TrainGap.tex"), "Evaluation of each model on the train set", methods, test_stats)
end
