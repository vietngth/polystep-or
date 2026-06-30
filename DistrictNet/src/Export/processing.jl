using DataFrames
using CSV


"""
Filter the geojson files with all BUs to have only the 120
BUs used in the main experimetns.
"""
function process_geojson_for_city(city::String)
    input_path = "data/geojson/" * city * ".geojson"
    output_path = "data/geojson/" * city * "_120_BUs.geojson"

    # Open input data file and read JSON data
    data = nothing
    open(input_path, "r") do file
        data = JSON.parse(file)
    end
    metadata = data["metadata"]
    features = data["features"]

    # Extract first 120 BUs and save to new geojson file
    new_geojson =
        Dict("type" => data["type"], "features" => features[1:120], "metadata" => metadata)
    open(output_path, "w") do file
        JSON.print(file, new_geojson)
    end
end


function parse_result_file(filepath)
    results = Dict{String,Float64}()
    open(filepath, "r") do file
        for line in eachline(file)
            method, value = split(line)
            results[method] = parse(Float64, value)
        end
    end
    return results
end


function export_city_size_results_to_csv(methods, sizes, city)
    folder_path = "output/solution/Comparaison/Experiment_General_multisize_cities/"
    # Initialize the dictionary for storing results
    all_results = Dict{String,Dict{Int,Float64}}()
    for method in methods
        all_results[method] = Dict{Int,Float64}()
    end

    # Read and parse each file, storing the results
    for size in sizes
        t = Int(floor(size / 6))
        filepath = folder_path * "$(city)_C_$(size)_$(t).txt"
        file_results = parse_result_file(filepath)
        for (method, value) in file_results
            all_results[method][size] = value
        end
    end

    # Add the results to the arrays
    districtNet = []
    predictGnn = []
    BD = []
    FIG = []
    for size in sizes
        push!(BD, 100 * all_results["BD"][size] / all_results["districtNet"][size])
        push!(FIG, 100 * all_results["FIG"][size] / all_results["districtNet"][size])
        push!(
            predictGnn,
            100 * all_results["predictGnn"][size] / all_results["districtNet"][size],
        )
        push!(
            districtNet,
            100 * all_results["districtNet"][size] / all_results["districtNet"][size],
        )
    end

    # Export results to csv file
    resDf = DataFrame([sizes BD FIG predictGnn districtNet], :auto)
    CSV.write("output\\csv\\$(city)_C_$(size)_k_6.csv", resDf)
end


"""
Read and store all results in dict.
"""
function dict_all_results_n(methods, cities, all_t, all_n)
    folder_path = "output/solution/Comparaison/Experiment_General_multisize_data/"
    # Initialize the dictionary for storing results
    all_results = Dict()
    for city in cities
        all_results[city] = Dict()
        for t in all_t
            all_results[city][t] = Dict()
            for method in methods
                all_results[city][t][method] = Dict()
                for n in all_n
                    all_results[city][t][method][n] = Dict()
                end
            end
        end
    end
    # Read and parse each file, storing the results
    for city in cities
        for t in all_t
            for n in all_n
                filepath = folder_path * "$(city)_C_120_$(t)_$(n).txt"
                file_results = parse_result_file(filepath)
                for (method, value) in file_results
                    all_results[city][t][method][n] = value
                end
            end
        end
    end
    return all_results
end


function export_training_size_results_to_csv(
    city::String,
    t::Int64,
    all_n::Vector{Int64},
    all_results,
)
    # Add the results to the arrays
    districtNet = []
    predictGnn = []
    BD = []
    FIG = []
    for n in all_n
        push!(BD, all_results["BD"][n])
        push!(FIG, all_results["FIG"][n])
        push!(predictGnn, all_results["predictGnn"][n])
        push!(districtNet, all_results["districtNet"][n])
    end

    # Export results to csv file
    resDf = DataFrame([all_n BD FIG predictGnn districtNet], :auto)
    CSV.write("output\\csv\\train_size_$(city)_C_120_$(t).csv", resDf)
end
