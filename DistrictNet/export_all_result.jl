using PyCall
using Statistics

include("src/Export/processing.jl")
include("src/Export/plots.jl")
include("src/Export/results.jl")

## Filter initial geojson to have only BUs used in experiments
process_geojson_for_city("London")
process_geojson_for_city("Manchester")


# - - - Districting maps - - -
# Download OSM background maps
@pyinclude("src\\map.py")

# Plot districts on top of city map
cities = ["London", "Manchester"]
methods = ["BD", "FIG", "predictGnn", "districtNet"]
target_sizes = [6, 20]

for city in cities
    for t in target_sizes
        id_colors = []
        for method in methods
            id_colors = generate_city_mapplot(city, t, method, id_colors)
        end
    end
end

# - - - Districting cost as boxplot - - -
# Plot district costs as boxplot
cities = ["London", "Leeds", "Manchester", "Bristol"]
target_sizes = [3, 6, 12, 20, 30]
output_path = "output/plots/boxplots/"
mkpath(output_path)
generate_cities_boxplots(cities, target_sizes)



#----------- Ablation study ----
# Generate ablation study table
cities =  ["London", "Leeds", "Bristol", "Manchester", "Lyon", "Paris", "Marseille"]
depots = ["C"]
times = [3, 6, 12, 20, 30]
methods = ["districtNet", "predictGnn", "BD", "FIG", "AvgTSP"]
output_path = "output/tables/ablation_study.tex"
input_path = "output/solution/Comparaison/Experiment_General_cities"
export_ablation_study_to_table(methods, input_path, output_path)

# - - - Main result tables - - -
# Get path and create folders if needed
input_path = "output/solution/Comparaison"
output_path = "output/tables/"
mkpath(output_path)
# Process all experiments and export results to LaTeX table
experimentList = ["Experiment_General_cities"]
for experiment in experimentList
    println("Processing experiment: $experiment")
    result_file = export_experiment_results_to_table(experiment, input_path, output_path)
    println("Result saved to LaTeX table: $result_file")
end


# - - - Cost as a function of city size: csv files - - -
output_path = "output/csv/"
mkpath(output_path)
methods = ["BD", "FIG", "predictGnn", "districtNet"]
# Leeds
city = "Bristol"
sizes = [110, 160, 210, 260] 
export_city_size_results_to_csv(methods, sizes, city)
# London
city = "London"
sizes = [100, 300]
export_city_size_results_to_csv(methods, sizes, city)
# Leeds
city = "Leeds"
sizes = [90, 150, 190, 240, 290] 
export_city_size_results_to_csv(methods, sizes, city)


 - - - Cost as a function of training examples: csv files - - -
output_path = "output/csv/"
methods = ["BD", "FIG", "predictGnn", "districtNet"]
cities = ["London", "Leeds", "Manchester", "Bristol"]
all_t = [3, 6, 12, 20, 30]
all_n = [20, 50, 100, 150, 200]
all_results = dict_all_results_n(methods, cities, all_t, all_n)
for city in cities
    for t in all_t
        export_training_size_results_to_csv(city, t, all_n, all_results[city][t])
    end
end

# Single summary statistic over cities and t's
# Average relative improvement compared to n=20
relativeCosts = zeros(length(all_n), length(cities) * length(all_t))
for (i, city) in enumerate(cities)
    for (j, t) in enumerate(all_t)
        for (k, n) in enumerate(all_n)
            ind = (i - 1) * length(all_t) + j
            relativeCosts[k, ind] = (
                100 * all_results[city][t]["districtNet"][n] /
                all_results[city][t]["districtNet"][20]
            )
        end
    end
end
# Export results to .csv
meanRealCost = mean(relativeCosts, dims = 2)
stdRealCost = std(relativeCosts, dims = 2)
upperConf = meanRealCost - 1.96 * stdRealCost / sqrt(length(cities) * length(all_t))
lowerConf = meanRealCost + 1.96 * stdRealCost / sqrt(length(cities) * length(all_t))
# Export results to csv file
resDf = DataFrame([all_n meanRealCost upperConf lowerConf], :auto)
# check if output folder exists
if !isdir("output/csv")
    mkdir("output/csv")
end
CSV.write("output\\csv\\train_size_all_cities.csv", resDf)

# Export statistical results for training and testing on small cities
export_statistical_results("output/solution/Comparaison/Experiment_General_small_cities", "output/tables")
