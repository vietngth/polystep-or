using Plots
using JSON
using Distributions
using StatsPlots, Distributions
using FileIO

function construct_filepath(city::String, target_size::Int)
    return "data/tspCosts/" *
           city *
           "_C_120_" *
           string(target_size) *
           "_tsp.train_and_test.json"
end


"""
Get cost of districting solution from path to solution file.
"""
function get_cost(path)
    # Read solution file from path
    data = open(path) do file
        JSON.parse(file)
    end
    t = split(path, "_")[4]
    int_t = parse(Int64, t)
    raw_districts = data["districts"]
    district_costs = []
    for i in range(1, length(raw_districts))
        if length(raw_districts[i]["list-blocks"]) < 0.8 * int_t
            continue
        end
        push!(district_costs, raw_districts[i]["average-cost"])
    end
    return Float64.(district_costs)
end

"""
Boxplot of district costs for a single city.
"""
function single_city_boxplot(target_sizes::Array{Int,1}, cost_data)
    plot_font = "times"
    default(fontfamily = plot_font)
    # Plot figure
    labels = string.(target_sizes)
    p = boxplot(
        cost_data,
        label = labels,
        legend = false,
        size = (220, 160),
        palette = :seaborn_colorblind6,
        xguidefontsize = 9,
        yguidefontsize = 9,
        titlefontsize = 12,
    )
    xticks!(p, (1:length(target_sizes), labels))
    return p
end


"""
Generate and save boxplots for a city.
"""
function generate_and_save_boxplot(city::String, target_sizes::Array{Int,1}, i::Int)
    cost_data = [get_cost(construct_filepath(city, size)) for size in target_sizes]
    p = single_city_boxplot(target_sizes, cost_data)
    # Save single city boxplot figure
    savefig("output/plots/boxplots/" * city * "_cost_boxplot.pdf")
    return p
end


"""
Generate boxplot of district costs for list of cities.
"""
function generate_cities_boxplots(cities::Array{String,1}, target_sizes::Array{Int,1})
    plots = []
    # Plot each city boxplot individually and save figure
    for (i, city) in enumerate(cities)
        push!(plots, generate_and_save_boxplot(city, target_sizes, i))
    end
    # Combine all plots in grid of width 2
    _ = plot(plots..., layout = (length(cities) ÷ 2, 2))
    savefig("output/plots/boxplots/all_cost_boxplot.pdf")
end


"""
Construct file names based on input parameters.
"""
function construct_filename(
    city::String,
    target_size::Int,
    method::String,
    file_type::String,
)
    if file_type == "data"
        return "data/geojson/" * city * ".geojson"
    elseif file_type == "solution"
        return "output/solution/Experiment_General_cities/" *
               city *
               "_C_120_" *
               string(target_size) *
               "." *
               method *
               ".txt"
    else
        throw(ArgumentError("Invalid file type"))
    end
end


"""
Parse IDs from file name.
"""
function parse_ids(filename)
    lines = readlines(filename)
    ids = []
    for i in range(6, length(lines))
        line = lines[i]
        try
            push!(ids, parse.(Int, split(line)) .- 1)
        catch e
            # not a line of IDs, skip it
        end
    end
    return ids
end


"""
Create large color palette.
"""
function create_color_palette()
    color_palette = [
        "red",
        "blue",
        "green",
        "yellow",
        "purple",
        "cyan",
        "magenta",
        "orange",
        "brown",
        "lime",
        "pink",
        "violet",
        "indigo",
        "coral",
        "teal",
        "olive",
        "navy",
        "maroon",
        "aquamarine",
        "turquoise",
        "silver",
        "goldenrod",
        "salmon",
        "tan",
        "royalblue",
        "plum",
        "peachpuff",
        "orchid",
        "mediumseagreen",
        "mediumorchid",
        "mediumpurple",
        "mediumblue",
        "lightcoral",
        "lawngreen",
        "lavender",
        "khaki",
        "hotpink",
        "dodgerblue",
        "deepskyblue",
        "darkviolet",
    ]
    color_palette_rgb = [parse(Colorant, color) for color in color_palette]
    return color_palette_rgb
end

"""
Sort array by frequency of occurence, from:
https://discourse.julialang.org/t/sort-vector-by-frequency/14946/7
"""
function count_unique(V::AbstractVector{T}) where {T}
    U = unique(V)
    l = length(U)
    counts = Dict{T,Int}(zip(U, zeros(l)))
    for v in V
        counts[v] += 1
    end
    return counts
end

function frequency_sort(V::AbstractVector)
    counts = count_unique(V)
    sorted = sort(collect(counts); by = last, rev = true)
    return first.(sorted)
end


"""
Assign a color for each district.

The colors are assigned to districts depending on the ID of the 
BU they contain. That is, the first BU is always assigned the
first color in the palette ; the second BU is always assigned the
second color (if not in the same district as the first BU) ; and
so on.
"""
function assign_colors(districts)
    all_BUs = vcat(districts...)
    is_dist_colored = falses(length(districts))
    # Initialize empty dictionary: bu -> RGB color
    id_colors = Dict{Int,Int}()

    color_index = 1

    for BU_index = 1:length(all_BUs)
        if !all(is_dist_colored)
            dist_index = findfirst(district -> BU_index in district, districts)
            if !is_dist_colored[dist_index]
                # Assign color to all bus in this district
                for bu in districts[dist_index]
                    id_colors[bu] = color_index
                end

                # Keep track of colored districts
                is_dist_colored[dist_index] = true
                color_index = color_index + 1
            end
        end
    end

    return id_colors
end


function assign_colors(districts, prev_id_colors)
    nb_colors = maximum(collect(values(prev_id_colors)))
    is_color_used = falses(nb_colors)
    # Initialize empty dictionary: bu -> RGB color
    id_colors = Dict{Int,Int}()

    # Read previous districts from assignment
    prev_districts = []
    for color in range(1, nb_colors)
        district = [k for (k, v) in prev_id_colors if v == color]
        push!(prev_districts, district)
    end

    # Rank districts in decreasing order of similarity with previous ones
    similarity = zeros(nb_colors)
    for d in range(1, nb_colors)
        for d_prime in range(1, nb_colors)
            nb_common = length(intersect(Set(districts[d]), Set(prev_districts[d_prime])))
            similarity[d] = max(nb_common, similarity[d])
        end
    end
    district_order = sortperm(similarity, rev = true)

    for indx in district_order
        district = districts[indx]

        # Read what colors were used in previous map
        colors_used = get.(Ref(prev_id_colors), district, missing)
        # Sort them in decreasing occurence
        colors = frequency_sort(colors_used)

        # Assign to this district the most frequent color
        # if it is available
        for color in colors
            if !is_color_used[color]
                district_color = color
                break
            end
            district_color = 0
        end

        # If no color assigned, assign first free color
        if district_color == 0
            district_color = findfirst(isequal(false), is_color_used)
        end

        # Assign color to all BUs in district
        for bu in district
            id_colors[bu] = district_color
        end
        is_color_used[district_color] = true

    end

    return id_colors
end


function get_bounding_rectangle(cityname::String)
    # Load GeoJSON data
    geojson_file = "data/geojson/" * cityname * "_120_BUs.geojson"
    geojson_str = read(geojson_file, String)
    geojson_data = JSON.parse(geojson_str)

    # Initialize min and max coordinates
    min_x, min_y = Inf, Inf
    max_x, max_y = -Inf, -Inf
    x_eps, y_eps = 1e-2, 1e-2

    # Iterate over each feature
    for feature in geojson_data["features"]
        # Check if the geometry type is Polygon or MultiPolygon
        if feature["geometry"]["type"] == "Polygon"
            for polygon in feature["geometry"]["coordinates"]
                for coord in polygon
                    min_x, min_y = min(min_x, coord[1]), min(min_y, coord[2])
                    max_x, max_y = max(max_x, coord[1]), max(max_y, coord[2])
                end
            end
        elseif feature["geometry"]["type"] == "MultiPolygon"
            for multipolygon in feature["geometry"]["coordinates"]
                for polygon in multipolygon
                    for coord in polygon
                        min_x, min_y = min(min_x, coord[1]), min(min_y, coord[2])
                        max_x, max_y = max(max_x, coord[1]), max(max_y, coord[2])
                    end
                end
            end
        end
    end

    # Add small epsilon to min and max coordinates
    min_x, min_y = min_x - x_eps, min_y - y_eps
    max_x, max_y = max_x + x_eps, max_y + y_eps
    # round to 7 decimal places
    min_x, min_y = round(min_x, digits = 7), round(min_y, digits = 7)
    max_x, max_y = round(max_x, digits = 7), round(max_y, digits = 7)

    return (min_x, max_x, min_y, max_y)
end


"""
Convert latitude into x-coordinate in Spherical Pseudo-Mercator projection.
See: https://wiki.openstreetmap.org/wiki/Mercator#Spherical_Pseudo-Mercator_projection
"""
function lat2y(lat)
    R = 6378137.0
    return log(tan(pi / 4 + deg2rad(lat) / 2)) * R
end


"""
Convert longitute into y-coordinate in Spherical Pseudo-Mercator projection.
See: https://wiki.openstreetmap.org/wiki/Mercator#Spherical_Pseudo-Mercator_projection
"""
function lon2x(lon)
    R = 6378137.0
    return deg2rad(lon) * R
end


"""
Plot and save the figure.
"""
function plot_and_save(
    city::String,
    target_size::Int,
    method::String,
    data,
    id_colors,
    output_path,
)
    # Create large color palette
    color_palette_rgb = create_color_palette()

    # Initialize empty plot
    plot(
        legend = false,
        aspect_ratio = :equal,
        axis = false,
        grid = false,
        framestyle = :none,
    )

    # Load and plot background image
    if city == "London"
        GPS_BOX = [-0.264807777, 0.0246201708, 51.450549979, 51.567122329]
        bg_img = load("data\\maps\\London_map.png")
    elseif city == "Manchester"
        GPS_BOX = [-2.489495333, -1.991144766, 53.3936437, 53.57961082480]
        bg_img = load("data\\maps\\Manchester_map.png")
    else
        throw(ArgumentError("Invalid city string argument."))
    end

    # Get bounding box by converting GPS coordinates to Mercator projection
    bbox = [lon2x(GPS_BOX[1]), lon2x(GPS_BOX[2]), lat2y(GPS_BOX[3]), lat2y(GPS_BOX[4])]

    # Plot background iamge
    sz = size(bg_img)
    x, y = LinRange(bbox[1], bbox[2], sz[1]), LinRange(bbox[3], bbox[4], sz[2])
    plot!(x, y, bg_img, ylimits = (bbox[3], bbox[4]), xlimits = (bbox[1], bbox[2]))

    # Calculate parameters of linear mirror transformation
    m = (lat2y(GPS_BOX[4]) - lat2y(GPS_BOX[3])) / (lat2y(GPS_BOX[3]) - lat2y(GPS_BOX[4]))
    q = lat2y(GPS_BOX[4]) - m * lat2y(GPS_BOX[3])

    for feature in data["features"]
        if feature["properties"]["ID"] < 120
            feature_id = feature["properties"]["ID"]
            coords = feature["geometry"]["coordinates"][1]
            x_coords = [lon2x(point[1]) for point in coords]
            # Apply mirror transformation to y coordinates
            y_coords = [m * lat2y(point[2]) + q for point in coords]
            plot!(
                x_coords,
                y_coords,
                fill = true,
                fillalpha = 0.45,
                linecolor = :black,
                linewidth = 0.75,
                fillcolor = color_palette_rgb[id_colors[feature_id]],
            )
        end
    end
    cenroid = data["metadata"]["REFERENCE_LONGLAT"]
    cenroid_x = lon2x(cenroid[1])
    cenroid_y = m * lat2y(cenroid[2]) + q
    scatter!([cenroid_x], [cenroid_y], markersize = 9, markercolor = :white, shape = :star5)

    # Save figure
    savefig(output_path * city * "_" * string(target_size) * "_" * method * ".pdf")
end


"""
Display districting solution on top of background city map.
"""
function generate_city_mapplot(city::String, target_size::Int, method::String, id_colors)
    # Read districting solution from file
    data_filename = construct_filename(city, target_size, method, "data")
    solution_filename = construct_filename(city, target_size, method, "solution")
    data = JSON.parsefile(data_filename)
    ids_list = parse_ids(solution_filename)

    # Assign colors to districts
    if isempty(id_colors)
        id_colors = assign_colors(ids_list)
    else
        id_colors = assign_colors(ids_list, id_colors)
    end

    # Plot districts in color on top of background map
    output_path = "output/plots/districts/"
    mkpath(output_path)
    plot_and_save(city, target_size, method, data, id_colors, output_path)
    return id_colors
end


# folder_path = "../output/solution/Comparaison/Experiment4/"
# # Function to read data from a file
# function read_data(filename)
#     return parse.(Float64, readlines(filename))
# end

# # Read data from each file
# bd_path = string(folder_path, "BD.txt")
# fig_path = string(folder_path, "FIG.txt")
# predictGnn_path = string(folder_path, "predictGnn.txt")
# districtNet_path = string(folder_path, "districtNet.txt")
# bd_data = read_data(bd_path)
# fig_data = read_data(fig_path)
# predictGnn_data = read_data(predictGnn_path)
# districtNet_data = read_data(districtNet_path)

# # print mean and max min of each method multiply by 100
# println("BD: ", mean(bd_data) * 100, " ", maximum(bd_data) * 100, " ", minimum(bd_data) * 100)
# println("FIG: ", mean(fig_data) * 100, " ", maximum(fig_data) * 100, " ", minimum(fig_data) * 100)
# println("predictGnn: ", mean(predictGnn_data) * 100, " ", maximum(predictGnn_data) * 100, " ", minimum(predictGnn_data) * 100)
# println("districtNet: ", mean(districtNet_data) * 100, " ", maximum(districtNet_data) * 100, " ", minimum(districtNet_data) * 100)

#data = []
#push!(data, districtNet_data .* 100)
#push!(data, bd_data .* 100)
#push!(data, fig_data .* 100)
#push!(data, predictGnn_data .* 100)
#
#labels = ["districtNet" "BD" "FIG" "predictGnn"]
#plot_font = "times"
#default(fontfamily=plot_font)
##box plot
#p = boxplot(labels, data, legend = false, ylabel = "Gap with respect to the optimal solution", xlabel = "Methods", size=(500, 400), title = "Comparison of methods",
#        palette=:seaborn_colorblind6, xguidefontsize=9, yguidefontsize=9, titlefontsize=12)
#savefig("../output/plots/Gapboxplot.pdf")
#
##violin plot
#p = violin(labels, data, legend = false, ylabel = "Gap with respect to the optimal solution", xlabel = "Methods", size=(500, 400), title = "Comparison of methods",
#        palette=:seaborn_colorblind6, xguidefontsize=9, yguidefontsize=9, titlefontsize=12)
#savefig("../output/plots/Gapviolin.pdf")
