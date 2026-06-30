using DataFrames, CSV
using PyCall

# Import the necessary Python libraries
@pyimport shapely.geometry as geom
@pyimport shapely.ops as ops
@pyimport shapely as shp

include("src/Export/plots.jl")

# Plot districts on top of city map
cities = ["Bristol", "Leeds", "London", "Lyon", "Manchester", "Marseille", "Paris"]
methods = ["BD", "FIG", "predictGnn", "AvgTSP", "districtNet"]
target_sizes = [3, 6, 12, 20, 30]

all_compactness = zeros(length(methods), length(cities), length(target_sizes))

for (i, city) in enumerate(cities)
    println("City: ", city)
    for (t, target_size) in enumerate(target_sizes)
        println("t = ", target_size)
        for (m, method) in enumerate(methods)
            println("Method: ", method)
            # Read instance from file
            data_filename = construct_filename(city, target_size, method, "data")
            data = JSON.parsefile(data_filename)
            # Read districting solution from file
            solution_filename = construct_filename(city, target_size, method, "solution")
            district_ids = parse_ids(solution_filename)

            # Extract polygons in each district
            district_geos = []
            for ids in district_ids
                district = []
                for feature in data["features"]
                    if feature["properties"]["ID"] in ids
                        push!(district, feature["geometry"])
                    end
                end
                push!(district_geos, district)
            end

            # Merge polygons in each district
            district_polygons = []
            district_compactness = []
            for geos in district_geos
                # Convert the GeoJSON polygons to Shapely polygons
                polygons = [geom.shape(geo) for geo in geos]
                # Check validity and print issues if any
                for (i, poly) in enumerate(polygons)
                    if !poly.is_valid
                        println("Polygon $i is invalid: ", poly)
                        # Optionally, attempt to fix or simplify invalid polygons
                        polygons[i] = poly.buffer(0)  # A common fix to make invalid geometries valid
                    end
                end
                # Merge the polygons
                merged_polygon = ops.unary_union(polygons)
                # print the new geometry
                push!(district_polygons, merged_polygon)
            end

            # Print the perimeter, area, and compactness of each district
            compactness = zeros(length(district_polygons))
            for (d, poly) in enumerate(district_polygons)
                circle = shp.minimum_bounding_circle(poly)
                D_area = poly.area
                C_area = circle.area
                compactness[d] = D_area/C_area
            end

            all_compactness[m, i, t] = mean(compactness)

        end
    end
end

# Average over all target sizes
all_compactness = mean(all_compactness, dims=3)
println(all_compactness[1, :])
println(all_compactness[4, :])
println(all_compactness[5, :])

# Add average compactness of the original districts as a column
all_compactness = hcat(all_compactness, mean(all_compactness, dims=2))

# Round to 3 decimals
all_compactness = round.(all_compactness, digits=3)

# Remove the last dimension
all_compactness = all_compactness[:, :, 1]

# Export to csv
if !isdir("output/csv")
    mkdir("output/csv")
end
CSV.write("output//csv//compactness.csv", DataFrame(all_compactness', :auto), writeheader=true)


