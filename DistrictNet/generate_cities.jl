using PyCall

py"""
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import os
import math
from shapely.geometry import Polygon, MultiPolygon
import random
import numpy as np
from scipy.stats import truncnorm
from shapely.geometry import Point
import contextily as ctx
import matplotlib.pyplot as plt
import math
from typing import Tuple, List

def determine_utm_zone(lon, lat):
    
    zone_number = int((lon + 180) / 6) + 1
    if lat >= 0:
        return f"EPSG:326{zone_number}"
    else:
        return f"EPSG:327{zone_number}"
    
def get_coordinates(feature):
    geom_type = feature["geometry"]["type"]
    coords = feature["geometry"]["coordinates"]
    if geom_type == "Polygon":
        return coords[0]
    elif geom_type == "MultiPolygon":
        return coords[0][0]
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")

def to_cartesian(WGS84Reference: Tuple[float, float], WGS84Position: Tuple[float, float]) -> Tuple[float, float]:
    M_PI = 3.141592653589793
    DEG_TO_RAD = M_PI / 180.0
    HALF_PI = M_PI / 2.0
    EPSILON10 = 1.0e-10
    EPSILON12 = 1.0e-12

    EQUATOR_RADIUS = 6378137.0
    FLATTENING = 1.0 / 298.257223563
    SQUARED_ECCENTRICITY = 2.0 * FLATTENING - FLATTENING * FLATTENING
    SQUARE_ROOT_ONE_MINUS_ECCENTRICITY = 0.996647189335
    POLE_RADIUS = EQUATOR_RADIUS * SQUARE_ROOT_ONE_MINUS_ECCENTRICITY

    C00 = 1.0
    C02 = 0.25
    C04 = 0.046875
    C06 = 0.01953125
    C08 = 0.01068115234375
    C22 = 0.75
    C44 = 0.46875
    C46 = 0.01302083333333333333
    C48 = 0.00712076822916666666
    C66 = 0.36458333333333333333
    C68 = 0.00569661458333333333
    C88 = 0.3076171875

    R0 = C00 - SQUARED_ECCENTRICITY * (C02 + SQUARED_ECCENTRICITY * (C04 + SQUARED_ECCENTRICITY * (C06 + SQUARED_ECCENTRICITY * C08)))
    R1 = SQUARED_ECCENTRICITY * (C22 - SQUARED_ECCENTRICITY * (C04 + SQUARED_ECCENTRICITY * (C06 + SQUARED_ECCENTRICITY * C08)))
    R2T = SQUARED_ECCENTRICITY * SQUARED_ECCENTRICITY
    R2 = R2T * (C44 - SQUARED_ECCENTRICITY * (C46 + SQUARED_ECCENTRICITY * C48))
    R3T = R2T * SQUARED_ECCENTRICITY
    R3 = R3T * (C66 - SQUARED_ECCENTRICITY * C68)
    R4 = R3T * SQUARED_ECCENTRICITY * C88

    def mlfn(lat):
        sin_phi = math.sin(lat)
        cos_phi = math.cos(lat) * sin_phi
        squared_sin_phi = sin_phi * sin_phi
        return (R0 * lat - cos_phi * (R1 + squared_sin_phi * (R2 + squared_sin_phi * (R3 + squared_sin_phi * R4))))

    ML0 = mlfn(WGS84Reference[0] * DEG_TO_RAD)

    def msfn(sinPhi, cosPhi, es):
        return (cosPhi / math.sqrt(1.0 - es * sinPhi * sinPhi))

    def project(lat, lon):
        retVal = [lon, -1.0 * ML0]
        if abs(lat) >= EPSILON10:
            ms_val = msfn(math.sin(lat), math.cos(lat), SQUARED_ECCENTRICITY) / math.sin(lat) if abs(math.sin(lat)) > EPSILON10 else 0.0
            retVal[0] = ms_val * math.sin(lon * math.sin(lat))
            retVal[1] = (mlfn(lat) - ML0) + ms_val * (1.0 - math.cos(lon))
        return retVal

    def fwd(lat, lon):
        D = abs(lat) - HALF_PI
        if (D > EPSILON12) or (abs(lon) > 10.0):
            return (0.0, 0.0)
        if abs(D) < EPSILON12:
            lat = -1.0 * HALF_PI if lat < 0.0 else HALF_PI
        lon -= WGS84Reference[1] * DEG_TO_RAD
        projectedRetVal = project(lat, lon)
        return (EQUATOR_RADIUS * projectedRetVal[0], EQUATOR_RADIUS * projectedRetVal[1])

    return fwd(WGS84Position[0] * DEG_TO_RAD, WGS84Position[1] * DEG_TO_RAD)

def convert_to_km(cartesian_coords: Tuple[float, float]) -> Tuple[float, float]:
    x_km = cartesian_coords[0] / 1000
    y_km = cartesian_coords[1] / 1000
    return (x_km, y_km)
def convert_to_cartesian(coords: List[Tuple[float, float]], WGS84Reference) -> List[Tuple[float, float]]:
    cartesian_coords = [to_cartesian(WGS84Reference, coord) for coord in coords]
    cartesian_coords = [convert_to_km(coord) for coord in cartesian_coords]
    return cartesian_coords
    

def build_connected_city(neighbors, start_polygon, max_bu):
    # Initialize the queue and set with the starting polygon index
    queue = [start_polygon]
    connected_set = {start_polygon}

    # List to keep track of all possible neighbors
    all_neighbors = []

    # BFS
    while queue and len(connected_set) < max_bu:
        current_polygon = queue.pop(0)

        # Add neighbors of the current polygon to the all_neighbors list
        for neighbor in neighbors[current_polygon]:
            if neighbor not in connected_set and neighbor not in all_neighbors:
                all_neighbors.append(neighbor)

        # Continue if there are no more neighbors to process
        if not all_neighbors:
            continue

        # Select a random neighbor from the entire list
        random_neighbor = random.choice(all_neighbors)
        all_neighbors.remove(random_neighbor)

        # Add the selected neighbor to the connected set and queue
        if random_neighbor not in connected_set:
            connected_set.add(random_neighbor)
            queue.append(random_neighbor)

    return connected_set

def generate_populations(n, mean=8000, std=2000, min_val=5000, max_val=20000):
    # Get the parameters for the truncated normal distribution
    a, b = (min_val - mean) / std, (max_val - mean) / std
    # Generate the populations
    populations = truncnorm.rvs(a, b, loc= mean, scale = std, size = n)
    return populations.astype(int)# Convert to integers for population counts

def pick_city(base_dir):
    cities = []
    for file in os.listdir(base_dir):
        cities.append(file)
    probabilities = []
    for city in cities:
        with open(os.path.join(base_dir, city), 'r', encoding='utf-8') as f:
            data = json.load(f)
        features = data["features"]
        probabilities.append(len(features))
    probabilities = np.array(probabilities)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(cities, p=probabilities)
    

def process_city(city, base_dir, folder_path, indice):
    # Reading city data
    with open(os.path.join(base_dir, city), 'r', encoding='utf-8') as f:
        data = json.load(f)
    features = data["features"]
    gdf = gpd.GeoDataFrame.from_features(features)
    print(f"Read {city}'s file with {len(gdf)} rows.")
    gdf['centroid'] = gdf.geometry.centroid

    # Find neighbors
    joined_gdf = gpd.sjoin(gdf, gdf, how="inner", predicate='intersects')
    joined_gdf = joined_gdf[joined_gdf.index != joined_gdf['index_right']]
    neighbors = joined_gdf.groupby(joined_gdf.index)['index_right'].apply(list)
    # Filter to a connected subset
    indices = gdf.index

    # Create a probability distribution
    probabilities = np.ones(len(indices))  # Initialize with equal probabilities
    higher_probability_indices = 300       # Number of indices with higher probability

    # Adjust probabilities for the first 300 indices
    if len(indices) > higher_probability_indices:
        probabilities[:higher_probability_indices] *= 10  
    # Normalize probabilities to sum to 1
    probabilities /= probabilities.sum()
    # Choose a start_polygon based on the custom probability distribution
    start_polygon = np.random.choice(indices, p=probabilities)
    max_bu = 30
    resulting_polygons = build_connected_city(neighbors, start_polygon, max_bu)
    filtered_gdf = gdf.loc[list(resulting_polygons)]

    # Convert to UTM and compute areas and perimeters
    filtered_gdf.set_crs(epsg=4326, inplace=True)
    
    # Determine the UTM zone based on centroid of the entire region
    region_centroid = filtered_gdf.unary_union.centroid
    utm_zone = determine_utm_zone(region_centroid.x, region_centroid.y)

    utm_gdf = filtered_gdf.to_crs(utm_zone)
    areas_m2 = utm_gdf.geometry.area
    perimeters = utm_gdf.geometry.length

    # Assign properties
    utm_gdf['AREA'] = areas_m2 / 1e6
    utm_gdf['PERIMETER'] = perimeters/ 1e3
    utm_gdf['LIST_ADJACENT'] = utm_gdf.index.map(neighbors)  # Ensure matching indices
    utm_gdf['POPULATION'] =generate_populations(len(utm_gdf))
    utm_gdf['DENSITY'] = utm_gdf['POPULATION'] / utm_gdf['AREA']
    utm_gdf['ID'] = utm_gdf.index
    utm_gdf['CENTROID_X'] = utm_gdf['centroid'].apply(lambda x: x.x)
    utm_gdf['CENTROID_Y'] = utm_gdf['centroid'].apply(lambda x: x.y)
    utm_gdf['REGION_CENTROID_X'] = region_centroid.x
    utm_gdf['REGION_CENTROID_Y'] = region_centroid.y

    # Convert back to longitude-latitude and save
    lonlat_gdf = utm_gdf.to_crs(epsg=4326)
    # Convert the LIST_ADJACENT column to strings
    lonlat_gdf.drop(columns=['centroid'], inplace=True)
    lonlat_gdf['LIST_ADJACENT'] = lonlat_gdf['LIST_ADJACENT'].apply(lambda x: ','.join(map(str, x)))
    columns_to_save = ['geometry', 'AREA', 'PERIMETER', 'LIST_ADJACENT', 
                'POPULATION', 'DENSITY', 'ID', 'CENTROID_X', 
                'CENTROID_Y', 'REGION_CENTROID_X', 'REGION_CENTROID_Y']

    # Create a new GeoDataFrame with only the specified columns
    lonlat_gdf = gpd.GeoDataFrame(lonlat_gdf[columns_to_save], geometry='geometry')
    lonlat_gdf.to_file(f"{folder_path}/{city.split('.geojson')[0]}_{indice}", driver="GeoJSON")


        
def swap_coordinates(coord_list):
    return [(lat, lon) for lon, lat in coord_list]

def process_geojson_files(folder_path):
    for file_name in os.listdir(folder_path):
        print(file_name)
        if file_name.startswith("London") or file_name.startswith("Manchester") or file_name.startswith("Leeds") or file_name.startswith("Bristol"):
            continue
        idx = 0
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as f:
            data = json.load(f)
        features = data["features"]
        #store the old id and the new one
        id_dict = {}
        for feature in features:
            id_dict[feature["properties"]["ID"]] = idx
            feature["properties"]["ID"] = idx
            idx +=1
        for feature in features:
            feature["properties"]["LIST_ADJACENT"] = ",".join([str(id_dict[int(x)]) for x in feature["properties"]["LIST_ADJACENT"].split(",") if int(x) in id_dict])
        #write the new file
        for feature in features:
            feature["properties"]["LIST_ADJACENT"] = [int(x) for x in feature["properties"]["LIST_ADJACENT"].split(",")]
            #convert the centroid to a list
            feature["properties"]["centre"] = [feature["properties"]["CENTROID_X"], feature["properties"]["CENTROID_Y"]]
            # add the points using the cartesian coordinates
            # Compute Cartesian points
        centroid = [data["features"][0]["properties"]["REGION_CENTROID_X"], data["features"][0]["properties"]["REGION_CENTROID_Y"]]
        swaped = swap_coordinates([centroid])[0]
        for feature in features:
            coords = get_coordinates(feature)
            coords = swap_coordinates(coords)
            coords = convert_to_cartesian(coords, swaped)
            feature["properties"]["POINTS"] = coords
        #add Metadata to the file with centroid of the region
        centroid = [data["features"][0]["properties"]["REGION_CENTROID_X"], data["features"][0]["properties"]["REGION_CENTROID_Y"]]
        data["metadata"] = {"REFERENCE_LONGLAT": centroid}
        
        with open(os.path.join(folder_path, file_name), 'w') as f:
            json.dump(data, f)

def  run(n, base_dir, folder_path):
    # creat folder_path if it does not exist
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    for i in range(n):
        #base_dir = "data/cities"
        city = pick_city(base_dir)
        process_city(city, base_dir, folder_path, i)

    idx = 1
    for file_name in os.listdir(folder_path):
        if file_name.startswith("London") or file_name.startswith("Manchester") or file_name.startswith("Leeds") or file_name.startswith("Bristol"):
            continue
        new_name = f"city{idx}.geojson" 
        os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_name))
        idx +=1
    process_geojson_files(folder_path)

"""


function main()
    n = parse(Int, ARGS[1])
    base_dir = ARGS[2]  #"data/cities" 
    folder_path = ARGS[3]  #"data/geojsons"
    py"run"(n, base_dir, folder_path)
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
