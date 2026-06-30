module AvgTSP
export fitGeneralModel, find_solution_city
using InferOpt, Plots
using MLUtils, GraphNeuralNetworks, Graphs, MetaGraphs, GraphPlot
using JSON, FilePaths, Random, LinearAlgebra
using Distributions, Statistics, UnionFind, Flux
using DataStructures, Serialization, FileIO
using Base.Filesystem: isfile
using Combinatorics, Optim
using JuMP, GLPK, CxxWrap, DistributedArrays
using PyCall



include("../utils.jl")
include("../struct.jl")
include("../instance.jl")
include("../district.jl")
include("../solution.jl")
include("../learning.jl")
include("../Solver/Kruskal.jl")
include("../Solver/localsearch.jl")
include("../Solver/exactsolver.jl")


using .CostEvaluator: EVmain
using .GenerateScenario: SCmain
# Constants

const NB_BU_LARGE = 120
const NB_BU_SMALL = 30
const DEPOT_LOCATION = "C"
const STRATEGY = "AvgTSP"
const NB_SCENARIO = 1

# Solver Hyperparameters
const MAX_TIME = 120
const PERTURBATION_PROBABILITY = 0.985
const PENALITY = 10000

using PyCall
py"""
import json
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
import matplotlib.pyplot as plt
import math
from typing import Tuple, List
import geopandas as gpd
    
def get_coordinates_centre(feature):
    geom_type = feature["properties"]["centre"]
    return [geom_type]
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

def swap_coordinates(coord_list):
    lon, lat = coord_list[0][0], coord_list[0][1]
    return [(lat, lon)]
    
def get_centre_of_mass(features, num_blocks):
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    for feature in features:
        if feature['properties']['ID'] >= num_blocks:
            break
        for coord in feature['properties']['POINTS']:
                x, y = coord
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y
    centre = ((min_x + max_x) / 2, (min_y + max_y) / 2)
    return centre

def get_scenario(city, num_blocks, t):
    with open(f"data/geojson/{city}.geojson", "r", encoding="utf-8") as f:
        data = json.load(f)
    features = data["features"]
    centroid = data["metadata"]["REFERENCE_LONGLAT"]
    lon, lat = centroid[0], centroid[1]
    swaped = swap_coordinates([centroid])[0]
    TSPSenario = dict()
    TSPSenario["blocks"] = []
    for feature in features:
        block = dict()
        if feature['properties']['ID'] >= num_blocks:
            break
        coords = get_coordinates_centre(feature)
        coords = swap_coordinates(coords)
        coords = convert_to_cartesian(coords, swaped)
        block["DEPOT_DIST"] = 0.5
        block["ID"] = feature['properties']['ID']
        block["Scenarios"] = [[[coords[0][0], coords[0][1]]]]
        TSPSenario["blocks"].append(block)
        centre = get_centre_of_mass(features, num_blocks)
    TSPSenario["metadata"] = dict()
    TSPSenario["metadata"]["DEPOT_XY"] = [centre[0], centre[1]]
    with open(f'deps/Scenario/output/{city}_C_{num_blocks}_{t}.json', 'w', encoding='utf-8') as f:
        json.dump(TSPSenario, f, ensure_ascii=False, separators=(',', ':'))

"""
function create_scenario(city, num_blocks, t)
    py"get_scenario"(city, num_blocks, t)
end

function fitGeneralModel(nb_data=100)
    return 0.0
end
function find_solution_city(
    city::String,
    target_district_size::Int,
    NB_BU::Int,
    depot_location::String,
    params
)
    instance = build_instance(city, NB_BU, target_district_size, depot_location)
    costloader = Costloader([], []) 
    pathScenario = "deps/Scenario/output/$(city)_$(depot_location)_$(NB_BU)_$(target_district_size).json"
    create_scenario(city, NB_BU, target_district_size)
    if !isfile(pathScenario)
        create_scenario(city, NB_BU, target_district_size)
    end
    params = nothing
    solution = ILS_solve_instance(instance, costloader, "AvgTSP", params)
    return solution
end

end
