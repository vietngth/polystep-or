from os import mkdir
from os.path import exists
from PIL import Image
from random import uniform
from time import sleep
from urllib.request import Request, urlopen
import math
import os 
import json
import sys

maps_dir = './data/maps/'
if not os.path.exists(maps_dir):
    os.makedirs(os.path.dirname('./data/maps/'), exist_ok=True)


def deg2float(lat_deg, lon_deg, zoom):
    """
    Taken from:
    https://allanrbo.blogspot.com/2021/08/download-openstreetmap-bounding-box-png.html
    Similar to:
    https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Python
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return (xtile, ytile)


def get_tile_url(zoom, x, y):
    """Return string of url of tile.

    Args:
        zoom (int): zoom level
        x (int): x index of tile
        y (int): y index of tile

    Returns:
        string: url of selected tile
    """
    # Change the provider for different tile style
    providerString = ("https://tile.openstreetmap.fr/hot/{}/{}/{}.png")
    return providerString.format(zoom, x, y)


def download_map(zoom, lat1, lon1, lat2, lon2, map_name):
    """"
    Taken from:
    https://allanrbo.blogspot.com/2021/08/download-openstreetmap-bounding-box-png.html
    """
    lon_start, lon_end = min(lon1, lon2), max(lon1, lon2)
    lat_start, lat_end = max(lat1, lat2), min(lat1, lat2)

    # Top left corner of bounding box.
    x1, y1 = deg2float(lat_start, lon_start, zoom)
    x1i, y1i = math.floor(x1), math.floor(y1)

    # Bottom right corner of bounding box.
    x2, y2 = deg2float(lat_end, lon_end, zoom)
    x2i, y2i = math.ceil(x2), math.ceil(y2)

    x_cnt, y_cnt = abs(x1i - x2i), abs(y1i - y2i)
    if x_cnt*y_cnt > 250:
        err = "Too many tiles. Area probably too big at too high a zoom level."
        err += " See https://operations.osmfoundation.org/policies/tiles/ ."
        raise Exception(err)

    if not exists("data/maptiles"):
        mkdir("data/maptiles")

    for x in range(x_cnt):
        for y in range(y_cnt):
            xt, yt = x + x1i, y + y1i
            path = "data/maptiles/{}_{}_{}.png".format(zoom, xt, yt)

            if not exists(path):
                sleep(uniform(0.5, 1.5))
                url = get_tile_url(zoom, xt, yt)
                print("Downloading tile {}".format(url))
                req = Request(url)
                ua = ("Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0)"
                      + "Gecko/20100101 Firefox/90.0")
                # OSM seems to not like Python's default UA.
                req.add_header("User-Agent", ua)
                resp = urlopen(req)
                body = resp.read()
                with open(path, "wb") as f:
                    f.write(body)

    im = Image.open("data/maptiles/{}_{}_{}.png".format(zoom, x1i, y1i))
    tile_w, tile_h = im.size
    total_w = x_cnt*tile_w
    total_h = y_cnt*tile_h

    new_im = Image.new("RGB", (total_w, total_h))

    for x in range(x_cnt):
        for y in range(y_cnt):
            xt, yt = x + x1i, y + y1i
            im = Image.open("data/maptiles/{}_{}_{}.png".format(zoom, xt, yt))
            new_im.paste(im, (x*tile_w, y*tile_h))

    cropped_w = round((x2 - x1)*tile_w)
    cropped_h = round((y2 - y1)*tile_h)
    cropped_im = Image.new("RGB", (cropped_w, cropped_h))
    translate_x = round(-(x1 - x1i)*tile_w)
    translate_y = round(-(y1 - y1i)*tile_h)
    cropped_im.paste(new_im, (translate_x, translate_y))
    cropped_im.save('./data/maps/' + map_name + '_map.png')


def get_city_box(city_name):
    """Get bounding box of city from GeoJSON file.

    Args:
        city_name (string): name of city

    Returns:
        tuple: bounding box of city
    """
    x_epsilons = 0.01
    y_epsilons = 0.01
    # Load GeoJSON data
    path = "data/geojson/" + city_name + "_120_BUs.geojson"
    with open(path, "r") as file:
        geojson_data = json.load(file)

    # Initialize min and max coordinates
    min_x, min_y = sys.float_info.max, sys.float_info.max
    max_x, max_y = -sys.float_info.max, -sys.float_info.max

    # Iterate over each feature
    for feature in geojson_data["features"]:
        # Check if the geometry type is Polygon or MultiPolygon
        if feature["geometry"]["type"] == "Polygon":
            for polygon in feature["geometry"]["coordinates"]:
                for coord in polygon:
                    min_x, min_y = min(min_x, coord[0]), min(min_y, coord[1])
                    max_x, max_y = max(max_x, coord[0]), max(max_y, coord[1])
        elif feature["geometry"]["type"] == "MultiPolygon":
            for multipolygon in feature["geometry"]["coordinates"]:
                for polygon in multipolygon:
                    for coord in polygon[0]:
                        min_x, min_y = min(min_x, coord[0]), min(min_y, coord[1])
                        max_x, max_y = max(max_x, coord[0]), max(max_y, coord[1])

    min_x -= x_epsilons
    max_x += x_epsilons
    min_y -= y_epsilons
    max_y += y_epsilons

    # round to 7 decimal places
    min_x = round(min_x, 7)
    max_x = round(max_x, 7)
    min_y = round(min_y, 7)
    max_y = round(max_y, 7)
    
    return (min_x, max_x, min_y,  max_y)

# - Main script -
"""
Download OSM maps to produce figures in paper.
Needs to download OSM tiles and store them locally.
"""
ZOOM_LEVEL = 12


# - London -
london_box = (-0.264807777, 0.0246201708, 51.450549979, 51.567122329)
download_map(ZOOM_LEVEL, london_box[2], london_box[0],
                london_box[3], london_box[1], 'London')
# - Manchester -
manchester_box = (-2.489495333, -1.991144766, 53.3936437, 53.57961082480)
download_map(ZOOM_LEVEL, manchester_box[2], manchester_box[0],
                manchester_box[3], manchester_box[1], 'Manchester')
