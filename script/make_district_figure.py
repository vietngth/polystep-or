"""DistrictNet Figure 2 reproduction: districting solutions for BD, FIG, PredGNN, DistrictNet, and PolyStep
on one city, in the style of Ferreira et al.: district polygons in their 40-colour palette over a REAL
street basemap (CartoDB/OSM via contextily), the depot as a white star at its true reference point, and
colours held consistent across panels (anchored on DistrictNet).
Usage: .venv/bin/python make_district_figure.py <City> <bu> <t>
"""
import sys, json, os
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection
from pyproj import Transformer
import contextily as ctx

DN = "DistrictNet"
SOLDIR = f"{DN}/output/solution/Experiment_General_multisize_cities"
METHODS = [("BD", "BD"), ("FIG", "FIG"), ("predictGnn", "PredGNN"),
           ("districtNet", "DistrictNet"), ("polyStep", "PolyStep (ours)")]
ANCHOR = "districtNet"
BASEMAP = ctx.providers.CartoDB.Positron
PALETTE = ["red", "blue", "green", "yellow", "purple", "cyan", "magenta", "orange", "brown", "lime",
           "pink", "violet", "indigo", "coral", "teal", "olive", "navy", "maroon", "aquamarine",
           "turquoise", "silver", "goldenrod", "salmon", "tan", "royalblue", "plum", "peachpuff",
           "orchid", "mediumseagreen", "mediumorchid", "mediumpurple", "mediumblue", "lightcoral",
           "lawngreen", "lavender", "khaki", "hotpink", "dodgerblue", "deepskyblue", "darkviolet"]

_T = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def to3857(arr):
    xs, ys = _T.transform(np.asarray(arr)[:, 0], np.asarray(arr)[:, 1])
    return np.column_stack([xs, ys])


def load_city(city, bu):
    d = json.load(open(f"{DN}/data/geojson/{city}.geojson"))
    rings = {}
    for f in d["features"]:
        pid = f["properties"]["ID"]
        if pid >= bu:
            continue
        g = f["geometry"]
        ring = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0]
        rings[pid] = np.array(ring)
    return rings, d["metadata"]["REFERENCE_LONGLAT"]


def read_districts(path):
    if not os.path.isfile(path):
        return None
    L = open(path).read().splitlines()
    return [[int(x) - 1 for x in l.split()] for l in L[5:] if l.split()]


def assign_colors(districts):
    all_bus = sorted({b for d in districts for b in d})
    colored = [False] * len(districts)
    id_colors, ci = {}, 1
    for bu in all_bus:
        di = next((j for j, d in enumerate(districts) if bu in d), None)
        if di is not None and not colored[di]:
            for b in districts[di]:
                id_colors[b] = ci
            colored[di] = True
            ci += 1
    return id_colors


def assign_colors_prev(districts, prev):
    nb = max(prev.values())
    prev_d = [[k for k, v in prev.items() if v == c] for c in range(1, nb + 1)]
    sim = [max((len(set(districts[d]) & set(pd)) for pd in prev_d), default=0) for d in range(len(districts))]
    order = sorted(range(len(districts)), key=lambda d: -sim[d])
    is_used = [False] * (nb + len(districts) + 2)
    id_colors = {}
    for idx in order:
        district = districts[idx]
        freq = [c for c, _ in Counter(prev.get(b) for b in district if b in prev).most_common()]
        dc = next((c for c in freq if c is not None and not is_used[c]), 0)
        if dc == 0:
            dc = next(i for i in range(1, len(is_used)) if not is_used[i])
        for b in district:
            id_colors[b] = dc
        is_used[dc] = True
    return id_colors


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "Manchester"
    bu = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    t = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    rings, depot = load_city(city, bu)
    rings_xy = {pid: to3857(ring) for pid, ring in rings.items()}
    depot_xy = to3857([depot])[0]
    present = [(m, lab) for m, lab in METHODS if read_districts(f"{SOLDIR}/{city}_C_{bu}_{t}.{m}.txt")]
    if not present:
        print("no solution files found for", city, bu, t); return
    anchor_colors = assign_colors(read_districts(f"{SOLDIR}/{city}_C_{bu}_{t}.{ANCHOR}.txt"))
    fig, axes = plt.subplots(1, len(present), figsize=(2.9 * len(present), 3.2))
    axes = np.atleast_1d(axes)
    allpts = np.vstack(list(rings_xy.values()))
    xs = np.append(allpts[:, 0], depot_xy[0]); ys = np.append(allpts[:, 1], depot_xy[1])
    mx = 0.08 * (xs.max() - xs.min()) + 1.0; my = 0.08 * (ys.max() - ys.min()) + 1.0
    xlim = (xs.min() - mx, xs.max() + mx); ylim = (ys.min() - my, ys.max() + my)
    for ax, (m, lab) in zip(axes, present):
        districts = read_districts(f"{SOLDIR}/{city}_C_{bu}_{t}.{m}.txt")
        cmap = anchor_colors if m == ANCHOR else assign_colors_prev(districts, anchor_colors)
        patches = [MplPoly(ring, closed=True) for ring in rings_xy.values()]
        cols = [PALETTE[(cmap.get(pid, 1) - 1) % len(PALETTE)] for pid in rings_xy]
        ax.add_collection(PatchCollection(patches, facecolor=cols, alpha=0.55,
                                          edgecolor="black", linewidths=0.4))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.scatter([depot_xy[0]], [depot_xy[1]], marker="*", s=300, c="white",
                   edgecolors="black", linewidths=1.0, zorder=5)
        try:
            ctx.add_basemap(ax, source=BASEMAP, attribution=False)
        except Exception as e:
            print("  (basemap unavailable:", repr(e)[:80], ")")
        ax.set_axis_off()
        ax.set_title(lab, fontsize=10)
    fig.suptitle(f"{city}, target size $t={t}$ ({len(districts)} districts); depot shown as a star",
                 fontsize=10)
    fig.tight_layout()
    out = "paper-template/figures/fig_district.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    print("wrote", out, "with methods:", [lab for _, lab in present])


if __name__ == "__main__":
    main()
