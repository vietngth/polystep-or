"""DistrictNet-style districting comparison: PredGNN vs DistrictNet vs PolyStep, one column per method,
one row per city. Replicates Ferreira et al. (DistrictNet, their Figure 2): district polygons in their
40-colour palette over a REAL street basemap (CartoDB/OSM tiles via contextily), with the DEPOT drawn as a
white star at its true reference point (metadata.REFERENCE_LONGLAT). Colours are kept CONSISTENT across the
method panels (ported from their assign_colors), anchored on DistrictNet, so real reassignments are visible.

Usage: .venv/bin/python make_compare_figure.py <City1,City2,...> <bu> <t> [out_suffix]
"""
import sys, os, json
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
CMPDIR = f"{DN}/output/solution/Comparaison/Experiment_General_multisize_cities"
METHODS = [("predictGnn", "PredGNN"), ("districtNet", "DistrictNet"), ("polyStep", "PolyStep (ours)")]
ANCHOR = "districtNet"
BASEMAP = ctx.providers.CartoDB.Positron       # light, clean under coloured districts
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
    # Depot = the actual evaluation depot the solver/SAA uses: location code "C" = center of the
    # BU bounding box (see DistrictNet/src/instance.jl::get_depot_location). We do NOT use the geojson
    # metadata REFERENCE_LONGLAT, which is a generation-time region centroid offset from the retained
    # BU subset for some cities (Paris/Marseille), placing the star outside the cluster in the figure.
    allpts = np.concatenate(list(rings.values()), axis=0)
    (lon_min, lat_min), (lon_max, lat_max) = allpts.min(axis=0), allpts.max(axis=0)
    depot = [(lon_min + lon_max) / 2.0, (lat_min + lat_max) / 2.0]
    return rings, depot


def read_sol(path):
    if not os.path.isfile(path):
        return None, None, None
    L = open(path).read().splitlines()
    feasible = (len(L) > 4 and L[4].strip().lower() == "true")
    hdr_cost = None
    if len(L) > 3 and L[3].upper().startswith("COST"):
        try:
            hdr_cost = float(L[3].split()[1])
        except (IndexError, ValueError):
            pass
    districts = [[int(x) - 1 for x in l.split()] for l in L[5:] if l.split()]
    return districts, feasible, hdr_cost


def sol_path(city, bu, t, m):
    # Prefer the feasibility-repaired (postprocessed) solution if present. This is the same kind of
    # post-hoc repair DistrictNet's ILS pipeline already applies, so a repaired-vs-repaired comparison
    # is the fair one; we use the repaired solution's in-header SAA cost.
    rp = f"{SOLDIR}/{city}_C_{bu}_{t}.{m}Repaired.txt"
    if os.path.isfile(rp):
        return rp, True
    return f"{SOLDIR}/{city}_C_{bu}_{t}.{m}.txt", False


def read_costs(city, bu, t):
    c = {}
    p = f"{CMPDIR}/{city}_C_{bu}_{t}.txt"
    if os.path.isfile(p):
        for ln in open(p).read().splitlines():
            s = ln.split()
            if len(s) >= 2:
                try:
                    c[s[0]] = float(s[1])
                except ValueError:
                    pass
    return c


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


def panel(ax, rings_xy, depot_xy, colors_idx, title):
    patches, cols = [], []
    for pid, ring in rings_xy.items():
        patches.append(MplPoly(ring, closed=True))
        cols.append(PALETTE[(colors_idx.get(pid, 1) - 1) % len(PALETTE)])
    ax.add_collection(PatchCollection(patches, facecolor=cols, alpha=0.55,
                                      edgecolor="black", linewidths=0.4))
    allpts = np.vstack(list(rings_xy.values()))
    xs = np.append(allpts[:, 0], depot_xy[0]); ys = np.append(allpts[:, 1], depot_xy[1])
    mx = 0.08 * (xs.max() - xs.min()) + 1.0
    my = 0.08 * (ys.max() - ys.min()) + 1.0
    ax.set_xlim(xs.min() - mx, xs.max() + mx)
    ax.set_ylim(ys.min() - my, ys.max() + my)
    ax.set_aspect("equal")
    # depot star at its TRUE position, over the real map
    ax.scatter([depot_xy[0]], [depot_xy[1]], marker="*", s=320, c="white",
               edgecolors="black", linewidths=1.1, zorder=5)
    try:
        ctx.add_basemap(ax, source=BASEMAP, attribution=False)
    except Exception as e:
        print("  (basemap unavailable:", repr(e)[:80], ")")
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def main():
    cities = (sys.argv[1].split(",") if len(sys.argv) > 1 else ["Manchester", "London"])
    bu = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    t = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    suffix = sys.argv[4] if len(sys.argv) > 4 else ""
    nrow, ncol = len(cities), len(METHODS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow))
    axes = np.atleast_2d(axes)
    for r, city in enumerate(cities):
        rings, depot = load_city(city, bu)
        rings_xy = {pid: to3857(ring) for pid, ring in rings.items()}
        depot_xy = to3857([depot])[0]
        costs = read_costs(city, bu, t)
        sols, is_rep = {}, {}
        for m, _ in METHODS:
            p, rep = sol_path(city, bu, t, m)
            sols[m] = read_sol(p); is_rep[m] = rep
        anchor_colors = assign_colors(sols[ANCHOR][0])
        dn_cost = (sols[ANCHOR][2] if is_rep[ANCHOR] and sols[ANCHOR][2] is not None else costs.get(ANCHOR))
        for cidx, (m, lab) in enumerate(METHODS):
            districts, feas, hdr_cost = sols[m]
            if districts is None:
                axes[r, cidx].axis("off"); axes[r, cidx].set_title(f"{lab}\n(missing)", fontsize=10); continue
            cmap = anchor_colors if m == ANCHOR else assign_colors_prev(districts, anchor_colors)
            cst = hdr_cost if (is_rep[m] and hdr_cost is not None) else costs.get(m)
            sub = f"cost {cst:.1f}" if cst is not None else ""
            if cst is not None and dn_cost and m != ANCHOR:
                sub += f"  ({100*(cst-dn_cost)/dn_cost:+.1f}%)"
            if not feas:
                sub += "  [infeasible]"
            ttl = f"{city} -- {lab}\n{sub}" if cidx == 1 else f"{lab}\n{sub}"
            panel(axes[r, cidx], rings_xy, depot_xy, cmap, ttl)
        print("  row done:", city)
    fig.tight_layout()
    out = f"paper-template/figures/fig_dn_compare{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    print("wrote", out, "| cities", cities)


if __name__ == "__main__":
    main()
