"""Districting size-generalization figure (Plotly): cost relative to DistrictNet at
target size t=20 as the city size N grows, with PolyStep added. Only feasible points are
drawn. Reads the Comparaison files produced by eval_driver for each size.
Usage: .venv/bin/python make_fig3.py <City> <bu1,bu2,...> [t]
"""
import sys
import os
import plotly.graph_objects as go
from plotly_style import finalize, save

DN = "DistrictNet"
CMP = f"{DN}/output/solution/Comparaison/Experiment_General_multisize_cities"
SOL = f"{DN}/output/solution/Experiment_General_multisize_cities"

# (key, label, color, symbol, is_ours)
SERIES = [("BD", "BD", "#8c564b", "circle", False),
          ("FIG", "FIG", "#e377c2", "square", False),
          ("predictGnn", "PredGNN", "#17becf", "triangle-up", False),
          ("polyStep", "PolyStep (ours)", "#d62728", "star", True)]


def feasible(city, N, t, m):
    p = f"{SOL}/{city}_C_{N}_{t}.{m}.txt"
    if not os.path.isfile(p):
        return False
    L = open(p).read().splitlines()
    return len(L) > 4 and L[4].strip().lower() == "true"


def read_cmp(path):
    d = {}
    if not os.path.isfile(path):
        return d
    for ln in open(path).read().splitlines():
        parts = ln.split()
        if len(parts) >= 2:
            try:
                d[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return d


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "Bristol"
    sizes = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["110", "160", "210", "260"])]
    t = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    per = {key: ([], []) for key, *_ in SERIES}
    for N in sizes:
        c = read_cmp(f"{CMP}/{city}_C_{N}_{t}.txt")
        if "districtNet" not in c or not c["districtNet"]:
            continue
        for key, *_ in SERIES:
            if key in c and c[key] and feasible(city, N, t, key):
                per[key][0].append(N)
                per[key][1].append(100.0 * c[key] / c["districtNet"])
    if not any(per[k][0] for k, *_ in SERIES):
        print("no comparison data for", city); return
    fig = go.Figure()
    for key, lab, col, sym, ours in SERIES:
        xs, ys = per[key]
        if not xs:
            continue
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=lab,
                                 line=dict(color=col, width=2.6 if ours else 1.8),
                                 marker=dict(color=col, symbol=sym, size=11 if ours else 8)))
    fig.add_hline(y=100, line=dict(dash="dot", color="black", width=1))
    xmin = min(min(per[k][0]) for k, *_ in SERIES if per[k][0])
    fig.add_annotation(x=xmin, y=100, text="DistrictNet", showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(size=12, color="#555"))
    fig.update_xaxes(title_text="city size <i>N</i>")
    fig.update_yaxes(title_text="cost relative to DistrictNet [%]")
    fig.update_layout(title=dict(text=f"{city}, target size <i>t</i>={t}", x=0.5, xanchor="center", font=dict(size=15)))
    out = "paper-template/figures/fig_dn_size.pdf"
    save(finalize(fig, 600, 430), out)


if __name__ == "__main__":
    main()
