"""Per-district cost distribution (Plotly): distribution of per-district SAA cost at
target size t=20, one box per method, pooled across the given cities. Shows that PolyStep
yields a per-district cost distribution comparable to DistrictNet (low, tight), unlike the
cost-estimator benchmarks. Reads the PerDistrict dump files.
Usage: .venv/bin/python make_fig4.py <City1,City2,...> <bu> [t] [suffix]
"""
import sys
import os
import plotly.graph_objects as go
from plotly_style import finalize, save

DN = "DistrictNet"
PD = f"{DN}/output/solution/PerDistrict/Experiment_General_multisize_cities"
ORDER = [("BD", "BD"), ("FIG", "FIG"), ("predictGnn", "PredGNN"), ("AvgTSP", "AvgTSP"),
         ("districtNet", "DistrictNet"), ("polyStep", "PolyStep<br>(ours)")]
COLORS = {"BD": "#8c564b", "FIG": "#e377c2", "PredGNN": "#17becf", "AvgTSP": "#bcbd22",
          "DistrictNet": "#ff7f0e", "PolyStep<br>(ours)": "#d62728"}


def read_pd(path):
    d = {}
    if not os.path.isfile(path):
        return d
    for ln in open(path).read().splitlines():
        parts = ln.split()
        if len(parts) >= 2:
            d[parts[0]] = [float(x) for x in parts[1:]]
    return d


def main():
    cities = (sys.argv[1].split(",") if len(sys.argv) > 1 else ["Manchester", "London", "Bristol"])
    bu = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    t = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    suffix = sys.argv[4] if len(sys.argv) > 4 else ""
    agg = {key: [] for key, _ in ORDER}
    present_cities = []
    for city in cities:
        d = read_pd(f"{PD}/{city}_C_{bu}_{t}.txt")
        if d:
            present_cities.append(city)
        for key, _ in ORDER:
            agg[key].extend(d.get(key, []))
    if not any(agg[k] for k, _ in ORDER):
        print("no per-district data found"); return
    all_vals = [v for k, _ in ORDER for v in agg[k]]
    hi = sorted(all_vals)[int(0.99 * (len(all_vals) - 1))]
    fig = go.Figure()
    for key, lab in ORDER:
        if agg[key]:
            fig.add_trace(go.Box(y=agg[key], name=lab, marker_color=COLORS[lab],
                                 line=dict(width=1.3), boxpoints="outliers",
                                 marker=dict(size=3, opacity=0.4), showlegend=False))
    loc = f"{', '.join(present_cities)}, {bu} BUs" if len(present_cities) == 1 else ", ".join(present_cities)
    fig.update_yaxes(title_text="per-district cost", range=[0, float(hi)])
    fig.update_layout(title=dict(text=f"district-cost distribution (<i>t</i>={t}, {loc})",
                                 x=0.5, xanchor="center", font=dict(size=15)))
    out = f"paper-template/figures/fig_dn_dist{suffix}.pdf"
    save(finalize(fig, 740, 430, legend=False), out)


if __name__ == "__main__":
    main()
