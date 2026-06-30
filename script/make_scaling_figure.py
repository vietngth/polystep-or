"""Scaling Pareto (Plotly). Top row: training wall-clock (log) vs the size knob;
bottom row: test regret vs the same knob, one column per problem. As n grows, SPO+'s
wall-clock climbs (one exact solve per instance per epoch) while PolyStep and SFGE stay
low (batched forward, no exact solver), at matched regret. Reads exp_results/scaling.json.
Run: .venv/bin/python make_scaling_figure.py
"""
import json
from plotly.subplots import make_subplots
from plotly_style import finalize, line_trace, save

DISP = {"SPO+": "SPO+", "PolyStep": "PolyStep (ours)", "SFGE": "SFGE"}
ORDER = ["SPO+", "SFGE", "PolyStep"]


def main():
    d = json.load(open("exp_results/scaling.json"))
    axes = d["axes"]; ncol = len(axes)
    fig = make_subplots(rows=2, cols=ncol, shared_xaxes=True,
                        column_titles=[d["results"][a]["xlabel"] for a in axes],
                        horizontal_spacing=0.10, vertical_spacing=0.08)
    first = True
    for j, a in enumerate(axes):
        R = d["results"][a]; xs = R["vals"]
        for m in ORDER:
            recs = R["rows"][m]; disp = DISP[m]
            wall = [p["wall"]["mean"] for p in recs]; wsd = [p["wall"]["std"] for p in recs]
            reg = [p["regret"]["mean"] for p in recs]; rsd = [p["regret"]["std"] for p in recs]
            fig.add_trace(line_trace(xs, wall, disp, err=wsd, show_legend=first), row=1, col=j + 1)
            fig.add_trace(line_trace(xs, reg, disp, err=rsd, show_legend=False), row=2, col=j + 1)
        first = False
        fig.update_yaxes(type="log", row=1, col=j + 1)
        fig.update_xaxes(title_text="training-set size <i>n</i>", row=2, col=j + 1)
    fig.update_yaxes(title_text="training wall-clock [s]", row=1, col=1)
    fig.update_yaxes(title_text="test regret", row=2, col=1)
    save(finalize(fig, 940, 600), "paper-template/figures/fig_scaling.pdf")


if __name__ == "__main__":
    main()
