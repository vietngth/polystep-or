"""Generate one result figure per experiment from exp_results/*.json into
paper-template/figures/*.pdf, rendered with Plotly (static PDF via kaleido) for a
clean, consistent visual family.

Run: .venv/bin/python make_figures.py
"""
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_style import PALETTE, SYMBOL, finalize, line_trace, bar_trace, save

OUT = "paper-template/figures"
os.makedirs(OUT, exist_ok=True)

METHODS5 = ["two-stage", "SPO+", "IMLE", "SFGE", "PolyStep"]
METHODS4 = ["two-stage", "SPO+", "SFGE", "PolyStep"]
LAB = {"sp": "shortest path (LP)", "knap": "knapsack (ILP)", "tsp": "TSP (ILP)", "port": "portfolio (SOCP)"}


def load(n):
    return json.load(open(f"exp_results/{n}.json"))


def fig_phase():
    pd_ = load("phase_diagram"); degs = pd_["degs"]; panel = pd_["panel"]; probs = pd_["problems"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[LAB.get(p, p) for p in probs],
                        horizontal_spacing=0.11, vertical_spacing=0.13)
    for idx, p in enumerate(probs):
        r, c = idx // 2 + 1, idx % 2 + 1
        for m in panel:
            ys = [pd_["results"][f"{p}|{d}"]["summary"][m]["mean"] for d in degs]
            es = [pd_["results"][f"{p}|{d}"]["summary"][m]["std"] for d in degs]
            fig.add_trace(line_trace(degs, ys, m, err=es, show_legend=(idx == 0)), row=r, col=c)
        fig.update_xaxes(tickvals=degs, title_text=("misspecification degree <i>d</i>" if r == 2 else ""), row=r, col=c)
        fig.update_yaxes(title_text=("normalized regret" if c == 1 else ""), row=r, col=c)
    save(finalize(fig, 920, 660), f"{OUT}/fig_phase.pdf")


def _pareto(name, lab, fname, w, h):
    d = load(name); data = d["results"]; probs = d["problems"]
    ncol = 2 if len(probs) > 1 else 1
    nrow = (len(probs) + ncol - 1) // ncol
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=[lab.get(p, p) for p in probs],
                        horizontal_spacing=0.10, vertical_spacing=0.16)
    for idx, p in enumerate(probs):
        r, c = idx // ncol + 1, idx % ncol + 1
        for m in ("SPO+", "SFGE", "PolyStep"):
            xs = [rr["wall_clock_s"]["mean"] for rr in data[p][m]]
            ys = [rr["regret"]["mean"] for rr in data[p][m]]
            fig.add_trace(line_trace(xs, ys, m, show_legend=(idx == 0)), row=r, col=c)
        fig.update_xaxes(title_text=("wall-clock training time (s)" if r == nrow else ""), row=r, col=c)
        fig.update_yaxes(title_text=("normalized regret" if c == 1 else ""), row=r, col=c)
    save(finalize(fig, w, h), f"{OUT}/{fname}")


def fig_pareto():
    _pareto("pareto", LAB, "fig_pareto.pdf", 920, 620)


def fig_pareto_common():
    lab = {"assignment": "assignment (LP)", "transportation": "transportation (LP)",
           "mdkp": "multi-dim. knapsack (ILP)"}
    _pareto("pareto_common", lab, "fig_pareto_common.pdf", 1180, 360)


def fig_biasvar():
    bv = load("bias_variance"); ss = bv["sample_size"]; ns = sorted(int(k) for k in ss); probs = bv["problems"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["(a) sample efficiency (knapsack)", "(b) variance (20 seeds)"],
                        horizontal_spacing=0.13)
    for m in METHODS4:
        ys = [ss[str(n)][m]["mean"] for n in ns]
        fig.add_trace(line_trace(ns, ys, m, show_legend=True), row=1, col=1)
    for m in METHODS4:
        xs = [LAB.get(p, p) for p in probs]
        ys = [bv["bias_variance"][p]["summary"][m]["std"] for p in probs]
        fig.add_trace(bar_trace(xs, ys, m, PALETTE[m], show_legend=False), row=1, col=2)
    fig.update_xaxes(type="log", tickvals=[50, 100, 200, 500, 1000, 3000],
                     title_text="training size <i>n</i> (log)", row=1, col=1)
    fig.update_yaxes(title_text="normalized regret", row=1, col=1)
    fig.update_xaxes(tickangle=-18, row=1, col=2)
    fig.update_yaxes(title_text="across-seed std of regret", row=1, col=2)
    fig.update_layout(barmode="group")
    save(finalize(fig, 1020, 400), f"{OUT}/fig_biasvar.pdf")


def fig_constraints():
    cs = load("constraints"); costs = cs["costs"]
    panels = [("fk", "fractional knapsack"), ("nv", "capacitated newsvendor")]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[t for _, t in panels], horizontal_spacing=0.12)
    for ci, (key, _) in enumerate(panels):
        res = cs["results"][key]
        for m in ("two-stage", "SFGE", "PolyStep"):
            ys = [res[str(c)]["summary"][m]["mean"] for c in costs]
            fig.add_trace(line_trace(costs, ys, m, show_legend=(ci == 0)), row=1, col=ci + 1)
        fig.update_xaxes(tickvals=costs, title_text="misprediction penalty", row=1, col=ci + 1)
        fig.update_yaxes(type="log", title_text=("normalized realized regret" if ci == 0 else ""), row=1, col=ci + 1)
    save(finalize(fig, 980, 390), f"{OUT}/fig_constraints.pdf")


def fig_ablation():
    ab = load("ot_ablation"); probs = ab["problems"]; methods = ab["results"][probs[0]]["methods"]
    bc = {"sp": "#1f77b4", "knap": "#d62728"}
    fig = go.Figure()
    for p in probs:
        reg = ab["results"][p]["regret"]
        ys = [reg[m]["mean"] for m in methods]; es = [reg[m]["std"] for m in methods]
        fig.add_trace(bar_trace(methods, ys, LAB.get(p, p), bc.get(p, "#888"), err=es, show_legend=True))
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-25)
    fig.update_yaxes(title_text="normalized regret (cold start)")
    save(finalize(fig, 980, 430), f"{OUT}/fig_ablation.pdf")


def fig_capacity():
    cap = load("capacity"); degs = cap["degs"]; caps = cap["caps"]
    name = {"linear": "linear", "mlp8": "MLP-8", "mlp32": "MLP-32"}
    col = {"linear": "#1f77b4", "mlp8": "#2ca02c", "mlp32": "#d62728"}
    sym = {"linear": "circle", "mlp8": "triangle-up", "mlp32": "star"}
    fig = go.Figure()
    for c in caps:
        ys = [cap["results"][f"{c}|{d}"]["adv_vs_two_stage"] * 100 for d in degs]
        fig.add_trace(go.Scatter(x=degs, y=ys, mode="lines+markers", name=name.get(c, c),
                                 line=dict(color=col.get(c, "#444"), width=2.4),
                                 marker=dict(color=col.get(c, "#444"), symbol=sym.get(c, "circle"), size=9)))
    fig.update_xaxes(tickvals=degs, title_text="misspecification degree <i>d</i>")
    fig.update_yaxes(title_text="advantage over two-stage (%)")
    fig.update_layout(title=dict(text="model-capacity arm (shortest path)", x=0.5, xanchor="center", font=dict(size=15)))
    save(finalize(fig, 640, 420), f"{OUT}/fig_capacity.pdf")


def fig_tsp_diag():
    td = load("tsp_diag"); degs = sorted(int(d) for d in td["results"])
    series = [("PolyStep(warm,sr=0.4)", "PolyStep r<sub>s</sub>=0.4 (default)", "#d62728"),
              ("PolyStep(warm,sr=1.6)", "PolyStep r<sub>s</sub>=1.6", "#ff7f0e"),
              ("PolyStep(cold,sr=0.8)", "PolyStep cold r<sub>s</sub>=0.8", "#8c564b"),
              ("SFGE", "SFGE", "#2ca02c"), ("IMLE", "IMLE", "#9467bd")]
    xs = [f"d={d}" for d in degs]
    fig = go.Figure()
    for key, lab, color in series:
        ys = [td["results"][str(d)][key]["mean"] for d in degs]
        fig.add_trace(bar_trace(xs, ys, lab, color, show_legend=True))
    fig.update_layout(barmode="group", title=dict(text="TSP: step radius and initialization", x=0.5, xanchor="center", font=dict(size=15)))
    fig.update_xaxes(title_text="misspecification degree")
    fig.update_yaxes(title_text="normalized regret (TSP)")
    save(finalize(fig, 880, 430), f"{OUT}/fig_tsp_diag.pdf")


def fig_interp():
    interp = {"knapsack": (9, 9), "shortest path": (6, 7), "TSP": (5, 14), "portfolio": (2, 2)}
    probs = list(interp)
    fig = go.Figure()
    fig.add_trace(bar_trace(probs, [interp[p][0] for p in probs], "d = 4", "#2ca02c"))
    fig.add_trace(bar_trace(probs, [interp[p][1] for p in probs], "d = 6", "#d62728"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="regret reduction from<br>decision-focused refinement (%)")
    save(finalize(fig, 720, 410), f"{OUT}/fig_interp.pdf")


if __name__ == "__main__":
    fig_phase(); fig_pareto(); fig_pareto_common(); fig_biasvar(); fig_constraints()
    fig_ablation(); fig_capacity(); fig_tsp_diag(); fig_interp()
    print("done:", sorted(f for f in os.listdir(OUT) if f.endswith(".pdf")))
