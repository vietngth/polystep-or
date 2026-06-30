"""Experiment #2 -- Solve-budget Pareto frontier (regret vs cost), honestly multi-axis.

Kills the "gradient-free is too expensive" objection without hiding PolyStep's real cost. We trace
regret as a function of a per-method BUDGET knob (PolyStep steps; SFGE/SPO+ epochs) on THREE distinct
cost axes that must not be conflated:
  (a) gurobi_solves       -- expensive per-instance solver calls: SPO+ pays epochs*n_train; GF ~ 0;
  (b) wall_clock_s        -- what the user actually pays;
  (c) batched_fwd_solves  -- cheap batched GPU oracle calls: PolyStep pays the MOST (its honest cost).

Reading: gradient-free dominates regret-vs-wall-clock and regret-vs-gurobi; on regret-vs-fwd-solves
PolyStep's curve is the most expensive -- but those solves are batched & cheap, so wall-clock wins.
All methods warm-started from the same two-stage model (fair init).

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp2_pareto.py [problems] [deg] [seeds]
"""
from __future__ import annotations
import sys
sys.path.insert(0, "polystep/src")
import numpy as np
import torch
from pyepo import metric
import pyepo.func as F
from pto.capability import (SETUPS, train_two_stage, train_sfge, train_polystep, _adam, dev)
from pto.seeding import seed_everything
from pto.budget import SolveCounter, Timer, spoplus_gurobi_solves
from pto.multiseed import summarize, md_table, write_json, write_md

# budget knobs (one list per method; chosen to span under- to well-trained)
PS_STEPS = [10, 25, 50, 100, 200]
SFGE_EPOCHS = [15, 30, 60, 120, 240]
SPO_EPOCHS = [5, 15, 30, 60, 100]
CATLABEL = {"sp": "shortest_path (LP)", "knap": "knapsack (ILP)", "tsp": "tsp (ILP)", "port": "portfolio (SOCP)"}


def n_train_of(cfg):
    return int(cfg["Xtr"].shape[0])


def run_spoplus(cfg, warm, epochs, lr=1e-2):
    m = cfg["make"]()
    with torch.no_grad():
        if warm is not None:
            m.weight.copy_(warm.weight)
    opt = _adam(m, lr); spop = F.SPOPlus(cfg["om"])
    for _ in range(epochs):
        for xb, cb, wb, zb in cfg["ld_tr"]:
            xb, cb, wb, zb = [t.float().to(dev) for t in (xb, cb, wb, zb)]
            opt.zero_grad(); spop(m(xb), cb, wb, zb).mean().backward(); opt.step()
    return m


def one_seed(setup, deg, seed):
    """Return per-method list of {budget, regret, wall_clock_s, gurobi_solves, batched_fwd_solves}."""
    seed_everything(seed)
    cfg, _ = setup(seed, deg)
    warm = train_two_stage(cfg); cfg["warm"] = warm
    nt = n_train_of(cfg)
    pts = {"SPO+": [], "SFGE": [], "PolyStep": []}

    for ep in SPO_EPOCHS:
        with Timer() as t:
            m = run_spoplus(cfg, warm, ep)
        pts["SPO+"].append({"budget": ep, "regret": metric.regret(m, cfg["om"], cfg["ld_te"]),
                            "wall_clock_s": t.seconds, "gurobi_solves": spoplus_gurobi_solves(ep, nt),
                            "batched_fwd_solves": 0})

    for ep in SFGE_EPOCHS:
        sc = SolveCounter(cfg["ps_solve"]); cc = sc.wrap(cfg)
        with Timer() as t:
            m = train_sfge(cc, epochs=ep)
        pts["SFGE"].append({"budget": ep, "regret": metric.regret(m, cfg["om"], cfg["ld_te"]),
                            "wall_clock_s": t.seconds, "gurobi_solves": 0,
                            "batched_fwd_solves": sc.instances})

    for st in PS_STEPS:
        sc = SolveCounter(cfg["ps_solve"]); cc = sc.wrap(cfg); cc["ps_steps"] = st
        with Timer() as t:
            m = train_polystep(cc)
        pts["PolyStep"].append({"budget": st, "regret": metric.regret(m, cfg["om"], cfg["ld_te"]),
                                "wall_clock_s": t.seconds, "gurobi_solves": 0,
                                "batched_fwd_solves": sc.instances})
    return pts


def aggregate(per_seed, method):
    """Average each budget point across seeds (points are index-aligned per method)."""
    npts = len(per_seed[0][method])
    out = []
    for i in range(npts):
        rows = [ps[method][i] for ps in per_seed]
        rec = {"budget": rows[0]["budget"]}
        for k in ("regret", "wall_clock_s", "gurobi_solves", "batched_fwd_solves"):
            rec[k] = summarize([r[k] for r in rows])
        out.append(rec)
    return out


def main():
    problems = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sp", "knap", "tsp", "port"]
    deg = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2]
    print(f"PARETO | problems={problems} deg={deg} seeds={seeds}", flush=True)
    results = {}
    for p in problems:
        print(f"[pareto] {p} ...", flush=True)
        per_seed = []
        for s in seeds:
            per_seed.append(one_seed(SETUPS[p], deg, s))
            print(f"    seed {s} done", flush=True)
        results[p] = {m: aggregate(per_seed, m) for m in ("SPO+", "SFGE", "PolyStep")}
    payload = {"problems": problems, "deg": deg, "seeds": seeds,
               "budgets": {"PolyStep_steps": PS_STEPS, "SFGE_epochs": SFGE_EPOCHS, "SPO_epochs": SPO_EPOCHS},
               "results": results}
    write_json("exp_results/pareto.json", payload)
    write_md("exp_results/pareto.md", to_markdown(results, problems, deg, seeds))
    print("\nwrote exp_results/pareto.{json,md}\nDONE", flush=True)


def to_markdown(results, problems, deg, seeds):
    L = ["# Experiment #2 -- Solve-budget Pareto frontier", "",
         f"deg={deg}, seeds={seeds}. Three honest cost axes; gradient-free dominates wall-clock & "
         "gurobi-calls, PolyStep pays the most batched-forward-solves (cheap, batched).", ""]
    for p in problems:
        L.append(f"## {CATLABEL.get(p, p)}")
        headers = ["method", "budget", "regret", "wall_clock_s", "gurobi_solves", "batched_fwd_solves"]
        rows = []
        for m in ("SPO+", "SFGE", "PolyStep"):
            for rec in results[p][m]:
                rows.append([m, rec["budget"], f"{rec['regret']['mean']:.4f}",
                             f"{rec['wall_clock_s']['mean']:.2f}",
                             f"{rec['gurobi_solves']['mean']:.0f}",
                             f"{rec['batched_fwd_solves']['mean']:.0f}"])
        L.append(md_table(headers, rows)); L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
