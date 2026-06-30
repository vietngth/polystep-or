"""Comprehensive PyEPO benchmark (cluster reproduction).

Reproduces the PyEPO-paper method comparison and adds PolyStep across THREE polytopes:
  * full DFL suite : SPO+, DBB, DPO, IMLE, PFYL, NCE, LTR  (the methods An flagged) + two-stage
  * PolyStep       : polytope in {orthoplex, simplex, cube}, each a SEPARATE method column
  * protocol       : 5 SEEDS, normalized regret mean±std, Wilcoxon(PolyStep < SPO+)

PolyStep hyperparameters (num_probe = probe points, probe_radius, step_radius, epsilon) are taken
per (problem, polytope) from exp_results/polystep_tune.json when present (the tuning phase), else
sensible defaults. RUN ON A SLURM COMPUTE NODE (never the login node).

Run:  .venv/bin/python exp_polystep_bench.py sp,knap,tsp,port 0,1,2,3,4
      POLY=orthoplex,simplex,cube DFL=SPO+,DBB,DPO,IMLE,PFYL,NCE,LTR .venv/bin/python exp_polystep_bench.py sp,knap 0,1,2,3,4
"""
from __future__ import annotations
import sys, json, os, time
import numpy as np
sys.path.insert(0, "polystep/src")
from pyepo import metric
from pto.capability import SETUPS, DFL, train_two_stage, train_dfl, train_polystep
from pto.multiseed import summarize, md_table, wilcoxon_pair, write_json, write_md

POLYTOPES = os.environ.get("POLY", "orthoplex,simplex,cube").split(",")
DFL_METHODS = os.environ.get("DFL", "SPO+,DBB,DPO,IMLE,PFYL,NCE,LTR").split(",")
CAT = {"sp": "shortest_path (LP)", "knap": "knapsack (ILP)",
       "tsp": "tsp (ILP)", "port": "portfolio (SOCP)"}


def load_tuned():
    p = "exp_results/polystep_tune.json"
    if os.path.exists(p):
        return json.load(open(p)).get("best", {})
    return {}


def run(problems, seeds, deg=4):
    tuned = load_tuned()
    if tuned:
        print(f"[using tuned PolyStep configs from exp_results/polystep_tune.json: {len(tuned)} entries]")
    dfl = [m for m in DFL_METHODS if m in DFL]
    ps_cols = [f"PolyStep-{p}" for p in POLYTOPES]
    methods = ["two-stage"] + dfl + ps_cols
    results = {}
    for pname in problems:
        print(f"\n[{pname}] {CAT.get(pname, pname)}  seeds={seeds}", flush=True)
        acc = {m: [] for m in methods}
        for seed in seeds:
            t0 = time.time()
            cfg, cat = SETUPS[pname](seed, deg)
            ts = train_two_stage(cfg)
            acc["two-stage"].append(float(metric.regret(ts, cfg["om"], cfg["ld_te"])))
            for name in dfl:
                try:
                    acc[name].append(float(metric.regret(train_dfl(cfg, name), cfg["om"], cfg["ld_te"])))
                except Exception as e:
                    acc[name].append(float("nan"))
            cfg["warm"] = ts
            for poly in POLYTOPES:
                c2 = dict(cfg); c2["ps_polytope"] = poly
                t = tuned.get(f"{pname}:{poly}", {})
                for k in ("ps_num_probe", "ps_probe_radius", "ps_step_radius", "ps_eps0", "ps_eps1"):
                    if k in t:
                        c2[k] = t[k]
                try:
                    acc[f"PolyStep-{poly}"].append(float(metric.regret(train_polystep(c2), cfg["om"], cfg["ld_te"])))
                except Exception:
                    acc[f"PolyStep-{poly}"].append(float("nan"))
            print(f"  seed {seed}: " + " ".join(f"{m}={acc[m][-1]:.3f}" for m in methods) +
                  f"  ({time.time()-t0:.0f}s)", flush=True)
        agg = {m: summarize(acc[m]) for m in methods}
        results[pname] = {"cat": cat, "agg": agg, "raw": acc,
                          "wilcoxon": {p: wilcoxon_pair(acc[f"PolyStep-{POLYTOPES[0]}"], acc[p])
                                       for p in dfl if p in acc}}
    payload = {"problems": problems, "seeds": seeds, "deg": deg, "polytopes": POLYTOPES,
               "dfl_methods": dfl, "results": results, "tuned": bool(tuned)}
    write_json("exp_results/polystep_bench.json", payload)
    write_md("exp_results/polystep_bench.md", to_md(payload))
    print("\nwrote exp_results/polystep_bench.{json,md}\nDONE", flush=True)


def to_md(payload):
    methods = ["two-stage"] + payload["dfl_methods"] + [f"PolyStep-{p}" for p in payload["polytopes"]]
    L = ["# Comprehensive PyEPO benchmark -- full DFL suite + PolyStep (3 polytopes) x seeds", "",
         f"seeds={payload['seeds']}, deg={payload['deg']}, polytopes={payload['polytopes']}, "
         f"tuned={'yes' if payload['tuned'] else 'no (defaults)'}.", "",
         "Normalized regret (lower is better), mean±std over seeds.", ""]
    for p in payload["problems"]:
        r = payload["results"][p]; agg = r["agg"]
        L.append(f"## {CAT.get(p, p)}")
        rows = [[m, f"{agg[m]['mean']:.4f}±{agg[m]['std']:.4f}" if agg[m]["n"] else "n/a"] for m in methods]
        L.append(md_table(["method", "regret"], rows))
        wl = r.get("wilcoxon", {})
        sig = ", ".join(f"vs {k}: p={v:.3f}" for k, v in wl.items() if v is not None)
        if sig:
            L.append(f"\n_Wilcoxon (PolyStep-{payload['polytopes'][0]} < method): {sig}_")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    problems = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sp", "knap", "tsp", "port"]
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3, 4]
    deg = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    run(problems, seeds, deg)
