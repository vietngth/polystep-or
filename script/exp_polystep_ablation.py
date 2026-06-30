"""PolyStep ablation: polytope x probe-count x seeds, on the PyEPO problems.

Shows how PolyStep's normalized regret varies with (a) the probe POLYTOPE (orthoplex / simplex / cube)
and (b) the number of probe points per direction (num_probe), each over 5 seeds. Complements the main
benchmark (exp_polystep_bench.py) by isolating the two PolyStep design knobs.

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp_polystep_ablation.py sp,knap,tsp,port 0,1,2,3,4
      POLY=orthoplex,simplex,cube PROBES=1,2,4 .venv/bin/python exp_polystep_ablation.py sp,knap 0,1,2,3,4
"""
from __future__ import annotations
import sys, json, os, time
import numpy as np
sys.path.insert(0, "polystep/src")
from pyepo import metric
from pto.capability import SETUPS, train_two_stage, train_polystep
from pto.multiseed import summarize, md_table, write_json, write_md

POLYTOPES = os.environ.get("POLY", "orthoplex,simplex,cube").split(",")
PROBES = [int(x) for x in os.environ.get("PROBES", "1,2,4").split(",")]
CAT = {"sp": "shortest_path (LP)", "knap": "knapsack (ILP)", "tsp": "tsp (ILP)", "port": "portfolio (SOCP)"}


def run(problems, seeds, deg=4):
    results = {}
    for p in problems:
        print(f"\n[{p}] {CAT.get(p, p)}  seeds={seeds}  polytopes={POLYTOPES}  probes={PROBES}", flush=True)
        cell = {}
        for poly in POLYTOPES:
            for K in PROBES:
                regs = []
                for seed in seeds:
                    t0 = time.time()
                    cfg, cat = SETUPS[p](seed, deg)
                    cfg["warm"] = train_two_stage(cfg)
                    cfg["ps_polytope"] = poly
                    cfg["ps_num_probe"] = K
                    try:
                        regs.append(float(metric.regret(train_polystep(cfg), cfg["om"], cfg["ld_te"])))
                    except Exception:
                        regs.append(float("nan"))
                key = f"{poly}/K{K}"
                cell[key] = summarize(regs)
                print(f"  {key:18s} regret={cell[key]['mean']:.4f}±{cell[key]['std']:.4f}", flush=True)
        results[p] = {"cat": cat, "cells": cell}
    payload = {"problems": problems, "seeds": seeds, "deg": deg, "polytopes": POLYTOPES,
               "probes": PROBES, "results": results}
    write_json("exp_results/polystep_ablation.json", payload)
    write_md("exp_results/polystep_ablation.md", to_md(payload))
    print("\nwrote exp_results/polystep_ablation.{json,md}\nDONE", flush=True)


def to_md(payload):
    L = ["# PolyStep ablation -- polytope x probe-count x seeds", "",
         f"seeds={payload['seeds']}, deg={payload['deg']}. Normalized regret mean±std.", ""]
    cols = [f"{po}/K{k}" for po in payload["polytopes"] for k in payload["probes"]]
    for p in payload["problems"]:
        L.append(f"## {CAT.get(p, p)}")
        cells = payload["results"][p]["cells"]
        rows = [[c, f"{cells[c]['mean']:.4f}±{cells[c]['std']:.4f}" if c in cells and cells[c]['n'] else "n/a"]
                for c in cols]
        L.append(md_table(["polytope / probes", "regret"], rows))
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    problems = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sp", "knap", "tsp", "port"]
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3, 4]
    deg = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    run(problems, seeds, deg)
