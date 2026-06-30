"""TSP diagnosis: characterize why PolyStep lags on high-degree TSP (phase-diagram weak spot).

The phase diagram shows PolyStep weakest on high-degree TSP (e.g. deg8: PolyStep ~0.26 vs IMLE ~0.11).
This isolates the cause by varying (a) the PolyStep step radius and (b) warm vs cold initialization,
against SFGE / IMLE / two-stage references, on TSP across degrees. If a larger step radius or a cold
start recovers PolyStep, the weakness is a step-size / warm-basin issue; if not, it is intrinsic to the
combinatorial landscape.

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp_tsp_diag.py [degs] [seeds]
"""
from __future__ import annotations
import sys
sys.path.insert(0, "polystep/src")
import numpy as np
from pyepo import metric
from pto.capability import setup_tsp, train_two_stage, train_sfge, dev
from pto.sweep_lr import train_polystep_sr
from challenge_established import train_dfl_warm
from pto.seeding import seed_everything
from pto.multiseed import summarize, md_table, write_json, write_md, fmt_mean_std

SR = [0.2, 0.4, 0.8, 1.6]


def main():
    degs = [int(d) for d in sys.argv[1].split(",")] if len(sys.argv) > 1 else [4, 6, 8]
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    cols = (["two-stage", "SFGE", "IMLE"] + [f"PolyStep(warm,sr={s})" for s in SR]
            + ["PolyStep(cold,sr=0.4)", "PolyStep(cold,sr=0.8)"])
    results = {}
    print(f"TSP DIAGNOSIS | degs={degs} seeds={seeds}", flush=True)
    for deg in degs:
        acc = {c: [] for c in cols}
        for seed in seeds:
            seed_everything(seed)
            cfg, _ = setup_tsp(seed, deg)
            ts = train_two_stage(cfg); cfg["warm"] = ts
            acc["two-stage"].append(metric.regret(ts, cfg["om"], cfg["ld_te"]))
            acc["SFGE"].append(metric.regret(train_sfge(cfg), cfg["om"], cfg["ld_te"]))
            try:
                acc["IMLE"].append(metric.regret(train_dfl_warm(cfg, "IMLE", ts), cfg["om"], cfg["ld_te"]))
            except Exception:
                acc["IMLE"].append(float("nan"))
            for s in SR:
                acc[f"PolyStep(warm,sr={s})"].append(
                    metric.regret(train_polystep_sr(cfg, ts, s), cfg["om"], cfg["ld_te"]))
            cold = cfg["make"]()                                  # fresh random init = cold start
            acc["PolyStep(cold,sr=0.4)"].append(metric.regret(train_polystep_sr(cfg, cold, 0.4), cfg["om"], cfg["ld_te"]))
            acc["PolyStep(cold,sr=0.8)"].append(metric.regret(train_polystep_sr(cfg, cold, 0.8), cfg["om"], cfg["ld_te"]))
        results[deg] = {c: summarize(acc[c]) for c in cols}
        print(f"  deg={deg}: " + "  ".join(f"{c}={results[deg][c]['mean']:.4f}" for c in cols), flush=True)
    write_json("exp_results/tsp_diag.json", {"degs": degs, "seeds": seeds, "results":
               {str(d): results[d] for d in degs}})
    L = ["# TSP diagnosis: PolyStep step-radius and initialization on TSP", "",
         f"seeds={seeds}. Normalized regret (lower better). Tests whether a larger step radius or a cold "
         "start recovers PolyStep on high-degree TSP.", ""]
    headers = ["deg"] + cols
    rows = [[d] + [f"{results[d][c]['mean']:.4f}" for c in cols] for d in degs]
    L.append(md_table(headers, rows))
    write_md("exp_results/tsp_diag.md", "\n".join(L))
    print("\nwrote exp_results/tsp_diag.{json,md}\nDONE", flush=True)


if __name__ == "__main__":
    main()
