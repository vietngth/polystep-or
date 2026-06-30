"""Cross-problem head-to-head of the two GRADIENT-FREE direct-regret methods (SFGE vs PolyStep),
with two-stage and SPO+ as references. deg=4, default per-setup data, 3 seeds. Tells us whether
SFGE's low-data knapsack edge generalizes, or whether PolyStep leads on LP/TSP/nonlinear.
"""
import sys, numpy as np
sys.path.insert(0, "polystep/src")
from pyepo import metric
from pto.capability import (setup_sp, setup_knap, setup_tsp, setup_port,
                            train_two_stage, train_dfl, train_sfge, train_polystep)

SETUPS = {"sp (LP)": setup_sp, "knap (ILP)": setup_knap, "tsp (ILP)": setup_tsp, "port (QP/SOCP)": setup_port}
SEEDS = [42, 43, 44]
print("Cross-problem | gradient-free rivals | deg=4, 3 seeds | normalized regret")
print(f"{'problem':>14} | {'two-stage':>9} {'SPO+':>9} {'SFGE':>9} {'PolyStep':>9} | winner")
print("-" * 70)
for name, setup in SETUPS.items():
    R = {m: [] for m in ("two-stage", "SPO+", "SFGE", "PolyStep")}
    for seed in SEEDS:
        cfg, cat = setup(seed, 4)
        ts = train_two_stage(cfg); cfg["warm"] = ts
        R["two-stage"].append(metric.regret(ts, cfg["om"], cfg["ld_te"]))
        R["SPO+"].append(metric.regret(train_dfl(cfg, "SPO+"), cfg["om"], cfg["ld_te"]))
        R["SFGE"].append(metric.regret(train_sfge(cfg), cfg["om"], cfg["ld_te"]))
        R["PolyStep"].append(metric.regret(train_polystep(cfg), cfg["om"], cfg["ld_te"]))
    m = {k: np.mean(v) for k, v in R.items()}
    win = min(m, key=m.get)
    print(f"{name:>14} | {m['two-stage']:>9.4f} {m['SPO+']:>9.4f} {m['SFGE']:>9.4f} {m['PolyStep']:>9.4f} | {win}", flush=True)
