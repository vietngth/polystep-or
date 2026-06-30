"""Objective 2: direct empirical regret (PolyStep) vs the SPO+ CONVEX SURROGATE.

For each optimization category (LP / ILP), with SIMPLE LINEAR predictors and 5 seeds:
  - train SPO+; measure its realized normalized regret AND its achieved SPO+ surrogate VALUE
    (the convex upper bound) on the test set -> the "surrogate gap" (looseness);
  - train PolyStep (direct true-regret); measure realized normalized regret.
Hypothesis (theory T1): the surrogate is ~tight on LPs (gap small -> parity) but loose on integer
programs (gap large -> PolyStep wins). Wilcoxon signed-rank PolyStep vs SPO+ across seeds.
"""
import sys, numpy as np, torch
sys.path.insert(0, "polystep/src")
from scipy import stats
from pyepo import metric
import pyepo.func as F
from pto.capability import (setup_sp, setup_knap, setup_tsp, train_two_stage,
                            train_dfl, train_polystep, dev)

def spoplus_norm_value(model, cfg):
    """Mean SPO+ surrogate loss on test, normalized by mean |z*| (comparable to norm regret)."""
    spop = F.SPOPlus(cfg["om"]); tot = nz = n = 0.0
    for xb, cb, wb, zb in cfg["ld_te"]:
        xb, cb, wb, zb = [t.float().to(dev) for t in (xb, cb, wb, zb)]
        with torch.no_grad():
            l = spop(model(xb), cb, wb, zb)
        tot += float(l) * xb.shape[0]; nz += float(zb.abs().sum()); n += xb.shape[0]
    return (tot / n) / (nz / n)

SETUPS = {"shortest path (LP)": setup_sp, "knapsack (ILP)": setup_knap, "TSP (ILP)": setup_tsp}
SEEDS = [0, 1, 2, 3, 4]
DEG = 4

print("Objective 2 | direct regret (PolyStep) vs SPO+ convex surrogate | linear models, deg=4, 5 seeds")
print(f"{'problem':>18} | {'SPO+ regret':>11} {'SPO+ surrog':>11} {'looseness':>9} | {'PolyStep':>9} | {'PS<SPO+ %':>9} {'Wilcoxon p':>10}")
print("-" * 96)
for name, setup in SETUPS.items():
    spo_reg, spo_sur, ps_reg = [], [], []
    for seed in SEEDS:
        cfg, cat = setup(seed, DEG)
        ts = train_two_stage(cfg); cfg["warm"] = ts
        m_spo = train_dfl(cfg, "SPO+")
        spo_reg.append(metric.regret(m_spo, cfg["om"], cfg["ld_te"]))
        spo_sur.append(spoplus_norm_value(m_spo, cfg))
        ps_reg.append(metric.regret(train_polystep(cfg), cfg["om"], cfg["ld_te"]))
    spo_reg, spo_sur, ps_reg = map(np.array, (spo_reg, spo_sur, ps_reg))
    loose = spo_sur - spo_reg                                          # surrogate upper-bound slack
    cut = (spo_reg.mean() - ps_reg.mean()) / spo_reg.mean() * 100
    try: p = stats.wilcoxon(ps_reg, spo_reg).pvalue
    except Exception: p = float("nan")
    print(f"{name:>18} | {spo_reg.mean():>11.4f} {spo_sur.mean():>11.4f} {loose.mean():>9.4f} | "
          f"{ps_reg.mean():>9.4f} | {cut:>+8.0f}% {p:>10.3f}", flush=True)
