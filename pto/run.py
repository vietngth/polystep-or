"""Aggregate experiment runner. Prints one table per setting, flushing as it goes.

Run:  .venv/bin/python -m pto.run            # all tables
      .venv/bin/python -m pto.run A          # just setting A, etc. (A|B|MDKP|TERRAIN)
"""
from __future__ import annotations
import sys, time, numpy as np, torch
sys.path.insert(0, "polystep/src")
from polystep.epsilon import CosineEpsilon
from pto import (ShortestPath, TerrainSP, MDKPConsumption, KnapsackWeights,
                 train_two_stage, train_spo_plus, train_polystep)

P = lambda *a: print(*a, flush=True)
LIN = dict(polytope_type="orthoplex", num_probe=1, use_momentum=True,
           momentum_init=0.5, momentum_final=0.9)


def mean_std(xs): return np.mean(xs), np.std(xs)


# ---- Setting A: objective prediction, parity vs SPO+ as constraints scale ----
def tableA(seeds=(0, 1, 2)):
    P("\n=== Setting A (objective prediction): shortest path, parity with SPO+ as #constraints grows ===")
    P(f"{'grid':>6} {'#cons':>6} {'#edges':>7} | {'two-stage':>18} {'SPO+':>18} {'PolyStep':>18}")
    cfg = dict(LIN, epsilon=CosineEpsilon(0.5, 0.05), step_radius=0.4, probe_radius=0.8)
    for (H, W) in [(5, 5), (8, 8), (12, 12)]:
        ts, sp_, ps = [], [], []
        for s in seeds:
            prob = ShortestPath(H=H, W=W, deg=6, n_train=800, n_val=200, n_test=200, seed=40 + s)
            m_ts = train_two_stage(prob, epochs=60); ts.append(prob.regret(m_ts))
            sp_.append(prob.regret(train_spo_plus(prob, epochs=40)))
            ps.append(prob.regret(train_polystep(prob, cfg, steps=120, warm=m_ts, seed=s)))
        a, b, c = mean_std(ts), mean_std(sp_), mean_std(ps)
        P(f"{H}x{W:<3} {H*W:>6} {prob.E:>7} | {a[0]:>8.4f}±{a[1]:<8.4f} {b[0]:>8.4f}±{b[1]:<8.4f} {c[0]:>8.4f}±{c[1]:<8.4f}")


# ---- Setting B: constraint prediction, clear advantage (SPO+ = N/A) ----
def tableB(seeds=(0, 1, 2)):
    P("\n=== Setting B (CONSTRAINT prediction): knapsack w/ predicted weights -- SPO+ = N/A ===")
    P(f"{'deg':>4} | {'two-stage MSE':>18} {'PolyStep':>18} | {'regret cut':>10}")
    cfg = dict(LIN, epsilon=CosineEpsilon(0.6, 0.05), step_radius=0.4, probe_radius=0.8)
    for deg in [1, 2, 4, 6, 8]:
        ts, ps = [], []
        for s in seeds:
            prob = KnapsackWeights(n_item=20, C=40, deg=deg, n_train=256, n_val=256,
                                   n_test=2000, seed=s)
            m_ts = train_two_stage(prob, epochs=60); ts.append(prob.regret(m_ts))
            ps.append(prob.regret(train_polystep(prob, cfg, steps=200, warm=m_ts, seed=s)))
        a, b = mean_std(ts), mean_std(ps)
        cut = (a[0] - b[0]) / a[0] * 100 if a[0] > 1e-6 else 0
        P(f"{deg:>4} | {a[0]:>8.4f}±{a[1]:<8.4f} {b[0]:>8.4f}±{b[1]:<8.4f} | {cut:>+9.0f}%")


# ---- Setting B companion: many resource constraints (MDKP, greedy solver) ----
def tableMDKP(seeds=(0, 1, 2)):
    P("\n=== Setting B companion: multi-dim knapsack (m resource constraints), predicted consumption ===")
    P(f"{'m_res':>6} {'fill':>5} | {'two-stage':>18} {'PolyStep':>18} | {'cut':>6}")
    cfg = dict(LIN, epsilon=CosineEpsilon(1.0, 0.08), step_radius=1.0, probe_radius=2.0)
    for m_res, fill in [(2, 0.2), (5, 0.2), (10, 0.2)]:
        ts, ps = [], []
        for s in seeds:
            prob = MDKPConsumption(n_item=40, m_res=m_res, deg=8, fill=fill,
                                   n_train=256, n_val=256, n_test=1500, seed=s)
            m_ts = train_two_stage(prob, epochs=60); ts.append(prob.regret(m_ts))
            ps.append(prob.regret(train_polystep(prob, cfg, steps=150, warm=m_ts, seed=s)))
        a, b = mean_std(ts), mean_std(ps)
        cut = (a[0] - b[0]) / a[0] * 100 if a[0] > 1e-6 else 0
        P(f"{m_res:>6} {fill:>5.2f} | {a[0]:>8.4f}±{a[1]:<8.4f} {b[0]:>8.4f}±{b[1]:<8.4f} | {cut:>+5.0f}%")


# ---- #4: scale the predictor to a CNN (image -> cost map), subspace mode ----
def tableTerrain(seeds=(0, 1)):
    P("\n=== #4 Scaling the predictor: Warcraft-style terrain image -> CNN cost map (subspace) ===")
    P(f"{'CNN params':>11} {'subspace rank':>13} | {'two-stage':>18} {'PolyStep':>18} | {'cut':>6}")
    cfg = dict(LIN, epsilon=CosineEpsilon(5.0, 0.5), step_radius=4.0, probe_radius=2.0)
    ts, ps, npar = [], [], 0
    for s in seeds:
        prob = TerrainSP(H=12, W=12, ps=3, n_train=400, n_val=150, n_test=300, seed=s)
        npar = sum(p.numel() for p in prob.predictor().parameters())
        m_ts = train_two_stage(prob, epochs=50); ts.append(prob.regret(m_ts))
        ps.append(prob.regret(train_polystep(prob, cfg, steps=60, warm=m_ts,
                                             subspace_rank=4, seed=s)))
    a, b = mean_std(ts), mean_std(ps)
    cut = (a[0] - b[0]) / a[0] * 100 if a[0] > 1e-6 else 0
    P(f"{npar:>11} {4:>13} | {a[0]:>8.4f}±{a[1]:<8.4f} {b[0]:>8.4f}±{b[1]:<8.4f} | {cut:>+5.0f}%")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    t0 = time.time()
    if which in ("A", "ALL"): tableA()
    if which in ("B", "ALL"): tableB()
    if which in ("MDKP", "ALL"): tableMDKP()
    if which in ("TERRAIN", "ALL"): tableTerrain()
    P(f"\n[done in {time.time()-t0:.0f}s]")
