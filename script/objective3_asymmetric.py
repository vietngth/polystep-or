"""Objective 3 / Theory T3: SPO+ consistency requires SYMMETRIC noise. Under ASYMMETRIC
(mean-preserving, right-skewed) cost noise, SPO+'s minimizer is biased, while direct empirical
regret minimization (PolyStep) adapts to the skew (cf. the newsvendor critical-fractile effect).

Knapsack (ILP), well-specified (deg=1) to isolate the noise-asymmetry effect from misspecification.
Symmetric vs asymmetric noise matched on std. Expect: symmetric -> methods close; asymmetric ->
PolyStep pulls ahead of SPO+ (and two-stage). 5 seeds.
"""
import sys, numpy as np, torch
sys.path.insert(0, "polystep/src")
from pyepo.data import knapsack
from pyepo.model.grb import knapsackModel
from pyepo import metric
from pto.capability import _common, train_two_stage, train_dfl, train_polystep, dev, PF
from pto.solvers import knap1_dp

NIT = 16

def setup(seed, deg, kind, a, n_train=400, n_test=1000):
    W_np, _, _ = knapsack.genData(2, PF, NIT, dim=1, deg=1, seed=1)
    weights = W_np[0].astype(int); CAP = int(weights.sum() * 0.5)
    om = knapsackModel(weights=W_np.astype(int), capacity=[CAP])
    Wt = torch.tensor(weights, dtype=torch.float32, device=dev)
    sb = lambda v: knap1_dp(v, Wt.expand(v.shape[0], -1), CAP)[1].float()
    _, x, c = knapsack.genData(n_train + n_test, PF, NIT, dim=1, deg=deg, noise_width=0, seed=seed)
    rng = np.random.RandomState(seed + 100)
    if kind == "sym":
        mult = 1 + a * np.sqrt(3) * (rng.rand(*c.shape) * 2 - 1)      # U, std a, symmetric
    else:
        mult = 1 + a * (rng.exponential(1.0, c.shape) - 1.0)         # std a, mean 1, right-skewed
    c = c * np.clip(mult, 0.05, None)
    return _common(om, x, c, NIT, sb, "max", "scale", seed, n_train)

print("Objective 3 / T3: knapsack (ILP), deg=1, noise std=0.5, 5 seeds")
print(f"{'noise':>10} | {'two-stage':>10} {'SPO+':>10} {'PolyStep':>10} | {'PS vs SPO+':>10}")
print("-" * 60)
for kind in ["sym", "asym"]:
    R = {m: [] for m in ("two-stage", "SPO+", "PolyStep")}
    for seed in range(5):
        cfg = setup(seed, 1, kind, a=0.5)
        ts = train_two_stage(cfg); cfg["warm"] = ts
        R["two-stage"].append(metric.regret(ts, cfg["om"], cfg["ld_te"]))
        R["SPO+"].append(metric.regret(train_dfl(cfg, "SPO+"), cfg["om"], cfg["ld_te"]))
        R["PolyStep"].append(metric.regret(train_polystep(cfg), cfg["om"], cfg["ld_te"]))
    m = {k: np.mean(v) for k, v in R.items()}
    cut = (m["SPO+"] - m["PolyStep"]) / m["SPO+"] * 100 if m["SPO+"] > 1e-9 else 0
    print(f"{kind:>10} | {m['two-stage']:>10.4f} {m['SPO+']:>10.4f} {m['PolyStep']:>10.4f} | {cut:>+9.0f}%", flush=True)
