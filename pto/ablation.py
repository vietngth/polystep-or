"""Track 2a: data-role ablations. Fix the solver; vary the DATA (training-set size,
noise, misspecification degree); show the data->predict->optimize pipeline trained
with PolyStep (and DFL) beats two-stage on decision quality where prediction is hard.

Reuses the capability machinery (train_two_stage/train_dfl/train_polystep, _common).
Compares a compact panel {two-stage, SPO+, IMLE (best differentiable DFL), PolyStep}.

Run: .venv/bin/python -m pto.ablation <problem> <axis>   (axis: size|noise|deg)
"""
from __future__ import annotations
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "polystep/src")
from pyepo.data import shortestpath, knapsack
from pyepo.model.grb import shortestPathModel, knapsackModel
from pyepo import metric
from pto.capability import (_common, train_two_stage, train_dfl, train_polystep,
                            train_sfge, PF, dev)
from pto.solvers import build_dag_solver, knap1_dp

# two-stage, incumbent surrogate, best gradient-DFL, the gradient-free rival, ours
PANEL = ["two-stage", "SPO+", "IMLE", "SFGE", "PolyStep"]


def setup_knap(seed, deg, n_train, noise, n_test=1000, NIT=16):
    W_np, _, _ = knapsack.genData(2, PF, NIT, dim=1, deg=1, seed=1)
    weights = W_np[0].astype(int); CAP = int(weights.sum() * 0.5)
    om = knapsackModel(weights=W_np.astype(int), capacity=[CAP])
    Wt = torch.tensor(weights, dtype=torch.float32, device=dev)
    sb = lambda v: knap1_dp(v, Wt.expand(v.shape[0], -1), CAP)[1].float()
    _, x, c = knapsack.genData(n_train + n_test, PF, NIT, dim=1, deg=deg, noise_width=noise, seed=seed)
    return _common(om, x, c, NIT, sb, "max", "scale", seed, n_train)

def setup_sp(seed, deg, n_train, noise, n_test=1000, H=5, W=5):
    om = shortestPathModel((H, W)); arcs = list(om.arcs)
    sb = build_dag_solver(arcs, H * W, 0, H * W - 1)
    x, c = shortestpath.genData(n_train + n_test, PF, (H, W), deg=deg, noise_width=noise, seed=seed)
    return _common(om, x, c, len(arcs), sb, "min", "affine", seed, n_train)

SETUPS = {"knap": setup_knap, "sp": setup_sp}


def panel(cfg):
    out = {}
    ts = train_two_stage(cfg); out["two-stage"] = metric.regret(ts, cfg["om"], cfg["ld_te"])
    out["SPO+"] = metric.regret(train_dfl(cfg, "SPO+"), cfg["om"], cfg["ld_te"])
    out["IMLE"] = metric.regret(train_dfl(cfg, "IMLE"), cfg["om"], cfg["ld_te"])
    cfg["warm"] = ts
    out["SFGE"] = metric.regret(train_sfge(cfg), cfg["om"], cfg["ld_te"])
    out["PolyStep"] = metric.regret(train_polystep(cfg), cfg["om"], cfg["ld_te"])
    return out


def ablate(problem, axis, seeds=(42, 43, 44)):
    setup = SETUPS[problem]
    if axis == "size":
        grid = [("n", n, dict(deg=4, n_train=n, noise=0.0)) for n in (50, 100, 200, 500, 1000, 3000)]
    elif axis == "noise":
        grid = [("noise", w, dict(deg=4, n_train=400, noise=w)) for w in (0.0, 0.25, 0.5, 1.0)]
    else:  # deg
        grid = [("deg", d, dict(deg=d, n_train=400, noise=0.0)) for d in (1, 2, 4, 6, 8)]
    print(f"{problem} | data axis = {axis} | normalized regret (3 seeds)")
    print(f"{axis:>8} | " + " ".join(f"{m:>10}" for m in PANEL), flush=True)
    for label, val, kw in grid:
        acc = {m: [] for m in PANEL}
        for seed in seeds:
            r = panel(setup(seed=seed, **kw))
            for m in PANEL: acc[m].append(r[m])
        print(f"{str(val):>8} | " + " ".join(f"{np.mean(acc[m]):>10.4f}" for m in PANEL), flush=True)


if __name__ == "__main__":
    prob = sys.argv[1] if len(sys.argv) > 1 else "knap"
    axis = sys.argv[2] if len(sys.argv) > 2 else "size"
    ablate(prob, axis)
