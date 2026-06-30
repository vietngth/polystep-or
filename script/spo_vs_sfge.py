#!/usr/bin/env python
"""
spo_vs_sfge.py — one-command, reproducible SPO+ vs. SFGE comparison for predict-then-optimize.

Compares ONLY two decision-focused-learning losses, under IDENTICAL data, predictor, initialization,
and evaluation metric, so the result is a fair head-to-head to share with others:

  * SPO+  — Smart Predict-then-Optimize+ convex surrogate (Elmachtoub & Grigas, Mgmt Sci 2022),
            reused directly from PyEPO  (pyepo.func.SPOPlus).
  * SFGE  — Score-Function Gradient Estimation, the gradient-free / black-box-solver method
            (Silvestri et al., JAIR 2024; arXiv:2307.05213). Not packaged in PyEPO, so this uses a
            faithful score-function (REINFORCE) implementation with a per-batch baseline.

Everything else is REUSED, not re-implemented:
  - problem data generators, optimization models, and the normalized-regret metric come from PyEPO;
  - the problem set-ups (optModel + DataLoaders + a batched forward solver) and the SFGE trainer come
    from this project's `pto/` package.

Both methods get the SAME predictor (linear by default), the SAME initialization (cold or warm), and
are scored by PyEPO's Gurobi-evaluated normalized regret. PolyStep is intentionally excluded.

Examples
--------
  python spo_vs_sfge.py --problem knapsack --deg 4 --seeds 0,1,2
  python spo_vs_sfge.py --problem shortest_path --deg 6 --n-train 200 --noise 0.5
  python spo_vs_sfge.py --problem tsp --deg 2 --seeds 0,1,2,3,4 --init warm
  python spo_vs_sfge.py --problem portfolio --deg 8 --spo-epochs 200 --sfge-epochs 150
"""
import argparse, sys
import numpy as np

sys.path.insert(0, "polystep/src")   # PolyStep package (only its deps; PolyStep itself is NOT used here)


def build_cfg(problem, seed, deg, n_train, n_test, noise):
    """Reuse the project's PyEPO-backed problem set-ups. Returns the harness cfg dict."""
    from pto.ablation import setup_knap, setup_sp            # these support a `noise` argument
    from pto.capability import setup_tsp, setup_port         # these return (cfg, category)
    if problem == "knapsack":
        return setup_knap(seed, deg, n_train, noise, n_test=n_test)
    if problem == "shortest_path":
        return setup_sp(seed, deg, n_train, noise, n_test=n_test)
    if problem == "tsp":
        if noise: print("  [note] --noise is ignored for tsp (no noise hook in the generator)")
        return setup_tsp(seed, deg, n_train=n_train, n_test=n_test)[0]
    if problem == "portfolio":
        if noise: print("  [note] --noise is ignored for portfolio")
        return setup_port(seed, deg, n_train=n_train, n_test=n_test)[0]
    raise ValueError(f"unknown problem {problem!r}")


def train_spoplus(cfg, warm, epochs, lr):
    """SPO+ reused straight from PyEPO (pyepo.func.SPOPlus)."""
    import torch
    import pyepo.func as F
    from pto.capability import dev
    m = cfg["make"]()
    if warm is not None:
        with torch.no_grad(): m.weight.copy_(warm.weight)
    opt = torch.optim.Adam(m.parameters(), lr)
    spop = F.SPOPlus(cfg["om"])
    for _ in range(epochs):
        for xb, cb, wb, zb in cfg["ld_tr"]:
            xb, cb, wb, zb = [t.float().to(dev) for t in (xb, cb, wb, zb)]
            opt.zero_grad(); spop(m(xb), cb, wb, zb).mean().backward(); opt.step()
    return m


def main():
    ap = argparse.ArgumentParser(description="Fair SPO+ vs. SFGE comparison (PyEPO-native).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--problem", default="knapsack",
                    choices=["shortest_path", "knapsack", "tsp", "portfolio"])
    ap.add_argument("--deg", type=int, default=4, help="polynomial misspecification degree (1=well-specified)")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--noise", type=float, default=0.0, help="multiplicative noise width (knapsack/shortest_path)")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    ap.add_argument("--init", default="cold", choices=["cold", "warm"],
                    help="cold = both from scratch; warm = both from a two-stage MSE model (SAME for both)")
    ap.add_argument("--spo-epochs", type=int, default=100)
    ap.add_argument("--spo-lr", type=float, default=1e-2)
    ap.add_argument("--sfge-epochs", type=int, default=120)
    ap.add_argument("--sfge-samples", type=int, default=8)
    ap.add_argument("--sfge-sigma", type=float, default=0.5)
    ap.add_argument("--sfge-lr", type=float, default=1e-2)
    args = ap.parse_args()

    from pyepo import metric
    from pto.capability import train_two_stage, train_sfge
    try:
        from scipy import stats
    except Exception:
        stats = None

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"\nSPO+ vs SFGE | problem={args.problem} deg={args.deg} n_train={args.n_train} "
          f"noise={args.noise} init={args.init} seeds={seeds}")
    print(f"SPO+: PyEPO SPOPlus ({args.spo_epochs} ep) | SFGE: score-function "
          f"({args.sfge_epochs} ep, {args.sfge_samples} samples, sigma={args.sfge_sigma}) | "
          f"metric: PyEPO normalized regret (Gurobi)\n", flush=True)

    spo, sfge = [], []
    for seed in seeds:
        cfg = build_cfg(args.problem, seed, args.deg, args.n_train, args.n_test, args.noise)
        warm = train_two_stage(cfg) if args.init == "warm" else None
        cfg["warm"] = warm                                  # SAME init for both methods
        r_spo = metric.regret(train_spoplus(cfg, warm, args.spo_epochs, args.spo_lr),
                              cfg["om"], cfg["ld_te"])
        r_sfge = metric.regret(train_sfge(cfg, epochs=args.sfge_epochs, n_samples=args.sfge_samples,
                                          sigma=args.sfge_sigma, lr=args.sfge_lr),
                               cfg["om"], cfg["ld_te"])
        spo.append(r_spo); sfge.append(r_sfge)
        print(f"  seed {seed}:  SPO+ = {r_spo:.4f}   SFGE = {r_sfge:.4f}", flush=True)

    spo, sfge = np.array(spo), np.array(sfge)
    print("\n--- summary (normalized regret, lower is better) ---")
    print(f"  SPO+ : {spo.mean():.4f} ± {spo.std():.4f}")
    print(f"  SFGE : {sfge.mean():.4f} ± {sfge.std():.4f}")
    winner = "SPO+" if spo.mean() < sfge.mean() else "SFGE"
    diff = abs(spo.mean() - sfge.mean()) / max(spo.mean(), 1e-9) * 100
    line = f"  winner: {winner}  ({diff:.0f}% lower regret"
    if stats is not None and len(seeds) >= 3:
        try: line += f", Wilcoxon p={stats.wilcoxon(spo, sfge).pvalue:.3f}"
        except Exception: pass
    print(line + ")")


if __name__ == "__main__":
    main()
