"""Learning-rate sweep in the MAIN repo: SPO+ vs SFGE vs PolyStep.

Ports the dfl-ablation robustness finding (SPO+ is highly learning-rate sensitive; a single-lr
comparison is unfair) into the PolyStep project and adds the third method. For each problem:

  - SPO+  : PyEPO SPOPlus, swept over an Adam learning-rate grid.
  - SFGE  : score-function gradient-free, swept over the SAME lr grid.
  - PolyStep : gradient-free OT optimizer. It has NO learning rate; its step is set by the OT
               radii. We report it at the project default AND sweep `step_radius` as the analog,
               to compare hyperparameter sensitivity on equal footing.

All three methods are warm-started from the SAME two-stage MSE model (the project's fair-init
convention, see challenge_established.py), so differences are due to the trainer alone.

Run:
  .venv/bin/python -m pto.sweep_lr sp,knap,tsp,port 4 0,1,2
"""
from __future__ import annotations
import sys, json, numpy as np, torch
from pyepo import metric
import pyepo.func as F
from pto.capability import (SETUPS, train_two_stage, train_polystep, train_sfge, _adam, dev)

LRS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
STEP_RADII = [0.1, 0.2, 0.4, 0.8]
CAT = {"sp": "LP", "knap": "ILP", "tsp": "ILP", "port": "SOCP"}


def train_spoplus_lr(cfg, warm, lr, epochs=100):
    m = cfg["make"]()
    with torch.no_grad():
        if warm is not None: m.weight.copy_(warm.weight)
    opt = _adam(m, lr); spop = F.SPOPlus(cfg["om"])
    for _ in range(epochs):
        for xb, cb, wb, zb in cfg["ld_tr"]:
            xb, cb, wb, zb = [t.float().to(dev) for t in (xb, cb, wb, zb)]
            opt.zero_grad(); spop(m(xb), cb, wb, zb).mean().backward(); opt.step()
    return m


def train_sfge_lr(cfg, warm, lr, epochs=120):
    cfg = dict(cfg); cfg["warm"] = warm
    return train_sfge(cfg, epochs=epochs, lr=lr)


def train_polystep_sr(cfg, warm, step_radius, steps=None):
    cfg = dict(cfg); cfg["warm"] = warm
    if steps is not None: cfg["ps_steps"] = steps
    # patch step_radius via a thin wrapper around train_polystep's optimizer config
    import polystep, polystep.epsilon as E
    m = cfg["make"]()
    with torch.no_grad(): m.weight.copy_(warm.weight)
    pso = polystep.PolyStepOptimizer(m, polytope_type="orthoplex",
                                     epsilon=E.CosineEpsilon(0.5, 0.05),
                                     step_radius=step_radius, probe_radius=2 * step_radius,
                                     num_probe=1, seed=cfg["seed"], use_momentum=True,
                                     momentum_init=0.5, momentum_final=0.9)
    X, Cs, solve, sgn = cfg["Xtr"], cfg["Cs"], cfg["ps_solve"], cfg["sign"]
    nstep = cfg.get("ps_steps", 150)
    def closure(bp):
        pred = torch.einsum("nef,bf->nbe", bp["weight"], X); N, nb, Eax = pred.shape
        w = solve(pred.reshape(N * nb, Eax)).reshape(N, nb, Eax)
        return sgn * (w * Cs.unsqueeze(0)).sum(-1).mean(-1)
    for _ in range(nstep): pso.step(closure)
    return m


def sweep(problems, deg, seeds):
    results = []
    for p in problems:
        print(f"\n=== {p} ({CAT[p]}) deg={deg} ===", flush=True)
        spo = {lr: [] for lr in LRS}; sfge = {lr: [] for lr in LRS}
        ps = {sr: [] for sr in STEP_RADII}
        for seed in seeds:
            cfg, _ = SETUPS[p](seed, deg)
            warm = train_two_stage(cfg)                       # SAME init for all methods
            for lr in LRS:
                spo[lr].append(metric.regret(train_spoplus_lr(cfg, warm, lr), cfg["om"], cfg["ld_te"]))
                sfge[lr].append(metric.regret(train_sfge_lr(cfg, warm, lr), cfg["om"], cfg["ld_te"]))
            for sr in STEP_RADII:
                ps[sr].append(metric.regret(train_polystep_sr(cfg, warm, sr), cfg["om"], cfg["ld_te"]))
            print(f"  seed {seed} done", flush=True)
        def best(d):
            ms = {k: (float(np.mean(v)), float(np.std(v))) for k, v in d.items()}
            bk = min(ms, key=lambda k: ms[k][0])
            return bk, ms[bk][0], ms[bk][1], ms
        bspo = best(spo); bsfge = best(sfge); bps = best(ps)
        winner = min([("SPO+", bspo[1]), ("SFGE", bsfge[1]), ("PolyStep", bps[1])], key=lambda t: t[1])[0]
        res = {"problem": p, "category": CAT[p], "deg": deg,
               "SPO+": {"by_lr": bspo[3], "best_lr": bspo[0], "best": bspo[1], "std": bspo[2]},
               "SFGE": {"by_lr": bsfge[3], "best_lr": bsfge[0], "best": bsfge[1], "std": bsfge[2]},
               "PolyStep": {"by_sr": bps[3], "best_sr": bps[0], "best": bps[1], "std": bps[2]},
               "winner_best_tuned": winner}
        results.append(res)
        sp_spread = max(v[0] for v in bspo[3].values()) - min(v[0] for v in bspo[3].values())
        ps_spread = max(v[0] for v in bps[3].values()) - min(v[0] for v in bps[3].values())
        print(f"  best-tuned: SPO+ {bspo[1]:.4f}@lr{bspo[0]:g}  SFGE {bsfge[1]:.4f}@lr{bsfge[0]:g}  "
              f"PolyStep {bps[1]:.4f}@sr{bps[0]:g}  -> {winner}", flush=True)
        print(f"  sensitivity spread: SPO+ {sp_spread:.4f} (over lr)  PolyStep {ps_spread:.4f} (over step_radius)",
              flush=True)
    return results


def main():
    problems = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sp", "knap", "tsp", "port"]
    deg = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2]
    print(f"MAIN-REPO LR SWEEP | SPO+ vs SFGE vs PolyStep | problems={problems} deg={deg} "
          f"seeds={seeds} lrs={LRS} step_radii={STEP_RADII}", flush=True)
    results = sweep(problems, deg, seeds)
    out = {"deg": deg, "seeds": seeds, "lrs": LRS, "step_radii": STEP_RADII, "results": results}
    with open("sweep_lr_polystep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n========= BEST-TUNED SUMMARY (normalized regret, lower better) =========", flush=True)
    print(f"{'problem':>8} {'cat':>5} | {'SPO+':>16} {'SFGE':>16} {'PolyStep':>16} | winner", flush=True)
    for r in results:
        print(f"{r['problem']:>8} {r['category']:>5} | "
              f"{r['SPO+']['best']:.4f}@{r['SPO+']['best_lr']:<6g} "
              f"{r['SFGE']['best']:.4f}@{r['SFGE']['best_lr']:<6g} "
              f"{r['PolyStep']['best']:.4f}@sr{r['PolyStep']['best_sr']:<5g} | {r['winner_best_tuned']}",
              flush=True)
    print("\nJSON -> sweep_lr_polystep.json\nDONE", flush=True)


if __name__ == "__main__":
    main()
