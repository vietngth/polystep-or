"""
"Abundant challenge": take the well-established predict-then-optimize benchmarks the surrogate /
gradient-DFL camp is known to work on, and challenge them with the gradient-free direct-regret camp.

Established surrogate / gradient camp : SPO+ (Elmachtoub-Grigas, Mgmt Sci 2022), PFYL (Berthet
                                        perturbed Fenchel-Young), IMLE (implicit MLE).
Gradient-free direct-regret camp      : SFGE (Silvestri et al., JAIR 2024), PolyStep.
Reference                             : two-stage MSE.

All on PyEPO's own data generators + Gurobi-evaluated normalized regret, across the four canonical
problems (shortest path LP, knapsack ILP, TSP ILP, portfolio SOCP) and several misspecification
degrees. FAIR PROTOCOL: every method gets the SAME two-stage warm start and the same data/predictor.

Not exhaustive, but abundant: 4 problems x 3 degrees x {6 methods} x seeds. Verdict per cell =
does the gradient-free camp match/beat the best of the established camp.

Run: .venv/bin/python challenge_established.py [degs] [seeds]
"""
import sys, numpy as np, torch
sys.path.insert(0, "polystep/src")
from pyepo import metric
from pto.capability import (setup_sp, setup_knap, setup_tsp, setup_port, train_two_stage,
                            train_sfge, train_polystep, train_dfl, DFL, _adam, dev)

SETUPS = {"sp (LP)": setup_sp, "knap (ILP)": setup_knap, "tsp (ILP)": setup_tsp, "port (SOCP)": setup_port}
SURROGATE = ["SPO+", "PFYL", "IMLE"]      # established gradient / surrogate camp
GRADFREE = ["SFGE", "PolyStep"]            # gradient-free direct-regret camp
ALL = ["two-stage"] + SURROGATE + GRADFREE


def train_dfl_warm(cfg, name, warm, epochs=30):
    """capability.train_dfl, but warm-start the predictor (same init as SFGE/PolyStep) for fairness."""
    build, kind, fwd = DFL[name]
    om = cfg["om"]; sense = om.modelSense
    m = cfg["make"]()
    if warm is not None:
        with torch.no_grad(): m.weight.copy_(warm.weight)
    opt = _adam(m); loss_mod = build(om, cfg["ds_tr"])
    for _ in range(epochs):
        for xb, cb, wb, zb in cfg["ld_tr"]:
            xb, cb, wb, zb = [t.float().to(dev) for t in (xb, cb, wb, zb)]
            pred = m(xb)
            if kind == "opt":
                loss = sense * (loss_mod(pred) * cb).sum(-1).mean()
            else:
                pick = {"pred": pred, "c": cb, "w": wb, "z": zb}
                out = loss_mod(*[pick[a] for a in fwd]); loss = out.mean() if out.dim() > 0 else out
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def cell(setup, deg, seeds):
    acc = {m: [] for m in ALL}
    for seed in seeds:
        cfg, _ = setup(seed, deg)
        ts = train_two_stage(cfg); cfg["warm"] = ts          # SAME warm start for everyone
        acc["two-stage"].append(metric.regret(ts, cfg["om"], cfg["ld_te"]))
        for name in SURROGATE:
            try: acc[name].append(metric.regret(train_dfl_warm(cfg, name, ts), cfg["om"], cfg["ld_te"]))
            except Exception: acc[name].append(float("nan"))
        acc["SFGE"].append(metric.regret(train_sfge(cfg), cfg["om"], cfg["ld_te"]))
        acc["PolyStep"].append(metric.regret(train_polystep(cfg), cfg["om"], cfg["ld_te"]))
    means = {m: float(np.nanmean(acc[m])) for m in ALL}
    stds = {m: float(np.nanstd(acc[m])) for m in ALL}
    return means, stds


def main():
    degs = [int(d) for d in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2, 4, 6]
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    print(f"ABUNDANT CHALLENGE | established surrogate/gradient camp vs gradient-free direct-regret")
    print(f"PyEPO data + Gurobi regret | warm-started | {len(seeds)} seeds | normalized regret (lower better)\n")
    hdr = f"{'problem':>11} {'deg':>3} | " + " ".join(f"{m:>9}" for m in ALL) + " | best-camp"
    print(hdr); print("-" * len(hdr))
    tally = {"gradient-free": 0, "surrogate": 0, "two-stage": 0}
    rows_json = []
    for pname, setup in SETUPS.items():
        for deg in degs:
            r, s = cell(setup, deg, seeds)
            best_sur = min(r[m] for m in SURROGATE)
            best_gf = min(r[m] for m in GRADFREE)
            if r["two-stage"] <= min(best_sur, best_gf): camp = "two-stage"
            elif best_gf <= best_sur: camp = "gradient-free"
            else: camp = "surrogate"
            tally[camp] += 1
            row = " ".join(f"{r[m]:>5.4f}+-{s[m]:<6.4f}" for m in ALL)
            rows_json.append({"problem": pname, "deg": deg, "mean": r, "std": s, "camp": camp})
            print(f"{pname:>11} {deg:>3} | {row} | {camp}", flush=True)
    import json, os
    os.makedirs("exp_results", exist_ok=True)
    json.dump({"seeds": len(seeds), "rows": rows_json},
              open("exp_results/challenge_grid.json", "w"), indent=2)
    print(f"\nVerdict over {len(SETUPS)*len(degs)} cells: " +
          ", ".join(f"{k} wins {v}" for k, v in tally.items()))
    print("(gradient-free = SFGE/PolyStep; surrogate = SPO+/PFYL/IMLE)")


if __name__ == "__main__":
    main()
