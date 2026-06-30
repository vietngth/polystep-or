"""PolyStep hyperparameter tuning -- probe points (num_probe) + radii, per (problem, polytope).

The tuning phase An asked for: sweep PolyStep's probe-point count and radii for EACH polytope
(orthoplex/simplex/cube) on each PyEPO problem, over a few tuning seeds, and record the config with
the lowest mean normalized regret -> exp_results/polystep_tune.json. exp_polystep_bench.py then runs
the STRONG benchmark (5 seeds) using these tuned configs. Run on a SLURM compute node.

Run:  .venv/bin/python exp_polystep_tune.py sp,knap,tsp,port 0,1
"""
from __future__ import annotations
import sys, json, os, itertools, time
import numpy as np
sys.path.insert(0, "polystep/src")
from pyepo import metric
from pto.capability import SETUPS, train_two_stage, train_polystep

POLYTOPES = os.environ.get("POLY", "orthoplex,simplex,cube").split(",")
# probe points are the headline knob An named; radii interact with the polytope geometry
GRID = dict(ps_num_probe=[1, 2, 4],
            ps_probe_radius=[0.5, 1.0, 2.0],
            ps_step_radius=[0.3, 0.6])


def run(problems, seeds, deg=4):
    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    best, allres = {}, {}
    print(f"tuning: {len(combos)} configs x {len(POLYTOPES)} polytopes x {len(seeds)} seeds "
          f"per problem; problems={problems}", flush=True)
    for pname in problems:
        cfgs = []
        for seed in seeds:
            cfg, cat = SETUPS[pname](seed, deg)
            cfg["warm"] = train_two_stage(cfg)
            cfgs.append(cfg)
        for poly in POLYTOPES:
            scored = []
            for combo in combos:
                ov = dict(zip(keys, combo)); ov["ps_polytope"] = poly
                regs = []
                for cfg in cfgs:
                    c2 = dict(cfg); c2.update(ov)
                    try:
                        regs.append(float(metric.regret(train_polystep(c2), cfg["om"], cfg["ld_te"])))
                    except Exception:
                        regs.append(float("nan"))
                m = float(np.nanmean(regs))
                scored.append((m, {k: ov[k] for k in keys}))
            scored.sort(key=lambda x: (np.inf if x[0] != x[0] else x[0]))
            bm, bo = scored[0]
            best[f"{pname}:{poly}"] = {**bo, "regret": bm}
            allres[f"{pname}:{poly}"] = [{"regret": r, "cfg": o} for r, o in scored]
            print(f"  BEST {pname:>5}:{poly:<9} -> {bo}  regret={bm:.4f}", flush=True)
    os.makedirs("exp_results", exist_ok=True)
    json.dump({"best": best, "all": allres, "grid": GRID, "polytopes": POLYTOPES, "seeds": seeds},
              open("exp_results/polystep_tune.json", "w"), indent=1)
    print("\nwrote exp_results/polystep_tune.json\nDONE", flush=True)


if __name__ == "__main__":
    problems = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sp", "knap", "tsp", "port"]
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1]
    deg = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    run(problems, seeds, deg)
