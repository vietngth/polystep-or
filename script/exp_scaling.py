"""Scaling Pareto: where gradient-free direct minimization pulls AWAY from SPO+ as complexity grows.

The fixed-size Pareto (exp2_pareto.py) shows SPO+ winning the cheap-solver scenes (LP shortest-path,
Hungarian assignment) because its few exact solves are fast. The structural cost asymmetry is:
  cost(SPO+)     ~ n_train * epochs * T_exact_solve         (one solver call per instance per epoch)
  cost(PolyStep) ~ steps * probes * T_batched_GPU_forward   (no exact solver; one batched GPU solve/step)
so SPO+'s wall-clock is the PRODUCT of (instance hardness) and (sample size), while PolyStep's is
decoupled from both. This experiment makes that asymmetry visible by sweeping each factor:

  A. grid-size scaling  -- shortest-path on H x H grids, H in {5..25}: the exact LP per instance grows,
                           the batched DAG forward stays cheap. Flips the SPO+-wins LP scene to PolyStep.
  B. sample-size scaling -- shortest-path 10x10, n_train in {500..8000}: SPO+ is linear in n, PolyStep
                           is ~flat (all n batched per GPU step).
  C. item-size scaling  -- knapsack (ILP) with NIT in {16..64}: the harder-solver case.

At each scale point all methods are warm-started identically and given a fixed budget chosen so their
regret is comparable; we then report regret (to confirm parity) and wall-clock (the scaling story).

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp_scaling.py [axes] [seeds]
      axes subset of {size_sp,n_sp,size_knap}; default all.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "polystep/src")
import numpy as np
import torch
from pyepo import metric
from pto.capability import setup_tsp, setup_knap, train_two_stage, train_sfge, train_polystep, dev
from pto.seeding import seed_everything
from pto.budget import SolveCounter, Timer, spoplus_gurobi_solves
from pto.multiseed import summarize, md_table, write_json, write_md
from exp2_pareto import run_spoplus

SPO_EP = 30
PS_ST = 100
SFGE_EP = 120

# Sample-size (n) scaling on problems whose EXACT solver (the one SPO+ must call per instance per epoch)
# is slow: TSP (NP-hard MTZ ILP, ~130 Gurobi solves/s) and knapsack (ILP). SPO+ cost ~ n*epochs*T_solve
# grows linearly in n; PolyStep/SFGE batch all n per GPU step, so they grow sublinearly. (Size scaling and
# cheap-LP problems like shortest-path are NOT shown: there Gurobi's LP is faster than the batched DAG
# forward, so SPO+ stays ahead -- an honest boundary, documented in the writeup.)
AXES = {
    # name: (setup_fn, scale_key, scale_values, fixed_kwargs, x_label)
    "n_tsp":  (setup_tsp,  "n_train", [100, 250, 500, 1000],         dict(N=8),    "n_train (TSP N=8, NP-hard ILP solver)"),
    "n_knap": (setup_knap, "n_train", [500, 1000, 2000, 4000, 8000], dict(NIT=16), "n_train (knapsack, ILP solver)"),
}


def measure_point(setup_fn, seed, deg, scale_key, scale_val, fixed):
    seed_everything(seed)
    kw = dict(fixed); kw[scale_key] = scale_val
    cfg, _ = setup_fn(seed, deg, **kw)
    warm = train_two_stage(cfg); cfg["warm"] = warm
    nt = int(cfg["Xtr"].shape[0])
    out = {}
    # SPO+ (exact solver in the loop)
    with Timer() as t:
        m = run_spoplus(cfg, warm, SPO_EP)
    out["SPO+"] = dict(regret=metric.regret(m, cfg["om"], cfg["ld_te"]), wall=t.seconds,
                       solver_calls=spoplus_gurobi_solves(SPO_EP, nt))
    del m
    # PolyStep (gradient-free, batched forward only)
    sc = SolveCounter(cfg["ps_solve"]); cc = sc.wrap(cfg); cc["ps_steps"] = PS_ST
    with Timer() as t:
        m = train_polystep(cc)
    out["PolyStep"] = dict(regret=metric.regret(m, cfg["om"], cfg["ld_te"]), wall=t.seconds,
                           solver_calls=0)
    del m
    # SFGE (gradient-free sibling)
    sc = SolveCounter(cfg["ps_solve"]); cc = sc.wrap(cfg)
    with Timer() as t:
        m = train_sfge(cc, epochs=SFGE_EP)
    out["SFGE"] = dict(regret=metric.regret(m, cfg["om"], cfg["ld_te"]), wall=t.seconds,
                       solver_calls=0)
    del m, cfg, cc, warm                       # release GPU tensors before the next scale point
    torch.cuda.empty_cache()
    return out


def run_axis(name, seeds, deg=4):
    setup_fn, key, vals, fixed, xlabel = AXES[name]
    print(f"[scaling] axis={name} key={key} vals={vals} seeds={seeds}", flush=True)
    rows = {m: [] for m in ("SPO+", "PolyStep", "SFGE")}
    for v in vals:
        per = {m: {"regret": [], "wall": [], "solver_calls": []} for m in rows}
        for s in seeds:
            pt = measure_point(setup_fn, s, deg, key, v, fixed)
            for m in rows:
                for k in ("regret", "wall", "solver_calls"):
                    per[m][k].append(pt[m][k])
            print(f"    {key}={v} seed={s}: "
                  + " | ".join(f"{m} r={np.mean(per[m]['regret']):.4f} t={pt[m]['wall']:.2f}s" for m in rows),
                  flush=True)
        for m in rows:
            rows[m].append(dict(scale=v, regret=summarize(per[m]["regret"]),
                                wall=summarize(per[m]["wall"]), solver_calls=summarize(per[m]["solver_calls"])))
    return dict(key=key, xlabel=xlabel, vals=vals, rows=rows)


def main():
    axes = sys.argv[1].split(",") if len(sys.argv) > 1 else list(AXES)
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    results = {a: run_axis(a, seeds) for a in axes}
    write_json("exp_results/scaling.json", {"axes": axes, "seeds": seeds, "budgets":
               {"SPO_EP": SPO_EP, "PS_ST": PS_ST, "SFGE_EP": SFGE_EP}, "results": results})
    write_md("exp_results/scaling.md", to_md(results, axes, seeds))
    print("\nwrote exp_results/scaling.{json,md}\nDONE", flush=True)


def to_md(results, axes, seeds):
    L = ["# Scaling Pareto -- gradient-free vs SPO+ as complexity grows", "",
         f"seeds={seeds}; budgets SPO+={SPO_EP}ep, PolyStep={PS_ST}steps, SFGE={SFGE_EP}ep. "
         "Regret confirms parity; wall_clock is the scaling story (SPO+ pays n*epochs exact solves).", ""]
    for a in axes:
        R = results[a]
        L.append(f"## {a} -- {R['xlabel']}")
        headers = [R["key"], "SPO+ regret", "SPO+ wall_s", "SPO+ solves",
                   "PolyStep regret", "PolyStep wall_s", "SFGE regret", "SFGE wall_s"]
        rows = []
        for i, v in enumerate(R["vals"]):
            sp, ps, sf = R["rows"]["SPO+"][i], R["rows"]["PolyStep"][i], R["rows"]["SFGE"][i]
            rows.append([v, f"{sp['regret']['mean']:.4f}", f"{sp['wall']['mean']:.2f}",
                         f"{sp['solver_calls']['mean']:.0f}", f"{ps['regret']['mean']:.4f}",
                         f"{ps['wall']['mean']:.2f}", f"{sf['regret']['mean']:.4f}", f"{sf['wall']['mean']:.2f}"])
        L.append(md_table(headers, rows)); L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
