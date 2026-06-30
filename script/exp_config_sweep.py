"""PolyStep configuration sweep -- the headline ROBUSTNESS study.

For the CHEAP predict-then-optimize problems, sweep PolyStep over
    polytope_type in {orthoplex, simplex, cube}   x
    num_probe      in {1, 3, 5, 8}                 x
    seeds          in {0,1,2,3,4}
recording normalized-regret mean+-std (and wall-clock) for every
(problem, polytope, probe) cell, then aggregating across problems via RANKS to
produce a GENERAL TREND: which polytope and which probe-count is best for most
cases, and a recommended default.

Two problem families share this one driver (the repo already exposes both):
  * sp, knap, tsp, port  -- PyEPO-backed, via pto.capability.SETUPS + capability
                            train_two_stage/train_polystep + pyepo.metric.regret.
                            Polytope/probe are set through cfg["ps_polytope"]/cfg["ps_num_probe"]
                            (exactly as exp_polystep_ablation.py / exp_polystep_bench.py do).
  * mdkp, knapw          -- CONSTRAINT-prediction problems (MDKPConsumption,
                            KnapsackWeights) from pto/problems.py, via the
                            pto.methods train_two_stage/train_polystep(problem, cfg=dict(
                            polytope_type=..., num_probe=...), ...) + prob.regret interface.

deg=4 for all problems (matches exp_polystep_bench.py) for comparability.

Run:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp_config_sweep.py \
      [problems_csv] [polytopes_csv] [probes_csv] [seeds_csv]
Defaults = the full grid:
  problems = sp,knap,tsp,port,mdkp,knapw
  polytopes= orthoplex,simplex,cube
  probes   = 1,3,5,8
  seeds    = 0,1,2,3,4

Smoke:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp_config_sweep.py sp,knap orthoplex,simplex 1,3 0,1

Writes exp_results/config_sweep.{json,md}. RUN ON A SLURM COMPUTE NODE.
"""
from __future__ import annotations
import sys, os, time
import numpy as np

# honor deterministic cuBLAS workspace (same as the rest of the suite)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, "polystep/src")
from pyepo import metric
from polystep.epsilon import CosineEpsilon

# --- interface A: PyEPO-backed cheap problems (capability.py) ---
from pto.capability import SETUPS as PYEPO_SETUPS, train_two_stage as cap_train_ts, \
    train_polystep as cap_train_ps
# --- interface B: constraint-prediction problems (problems.py + methods.py) ---
from pto.problems import MDKPConsumption, KnapsackWeights
from pto.methods import train_two_stage as prob_train_ts, train_polystep as prob_train_ps
from pto.multiseed import summarize, md_table, write_json, write_md

try:
    from scipy.stats import rankdata as _rankdata
except Exception:  # pragma: no cover
    _rankdata = None

CAT = {"sp": "shortest_path (LP)", "knap": "knapsack (ILP)", "tsp": "tsp (ILP)",
       "port": "portfolio (SOCP)", "mdkp": "MDKP-consumption (constraint pred.)",
       "knapw": "knapsack-weights (constraint pred.)"}

PYEPO_PROBLEMS = set(PYEPO_SETUPS)            # {"sp","knap","tsp","port"}
CONSTRAINT_PROBLEMS = {"mdkp", "knapw"}
ALL_PROBLEMS = ["sp", "knap", "tsp", "port", "mdkp", "knapw"]

# constraint-regime PolyStep base cfg (mirrors exp_manyconstraints / exp_constraint_scaling CFG_CONS);
# polytope_type and num_probe are overridden per cell.
_CONS_BASE = dict(use_momentum=True, momentum_init=0.5, momentum_final=0.9,
                  epsilon=CosineEpsilon(1.0, 0.08), step_radius=1.0, probe_radius=2.0)
_CONS_STEPS = 150


def _rank(means):
    """Ascending average ranks (1 = best/lowest). NaN -> worst. Ties averaged."""
    a = np.asarray([m if np.isfinite(m) else np.inf for m in means], dtype=float)
    if _rankdata is not None:
        return _rankdata(a, method="average")
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a)); r[order] = np.arange(1, len(a) + 1)
    return r


# ---------------- per-cell runners ----------------
def _run_cell_pyepo(pname, poly, K, seeds, deg):
    regs, walls = [], []
    for seed in seeds:
        t0 = time.time()
        try:
            cfg, _ = PYEPO_SETUPS[pname](seed, deg)
            cfg["warm"] = cap_train_ts(cfg)
            cfg["ps_polytope"] = poly
            cfg["ps_num_probe"] = K
            r = float(metric.regret(cap_train_ps(cfg), cfg["om"], cfg["ld_te"]))
        except Exception as e:
            print(f"      ! {pname}/{poly}/K{K}/seed{seed} failed: {e}", flush=True)
            r = float("nan")
        regs.append(r); walls.append(time.time() - t0)
    return regs, walls


def _make_constraint_problem(pname, seed, deg):
    if pname == "mdkp":
        return MDKPConsumption(deg=deg, seed=seed)
    if pname == "knapw":
        return KnapsackWeights(deg=deg, seed=seed)
    raise KeyError(pname)


def _run_cell_constraint(pname, poly, K, seeds, deg):
    regs, walls = [], []
    for seed in seeds:
        t0 = time.time()
        try:
            prob = _make_constraint_problem(pname, seed, deg)
            ts = prob_train_ts(prob)
            cfg = dict(_CONS_BASE, polytope_type=poly, num_probe=K)
            m = prob_train_ps(prob, cfg, steps=_CONS_STEPS, warm=ts, seed=seed)
            r = float(prob.regret(m))
        except Exception as e:
            print(f"      ! {pname}/{poly}/K{K}/seed{seed} failed: {e}", flush=True)
            r = float("nan")
        regs.append(r); walls.append(time.time() - t0)
    return regs, walls


def _run_cell(pname, poly, K, seeds, deg):
    if pname in PYEPO_PROBLEMS:
        return _run_cell_pyepo(pname, poly, K, seeds, deg)
    if pname in CONSTRAINT_PROBLEMS:
        return _run_cell_constraint(pname, poly, K, seeds, deg)
    raise KeyError(f"unknown problem '{pname}'")


# ---------------- driver ----------------
def run(problems, polytopes, probes, seeds, deg=4):
    results = {}
    for p in problems:
        print(f"\n[{p}] {CAT.get(p, p)}  polytopes={polytopes} probes={probes} seeds={seeds}", flush=True)
        cells = {}
        for poly in polytopes:
            for K in probes:
                regs, walls = _run_cell(p, poly, K, seeds, deg)
                s = summarize(regs)
                s["wall_mean"] = float(np.mean(walls)) if walls else float("nan")
                s["wall_total"] = float(np.sum(walls)) if walls else 0.0
                key = f"{poly}/K{K}"
                cells[key] = s
                print(f"  {key:14s} regret={s['mean']:.4f}±{s['std']:.4f}  "
                      f"(n={s['n']}, {s['wall_mean']:.1f}s/seed)", flush=True)
        results[p] = {"cat": CAT.get(p, p), "cells": cells}

    trend = compute_trend(results, polytopes, probes)
    payload = {"problems": problems, "polytopes": polytopes, "probes": probes,
               "seeds": seeds, "deg": deg, "results": results, "trend": trend}
    # Top-level combined file (literal artifact; used by the full local run + smoke).
    write_json("exp_results/config_sweep.json", payload)
    write_md("exp_results/config_sweep.md", to_md(payload))
    # Per-run SHARD (collision-free) so per-problem parallel cluster jobs never
    # clobber each other; merge_shards() rebuilds the combined file at fetch time.
    shard = "_".join(problems)
    write_json(f"exp_results/config_sweep/{shard}.json", payload)
    write_md(f"exp_results/config_sweep/{shard}.md", to_md(payload))
    print(f"\nwrote exp_results/config_sweep.{{json,md}} and "
          f"exp_results/config_sweep/{shard}.{{json,md}}\nDONE", flush=True)
    return payload


def merge_shards(shard_dir="exp_results/config_sweep"):
    """Combine all per-problem shard JSONs in shard_dir into one payload + md, in the
    canonical problem order. Run after fetching parallel per-problem cluster jobs:
        .venv/bin/python -c 'import exp_config_sweep as e; e.merge_shards()'
    """
    import glob, json
    results, polytopes, probes, seeds, deg = {}, None, None, None, 4
    for f in sorted(glob.glob(os.path.join(shard_dir, "*.json"))):
        d = json.load(open(f))
        results.update(d["results"])
        polytopes = d["polytopes"]; probes = d["probes"]; seeds = d["seeds"]; deg = d["deg"]
    if not results:
        print(f"no shards found in {shard_dir}"); return None
    problems = [p for p in ALL_PROBLEMS if p in results] + \
               [p for p in results if p not in ALL_PROBLEMS]
    ordered = {p: results[p] for p in problems}
    trend = compute_trend(ordered, polytopes, probes)
    payload = {"problems": problems, "polytopes": polytopes, "probes": probes,
               "seeds": seeds, "deg": deg, "results": ordered, "trend": trend}
    write_json("exp_results/config_sweep.json", payload)
    write_md("exp_results/config_sweep.md", to_md(payload))
    print(f"merged {len(problems)} problems -> exp_results/config_sweep.{{json,md}}")
    return payload


def compute_trend(results, polytopes, probes):
    """Aggregate across problems using RANKS so different regret scales combine fairly.

    Per problem we compute a polytope-marginal (mean regret over probes) and a
    probe-marginal (mean regret over polytopes), rank them (1=best), and average
    ranks / count #1s across problems. We also track which FULL (polytope,probe)
    cell is best per problem.
    """
    probs = list(results)
    poly_ranks = {po: [] for po in polytopes}
    poly_wins = {po: 0 for po in polytopes}
    probe_ranks = {k: [] for k in probes}
    probe_wins = {k: 0 for k in probes}
    cell_wins = {}                     # "poly/Kk" -> count of problems where it is the single best cell
    best_cell_per_problem = {}

    for p in probs:
        cells = results[p]["cells"]

        def cmean(po, k):
            c = cells.get(f"{po}/K{k}")
            return c["mean"] if c and c["n"] else float("nan")

        # polytope marginals (mean over probes)
        poly_marg = [np.nanmean([cmean(po, k) for k in probes]) for po in polytopes]
        pr = _rank(poly_marg)
        for po, rk in zip(polytopes, pr):
            poly_ranks[po].append(float(rk))
        poly_wins[polytopes[int(np.argmin([m if np.isfinite(m) else np.inf for m in poly_marg]))]] += 1

        # probe marginals (mean over polytopes)
        probe_marg = [np.nanmean([cmean(po, k) for po in polytopes]) for k in probes]
        kr = _rank(probe_marg)
        for k, rk in zip(probes, kr):
            probe_ranks[k].append(float(rk))
        probe_wins[probes[int(np.argmin([m if np.isfinite(m) else np.inf for m in probe_marg]))]] += 1

        # best full cell
        keys = [f"{po}/K{k}" for po in polytopes for k in probes]
        vals = [cells[key]["mean"] if cells.get(key) and cells[key]["n"] else float("nan") for key in keys]
        bk = keys[int(np.argmin([v if np.isfinite(v) else np.inf for v in vals]))]
        best_cell_per_problem[p] = bk
        cell_wins[bk] = cell_wins.get(bk, 0) + 1

    poly_meanrank = {po: (float(np.mean(poly_ranks[po])) if poly_ranks[po] else float("nan")) for po in polytopes}
    probe_meanrank = {k: (float(np.mean(probe_ranks[k])) if probe_ranks[k] else float("nan")) for k in probes}

    # recommendations: lowest mean rank (tie-break: most wins)
    rec_poly = min(polytopes, key=lambda po: (poly_meanrank[po], -poly_wins[po]))
    rec_probe = min(probes, key=lambda k: (probe_meanrank[k], -probe_wins[k]))

    return {"n_problems": len(probs),
            "polytope_wins": poly_wins, "polytope_mean_rank": poly_meanrank,
            "probe_wins": probe_wins, "probe_mean_rank": probe_meanrank,
            "best_cell_per_problem": best_cell_per_problem, "cell_wins": cell_wins,
            "recommended_polytope": rec_poly, "recommended_probe": rec_probe}


def to_md(payload):
    polytopes, probes = payload["polytopes"], payload["probes"]
    L = ["# PolyStep configuration sweep -- polytope x probe-count x seeds (robustness study)", "",
         f"problems={payload['problems']}, polytopes={polytopes}, probes={probes}, "
         f"seeds={payload['seeds']}, deg={payload['deg']}.", "",
         "Normalized regret (lower is better), mean±std over seeds. One row per "
         "(polytope, probe) cell.", ""]

    # per-problem tables (rows = polytope x probe; cols = regret mean±std, wall)
    for p in payload["problems"]:
        cells = payload["results"][p]["cells"]
        L.append(f"## {payload['results'][p]['cat']}  (`{p}`)")
        rows = []
        for po in polytopes:
            for k in probes:
                c = cells.get(f"{po}/K{k}")
                reg = f"{c['mean']:.4f}±{c['std']:.4f}" if c and c["n"] else "n/a"
                wall = f"{c['wall_mean']:.1f}" if c and c.get("wall_mean") == c.get("wall_mean") else "-"
                rows.append([po, str(k), reg, wall])
        L.append(md_table(["polytope", "num_probe", "regret (mean±std)", "wall (s/seed)"], rows))
        bc = payload["trend"]["best_cell_per_problem"].get(p, "?")
        L.append(f"\n_best cell: **{bc}**_\n")

    # ---- trend analysis ----
    t = payload["trend"]
    L.append("## TREND ANALYSIS (aggregated across problems via ranks)")
    L.append("")
    L.append(f"Aggregated over {t['n_problems']} problems. Ranks computed PER PROBLEM "
             "(1 = best) on marginal mean regret, then averaged, so problems with "
             "different regret scales combine fairly. Lower mean-rank = better.")
    L.append("")

    L.append("### Polytope (marginal = mean regret over probe-counts)")
    prows = [[po, str(t["polytope_wins"][po]), f"{t['polytope_mean_rank'][po]:.2f}"]
             for po in polytopes]
    L.append(md_table(["polytope", "#problems best", "mean rank"], prows))
    L.append("")

    L.append("### Probe-count (marginal = mean regret over polytopes)")
    krows = [[str(k), str(t["probe_wins"][k]), f"{t['probe_mean_rank'][k]:.2f}"]
             for k in probes]
    L.append(md_table(["num_probe", "#problems best", "mean rank"], krows))
    L.append("")

    L.append("### Best full (polytope, probe) cell -- count of problems")
    cw = sorted(t["cell_wins"].items(), key=lambda kv: -kv[1])
    L.append(md_table(["cell", "#problems best"], [[c, str(n)] for c, n in cw]))
    L.append("")

    # ---- recommendation ----
    rp, rk = t["recommended_polytope"], t["recommended_probe"]
    L.append("### Recommended default")
    L.append("")
    L.append(f"**polytope = `{rp}`**, **num_probe = `{rk}`**.")
    L.append("")
    L.append(f"Evidence: `{rp}` has the lowest mean rank "
             f"({t['polytope_mean_rank'][rp]:.2f}) among polytopes and wins "
             f"{t['polytope_wins'][rp]}/{t['n_problems']} problems; "
             f"num_probe=`{rk}` has the lowest mean rank "
             f"({t['probe_mean_rank'][rk]:.2f}) among probe-counts and wins "
             f"{t['probe_wins'][rk]}/{t['n_problems']} problems.")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    problems = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ALL_PROBLEMS)
    polytopes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["orthoplex", "simplex", "cube"]
    probes = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 3, 5, 8]
    seeds = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1, 2, 3, 4]
    deg = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    run(problems, polytopes, probes, seeds, deg)
