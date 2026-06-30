"""Experiment #1 -- Misspecification x problem-class phase diagram (HKM/Elmachtoub boundary overlaid).

Maps WHERE decision-focused learning helps, WHERE SPO+ helps, and (separately) where the surrogate
camp is structurally N/A. Grid = misspecification degree (x) x problem class (y, ordered by surrogate
looseness LP -> ILP -> SOCP). Per cell: two-stage / SPO+ / IMLE / SFGE / PolyStep, multi-seed,
PyEPO + Gurobi-evaluated normalized regret. All methods share the SAME two-stage warm start (fair).

Theory overlay (verified this session):
  HKM (Hu-Kallus-Mao, Mgmt Sci 2022, arXiv:2011.03030): on the LP/polyhedral row, two-stage/ETO gets
  a FAST rate under a margin (near-degeneracy) condition while end-to-end/SPO+ is stuck at the SLOW
  n^-1/2. Phase behavior: well-specified -> two-stage wins; misspecified-but-simple -> DFL wins;
  flexible model -> two-stage wins again. Rigorous ONLY on the LP row; for ILP/SOCP "no separation
  proven" and the DFL-helps region expands. We therefore (a) report deg=1 separately (the well-spec
  left edge) because the ILP advantage is CONFOUNDED with decision-sensitivity, and (b) provide a
  model-capacity arm (exp1_capacity.py) to exhibit the flexible-model recovery.

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp1_phase_diagram.py [problems] [degs] [seeds]
      e.g. ... exp1_phase_diagram.py sp,knap,tsp,port 1,2,4,6,8 0,1,2,3,4
"""
from __future__ import annotations
import sys
sys.path.insert(0, "polystep/src")
import numpy as np
from pyepo import metric
from pto.capability import (SETUPS, train_two_stage, train_sfge, train_polystep, dev)
from challenge_established import train_dfl_warm
from pto.seeding import seed_everything
from pto.multiseed import summarize, wilcoxon_pair, md_table, write_json, write_md, fmt_mean_std

PANEL = ["two-stage", "SPO+", "IMLE", "SFGE", "PolyStep"]       # SPO+ antagonist, IMLE differentiable floor
SURR = ["SPO+", "IMLE"]                                         # trained via train_dfl_warm
GRADFREE = ["SFGE", "PolyStep"]
CATLABEL = {"sp": "shortest_path (LP)", "knap": "knapsack (ILP)", "tsp": "tsp (ILP)", "port": "portfolio (SOCP)"}


def cell_detailed(setup, deg, seeds):
    """Return {method: [per-seed regret]} with a shared, reproducible warm start per seed."""
    acc = {m: [] for m in PANEL}
    for seed in seeds:
        seed_everything(seed)                                  # reproducible model init; data seeded in setup
        cfg, _ = setup(seed, deg)
        ts = train_two_stage(cfg); cfg["warm"] = ts            # same warm start for everyone
        acc["two-stage"].append(metric.regret(ts, cfg["om"], cfg["ld_te"]))
        for name in SURR:
            try:
                acc[name].append(metric.regret(train_dfl_warm(cfg, name, ts), cfg["om"], cfg["ld_te"]))
            except Exception:
                acc[name].append(float("nan"))
        acc["SFGE"].append(metric.regret(train_sfge(cfg), cfg["om"], cfg["ld_te"]))
        acc["PolyStep"].append(metric.regret(train_polystep(cfg), cfg["om"], cfg["ld_te"]))
    return acc


def analyze(acc):
    """Per-cell summary: means, DFL-advantage vs two-stage and vs SPO+, winner, Wilcoxon(best_gf<SPO+)."""
    summ = {m: summarize(acc[m]) for m in PANEL}
    means = {m: summ[m]["mean"] for m in PANEL}
    best_gf = min(GRADFREE, key=lambda m: means[m])
    ts_, spo_ = means["two-stage"], means["SPO+"]
    adv_ts = (ts_ - means[best_gf]) / ts_ if ts_ and not np.isnan(ts_) else float("nan")
    adv_spo = (spo_ - means[best_gf]) / spo_ if spo_ and not np.isnan(spo_) else float("nan")
    winner = min(PANEL, key=lambda m: means[m] if not np.isnan(means[m]) else float("inf"))
    p_gf_spo = wilcoxon_pair(acc[best_gf], acc["SPO+"], alternative="less")
    return {"summary": summ, "best_gradfree": best_gf, "adv_vs_two_stage": adv_ts,
            "adv_vs_spo": adv_spo, "winner": winner, "wilcoxon_bestgf_lt_spo": p_gf_spo}


def run_grid(problems, degs, seeds):
    results = {}
    for p in problems:
        for deg in degs:
            print(f"[grid] {p} deg={deg} ...", flush=True)
            acc = cell_detailed(SETUPS[p], deg, seeds)
            results[f"{p}|{deg}"] = {"problem": p, "deg": deg, **analyze(acc)}
            a = results[f"{p}|{deg}"]
            print(f"    winner={a['winner']}  best_gf={a['best_gradfree']}  "
                  f"adv_vs_two_stage={a['adv_vs_two_stage']:+.1%}  adv_vs_SPO+={a['adv_vs_spo']:+.1%}  "
                  f"(p={a['wilcoxon_bestgf_lt_spo']})", flush=True)
    return results


def to_markdown(results, problems, degs, seeds):
    L = ["# Experiment #1 -- Misspecification x problem-class phase diagram",
         "",
         f"PyEPO data + Gurobi normalized regret (lower better). Seeds={seeds}. All methods share the "
         "same two-stage warm start.", "",
         "**HKM overlay (LP row only):** two-stage/ETO is fast-rate-optimal at the well-specified left "
         "edge; DFL helps in the misspecified-but-simple band. For ILP/SOCP no fast-rate separation is "
         "proven and the DFL-helps region expands. deg=1 reported separately because the ILP advantage "
         "is confounded with decision-sensitivity.", ""]
    # main per-method regret table
    L.append("## Per-cell normalized regret (mean±std)")
    headers = ["problem", "deg"] + PANEL + ["winner", "best_gf vs SPO+ (p)"]
    rows = []
    for p in problems:
        for deg in degs:
            r = results.get(f"{p}|{deg}")
            if not r:
                continue
            cells = [fmt_mean_std(r["summary"][m]) for m in PANEL]
            best = r["winner"]
            cells = [f"**{c}**" if PANEL[i] == best else c for i, c in enumerate(cells)]
            p_ = r["wilcoxon_bestgf_lt_spo"]
            rows.append([CATLABEL[p], deg] + cells + [best, f"{p_:.3f}" if p_ is not None else "—"])
    L.append(md_table(headers, rows))
    # advantage heatmap-as-table
    L += ["", "## DFL advantage: best gradient-free vs two-stage and vs SPO+ (regret reduction)"]
    h2 = ["problem"] + [f"deg={d}" for d in degs]
    rt, rs = ["vs two-stage"], ["vs SPO+"]
    rows_ts, rows_spo = [], []
    for p in problems:
        rt = [CATLABEL[p]]; rs = [CATLABEL[p]]
        for deg in degs:
            r = results.get(f"{p}|{deg}")
            rt.append(f"{r['adv_vs_two_stage']:+.0%}" if r else "—")
            rs.append(f"{r['adv_vs_spo']:+.0%}" if r else "—")
        rows_ts.append(rt); rows_spo.append(rs)
    L.append("**vs two-stage:**"); L.append(md_table(h2, rows_ts))
    L.append(""); L.append("**vs SPO+:**"); L.append(md_table(h2, rows_spo))
    return "\n".join(L)


def main():
    problems = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sp", "knap", "tsp", "port"]
    degs = [int(d) for d in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 4, 6, 8]
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2, 3, 4]
    print(f"PHASE DIAGRAM | problems={problems} degs={degs} seeds={seeds} | panel={PANEL}", flush=True)
    results = run_grid(problems, degs, seeds)
    payload = {"problems": problems, "degs": degs, "seeds": seeds, "panel": PANEL, "results": results}
    write_json("exp_results/phase_diagram.json", payload)
    write_md("exp_results/phase_diagram.md", to_markdown(results, problems, degs, seeds))
    print("\nwrote exp_results/phase_diagram.{json,md}\nDONE", flush=True)


if __name__ == "__main__":
    main()
