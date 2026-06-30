"""Experiment #3 -- PolyStep vs SFGE bias-variance study (>=20 seeds).

Resolves the open "PolyStep ~= SFGE peers" question by separating the two gradient-free methods on
their DISTINCT profiles. Hypotheses: (H1) PolyStep is lower-variance / more deterministic across seeds;
(H2) SFGE is more sample-efficient at low n. SPO+ (surrogate reference) + two-stage (floor) included.

Three views:
  (A) bias-variance table  : >=20 seeds, each method at its OWN sweep-best hyperparameter (so spread is
      not a tuning artifact); report mean (bias) and std (across-seed variance) of normalized regret.
  (B) optimizer variance   : fix the DATA (one seed), re-run SFGE & PolyStep across optimizer/sampling
      seeds -> std isolates the trainer's intrinsic stochasticity (SFGE's REINFORCE sampling noise vs
      PolyStep's random-rotation noise) from data variance.
  (C) sample-size curve     : knapsack, n in {50..3000}, multi-seed -> the sample-efficiency curve.

Hyperparameter-sensitivity (the 2nd variance axis) is already quantified in LR_SWEEP_RESULTS.md
(PolyStep's step_radius is a bounded 1-decade knob vs SPO+/SFGE lr over 2+ decades); we reuse those
best-hp settings here rather than recomputing.

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp3_bias_variance.py [problems] [deg] [nseeds]
"""
from __future__ import annotations
import sys
sys.path.insert(0, "polystep/src")
import numpy as np
from pyepo import metric
from pto.capability import SETUPS, train_two_stage, train_sfge, dev
from pto.sweep_lr import train_spoplus_lr, train_polystep_sr
from pto.seeding import seed_everything
from pto.multiseed import summarize, wilcoxon_pair, md_table, write_json, write_md, fmt_mean_std

# each method's sweep-best hyperparameter per problem (from LR_SWEEP_RESULTS.md / sweep_lr_polystep.json)
BEST_HP = {
    "sp":   {"spo_lr": 3e-3, "sfge_lr": 1e-1, "ps_sr": 0.2},
    "knap": {"spo_lr": 3e-2, "sfge_lr": 3e-2, "ps_sr": 0.8},
    "tsp":  {"spo_lr": 1e-1, "sfge_lr": 3e-2, "ps_sr": 0.8},
    "port": {"spo_lr": 1e-3, "sfge_lr": 3e-3, "ps_sr": 0.1},
}
METHODS = ["two-stage", "SPO+", "SFGE", "PolyStep"]
CATLABEL = {"sp": "shortest_path (LP)", "knap": "knapsack (ILP)", "tsp": "tsp (ILP)", "port": "portfolio (SOCP)"}


def _train_all(cfg, hp, seed):
    """Return regret of each method at its best hp; all warm-started from the same two-stage."""
    ts = train_two_stage(cfg); cfg["warm"] = ts
    out = {"two-stage": metric.regret(ts, cfg["om"], cfg["ld_te"])}
    try:
        out["SPO+"] = metric.regret(train_spoplus_lr(cfg, ts, hp["spo_lr"]), cfg["om"], cfg["ld_te"])
    except Exception:
        out["SPO+"] = float("nan")
    cc = dict(cfg); cc["warm"] = ts
    out["SFGE"] = metric.regret(train_sfge(cc, lr=hp["sfge_lr"]), cfg["om"], cfg["ld_te"])
    out["PolyStep"] = metric.regret(train_polystep_sr(cfg, ts, hp["ps_sr"]), cfg["om"], cfg["ld_te"])
    return out


def bias_variance(problem, deg, seeds):
    hp = BEST_HP[problem]
    acc = {m: [] for m in METHODS}
    for seed in seeds:
        seed_everything(seed)
        cfg, _ = SETUPS[problem](seed, deg)
        r = _train_all(cfg, hp, seed)
        for m in METHODS:
            acc[m].append(r[m])
        print(f"    seed {seed}: " + " ".join(f"{m}={r[m]:.4f}" for m in METHODS), flush=True)
    summ = {m: summarize(acc[m]) for m in METHODS}
    return {"summary": summ, "raw": acc,
            "wilcoxon_PS_lt_SFGE": wilcoxon_pair(acc["PolyStep"], acc["SFGE"]),
            "wilcoxon_SFGE_lt_PS": wilcoxon_pair(acc["SFGE"], acc["PolyStep"])}


def optimizer_variance(problem, deg, data_seed, opt_seeds):
    """Fix the data; vary only the optimizer/sampling seed -> intrinsic trainer stochasticity."""
    hp = BEST_HP[problem]
    seed_everything(data_seed)
    cfg, _ = SETUPS[problem](data_seed, deg)
    ts = train_two_stage(cfg); cfg["warm"] = ts
    sfge_v, ps_v = [], []
    for s in opt_seeds:
        seed_everything(s)
        cc = dict(cfg); cc["warm"] = ts
        sfge_v.append(metric.regret(train_sfge(cc, lr=hp["sfge_lr"]), cfg["om"], cfg["ld_te"]))
        cc2 = dict(cfg); cc2["seed"] = s
        ps_v.append(metric.regret(train_polystep_sr(cc2, ts, hp["ps_sr"]), cfg["om"], cfg["ld_te"]))
    return {"SFGE": summarize(sfge_v), "PolyStep": summarize(ps_v)}


def sample_size(problem, deg, ns, seeds):
    hp = BEST_HP[problem]
    curve = {}
    for n in ns:
        acc = {m: [] for m in METHODS}
        for seed in seeds:
            seed_everything(seed)
            cfg, _ = SETUPS[problem](seed, deg, n_train=n)
            r = _train_all(cfg, hp, seed)
            for m in METHODS:
                acc[m].append(r[m])
        curve[n] = {m: summarize(acc[m]) for m in METHODS}
        print(f"  n={n:>4}: " + " ".join(f"{m}={curve[n][m]['mean']:.4f}" for m in METHODS), flush=True)
    return curve


def main():
    problems = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sp", "knap", "port"]
    deg = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    nseeds = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    seeds = list(range(nseeds))
    print(f"BIAS-VARIANCE | problems={problems} deg={deg} nseeds={nseeds} | each method @ sweep-best hp", flush=True)
    bv = {}
    for p in problems:
        print(f"[bias-var] {p}", flush=True)
        bv[p] = bias_variance(p, deg, seeds)
    print("[optimizer-variance] knap (fixed data, vary optimizer seed)", flush=True)
    ov = optimizer_variance("knap", deg, 0, list(range(10))) if "knap" in problems else {}
    print("[sample-size] knap", flush=True)
    ss = sample_size("knap", deg, [50, 100, 200, 500, 1000, 3000], list(range(5))) if "knap" in problems else {}
    payload = {"problems": problems, "deg": deg, "nseeds": nseeds, "best_hp": BEST_HP,
               "bias_variance": bv, "optimizer_variance": ov, "sample_size": ss}
    write_json("exp_results/bias_variance.json", payload)
    write_md("exp_results/bias_variance.md", to_markdown(bv, ov, ss, problems, deg, nseeds))
    print("\nwrote exp_results/bias_variance.{json,md}\nDONE", flush=True)


def to_markdown(bv, ov, ss, problems, deg, nseeds):
    L = [f"# Experiment #3 -- PolyStep vs SFGE bias-variance ({nseeds} seeds)", "",
         f"deg={deg}. Each method at its own sweep-best hyperparameter. Normalized regret (lower better); "
         "**std = across-seed variance** is the headline.", "",
         "## (A) Bias (mean) and variance (std) across seeds"]
    headers = ["problem"] + [f"{m} (mean±std)" for m in METHODS] + ["PS<SFGE p", "SFGE<PS p"]
    rows = []
    for p in problems:
        s = bv[p]["summary"]
        rows.append([CATLABEL[p]] + [fmt_mean_std(s[m]) for m in METHODS] +
                    [f"{bv[p]['wilcoxon_PS_lt_SFGE']:.3f}" if bv[p]['wilcoxon_PS_lt_SFGE'] is not None else "—",
                     f"{bv[p]['wilcoxon_SFGE_lt_PS']:.3f}" if bv[p]['wilcoxon_SFGE_lt_PS'] is not None else "—"])
    L.append(md_table(headers, rows))
    L += ["", "**Variance (std) only** — the H1 test (PolyStep lower-variance?):"]
    rows2 = [[CATLABEL[p]] + [f"{bv[p]['summary'][m]['std']:.4f}" for m in METHODS] for p in problems]
    L.append(md_table(["problem"] + METHODS, rows2))
    if ov:
        L += ["", "## (B) Optimizer-only variance (fixed data, vary optimizer/sampling seed) -- knapsack",
              md_table(["method", "mean±std", "std"],
                       [[m, fmt_mean_std(ov[m]), f"{ov[m]['std']:.4f}"] for m in ("SFGE", "PolyStep")])]
    if ss:
        L += ["", "## (C) Sample-efficiency curve -- knapsack (mean regret)"]
        ns = sorted(ss.keys())
        rows3 = [[m] + [f"{ss[n][m]['mean']:.4f}" for n in ns] for m in METHODS]
        L.append(md_table(["method"] + [f"n={n}" for n in ns], rows3))
    return "\n".join(L)


if __name__ == "__main__":
    main()
