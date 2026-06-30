"""Experiment #4 -- Prediction-in-CONSTRAINTS: PolyStep vs SFGE (the strongest-novelty experiment).

Two problems where the PREDICTED quantity defines the feasible region, so SPO+ / cvxpylayers / PFYL /
IMLE cannot be FORMULATED (no fixed S, no objective cost vector). Only methods that evaluate the
realized outcome of a deployed decision run: two-stage (MSE), SFGE, PolyStep.

  (1) fractional knapsack -- item VALUES v known/fixed; per-item WEIGHTS predicted (in the constraint);
      capacity C known. deploy = batched fractional-knapsack greedy on predicted weights; realize =
      v^T x - lambda * max(0, w_true^T x - C). Sweep lambda in {1,5,20} (cost of infeasibility).
  (2) capacitated newsvendor -- per-item DEMAND predicted (caps each order: box constraint q_i<=dhat_i);
      shared budget C; per-item criticality b known. deploy = batched newsvendor greedy on predicted
      demand; realize asymmetric cost = sum_i [b_i (d_i-q_i)^+ + h_i (q_i-d_i)^+] on TRUE demand.
      Sweep overage penalty h in {1,5,20} (cost of over-prediction).

Baselines: SFGE + strong-Adam two-stage (the only structurally-applicable methods). The SOTA
constraint-DFL methods (ODECE NeurIPS'25, Branch&Learn, Post-hoc Regret -- cloned under baselines/)
are per-instance / non-batchable references; the surrogate camp is N/A. DFF (arXiv:2501.01874) trains
its correction with SPO+ (objective-only) -> a foil, not a rival.

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp4_constraints.py [problems] [deg] [seeds]
      problems in {fk,nv}; e.g. ... exp4_constraints.py fk,nv 4 0,1,2,3,4
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, "polystep/src")
import numpy as np
import torch
import torch.nn as nn
from pyepo.data import knapsack
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from pto.solvers import solve_fractional_knapsack, solve_newsvendor_cap
from pto.seeding import seed_everything, device_generator
from pto.multiseed import summarize, wilcoxon_pair, md_table, write_json, write_md, fmt_mean_std

dev = "cuda" if torch.cuda.is_available() else "cpu"
PF = 5; NIT = 16; NTR = 400
# --- polytope x probe sweep (env-overridable; defaults preserve original behavior) ---
PS_POLYTOPE = os.environ.get("PS_POLYTOPE", "orthoplex")
PS_PROBES = int(os.environ.get("PS_PROBES", "1"))
OUT_TAG = os.environ.get("OUT_TAG", "")
_OUT_SFX = f"_{OUT_TAG}" if OUT_TAG else ""


def make_pred(n):
    return nn.Linear(PF, n, bias=True).to(dev)      # bias lets it predict conservatively (see gotchas)


def gen(seed, deg):
    """Features x and a positive per-instance quantity (weights / demand) with deg misspecification."""
    Vfix, _, _ = knapsack.genData(2, PF, NIT, dim=1, deg=1, seed=1)
    v = torch.tensor(Vfix[0], dtype=torch.float32, device=dev)            # known item values / criticality base
    _, x, q = knapsack.genData(NTR + 600, PF, NIT, dim=1, deg=deg, noise_width=0, seed=seed)
    X = torch.tensor(x, dtype=torch.float32, device=dev)
    Q = torch.tensor(q, dtype=torch.float32, device=dev)                  # true weights (fk) / true demand (nv)
    return X[:NTR], Q[:NTR], X[NTR:], Q[NTR:], v


# --------------------------- realized objectives (sense) ---------------------------
def fk_realized(x, w_true, v, C, lam):
    """fractional knapsack realized VALUE (maximize). x (...,n), w_true (B,n) broadcasts."""
    value = (x * v).sum(-1)
    overflow = (x * w_true).sum(-1) - C
    return value - lam * overflow.clamp(min=0)


def nv_realized(q, d_true, b, h, C):
    """newsvendor realized COST (minimize). q (...,n), d_true (B,n) broadcasts; b,h (n,)."""
    return (b * (d_true - q).clamp(min=0) + h * (q - d_true).clamp(min=0)).sum(-1)


# --------------------------- trainers (bias predictor, warm-started) ---------------------------
def train_two_stage(Xtr, Ytr, n, epochs=60):
    m = make_pred(n); opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(epochs):
        opt.zero_grad(); ((m(Xtr) - Ytr) ** 2).mean().backward(); opt.step()
    return m


def train_polystep(Xtr, n, closure_obj, warm, scale, steps=200, seed=0):
    """closure_obj(dec)->(K,B) realized objective to MINIMIZE (already signed); dec (K,B,n)."""
    m = make_pred(n)
    with torch.no_grad():
        m.weight.copy_(warm.weight); m.bias.copy_(warm.bias)
    pso = PolyStepOptimizer(m, polytope_type=PS_POLYTOPE, epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=0.4, probe_radius=0.8, num_probe=PS_PROBES, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)

    def closure(bp):
        pred = torch.einsum("mnf,bf->mbn", bp["weight"], Xtr) + bp["bias"].unsqueeze(1)
        M, B, nn_ = pred.shape
        dec = SOLVE(pred.reshape(M * B, nn_)).reshape(M, B, nn_)
        return (closure_obj(dec) / scale).mean(-1)
    for _ in range(steps):
        pso.step(closure)
    return m


def train_sfge(Xtr, n, closure_obj, warm, scale, epochs=200, n_samples=8, sigma=0.5, lr=1e-2, seed=0):
    m = make_pred(n)
    with torch.no_grad():
        m.weight.copy_(warm.weight); m.bias.copy_(warm.bias)
    opt = torch.optim.Adam(m.parameters(), lr); g = device_generator(seed, dev)
    for _ in range(epochs):
        pred = m(Xtr)
        with torch.no_grad():
            eps = torch.randn(n_samples, *pred.shape, device=dev, generator=g)
            chat = pred.unsqueeze(0) + sigma * eps
            S, B, nn_ = chat.shape
            dec = SOLVE(chat.reshape(S * B, nn_)).reshape(S, B, nn_)
            r = closure_obj(dec) / scale                          # (S,B) minimize
            adv = r - r.mean(0, keepdim=True)
        logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
        surrogate = (adv * logp).mean()
        opt.zero_grad(); surrogate.backward(); opt.step()
    return m


# SOLVE is set per-problem (module-level so the closures pick it up without re-plumbing)
SOLVE = None


def run_problem(prob, deg, seeds, costs):
    global SOLVE
    rows = {}
    for cost in costs:                                            # lambda (fk) or overage h (nv)
        acc = {m: [] for m in ("two-stage", "SFGE", "PolyStep")}
        for seed in seeds:
            seed_everything(seed)
            Xtr, Qtr, Xte, Qte, v = gen(seed, deg)
            if prob == "fk":
                C = float(Qtr.sum(-1).mean() * 0.5)               # binding capacity
                SOLVE = lambda p: solve_fractional_knapsack(v.unsqueeze(0).expand(p.shape[0], -1), p.clamp(min=1e-3), C)
                scale = float(v.sum())
                obj = lambda dec, W=Qtr: -fk_realized(dec, W, v, C, cost)        # maximize value -> minimize neg
                ev_obj = lambda dec, W: fk_realized(dec, W, v, C, cost)
                sense = "max"
            else:  # nv
                b = v.clamp(min=0.2)                              # per-item criticality (known)
                h = float(cost)                                   # overage penalty (swept)
                C = float(Qtr.sum(-1).mean() * 0.7)               # binding shared budget
                SOLVE = lambda p: solve_newsvendor_cap(p.clamp(min=0.0), b, C)
                scale = float((b * Qtr).sum(-1).mean())
                obj = lambda dec, D=Qtr: nv_realized(dec, D, b, h, C)            # minimize cost
                ev_obj = lambda dec, D: nv_realized(dec, D, b, h, C)
                sense = "min"
            nv_b = b if prob == "nv" else None
            ts = train_two_stage(Xtr, Qtr, NIT)
            ps = train_polystep(Xtr, NIT, obj, ts, scale, seed=seed)
            sf = train_sfge(Xtr, NIT, obj, ts, scale, seed=seed)
            acc["two-stage"].append(evaluate(ts, Xte, Qte, ev_obj, sense, nv_b))
            acc["SFGE"].append(evaluate(sf, Xte, Qte, ev_obj, sense, nv_b))
            acc["PolyStep"].append(evaluate(ps, Xte, Qte, ev_obj, sense, nv_b))
        summ = {m: summarize(acc[m]) for m in acc}
        best = min(summ, key=lambda m: summ[m]["mean"])
        p_ps = wilcoxon_pair(acc["PolyStep"], acc["two-stage"])
        p_sf = wilcoxon_pair(acc["SFGE"], acc["two-stage"])
        rows[cost] = {"summary": summ, "best": best, "p_polystep_lt_ts": p_ps, "p_sfge_lt_ts": p_sf}
        print(f"  cost={cost:>4}: " + "  ".join(f"{m}={fmt_mean_std(summ[m])}" for m in acc) +
              f"  best={best}", flush=True)
    return rows


def evaluate(m, Xte, Qte, ev_obj, sense, nv_b=None):
    """Normalized realized regret (aggregate, PyEPO-style). For min-sense (newsvendor) the oracle cost
    can be 0 on non-budget-binding instances, so we normalize the excess cost by the always-positive
    do-nothing cost scale sum_i sum_j b_j*d_ij (cost of ordering nothing) instead of by the oracle."""
    with torch.no_grad():
        dec = SOLVE(m(Xte))
        achieved = ev_obj(dec.unsqueeze(0), Qte).squeeze(0)         # (B,)
        oracle = ev_obj(SOLVE(Qte).unsqueeze(0), Qte).squeeze(0)    # decision on TRUE params
        if sense == "max":
            reg = (oracle - achieved).sum().item() / oracle.sum().clamp(min=1e-6).item()
        else:
            denom = (nv_b * Qte).sum().clamp(min=1e-6).item()      # do-nothing (q=0) cost, always > 0
            reg = (achieved - oracle).sum().item() / denom
    return reg


PNAME = {"fk": "fractional knapsack (weights in constraint)", "nv": "capacitated newsvendor (demand in constraint)"}
COSTLBL = {"fk": "lambda (overflow penalty)", "nv": "h (overage penalty)"}


def main():
    problems = sys.argv[1].split(",") if len(sys.argv) > 1 else ["fk", "nv"]
    deg = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2, 3, 4]
    costs = [1.0, 5.0, 20.0]
    print(f"PREDICTION-IN-CONSTRAINTS | problems={problems} deg={deg} seeds={seeds}", flush=True)
    print("  surrogate camp (SPO+/cvxpylayers/PFYL/IMLE): N/A -- prediction defines the feasible region\n")
    results = {}
    for p in problems:
        print(f"[{p}] {PNAME[p]}", flush=True)
        results[p] = run_problem(p, deg, seeds, costs)
    payload = {"problems": problems, "deg": deg, "seeds": seeds, "costs": costs, "results": results}
    write_json(f"exp_results/constraints{_OUT_SFX}.json", payload)
    write_md(f"exp_results/constraints{_OUT_SFX}.md", to_markdown(results, problems, deg, seeds, costs))
    print(f"\nwrote exp_results/constraints{_OUT_SFX}.{{json,md}}\nDONE", flush=True)


def to_markdown(results, problems, deg, seeds, costs):
    L = ["# Experiment #4 -- Prediction-in-constraints (PolyStep vs SFGE)", "",
         f"deg={deg}, seeds={seeds}, normalized realized-regret (lower better). The predicted quantity "
         "defines the feasible region, so **SPO+ / cvxpylayers / PFYL / IMLE are structurally N/A**; only "
         "two-stage, SFGE and PolyStep run.", ""]
    for p in problems:
        L.append(f"## {PNAME[p]}")
        headers = [COSTLBL[p], "two-stage", "SFGE", "PolyStep", "best", "PS<TS (p)", "SFGE<TS (p)"]
        rows = []
        for c in costs:
            r = results[p][c]; s = r["summary"]
            cells = [fmt_mean_std(s["two-stage"]), fmt_mean_std(s["SFGE"]), fmt_mean_std(s["PolyStep"])]
            cells = [f"**{cells[i]}**" if ["two-stage", "SFGE", "PolyStep"][i] == r["best"] else cells[i]
                     for i in range(3)]
            rows.append([c] + cells + [r["best"],
                         f"{r['p_polystep_lt_ts']:.3f}" if r["p_polystep_lt_ts"] is not None else "—",
                         f"{r['p_sfge_lt_ts']:.3f}" if r["p_sfge_lt_ts"] is not None else "—"])
        L.append(md_table(headers, rows)); L.append("")
    L.append("**Reading:** two-stage degrades as the cost of misprediction rises (it predicts unbiasedly "
             "and pays the penalty); gradient-free learns to predict conservatively. SFGE is typically the "
             "more robust at low/mid cost, PolyStep strongest at high cost (strong decision signal).")
    return "\n".join(L)


if __name__ == "__main__":
    main()
