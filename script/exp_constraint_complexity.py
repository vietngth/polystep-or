"""Constraint-TYPE complexity ladder: does SFGE degrade vs PolyStep as the CONSTRAINTS get
structurally harder (not just more numerous)?

Same base decision problem at every rung -- a feature->parameter predictor over a FIXED set of
n_items produces per-item predicted VALUES v_hat (objective) and per-item predicted SIZES s_hat
(the quantity that lives in the constraints). We "pick items" x in {0,1}^n to maximize predicted
value. Only the CONSTRAINT TYPE changes as we climb the ladder, so the feasible region goes from
trivial -> linear -> combinatorial/non-smooth -> conic -> prediction-coupled:

  L0  box only           : max v_hat^T x,  x in {0,1}^n            (near-trivial region)
  L1  + linear capacity  : ... s.t. sum s_hat_i x_i <= C          (predicted weights in constraint)
  L2  + cardinality      : ... and sum x_i <= k                   (combinatorial / non-smooth)
  L3  + quadratic/conic  : ... and x^T Q(s_hat) x <= b,  Q=diag(s_hat) R diag(s_hat)   (SOC/conic)
  L4  + prediction-coupled: capacity RHS becomes C_hat = alpha * sum_i s_hat_i (+ card + quad)
                            -- the feasible region itself MOVES with the prediction.

Deploy = batched GPU greedy on the PREDICTED params (verified vs a Gurobi exact MIQCP per level).
Realized cost (MIN-sense) is evaluated on the TRUE params:
    cost(x) = -(v_true^T x) + lambda * [ capacity overflow + quadratic overflow + coupled overflow ]
n_items and the feature dim are FIXED across levels; ONLY the constraint type varies.

METHODS per level: two-stage (MSE floor) / SFGE / PolyStep, >=5 seeds.
  SPO+/IMLE/PFYL/cvxpylayers are marked N/A from L2 onward (cardinality/non-linear/coupled break the
  linear-objective + fixed-polytope assumption these surrogates require). Applicability note per level.

KEY METRICS (the point of the experiment):
  * regret mean+-std, Wilcoxon PolyStep vs SFGE;
  * SFGE HP fragility: sweep sigma per level -> "usable sigma band" (range within 10% of best) and the
    count of catastrophic divergences. HYPOTHESIS: the band NARROWS as complexity rises, while
    PolyStep's probe_radius band stays wide (we sweep probe_radius the same way as a control);
  * across-seed variance per method per level;
  * forward-solve counts (pto.budget.SolveCounter).

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp_constraint_complexity.py [--smoke]
"""
from __future__ import annotations
import os, sys, time, math
sys.path.insert(0, "polystep/src")
import numpy as np
import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from pto.seeding import seed_everything, device_generator
from pto.budget import SolveCounter
from pto.multiseed import summarize, wilcoxon_pair, md_table, write_json, write_md, fmt_mean_std

dev = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = ("--smoke" in sys.argv) or (os.environ.get("SMOKE", "0") == "1")
OUT_TAG = os.environ.get("OUT_TAG", "")
_SFX = f"_{OUT_TAG}" if OUT_TAG else ""

# ----------------------------- config (shrunk under SMOKE) -----------------------------
PF = 5                                   # feature dim (fixed across levels)
NIT = 12 if SMOKE else 20                # n_items (fixed across levels)
DEG = 4                                  # feature->param misspecification degree (linear head is wrong)
NTR = 80 if SMOKE else 400
NTE = 120 if SMOKE else 600
LAM = 10.0                               # violation penalty (fixed; the ladder varies TYPE, not lambda)
TS_EP = 200 if SMOKE else 300
TS_LR = 5e-2
PS_STEPS = 40 if SMOKE else 200
SF_EP = 40 if SMOKE else 200
SF_NS = 8                                # SFGE MC samples
LEVELS = [0, 1, 2, 3, 4]
SEEDS = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
SIGMA_GRID = [0.1, 0.5, 1.0, 2.0] if SMOKE else [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
PROBE_GRID = [0.4, 0.8, 1.6] if SMOKE else [0.2, 0.4, 0.6, 0.8, 1.2, 1.6]
SIGMA_DEFAULT = 0.5
PROBE_DEFAULT = 0.8
PS_POLYTOPE = os.environ.get("PS_POLYTOPE", "orthoplex")
PS_PROBES = int(os.environ.get("PS_PROBES", "1"))
BAND_TOL = 0.10                          # "usable" = within 10% of best HP setting
DIVERGE_MULT = 5.0                       # catastrophic = run regret > DIVERGE_MULT * best-HP mean regret

LEVEL_NAME = {0: "L0 box-only", 1: "L1 +linear-capacity", 2: "L2 +cardinality",
              3: "L3 +quadratic/conic", 4: "L4 +prediction-coupled"}
LEVEL_CONSTRAINT = {
    0: "x in {0,1}^n  (box bounds only)",
    1: "+ sum_i s_hat_i x_i <= C  (predicted weights in a linear capacity constraint)",
    2: "+ sum_i x_i <= k  (cardinality; combinatorial / non-smooth)",
    3: "+ x^T Q x <= b,  Q = diag(s_hat) R diag(s_hat)  (conic / quadratic risk budget)",
    4: "capacity RHS coupled: C_hat = alpha * sum_i s_hat_i  (feasible region moves with prediction; + card + quad)",
}
# Surrogate camp applicability per level (we never RUN them; this documents the N/A frontier).
LEVEL_NA = {
    0: ("applicable*", "linear objective in predicted values over a fixed box (in principle SPO+/PFYL/IMLE "
                       "could run; we report the two-stage/SFGE/PolyStep family for a like-for-like ladder)"),
    1: ("marginal", "predicted weights already sit in the CONSTRAINT (SFGE-suite regime) -> SPO+/PFYL/IMLE/"
                    "cvxpylayers lose their fixed-polytope / objective-cost-vector footing; treated as out-of-scope"),
    2: ("N/A", "cardinality makes the region combinatorial & non-smooth; no linear-objective fixed polytope"),
    3: ("N/A", "quadratic/conic constraint; not a polytope, perturbed-argmin surrogates undefined"),
    4: ("N/A", "feasible region MOVES with the prediction; no fixed S to argmin over, predicted optimum can be infeasible"),
}

# Fixed PSD correlation matrix R for the conic level (seed-independent, depends only on NIT).
_rs = np.random.RandomState(20240601)
_Lf = _rs.randn(NIT, max(2, NIT // 3)).astype(np.float32)
_cov = _Lf @ _Lf.T + np.eye(NIT, dtype=np.float32)
_d = np.sqrt(np.diag(_cov))
R = torch.tensor((_cov / np.outer(_d, _d)).astype(np.float32), device=dev)   # unit-diagonal PSD corr


# ----------------------------- data (shared X, two positive deg-DEG targets) -----------------------------
def _poly_target(X, B, deg):
    """pyepo-style multiplicative polynomial: positive target, degree-`deg` in features so a LINEAR
    head is misspecified. X (N,p), B (p,n) fixed map -> (N,n) strictly positive."""
    lin = X @ B / math.sqrt(X.shape[1])
    return (lin + 3.0).clamp(min=0.1) ** deg


def gen(seed):
    """Features X + true per-item VALUES v_true (objective) and SIZES s_true (constraints).
    Both are deg-DEG nonlinear functions of the SAME X via two distinct fixed maps."""
    rs = np.random.RandomState(1000 + seed)
    X = rs.randn(NTR + NTE, PF).astype(np.float32)
    Bv = np.random.RandomState(7).randn(PF, NIT).astype(np.float32)      # fixed value map
    Bs = np.random.RandomState(13).randn(PF, NIT).astype(np.float32)     # fixed size  map
    Xt = torch.tensor(X, device=dev)
    V = _poly_target(Xt, torch.tensor(Bv, device=dev), DEG)
    S = _poly_target(Xt, torch.tensor(Bs, device=dev), DEG)
    V = V / V.mean(); S = S / S.mean()                  # O(1) scale (keeps deg-DEG misspecification)
    return Xt[:NTR], V[:NTR], S[:NTR], Xt[NTR:], V[NTR:], S[NTR:]


class P:  # per-(seed,level) feasible-region parameters (data-derived, method-independent)
    pass


def calibrate(Vtr, Str):
    """Set C (capacity), k (cardinality), b (risk budget), alpha (coupling) so each constraint binds."""
    p = P()
    p.C = float(Str.sum(-1).mean() * 0.5)                 # ~half the true total size
    p.k = max(2, NIT // 2)
    p.alpha = 0.5
    # risk budget b: 0.7 * mean risk of the capacity+cardinality solution on TRUE params
    x2 = solve_ladder(Vtr, Str, 2, p)
    p.b = float(quad_risk(x2, Str).mean() * 0.7) + 1e-6
    return p


# ----------------------------- batched GPU greedy solver (deployable, verified vs Gurobi) -----------------------------
def quad_risk(x, s):
    """x^T Q x with Q = diag(s) R diag(s)  ->  (us)^T R (us), us = s*x.  x (...,n), s (...,n)."""
    us = x * s
    return (us @ R * us).sum(-1)


def solve_ladder(vhat, shat, level, p):
    """Greedy item selection under the level's active constraints. Predicted params in; x (M,n) in {0,1}.
    Items added in value/size-density order if every active constraint stays feasible (incremental check)."""
    M, n = vhat.shape
    shat = shat.clamp(min=1e-3)
    if level == 0:
        return (vhat > 0).float()
    midx = torch.arange(M, device=dev)
    order = (vhat / shat).argsort(-1, descending=True)
    x = torch.zeros(M, n, device=dev)
    used = torch.zeros(M, device=dev)
    cnt = torch.zeros(M, device=dev)
    Cdep = (p.alpha * shat.sum(-1)) if level == 4 else torch.full((M,), float(p.C), device=dev)
    if level >= 3:
        u = torch.zeros(M, n, device=dev)                 # u = shat * x  (for incremental risk)
        risk = torch.zeros(M, device=dev)
        Rdiag = torch.diag(R)
    for t in range(n):
        i = order[:, t]
        si = shat.gather(1, i[:, None]).squeeze(1)
        vi = vhat.gather(1, i[:, None]).squeeze(1)
        feas = vi > 0                                     # never helps the objective to add value<=0
        feas &= (used + si) <= Cdep                       # capacity (fixed L1-3, coupled L4)
        if level >= 2:
            feas &= cnt < p.k                             # cardinality
        if level >= 3:
            Ri = R[i]                                     # (M,n) rows for the candidate item
            dot = (Ri * u).sum(-1)                        # s_i * R[i,:] . (s*x)
            drisk = 2.0 * si * dot + si * si * Rdiag[i]
            feas &= (risk + drisk) <= p.b
        add = feas.float()
        x[midx, i] = add
        used = used + si * add
        cnt = cnt + add
        if level >= 3:
            u[midx, i] = si * add
            risk = risk + drisk * add
    return x


def realized_cost(x, v_true, s_true, level, p):
    """MIN-sense realized cost on TRUE params (signed). x (...,n), v_true/s_true (B,n) broadcast.
    Each constraint violation is FRACTIONAL (overflow / its own RHS) so the linear, conic and coupled
    penalties live on a comparable O(1) scale to the collected value -- no single term swamps the metric."""
    cost = -(x * v_true).sum(-1)
    if 1 <= level <= 3:                                   # fixed linear capacity (fractional overflow)
        cost = cost + LAM * ((x * s_true).sum(-1) - p.C).clamp(min=0) / p.C
    if level >= 3:                                        # quadratic/conic risk budget (true covariance)
        cost = cost + LAM * (quad_risk(x, s_true) - p.b).clamp(min=0) / p.b
    if level == 4:                                        # prediction-coupled capacity (true RHS)
        Ctrue = p.alpha * s_true.sum(-1)
        cost = cost + LAM * ((x * s_true).sum(-1) - Ctrue).clamp(min=0) / Ctrue
    return cost


# ----------------------------- Gurobi exactness reference (a few instances per level) -----------------------------
def gurobi_gap(Vte, Ste, level, p, n_check=6):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        return None, f"gurobi unavailable ({e})"
    Rn = R.cpu().numpy()
    idx = list(range(min(n_check, Vte.shape[0])))
    xg = solve_ladder(Vte[idx], Ste[idx], level, p)                         # greedy on TRUE params
    gval = (xg * Vte[idx]).sum(-1).cpu().numpy()
    ratios = []
    for k, ii in enumerate(idx):
        v = Vte[ii].cpu().numpy(); s = Ste[ii].clamp(min=1e-3).cpu().numpy()
        md = gp.Model(); md.Params.OutputFlag = 0; md.Params.TimeLimit = 5
        x = md.addVars(NIT, vtype=GRB.BINARY)
        md.setObjective(gp.quicksum(float(v[j]) * x[j] for j in range(NIT)), GRB.MAXIMIZE)
        if level in (1, 2, 3):
            md.addConstr(gp.quicksum(float(s[j]) * x[j] for j in range(NIT)) <= p.C)
        if level == 4:
            md.addConstr(gp.quicksum(float(s[j]) * x[j] for j in range(NIT)) <= p.alpha * float(s.sum()))
        if level >= 2:
            md.addConstr(gp.quicksum(x[j] for j in range(NIT)) <= p.k)
        if level >= 3:
            us = [float(s[j]) * x[j] for j in range(NIT)]
            quad = gp.quicksum(float(Rn[a, bb]) * us[a] * us[bb] for a in range(NIT) for bb in range(NIT))
            md.addConstr(quad <= p.b)
        md.optimize()
        if md.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and md.SolCount > 0:
            opt = md.objVal
            if opt > 1e-9:
                ratios.append(gval[k] / opt)
    if not ratios:
        return None, "no reference solved"
    return float(np.mean(ratios)), f"greedy/opt value n={len(ratios)} (min {min(ratios):.3f})"


# ----------------------------- predictor + trainers (warm-started from two-stage) -----------------------------
def make_pred():
    return nn.Linear(PF, 2 * NIT, bias=True).to(dev)      # outputs [v_hat | s_hat] per item


def split(out):
    return out[..., :NIT], out[..., NIT:]


def train_two_stage(Xtr, Vtr, Str):
    m = make_pred(); opt = torch.optim.Adam(m.parameters(), TS_LR)
    tgt = torch.cat([Vtr, Str], -1)
    for _ in range(TS_EP):
        opt.zero_grad(); ((m(Xtr) - tgt) ** 2).mean().backward(); opt.step()
    return m


def train_polystep(Xtr, Vtr, Str, level, p, warm, scale, solve, probe_radius=PROBE_DEFAULT, seed=0):
    m = make_pred()
    with torch.no_grad():
        m.weight.copy_(warm.weight); m.bias.copy_(warm.bias)
    pso = PolyStepOptimizer(m, polytope_type=PS_POLYTOPE, epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=0.4, probe_radius=probe_radius, num_probe=PS_PROBES, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)

    def closure(bp):
        out = torch.einsum("mof,bf->mbo", bp["weight"], Xtr) + bp["bias"].unsqueeze(1)   # (M,B,2n)
        M, B, _ = out.shape
        vh, sh = split(out.reshape(M * B, 2 * NIT))
        x = solve(torch.cat([vh, sh], -1)).reshape(M, B, NIT)
        return (realized_cost(x, Vtr, Str, level, p) / scale).mean(-1)
    for _ in range(PS_STEPS):
        pso.step(closure)
    return m


def train_sfge(Xtr, Vtr, Str, level, p, warm, scale, solve, sigma=SIGMA_DEFAULT, lr=1e-2, seed=0):
    m = make_pred()
    with torch.no_grad():
        m.weight.copy_(warm.weight); m.bias.copy_(warm.bias)
    opt = torch.optim.Adam(m.parameters(), lr); g = device_generator(seed, dev)
    for _ in range(SF_EP):
        pred = m(Xtr)                                                       # (B,2n)
        with torch.no_grad():
            eps = torch.randn(SF_NS, *pred.shape, device=dev, generator=g)
            chat = pred.unsqueeze(0) + sigma * eps                          # (S,B,2n)
            S, B, _ = chat.shape
            vh, sh = split(chat.reshape(S * B, 2 * NIT))
            x = solve(torch.cat([vh, sh], -1)).reshape(S, B, NIT)
            r = realized_cost(x, Vtr, Str, level, p) / scale               # (S,B)
            adv = r - r.mean(0, keepdim=True)
        logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
        surrogate = (adv * logp).mean()
        opt.zero_grad(); surrogate.backward(); opt.step()
    return m


def evaluate(m, Xte, Vte, Ste, level, p, solve):
    """Normalized realized regret = (achieved - oracle)/|oracle|, oracle = same solver on TRUE params."""
    with torch.no_grad():
        out = m(Xte); x = solve(out)
        achieved = realized_cost(x.unsqueeze(0), Vte, Ste, level, p).squeeze(0)
        x_or = solve(torch.cat([Vte, Ste], -1))
        oracle = realized_cost(x_or.unsqueeze(0), Vte, Ste, level, p).squeeze(0)
        denom = oracle.sum().abs().clamp(min=1e-6).item()
        return (achieved - oracle).sum().item() / denom


# ----------------------------- HP-fragility band -----------------------------
def usable_band(grid, means, scale):
    """grid HP values, means per HP (across-seed). 'usable' = within BAND_TOL * scale of the best setting
    (scale anchors the tolerance so a near-zero best regret does not make the band spuriously tiny).
    Returns (best, lo, hi, width, n_usable)."""
    best = min(means)
    thr = best + BAND_TOL * max(abs(best), abs(scale), 1e-6)
    usable = [g for g, mn in zip(grid, means) if mn <= thr]
    if not usable:
        usable = [grid[int(np.argmin(means))]]
    lo, hi = min(usable), max(usable)
    return best, lo, hi, hi - lo, len(usable)


def count_catastrophic(runs_by_hp, ts_mean):
    """A single (HP, seed) run is CATASTROPHIC if its regret blew past the two-stage floor by a wide
    margin (DFL actively backfired) or is non-finite. Robust to a near-zero / negative best regret."""
    cat_thr = abs(ts_mean) + max(abs(ts_mean), 0.05)        # > ~2x the two-stage floor (or +0.05 absolute)
    return int(sum(1 for hp in runs_by_hp for v in runs_by_hp[hp]
                   if (not math.isfinite(v)) or v > cat_thr))


# ----------------------------- driver -----------------------------
def run_level(level, seeds):
    main = {mname: [] for mname in ("two-stage", "SFGE", "PolyStep")}
    sigma_runs = {sg: [] for sg in SIGMA_GRID}      # per-sigma list of per-seed SFGE regrets
    probe_runs = {pr: [] for pr in PROBE_GRID}      # per-probe_radius list of per-seed PolyStep regrets
    solves = {mname: 0 for mname in ("SFGE", "PolyStep")}
    p_ref = None
    for seed in seeds:
        seed_everything(seed)
        Xtr, Vtr, Str, Xte, Vte, Ste = gen(seed)
        p = calibrate(Vtr, Str)
        if p_ref is None:
            p_ref = (Vte, Ste, p)
        scale = float(Vtr.sum(-1).mean()) + 1e-6
        counter = SolveCounter(lambda pred, _p=p, _l=level: solve_ladder(*split(pred), _l, _p))

        ts = train_two_stage(Xtr, Vtr, Str)
        main["two-stage"].append(evaluate(ts, Xte, Vte, Ste, level, p, counter))

        # default-HP main runs (counted) ----------------------------------
        counter.reset()
        sf = train_sfge(Xtr, Vtr, Str, level, p, ts, scale, counter, sigma=SIGMA_DEFAULT, seed=seed)
        solves["SFGE"] += counter.instances
        main["SFGE"].append(evaluate(sf, Xte, Vte, Ste, level, p, counter))

        counter.reset()
        ps = train_polystep(Xtr, Vtr, Str, level, p, ts, scale, counter, probe_radius=PROBE_DEFAULT, seed=seed)
        solves["PolyStep"] += counter.instances
        main["PolyStep"].append(evaluate(ps, Xte, Vte, Ste, level, p, counter))

        # SFGE sigma sweep (fragility) ------------------------------------
        for sg in SIGMA_GRID:
            msf = train_sfge(Xtr, Vtr, Str, level, p, ts, scale, counter, sigma=sg, seed=seed)
            sigma_runs[sg].append(evaluate(msf, Xte, Vte, Ste, level, p, counter))
        # PolyStep probe_radius sweep (control) ---------------------------
        for pr in PROBE_GRID:
            mps = train_polystep(Xtr, Vtr, Str, level, p, ts, scale, counter, probe_radius=pr, seed=seed)
            probe_runs[pr].append(evaluate(mps, Xte, Vte, Ste, level, p, counter))

    summ = {mname: summarize(main[mname]) for mname in main}
    best = min(summ, key=lambda mn: summ[mn]["mean"])
    p_ps_sf = wilcoxon_pair(main["PolyStep"], main["SFGE"])
    ts_mean = summ["two-stage"]["mean"]

    # ---- SFGE sigma band + catastrophic divergences ----
    sig_means = [float(np.mean(sigma_runs[sg])) for sg in SIGMA_GRID]
    sb_best, sb_lo, sb_hi, sb_w, sb_n = usable_band(SIGMA_GRID, sig_means, ts_mean)
    sig_diverge = count_catastrophic(sigma_runs, ts_mean)
    # ---- PolyStep probe_radius band (control) ----
    pr_means = [float(np.mean(probe_runs[pr])) for pr in PROBE_GRID]
    pb_best, pb_lo, pb_hi, pb_w, pb_n = usable_band(PROBE_GRID, pr_means, ts_mean)
    pr_diverge = count_catastrophic(probe_runs, ts_mean)

    gap = gurobi_gap(*p_ref[:2], level, p_ref[2]) if p_ref is not None else (None, "n/a")

    return {
        "level": level, "name": LEVEL_NAME[level], "constraint": LEVEL_CONSTRAINT[level],
        "na_flag": LEVEL_NA[level][0], "na_note": LEVEL_NA[level][1],
        "summary": summ, "best": best, "p_polystep_lt_sfge": p_ps_sf,
        "solves": solves,
        "sfge_sigma": {"grid": SIGMA_GRID, "means": sig_means,
                       "best_regret": sb_best, "band_lo": sb_lo, "band_hi": sb_hi,
                       "band_width": sb_w, "n_usable": sb_n, "frac_usable": sb_n / len(SIGMA_GRID),
                       "catastrophic": sig_diverge, "n_runs": len(SIGMA_GRID) * len(seeds)},
        "polystep_probe": {"grid": PROBE_GRID, "means": pr_means,
                           "best_regret": pb_best, "band_lo": pb_lo, "band_hi": pb_hi,
                           "band_width": pb_w, "n_usable": pb_n, "frac_usable": pb_n / len(PROBE_GRID),
                           "catastrophic": pr_diverge, "n_runs": len(PROBE_GRID) * len(seeds)},
        "gurobi_gap": gap,
    }


# ----------------------------- reporting -----------------------------
def to_markdown(results, seeds):
    L = ["# Constraint-TYPE complexity ladder -- SFGE vs PolyStep", "",
         f"seeds={seeds}, n_items={NIT}, feat={PF}, deg={DEG}, lambda={LAM}, SMOKE={int(SMOKE)}. "
         "Same base predict-then-pick problem at every rung; ONLY the constraint TYPE changes. "
         "Regret = (achieved_cost - oracle_cost)/|oracle_cost| (lower better; oracle = same batched "
         "greedy on TRUE params). The hypothesis under test: as constraint complexity rises, SFGE's "
         "**usable sigma band narrows** and its regret/variance grow faster than PolyStep's "
         "(probe_radius band = control).", "",
         "## The ladder (exact constraint at each rung)", ""]
    for lv in LEVELS:
        r = results[lv]
        L.append(f"- **{r['name']}**: {r['constraint']}  -- surrogate camp: *{r['na_flag']}* ({r['na_note']})")
    L.append("")

    # regret table
    L.append("## Regret by method (mean +- std)")
    headers = ["level", "two-stage", "SFGE", "PolyStep", "best", "PS<SFGE (p)", "greedy/opt"]
    rows = []
    for lv in LEVELS:
        r = results[lv]; s = r["summary"]; names = ["two-stage", "SFGE", "PolyStep"]
        cells = [fmt_mean_std(s[nm]) for nm in names]
        cells = [f"**{cells[i]}**" if names[i] == r["best"] else cells[i] for i in range(3)]
        gp = r["gurobi_gap"][0]
        pps = r["p_polystep_lt_sfge"]
        rows.append([r["name"]] + cells + [r["best"],
                     f"{pps:.3f}" if pps is not None else "-",
                     f"{gp:.3f}" if gp is not None else "-"])
    L.append(md_table(headers, rows)); L.append("")

    # across-seed variance
    L.append("## Across-seed std (stability)")
    headers = ["level", "two-stage std", "SFGE std", "PolyStep std"]
    rows = [[results[lv]["name"]] + [f"{results[lv]['summary'][nm]['std']:.4f}"
            for nm in ("two-stage", "SFGE", "PolyStep")] for lv in LEVELS]
    L.append(md_table(headers, rows)); L.append("")

    # SFGE sigma fragility vs PolyStep probe robustness
    L.append("## HP fragility -- SFGE usable-sigma band vs PolyStep probe_radius band")
    L.append(f"Usable = HP settings whose across-seed mean regret is within {int(BAND_TOL*100)}% of the best "
             "setting (tolerance anchored to the two-stage floor). Catastrophic = single (HP,seed) runs that "
             "blew past ~2x the two-stage floor or diverged (non-finite). HYPOTHESIS: SFGE band narrows up the "
             "ladder; PolyStep probe_radius band stays wide.")
    headers = ["level", "SFGE band [lo,hi]", "SFGE width", "SFGE usable", "SFGE catastrophic",
               "PS probe band", "PS width", "PS usable", "PS catastrophic"]
    rows = []
    for lv in LEVELS:
        sgr = results[lv]["sfge_sigma"]; psr = results[lv]["polystep_probe"]
        rows.append([results[lv]["name"],
                     f"[{sgr['band_lo']:g},{sgr['band_hi']:g}]", f"{sgr['band_width']:g}",
                     f"{sgr['n_usable']}/{len(SIGMA_GRID)}", f"{sgr['catastrophic']}/{sgr['n_runs']}",
                     f"[{psr['band_lo']:g},{psr['band_hi']:g}]", f"{psr['band_width']:g}",
                     f"{psr['n_usable']}/{len(PROBE_GRID)}", f"{psr['catastrophic']}/{psr['n_runs']}"])
    L.append(md_table(headers, rows)); L.append("")

    # forward solves
    L.append("## Forward-solve counts (per method, summed over seeds, default-HP main run)")
    headers = ["level", "SFGE forward-solves", "PolyStep forward-solves"]
    rows = [[results[lv]["name"], results[lv]["solves"]["SFGE"], results[lv]["solves"]["PolyStep"]]
            for lv in LEVELS]
    L.append(md_table(headers, rows)); L.append("")

    # verdict (3 sentences, honest)
    sf_w = [results[lv]["sfge_sigma"]["frac_usable"] for lv in LEVELS]
    ps_w = [results[lv]["polystep_probe"]["frac_usable"] for lv in LEVELS]
    sf_narrows = sf_w[-1] < sf_w[0] - 1e-9
    sf_var = [results[lv]["summary"]["SFGE"]["std"] for lv in LEVELS]
    ps_var = [results[lv]["summary"]["PolyStep"]["std"] for lv in LEVELS]
    var_faster = (sf_var[-1] - sf_var[0]) > (ps_var[-1] - ps_var[0])
    reg_better = sum(results[lv]["summary"]["PolyStep"]["mean"] < results[lv]["summary"]["SFGE"]["mean"]
                     for lv in LEVELS)
    L.append("## Verdict")
    L.append(
        f"Across the 5-rung ladder PolyStep had lower mean regret than SFGE at {reg_better}/5 levels; "
        f"SFGE's usable-sigma fraction went from {sf_w[0]:.2f} (L0) to {sf_w[-1]:.2f} (L4), so the band "
        f"{'NARROWS' if sf_narrows else 'does NOT clearly narrow'} as constraint complexity rises, while "
        f"PolyStep's probe_radius usable fraction stayed {ps_w[0]:.2f}->{ps_w[-1]:.2f}. "
        f"SFGE's across-seed variance grew {'FASTER' if var_faster else 'no faster'} than PolyStep's from "
        f"L0 to L4 (SFGE {sf_var[0]:.3f}->{sf_var[-1]:.3f} vs PolyStep {ps_var[0]:.3f}->{ps_var[-1]:.3f}). "
        f"{'This supports the hypothesis that harder constraint TYPES disproportionately hurt SFGE.' if (sf_narrows or var_faster) else 'On this run the methods are broadly comparable -- no clear PolyStep advantage from constraint type; reported honestly as a tie.'}")
    return "\n".join(L)


# ----------------------------- figures (colorblind-safe Okabe-Ito) -----------------------------
def make_figures(results):
    try:
        import plotly.graph_objects as go
    except Exception as e:
        print(f"[figs] plotly unavailable ({e}); skipping figures", flush=True)
        return []
    CB = {"two-stage": "#999999", "SFGE": "#E69F00", "PolyStep": "#0072B2"}
    SYM = {"two-stage": "circle", "SFGE": "triangle-up", "PolyStep": "star"}
    os.makedirs("exp_results/figs", exist_ok=True)
    xs = [results[lv]["name"] for lv in LEVELS]
    written = []

    # fig 1: regret vs ladder level
    fig = go.Figure()
    for nm in ("two-stage", "SFGE", "PolyStep"):
        ys = [results[lv]["summary"][nm]["mean"] for lv in LEVELS]
        es = [results[lv]["summary"][nm]["std"] for lv in LEVELS]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=nm,
                                 line=dict(color=CB[nm], width=2.4),
                                 marker=dict(symbol=SYM[nm], size=11, color=CB[nm]),
                                 error_y=dict(type="data", array=es, visible=True, thickness=1.2)))
    # N/A frontier annotation (SPO+/IMLE/PFYL N/A from L2 onward)
    fig.add_vrect(x0=1.5, x1=4.5, fillcolor="#cccccc", opacity=0.18, line_width=0,
                  annotation_text="SPO+/IMLE/PFYL/cvxpylayers N/A", annotation_position="top left",
                  annotation_font=dict(size=11, color="#555555"))
    fig.update_layout(template="simple_white", width=760, height=470,
                      font=dict(family="Times New Roman, serif", size=15),
                      title="Realized regret vs constraint-type complexity",
                      xaxis_title="constraint complexity (ladder level)", yaxis_title="normalized realized regret",
                      legend=dict(x=0.02, y=0.98))
    for ext in ("png", "pdf"):
        path = f"exp_results/figs/fig_complexity_regret.{ext}"
        try:
            fig.write_image(path); written.append(path)
        except Exception as e:
            print(f"[figs] could not write {path} ({e})", flush=True)

    # fig 2: HP usable-band fraction vs level (does SFGE narrow?)
    fig2 = go.Figure()
    sf_frac = [results[lv]["sfge_sigma"]["frac_usable"] for lv in LEVELS]
    ps_frac = [results[lv]["polystep_probe"]["frac_usable"] for lv in LEVELS]
    fig2.add_trace(go.Scatter(x=xs, y=sf_frac, mode="lines+markers", name="SFGE (sigma)",
                              line=dict(color=CB["SFGE"], width=2.4),
                              marker=dict(symbol="triangle-up", size=12, color=CB["SFGE"])))
    fig2.add_trace(go.Scatter(x=xs, y=ps_frac, mode="lines+markers", name="PolyStep (probe_radius)",
                              line=dict(color=CB["PolyStep"], width=2.4),
                              marker=dict(symbol="star", size=13, color=CB["PolyStep"])))
    fig2.update_layout(template="simple_white", width=760, height=470,
                       font=dict(family="Times New Roman, serif", size=15),
                       title="HP robustness vs complexity (usable band fraction within 10% of best)",
                       xaxis_title="constraint complexity (ladder level)",
                       yaxis_title="fraction of HP grid that is 'usable'",
                       yaxis=dict(range=[0, 1.05]), legend=dict(x=0.02, y=0.06))
    for ext in ("png", "pdf"):
        path = f"exp_results/figs/fig_complexity_sigmaband.{ext}"
        try:
            fig2.write_image(path); written.append(path)
        except Exception as e:
            print(f"[figs] could not write {path} ({e})", flush=True)
    return written


def main():
    print(f"CONSTRAINT-TYPE COMPLEXITY LADDER | SMOKE={int(SMOKE)} | seeds={SEEDS} | n_items={NIT}", flush=True)
    print(f"  sigma grid={SIGMA_GRID}\n  probe grid={PROBE_GRID}\n", flush=True)
    results = {}
    t0 = time.time()
    for lv in LEVELS:
        print(f"[L{lv}] {LEVEL_NAME[lv]} :: {LEVEL_CONSTRAINT[lv]}", flush=True)
        r = run_level(lv, SEEDS)
        results[lv] = r
        s = r["summary"]
        print(f"   regret: " + "  ".join(f"{m}={fmt_mean_std(s[m])}" for m in ("two-stage", "SFGE", "PolyStep"))
              + f"  best={r['best']}", flush=True)
        sg = r["sfge_sigma"]; pr = r["polystep_probe"]
        print(f"   SFGE sigma-band [{sg['band_lo']:g},{sg['band_hi']:g}] usable={sg['n_usable']}/{len(SIGMA_GRID)} "
              f"catastrophic={sg['catastrophic']}/{sg['n_runs']} | "
              f"PolyStep probe-band usable={pr['n_usable']}/{len(PROBE_GRID)} | gurobi greedy/opt="
              f"{r['gurobi_gap'][0] if r['gurobi_gap'][0] is not None else 'n/a'}", flush=True)
    payload = {"config": {"smoke": SMOKE, "seeds": SEEDS, "n_items": NIT, "feat": PF, "deg": DEG,
                          "lambda": LAM, "sigma_grid": SIGMA_GRID, "probe_grid": PROBE_GRID,
                          "band_tol": BAND_TOL, "diverge_mult": DIVERGE_MULT},
               "results": {str(lv): results[lv] for lv in LEVELS}}
    write_json(f"exp_results/constraint_complexity{_SFX}.json", payload)
    write_md(f"exp_results/constraint_complexity{_SFX}.md", to_markdown(results, SEEDS))
    figs = make_figures(results)
    print(f"\nwrote exp_results/constraint_complexity{_SFX}.{{json,md}}", flush=True)
    print(f"figures: {figs}", flush=True)
    print(f"DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
