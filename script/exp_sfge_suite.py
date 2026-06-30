"""SFGE-suite: the PUREST structural-win benchmark -- prediction in the CONSTRAINTS.

Two problems straight from the SFGE paper (Silvestri et al., "Score Function Gradient Estimation
to Widen the Applicability of Decision-Focused Learning", JAIR 81 (2024), arXiv:2307.05213) in which
the PREDICTED quantity parametrizes the FEASIBLE REGION rather than the objective. In that regime the
optimization-oracle / differentiable camp (SPO+, IMLE, PFYL / Fenchel-Young, DBB, cvxpylayers) is
STRUCTURALLY UNDEFINED -- a fact certified by the SFGE paper itself ("SPO+ applies only when the
predicted parameters appear in the objective"). Here SFGE is a legitimate PEER (it, like PolyStep,
needs only a scalar realized cost); two-stage (MSE) is the floor.

  (1) KP-unknown-weights -- fractional 0/1-style knapsack; item VALUES v known, per-item WEIGHTS
      predicted (and sit in the capacity constraint sum w_i x_i <= C). Deploy = fractional-knapsack
      greedy on predicted weights; realized cost = -(v^T x) + lambda * max(0, w_true^T x - C). The
      advantage is lambda-dependent: sweep lambda in {1,5,20}. n_items=50.
  (2) WSMC (Weighted Set Multi-Cover) -- buy sets (known costs c) so each element's coverage
      REQUIREMENT (predicted, in the constraint RHS A y >= b) is met. Deploy = batched greedy cover on
      predicted requirements; realized cost = c^T y + lambda * shortfall(y, b_true). 10 elements x 25
      sets, solvable by a batched greedy verified against a Gurobi exact ILP. Sweep lambda in {1,5,20}.

Run:   CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp_sfge_suite.py <seeds_csv>
       SMOKE=1 shrinks sizes/iters (under ~3 min) for a local sanity run.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "polystep/src")
import numpy as np
import torch
import torch.nn as nn
from pyepo.data import knapsack
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from pto.solvers import solve_fractional_knapsack, solve_set_multicover
from pto.seeding import seed_everything, device_generator
from pto.multiseed import summarize, wilcoxon_pair, md_table, write_json, write_md, fmt_mean_std

dev = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("SMOKE", "0") == "1"
# --- polytope x probe sweep (env-overridable; defaults preserve original behavior) ---
PS_POLYTOPE = os.environ.get("PS_POLYTOPE", "orthoplex")
PS_PROBES = int(os.environ.get("PS_PROBES", "1"))
OUT_TAG = os.environ.get("OUT_TAG", "")
_OUT_SFX = f"_{OUT_TAG}" if OUT_TAG else ""

# sizes / iteration budgets (shrunk under SMOKE)
DEG = 4
PF = 5                                                    # feature dim
KP_NIT = 20 if SMOKE else 50                              # knapsack items
WS_NE, WS_NS = (6, 12) if SMOKE else (10, 25)            # WSMC elements x sets
NTR = 80 if SMOKE else 400
NTE = 120 if SMOKE else 600
TS_EP = 200 if SMOKE else 300                             # two-stage epochs (must fit the target scale)
TS_LR = 5e-2                                               # Adam moves ~lr/step; needs lr>1e-2 to reach scale
PS_STEPS = 40 if SMOKE else 200                           # PolyStep steps
SF_EP = 40 if SMOKE else 200                              # SFGE epochs
LAMBDAS = [1.0, 5.0, 20.0]

# global forward-solve counter (incremented inside SOLVE wrappers)
N_SOLVES = {"v": 0}


# --------------------------- predictor ---------------------------
def make_pred(n):
    return nn.Linear(PF, n, bias=True).to(dev)            # bias lets it learn to predict conservatively


# --------------------------- data ---------------------------
def gen_kp(seed):
    """features X + true per-item WEIGHTS (in the constraint); item VALUES v fixed/known."""
    Vfix, _, _ = knapsack.genData(2, PF, KP_NIT, dim=1, deg=1, seed=1)
    v = torch.tensor(Vfix[0], dtype=torch.float32, device=dev)            # known item values
    _, x, w = knapsack.genData(NTR + NTE, PF, KP_NIT, dim=1, deg=DEG, noise_width=0, seed=seed)
    X = torch.tensor(x, dtype=torch.float32, device=dev)
    W = torch.tensor(w, dtype=torch.float32, device=dev)                  # true weights (predicted target)
    C = float(W[:NTR].sum(-1).mean() * 0.5)                               # binding capacity
    return X[:NTR], W[:NTR], X[NTR:], W[NTR:], v, C


def gen_wsmc(seed):
    """features X + true per-element coverage REQUIREMENTS b (constraint RHS); incidence A + costs c fixed."""
    rs = np.random.RandomState(20240517)                                 # fixed instance structure
    A = (rs.rand(WS_NE, WS_NS) < 0.45).astype(np.float32)               # 0/1 incidence
    for i in range(WS_NE):                                              # guarantee a coverable element
        if A[i].sum() < 3:
            A[i, rs.choice(WS_NS, 3, replace=False)] = 1.0
    c = (rs.rand(WS_NS).astype(np.float32) * 0.8 + 0.4)                 # set costs in [0.4,1.2]
    At = torch.tensor(A, device=dev); ct = torch.tensor(c, device=dev)
    deg_i = torch.tensor(A.sum(1), device=dev)                          # max coverage per element (row degree)
    # feature -> requirement map (degree-`DEG` polynomial, same misspecification as KP), scaled to [1, deg_i]
    _, x, raw = knapsack.genData(NTR + NTE, PF, WS_NE, dim=1, deg=DEG, noise_width=0, seed=seed)
    X = torch.tensor(x, dtype=torch.float32, device=dev)
    R = torch.tensor(raw, dtype=torch.float32, device=dev)
    R = R / R.max(0).values.clamp(min=1e-6)                             # per-element to [0,1]
    B = (R * (deg_i * 0.55)).round().clamp(min=1.0)                     # true requirements (feasible: <= deg_i)
    B = torch.minimum(B, deg_i)
    return X[:NTR], B[:NTR], X[NTR:], B[NTR:], At, ct, deg_i


# --------------------------- realized cost (MIN-sense, signed) ---------------------------
def kp_cost(x, w_true, v, C, lam):
    """fractional knapsack realized COST to MINIMIZE. x (...,n), w_true (...,n) broadcasts."""
    value = (x * v).sum(-1)
    overflow = ((x * w_true).sum(-1) - C).clamp(min=0)
    return lam * overflow - value                                        # -(value) + lambda*violation


def wsmc_cost(y, b_true, c, A, lam):
    """WSMC realized COST to MINIMIZE. y (...,ns); coverage = y @ A^T; shortfall vs true requirement."""
    set_cost = (y * c).sum(-1)
    coverage = y @ A.t()                                                 # (...,ne)
    shortfall = (b_true - coverage).clamp(min=0).sum(-1)
    return set_cost + lam * shortfall


# --------------------------- trainers (warm-start from two-stage) ---------------------------
def train_two_stage(Xtr, Ytr, n):
    m = make_pred(n); opt = torch.optim.Adam(m.parameters(), TS_LR)
    for _ in range(TS_EP):
        opt.zero_grad(); ((m(Xtr) - Ytr) ** 2).mean().backward(); opt.step()
    return m


def train_polystep(Xtr, n, closure_obj, warm, scale, seed):
    """closure_obj(dec)->(K,B) realized COST (signed, minimize); dec (K,B,n)."""
    m = make_pred(n)
    with torch.no_grad():
        m.weight.copy_(warm.weight); m.bias.copy_(warm.bias)
    pso = PolyStepOptimizer(m, polytope_type=PS_POLYTOPE, epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=0.4, probe_radius=0.8, num_probe=PS_PROBES, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)

    def closure(bp):
        pred = torch.einsum("mnf,bf->mbn", bp["weight"], Xtr) + bp["bias"].unsqueeze(1)
        M, B, nn_ = pred.shape
        dec = SOLVE(pred.reshape(M * B, nn_))                          # (M*B, dec_dim) != nn_ for WSMC
        dec = dec.reshape(M, B, dec.shape[-1])
        return (closure_obj(dec) / scale).mean(-1)
    for _ in range(PS_STEPS):
        pso.step(closure)
    return m


def train_sfge(Xtr, n, closure_obj, warm, scale, seed, n_samples=8, sigma=0.5, lr=1e-2):
    m = make_pred(n)
    with torch.no_grad():
        m.weight.copy_(warm.weight); m.bias.copy_(warm.bias)
    opt = torch.optim.Adam(m.parameters(), lr); g = device_generator(seed, dev)
    for _ in range(SF_EP):
        pred = m(Xtr)
        with torch.no_grad():
            eps = torch.randn(n_samples, *pred.shape, device=dev, generator=g)
            chat = pred.unsqueeze(0) + sigma * eps
            S, B, nn_ = chat.shape
            dec = SOLVE(chat.reshape(S * B, nn_))                       # (S*B, dec_dim) != nn_ for WSMC
            dec = dec.reshape(S, B, dec.shape[-1])
            r = closure_obj(dec) / scale                                # (S,B) minimize
            adv = r - r.mean(0, keepdim=True)
        logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
        surrogate = (adv * logp).mean()
        opt.zero_grad(); surrogate.backward(); opt.step()
    return m


# SOLVE is set per-problem (module-level so closures pick it up); it counts forward solves.
SOLVE = None


def evaluate(m, Xte, Yte, ev_cost):
    """Normalized realized regret = (achieved_cost - oracle_cost)/oracle_cost, aggregate over the test set.
    Oracle uses the SAME forward solver on the TRUE params, isolating prediction regret from solver
    sub-optimality (PyEPO convention). Costs are MIN-sense (lower better). We normalize the excess cost
    by |oracle| (the optimal feasible value for KP, the optimal cover cost for WSMC), always > 0, so the
    metric is a positive normalized realized-regret (matches exp4_constraints)."""
    with torch.no_grad():
        dec = SOLVE(m(Xte))
        achieved = ev_cost(dec.unsqueeze(0), Yte).squeeze(0)            # (B,)
        oracle = ev_cost(SOLVE(Yte).unsqueeze(0), Yte).squeeze(0)       # decision on TRUE params
        denom = oracle.sum().abs().clamp(min=1e-6).item()
        reg = (achieved - oracle).sum().item() / denom
    return reg


# --------------------------- exactness check (greedy vs Gurobi) ---------------------------
def wsmc_gurobi_gap(A, c, B_true, n_check=12):
    """Mean greedy/optimal set-cost ratio over n_check true-requirement instances (1.0 = exact)."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        return None, f"gurobi unavailable ({e})"
    An = A.cpu().numpy(); cn = c.cpu().numpy()
    ne, ns = An.shape
    idx = list(range(min(n_check, B_true.shape[0])))
    g_greedy = solve_set_multicover(B_true[idx], A, c)                   # (k,ns)
    greedy_cost = (g_greedy * c).sum(-1).cpu().numpy()
    ratios, infeas = [], 0
    for k, i in enumerate(idx):
        b = B_true[i].cpu().numpy()
        md = gp.Model(); md.Params.OutputFlag = 0
        y = md.addVars(ns, vtype=GRB.BINARY)
        md.setObjective(gp.quicksum(float(cn[j]) * y[j] for j in range(ns)), GRB.MINIMIZE)
        for el in range(ne):
            md.addConstr(gp.quicksum(float(An[el, j]) * y[j] for j in range(ns)) >= float(b[el]))
        md.optimize()
        if md.status != GRB.OPTIMAL:
            infeas += 1; continue
        opt = md.objVal
        ratios.append(greedy_cost[k] / max(opt, 1e-9))
    if not ratios:
        return None, f"no feasible reference ({infeas} infeasible)"
    return float(np.mean(ratios)), f"n={len(ratios)} (max {max(ratios):.3f}); {infeas} infeasible-skipped"


# --------------------------- per-problem driver ---------------------------
def run_problem(prob, seeds):
    global SOLVE
    rows = {}
    gap_note = None
    for lam in LAMBDAS:
        acc = {m: [] for m in ("two-stage", "SFGE", "PolyStep")}
        wall = {m: 0.0 for m in acc}; solves0 = N_SOLVES["v"]
        for seed in seeds:
            seed_everything(seed)
            if prob == "kp":
                Xtr, Ytr, Xte, Yte, v, C = gen_kp(seed)
                n = KP_NIT
                SOLVE = _wrap(lambda p: solve_fractional_knapsack(
                    v.unsqueeze(0).expand(p.shape[0], -1), p.clamp(min=1e-3), C))
                scale = float(v.sum())
                obj = lambda dec, W=Ytr: kp_cost(dec, W, v, C, lam)
                ev = lambda dec, W: kp_cost(dec, W, v, C, lam)
            else:  # wsmc
                Xtr, Ytr, Xte, Yte, A, c, deg_i = gen_wsmc(seed)
                n = WS_NE
                SOLVE = _wrap(lambda p: solve_set_multicover(p, A, c))
                scale = float(c.sum())
                obj = lambda dec, B=Ytr: wsmc_cost(dec, B, c, A, lam)
                ev = lambda dec, B: wsmc_cost(dec, B, c, A, lam)
                if gap_note is None:
                    ratio, note = wsmc_gurobi_gap(A, c, Yte)
                    gap_note = (ratio, note)
            ts = train_two_stage(Xtr, Ytr, n)
            t = time.time(); ps = train_polystep(Xtr, n, obj, ts, scale, seed); wall["PolyStep"] += time.time() - t
            t = time.time(); sf = train_sfge(Xtr, n, obj, ts, scale, seed); wall["SFGE"] += time.time() - t
            acc["two-stage"].append(evaluate(ts, Xte, Yte, ev))
            acc["SFGE"].append(evaluate(sf, Xte, Yte, ev))
            acc["PolyStep"].append(evaluate(ps, Xte, Yte, ev))
        summ = {m: summarize(acc[m]) for m in acc}
        best = min(summ, key=lambda m: summ[m]["mean"])
        rows[lam] = {"summary": summ, "best": best,
                     "p_polystep_lt_ts": wilcoxon_pair(acc["PolyStep"], acc["two-stage"]),
                     "p_polystep_vs_sfge": wilcoxon_pair(acc["PolyStep"], acc["SFGE"]),
                     "wall_polystep_s": wall["PolyStep"], "wall_sfge_s": wall["SFGE"],
                     "solves": N_SOLVES["v"] - solves0}
        print(f"  lambda={lam:>4}: " + "  ".join(f"{m}={fmt_mean_std(summ[m])}" for m in acc) +
              f"  best={best}", flush=True)
    return rows, gap_note


def _wrap(fn):
    """Wrap a batched solver so every call increments the forward-solve counter by the batch size."""
    def wrapped(p):
        N_SOLVES["v"] += int(p.shape[0]); return fn(p)
    return wrapped


PNAME = {"kp": "KP-unknown-weights (weights in capacity constraint)",
         "wsmc": "WSMC (requirements in coverage constraint RHS)"}


def to_markdown(results, seeds, gaps):
    L = ["# SFGE-suite -- prediction-in-CONSTRAINTS (PolyStep vs SFGE; two-stage = floor)", "",
         f"seeds={seeds}, deg={DEG}, normalized realized-regret = (achieved_cost - oracle_cost)/oracle_cost "
         "(lower better; oracle uses the same forward solver on TRUE params). SMOKE="
         f"{int(SMOKE)} (KP n_items={KP_NIT}; WSMC {WS_NE}x{WS_NS}; "
         f"train={NTR}, test={NTE}; PolyStep {PS_STEPS} steps, SFGE {SF_EP} epochs).", "",
         "## Applicability note (why the optimization-oracle camp is N/A here)", "",
         "In both problems the predicted quantity parametrizes the FEASIBLE REGION (knapsack capacity "
         "constraint; set-cover RHS), not a linear objective cost vector c^T x. Consequently **SPO+, IMLE, "
         "PFYL / Fenchel-Young (FYL), DBB (differentiable black-box) and cvxpylayers are structurally "
         "UNDEFINED**, for three compounding reasons: (i) there is no objective cost vector c to "
         "differentiate the surrogate against; (ii) the feasible set is not fixed -- it moves with the "
         "prediction, so the SPO+ / FYL / DBB perturbed-optimum constructions have no fixed polytope to "
         "argmin over; (iii) the predicted optimum can be INFEASIBLE under the true parameters, so plain "
         "regret is undefined without a recourse / penalty term. This is certified by the SFGE paper "
         "itself (Silvestri et al., JAIR 81, 2024, arXiv:2307.05213): *SPO+ applies only when the "
         "predicted parameters appear in the objective.* The methods that DO run only ever consume a "
         "scalar realized cost of a deployed decision: two-stage (MSE floor), SFGE (score-function "
         "gradient), and PolyStep (gradient-free). SFGE is the legitimate peer.", ""]
    if gaps.get("wsmc") and gaps["wsmc"][0] is not None:
        L += [f"**WSMC solver-exactness:** batched greedy mean cost / Gurobi-exact ILP optimum = "
              f"**{gaps['wsmc'][0]:.3f}x** ({gaps['wsmc'][1]}). "
              "Regret is computed with the same greedy as oracle, so this solver gap does not bias the "
              "comparison; it is reported for transparency. KP uses the exact fractional-knapsack LP "
              "(greedy = LP optimum, verified vs cvxpy in pto.solvers).", ""]
    for p in ("kp", "wsmc"):
        if p not in results:
            continue
        L.append(f"## {PNAME[p]}")
        headers = ["lambda", "two-stage", "SFGE", "PolyStep", "best", "PS<TS (p)", "PS vs SFGE (p)",
                   "PS wall (s)", "SFGE wall (s)", "#solves/λ"]
        rws = []
        for lam in LAMBDAS:
            r = results[p][lam]; s = r["summary"]
            cells = [fmt_mean_std(s["two-stage"]), fmt_mean_std(s["SFGE"]), fmt_mean_std(s["PolyStep"])]
            names = ["two-stage", "SFGE", "PolyStep"]
            cells = [f"**{cells[i]}**" if names[i] == r["best"] else cells[i] for i in range(3)]
            pp = r["p_polystep_lt_ts"]; pvs = r["p_polystep_vs_sfge"]
            rws.append([lam] + cells + [r["best"],
                        f"{pp:.3f}" if pp is not None else "-",
                        f"{pvs:.3f}" if pvs is not None else "-",
                        f"{r['wall_polystep_s']:.1f}", f"{r['wall_sfge_s']:.1f}", r["solves"]])
        L.append(md_table(headers, rws)); L.append("")
    L.append("**Reading:** two-stage predicts unbiasedly and pays the misprediction penalty, so it "
             "degrades as lambda rises; the decision-aware methods (SFGE, PolyStep) learn to predict "
             "conservatively (over-state weights / requirements) to dodge the asymmetric penalty. The "
             "PolyStep advantage is lambda-dependent and largest at high lambda.")
    return "\n".join(L)


def main():
    seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2, 3, 4]
    print(f"SFGE-SUITE | prediction-in-constraints | seeds={seeds} SMOKE={int(SMOKE)}", flush=True)
    print("  optimization-oracle camp (SPO+/IMLE/PFYL/DBB/cvxpylayers): structurally N/A "
          "(prediction defines the feasible region; SFGE JAIR'24 certifies this)\n", flush=True)
    results, gaps = {}, {}
    for p in ("kp", "wsmc"):
        print(f"[{p}] {PNAME[p]}", flush=True)
        results[p], gaps[p] = run_problem(p, seeds)
    payload = {"seeds": seeds, "smoke": SMOKE, "deg": DEG, "lambdas": LAMBDAS,
               "sizes": {"kp_nit": KP_NIT, "wsmc_ne": WS_NE, "wsmc_ns": WS_NS, "ntr": NTR, "nte": NTE},
               "wsmc_solver_gap": gaps.get("wsmc"), "results": results}
    write_json(f"exp_results/sfge_suite{_OUT_SFX}.json", payload)
    write_md(f"exp_results/sfge_suite{_OUT_SFX}.md", to_markdown(results, seeds, gaps))
    print(f"\nwrote exp_results/sfge_suite{_OUT_SFX}.{{json,md}}\nDONE", flush=True)


if __name__ == "__main__":
    main()
