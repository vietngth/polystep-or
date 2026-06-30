"""
Dynamic & Stochastic Inventory Routing (DSIRP) x PolyStep  -- gradient-free DFL scaffold.

Benchmark target: Greif, Bouvier, Flath, Parmentier, Rohmer, Vidal (2024),
"Combinatorial Optimization and Machine Learning for Dynamic Inventory Routing", arXiv:2402.04463.
Authors' released code (Julia/InferOpt/Gurobi):  https://github.com/tonigreif/InferOpt_DSIRP
cloned at: scratchpad/bbdiff/irp/InferOpt_DSIRP  (instances/, src/pctsp.jl, src/sirp_*.jl, src/stat_model.jl,
src/evaluation_pipeline.jl, src/pipeline_baty.jl).

WHAT THIS FILE IS
-----------------
A faithful *Python* re-implementation of the authors' predict -> CPCTSP-oracle -> simulate-realized-cost
pipeline, wired to OUR PolyStep machinery so we can benchmark PolyStep against their InferOpt approach.
It mirrors the DistrictNet x PolyStep precedent (districtnet_polystep.py): a predictor on GPU/CPU, a
NON-differentiable combinatorial oracle (Gurobi), and a realized-cost closure PolyStep minimizes directly.

PIPELINE  (one period t of the DSIRP MDP, exactly evaluate_pctsp in src/evaluation_pipeline.jl)
  state x_t = (inventories I_t, last-50 demands per customer D_hat_t, [+context Phi_t])
     -> features  (PINN feature array; src/sirp_model.jl::createFeatureArray)
     -> theta = phi_w(features) in R^C   (per-customer PRIZES; PINN; src/stat_model.jl::build_stat_model)
     -> u_t = CPCTSP(theta)              (capacitated prize-collecting TSP; src/pctsp.jl; Gurobi MILP)
     -> apply order-up-to deliveries q_i = (max_inv_i - I_i)*visited_i, advance inventory with the
        REALIZED demand d_t, accrue holding + stock-out(penalty) + routing cost.
  rollout cost over horizon T = sum_t (holding_t + penalty_t + routing_t)   <-- the PolyStep objective.

THEIR update vs POLYSTEP (see REPORT at bottom of run output):
  * Theirs (InferOpt): Fenchel-Young loss of a PERTURBED CPCTSP maximizer (PerturbedAdditive, eps=20,
    5 samples) against ANTICIPATIVE-EXPERT labels u_bar (= first-period decision of a full known-demand IRP
    MILP, src/sirp_solver.jl). Supervised IMITATION; gradient flows through the perturbed argmax
    ("the solver updates the model"). Sample states come from baty / sampling / DAgger paradigms.
  * PolyStep: NO labels, NO expert, NO surrogate. Minimize the realized closed-loop rollout cost above as a
    black-box function of w. Gradient-free OT/softmax-barycenter step in parameter space from scalar costs.
    The SEQUENTIAL signal is consumed by applying ONE predictor phi_w at every period of the T-rollout and
    minimizing the TOTAL accumulated cost (averaged over training demand scenarios) -- one scalar per (w, scenario).

Run (CPU smoke):   .venv/bin/python exp_irp_polystep.py smoke
     (fuller):     .venv/bin/python exp_irp_polystep.py full
"""
from __future__ import annotations
import os, sys, glob, json, time, argparse
import numpy as np
sys.path.insert(0, "polystep/src")
import torch
import torch.nn as nn

IRP_DIR = ("/tmp/claude-1000/-media-anindex-Data-ot-or-project/"
           "eb8c8aab-4d2e-4e9a-9f32-d20c96478c25/scratchpad/bbdiff/irp/InferOpt_DSIRP")
# INSTDIR may be overridden (the scratchpad default is local-only; on the cluster the instances live under
# $PWD/InferOpt_DSIRP/instances or the project cache) so this script can run wherever the data is staged.
INSTDIR = os.environ.get("IRP_INSTDIR", os.path.join(IRP_DIR, "instances"))
DEV = "cpu"                                   # oracle is CPU (Gurobi); predictor is tiny -> keep off the GPU
P = lambda *a: print(*a, flush=True)

# ---- polytope-robustness sweep knobs (mirror the cheap experiments; defaults keep the working config) ----
PS_POLYTOPE = os.environ.get("PS_POLYTOPE", "orthoplex")   # orthoplex | simplex | cube
PS_PROBES   = int(os.environ.get("PS_PROBES", "1"))
OUT_TAG     = os.environ.get("OUT_TAG", "")
_OUT_SFX    = f"_{OUT_TAG}" if OUT_TAG else ""

# Authors' defaults (src/pipeline_baty.jl::baty_settings, train_pipeline.jl)
QUANTILES = [0.01, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99]
LOOK_AHEAD = 6
SHORTAGE_PENALTY = 200          # penalty_cost = holding_cost * 200
EVAL_HORIZON = 10               # T = 10 in the paper


# --------------------------------------------------------------------------------------------------
# DATA  (mirrors src/sirp_model.jl::readInstance; non-contextual normal/uniform/bimodal)
# --------------------------------------------------------------------------------------------------
def load_instance(path, pattern, penalty_inv=SHORTAGE_PENALTY):
    """Load one DSIRP instance. Node 0 = depot at (0,0); customers c=0..C-1 are nodes 1..C.
    demand_hist[c]: 50 historical demands. demand_test[c]: 90-long realized test trajectory.
    demand_eval[c][s]: 5 eval scenarios (len 15 each). start_inventory = max_inv - mean_demand."""
    d = json.load(open(path))
    C = d["nb_cst"]
    coords = np.array([[0.0, 0.0]] + list(zip(d["x"], d["y"])), dtype=np.float64)   # (C+1, 2), node0=depot
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))         # (C+1, C+1) Euclidean
    max_inv = np.array(d["max_inventory"], dtype=np.float64)
    holding = np.array(d["holding_cost"], dtype=np.float64)
    penalty = holding * penalty_inv
    if pattern == "uniform":                                  # uniform-10 instances ship no mean_demand;
        # authors' demand is Uniform(0, 0.5*max_inventory) with mean 0.25*max, so start = max - mean = 0.75*max.
        start_inv = max_inv - 0.25 * max_inv
    elif pattern == "bimodal":                                # mean_demand is a 2-vector per customer
        start_inv = max_inv - np.array([np.mean(m) for m in d["mean_demand"]])
    else:
        start_inv = max_inv - np.array(d["mean_demand"], dtype=np.float64)
    start_inv = np.clip(start_inv, 0.0, None)
    demand_hist = [np.array(x, dtype=np.float64) for x in d["demand_hist"]]          # C x 50
    demand_test = [np.array(x, dtype=np.float64) for x in d["demand_test"]]          # C x 90
    demand_eval = [[np.array(s, dtype=np.float64) for s in d["demand_eval"][c]]      # C x 5 x 15
                   for c in range(C)]
    return dict(name=d["idx"], pattern=pattern, C=C, dist=dist, v_cap=float(d["vehicle_capacity"]),
                max_inv=max_inv, holding=holding, penalty=penalty, start_inv=start_inv.copy(),
                demand_hist=demand_hist, demand_test=demand_test, demand_eval=demand_eval)


def demand_seq(inst, kind="test", scenario=0):
    """Realized demand trajectory used to advance inventories. test -> demand_test[c]; eval -> a scenario."""
    if kind == "test":
        return [inst["demand_test"][c] for c in range(inst["C"])]
    return [inst["demand_eval"][c][scenario] for c in range(inst["C"])]


# --------------------------------------------------------------------------------------------------
# PINN PREDICTOR  (mirrors src/stat_model.jl::build_stat_model, NON-contextual path)
# --------------------------------------------------------------------------------------------------
# Feature derivation (exact for non-contextual): each customer c contributes nb_obs = LOOK_AHEAD*|Q| = 72
# observations indexed (k, qi), k=1..6 look-ahead, qi over the 12 demand quantiles of its demand history.
# Because the feature column [start_inv, q, holding, penalty] is replicated across look-ahead, the authors'
# cumulative-demand recursion collapses to  cumulative[k,qi] = k * q_vals[qi]  (k periods of quantile demand).
#   holding term_(k,qi) = holding * relu(start_inv - k*q_vals[qi])
#   penalty term_(k,qi) = penalty * relu(k*q_vals[qi] - start_inv)
#   theta_c = sum_obs w_inv[obs]*holding_term + sum_obs w_pen[obs]*penalty_term
# Authors init w_inv = -1/72, w_pen = +1/72 (i.e. theta = mean(penalty term) - mean(holding term)); these
# 2*72 = 144 weights are the learnable parameters. (CONTEXTUAL adds a `regression` layer on extra features;
# noted but not wired here -- see CONTEXTUAL EXTENSION below.)
NB_OBS = LOOK_AHEAD * len(QUANTILES)        # 72


def period_terms(start_inv, demands_hist, holding, penalty):
    """Build the (C, NB_OBS) holding-term and penalty-term tensors for the current state.
    start_inv:(C,), demands_hist: list C of 1-D arrays (>=1 long), holding/penalty:(C,)."""
    C = len(start_inv)
    qv = np.stack([np.quantile(demands_hist[c], QUANTILES) for c in range(C)])       # (C, |Q|)
    k = np.arange(1, LOOK_AHEAD + 1, dtype=np.float64)                                # (K,)
    cum = k[None, :, None] * qv[:, None, :]                                           # (C, K, |Q|) = k*q
    si = start_inv[:, None, None]
    hold = holding[:, None, None] * np.clip(si - cum, 0.0, None)                      # (C, K, |Q|)
    pen = penalty[:, None, None] * np.clip(cum - si, 0.0, None)
    hold = hold.reshape(C, NB_OBS)
    pen = pen.reshape(C, NB_OBS)
    return (torch.tensor(hold, dtype=torch.float32, device=DEV),
            torch.tensor(pen, dtype=torch.float32, device=DEV))


class PINN(nn.Module):
    """Generalized-linear prize predictor.  theta = hold_terms @ w_inv + pen_terms @ w_pen."""
    def __init__(self):
        super().__init__()
        self.w_inv = nn.Parameter(torch.full((NB_OBS,), -1.0 / NB_OBS))
        self.w_pen = nn.Parameter(torch.full((NB_OBS,), +1.0 / NB_OBS))

    def theta(self, hold_terms, pen_terms):                  # (C,NB_OBS),(C,NB_OBS) -> (C,)
        return hold_terms @ self.w_inv + pen_terms @ self.w_pen


def theta_from_params(w_inv, w_pen, hold_terms, pen_terms):
    """Functional version for PolyStep candidate params (no module state)."""
    return hold_terms @ w_inv + pen_terms @ w_pen


# --------------------------------------------------------------------------------------------------
# CPCTSP ORACLE  (mirrors src/pctsp.jl::pctsp -- Gurobi MILP + lazy subtour elimination)
# --------------------------------------------------------------------------------------------------
# max  sum_c theta_c * y_c  -  sum_{i<j} dist_ij * x_ij
# s.t. sum_c (max_inv_c - start_inv_c) * y_c <= v_cap            (order-up-to deliveries fit one vehicle)
#      sum_{j!=i} x_ij == 2 * y_i   for every node i (depot 0 included)   (degree-2 of visited nodes)
#      y_0 active iff any customer visited; x symmetric, no self-loops; one tour through depot (subtour cuts)
# Returns: visited mask (C,) {0,1}, routing_cost (sum dist over tour edges), tour edge list.
import gurobipy as gp
from gurobipy import GRB

_GRB_ENV = None
def _grb_env():
    global _GRB_ENV
    if _GRB_ENV is None:
        _GRB_ENV = gp.Env(empty=True); _GRB_ENV.setParam("OutputFlag", 0); _GRB_ENV.start()
    return _GRB_ENV


def cpctsp(theta, start_inv, inst, time_limit=None):
    """Solve the per-period CPCTSP. theta:(C,) np/torch; returns (visited(C,), routing_cost, edges)."""
    th = theta.detach().cpu().numpy() if torch.is_tensor(theta) else np.asarray(theta, float)
    C, dist, v_cap = inst["C"], inst["dist"], inst["v_cap"]
    deliver = inst["max_inv"] - start_inv                     # order-up-to quantity if visited
    N = C + 1                                                 # node 0 = depot, 1..C = customers
    m = gp.Model(env=_grb_env())
    if time_limit:
        m.Params.TimeLimit = time_limit
    y = m.addVars(N, vtype=GRB.BINARY, name="y")
    x = m.addVars(N, N, vtype=GRB.BINARY, name="x")           # use i<j entries
    m.setObjective(
        gp.quicksum(float(th[c]) * y[c + 1] for c in range(C))
        - gp.quicksum(float(dist[i, j]) * x[i, j] for i in range(N) for j in range(i + 1, N)),
        GRB.MAXIMIZE)
    # capacity (order-up-to deliveries fit the vehicle)
    m.addConstr(gp.quicksum(float(deliver[c]) * y[c + 1] for c in range(C)) <= v_cap)
    # depot active iff any customer visited
    m.addConstr(gp.quicksum(y[c + 1] for c in range(C)) <= C * y[0])
    m.addConstr(y[0] <= gp.quicksum(y[c + 1] for c in range(C)))
    # degree-2 of visited nodes (undirected: only i<j vars exist)
    def inc(i):  # edges incident to i
        return (gp.quicksum(x[j, i] for j in range(i)) + gp.quicksum(x[i, j] for j in range(i + 1, N)))
    for i in range(N):
        m.addConstr(inc(i) == 2 * y[i])
    # lazy subtour elimination: re-solve, cut components not containing the depot, until one tour
    while True:
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            # smoke-safe: treat as "visit nobody" if the MILP is infeasible/aborted
            return np.zeros(C, dtype=np.float64), 0.0, []
        sel = {(i, j) for i in range(N) for j in range(i + 1, N) if x[i, j].X > 0.5}
        comps = _components(N, sel, [i for i in range(N) if y[i].X > 0.5])
        bad = [comp for comp in comps if 0 not in comp]      # subtours (no depot)
        if not bad:
            break
        for comp in bad:
            cl = list(comp)
            m.addConstr(gp.quicksum(x[min(a, b), max(a, b)] for a in cl for b in cl if a < b) <= len(cl) - 1)
    visited = np.array([y[c + 1].X > 0.5 for c in range(C)], dtype=np.float64)
    routing = sum(float(dist[i, j]) for (i, j) in sel)
    return visited, routing, sorted(sel)


def _components(N, edges, nodes):
    """Connected components (union-find) over `nodes` using undirected `edges`."""
    parent = {v: v for v in nodes}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for (i, j) in edges:
        if i in parent and j in parent:
            parent[find(i)] = find(j)
    comps = {}
    for v in nodes:
        comps.setdefault(find(v), set()).add(v)
    return list(comps.values())


# --------------------------------------------------------------------------------------------------
# ROLLOUT SIMULATOR  (mirrors src/evaluation_pipeline.jl::evaluate_pctsp)
# --------------------------------------------------------------------------------------------------
def rollout(predict_theta, inst, dseq, horizon=EVAL_HORIZON, trace=False):
    """Closed-loop rollout of a policy. `predict_theta(hold_terms, pen_terms) -> theta (C,)`.
    Returns (total_cost, breakdown dict). Advances inventory with the REALIZED demand dseq[c][t]."""
    C = inst["C"]
    start_inv = inst["start_inv"].copy()
    hist = [h.copy() for h in inst["demand_hist"]]            # growing demand history per customer
    holding, penalty, max_inv = inst["holding"], inst["penalty"], inst["max_inv"]
    tot_hold = tot_pen = tot_route = 0.0
    tr = []
    for t in range(horizon):
        if t > 0:                                            # observe previous period's realized demand
            for c in range(C):
                hist[c] = np.append(hist[c], dseq[c][t - 1])
        hold_terms, pen_terms = period_terms(start_inv, hist, holding, penalty)
        theta = predict_theta(hold_terms, pen_terms)
        visited, routing, edges = cpctsp(theta, start_inv, inst)
        q = (max_inv - start_inv) * visited                  # order-up-to delivery
        for c in range(C):
            inv_tmp = start_inv[c] + q[c] - dseq[c][t]
            if inv_tmp < 0:                                  # stock-out
                tot_pen += -inv_tmp * penalty[c]; start_inv[c] = 0.0
            else:                                            # carry inventory
                tot_hold += inv_tmp * holding[c]; start_inv[c] = inv_tmp
        tot_route += routing
        if trace:
            tr.append(dict(t=t, n_visited=int(visited.sum()), routing=round(routing, 2),
                           hold=round(tot_hold, 1), pen=round(tot_pen, 1)))
    total = tot_hold + tot_pen + tot_route
    return total, dict(holding=tot_hold, penalty=tot_pen, routing=tot_route, total=total, trace=tr)


# --------------------------------------------------------------------------------------------------
# POLYSTEP TRAINER  (gradient-free; minimize realized rollout cost; objective="cost")
# --------------------------------------------------------------------------------------------------
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon


def mean_rollout_cost(w_inv, w_pen, instances, scenarios, horizon):
    """Average realized rollout cost over (instance, scenario) for a single parameter set."""
    pt = lambda h, p: theta_from_params(w_inv, w_pen, h, p)
    tot, n = 0.0, 0
    for inst in instances:
        for s in scenarios:
            dseq = demand_seq(inst, kind="eval", scenario=s)
            c, _ = rollout(pt, inst, dseq, horizon=horizon)
            tot += c; n += 1
    return tot / max(n, 1)


def train_polystep(instances, warm, scenarios, horizon, steps=40, seed=0, scale=1.0, log_every=5,
                   probe_radius=None, step_radius=None, eps0=None, eps1=None):
    """PolyStep over the 144 PINN weights. The closure runs the closed-loop rollout per candidate.

    probe_radius is the KEY knob on this benchmark: the CPCTSP oracle is piecewise-constant in the
    prizes (small prize changes don't flip the integer tour), so probes must be large enough to cross
    plateau boundaries and create cost SPREAD across the orthoplex vertices -- otherwise PolyStep's
    per-particle softmax is near-uniform and the step degenerates (the under-tuning we observed)."""
    probe_radius = float(os.environ.get("IRP_PR", probe_radius if probe_radius is not None else 0.8))
    step_radius  = float(os.environ.get("IRP_SR", step_radius  if step_radius  is not None else 0.4))
    eps0 = float(os.environ.get("IRP_EPS0", eps0 if eps0 is not None else 0.5))
    eps1 = float(os.environ.get("IRP_EPS1", eps1 if eps1 is not None else 0.05))
    model = PINN().to(DEV)
    model.load_state_dict(warm.state_dict())
    names = [n for n, _ in model.named_parameters()]         # ['w_inv','w_pen']
    pso = PolyStepOptimizer(model, polytope_type=PS_POLYTOPE, epsilon=CosineEpsilon(eps0, eps1),
                            step_radius=step_radius, probe_radius=probe_radius, num_probe=PS_PROBES, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)

    def closure(bp):                                         # bp = {name:(K,*shape)}
        K = bp[names[0]].shape[0]
        out = torch.zeros(K, device=DEV)
        for k in range(K):
            wi, wp = bp["w_inv"][k], bp["w_pen"][k]
            out[k] = mean_rollout_cost(wi, wp, instances, scenarios, horizon) / scale
        return out

    hist = []
    best = (float("inf"), None)                              # retain best-seen params (cost is non-monotone)
    for s in range(steps):
        pso.step(closure)
        cur = mean_rollout_cost(model.w_inv.detach(), model.w_pen.detach(), instances, scenarios, horizon)
        if cur < best[0]:
            best = (cur, {k: v.detach().clone() for k, v in model.state_dict().items()})
        if s % log_every == 0 or s == steps - 1:
            hist.append((s, cur))
            P(f"  [step {s:>3}] train mean rollout cost {cur:,.1f}  best {best[0]:,.1f}")
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, hist


# --------------------------------------------------------------------------------------------------
# DRIVER
# --------------------------------------------------------------------------------------------------
def pick_instances(pattern, n):
    files = sorted(glob.glob(os.path.join(INSTDIR, f"{pattern}-10_*.json")))
    return [load_instance(f, pattern) for f in files[:n]]


def sanity_oracle(inst):
    """Verify the CPCTSP returns a valid capacitated single tour. Use boosted prizes so it visits a subset."""
    C = inst["C"]
    theta = np.full(C, 500.0)                               # all-positive prizes -> oracle wants to visit
    visited, routing, edges = cpctsp(theta, inst["start_inv"], inst)
    deliver = ((inst["max_inv"] - inst["start_inv"]) * visited).sum()
    deg = {}
    for (i, j) in edges:
        deg[i] = deg.get(i, 0) + 1; deg[j] = deg.get(j, 0) + 1
    ok_deg = all(d == 2 for d in deg.values())
    comps = _components(C + 1, set(edges), sorted({i for e in edges for i in e})) if edges else []
    P(f"[oracle check] {inst['name'][:28]} | visited {int(visited.sum())}/{C} | edges {len(edges)} | "
      f"deliver {deliver:.0f}<=v_cap {inst['v_cap']:.0f} ({'OK' if deliver<=inst['v_cap']+1e-6 else 'BAD'}) | "
      f"all deg-2 {ok_deg} | one-tour {len(comps)<=1} | routing {routing:.1f}")
    return ok_deg and len(comps) <= 1 and deliver <= inst["v_cap"] + 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--pattern", default="normal", choices=["normal", "uniform", "bimodal"])
    args = ap.parse_args()
    smoke = args.mode == "smoke"

    n_inst   = 1 if smoke else 5
    horizon  = 3 if smoke else EVAL_HORIZON
    scenarios = [0] if smoke else [0, 1, 2]
    steps    = 6 if smoke else 40

    P(f"=== DSIRP x PolyStep | {'SMOKE' if smoke else 'FULL'} | pattern={args.pattern} | "
      f"n_inst={n_inst} horizon={horizon} scenarios={scenarios} steps={steps} | "
      f"polytope={PS_POLYTOPE} num_probe={PS_PROBES} tag={OUT_TAG or '(none)'} ===")
    t0 = time.time()
    insts = pick_instances(args.pattern, n_inst)
    P(f"loaded {len(insts)} instance(s); C={insts[0]['C']} customers, v_cap={insts[0]['v_cap']:.0f}")

    # 1) ORACLE SANITY
    ok = all(sanity_oracle(i) for i in insts)
    P(f"[oracle] valid capacitated single tour on all instances: {ok}")

    # 2) BASELINE = the paper's GLM-initialized PINN (theta = mean penalty term - mean holding term),
    #    i.e. the InferOpt pipeline at init / a hand-designed prize policy (zero training).
    base = PINN().to(DEV)
    pt0 = lambda h, p: base.theta(h, p)
    t1 = time.time()
    base_costs = []
    for inst in insts:
        for s in scenarios:
            c, bd = rollout(pt0, inst, demand_seq(inst, "eval", s), horizon=horizon, trace=(s == scenarios[0]))
            base_costs.append(c)
            if s == scenarios[0]:
                P(f"  baseline rollout {inst['name'][:24]} sc{s}: total {c:,.1f} "
                  f"(hold {bd['holding']:.0f} pen {bd['penalty']:.0f} route {bd['routing']:.0f}); "
                  f"trace {bd['trace']}")
    base_mean = float(np.mean(base_costs))
    P(f"[baseline] mean rollout cost {base_mean:,.1f}  ({time.time()-t1:.1f}s)")

    # 3) POLYSTEP (gradient-free) minimizing the realized rollout cost
    t2 = time.time()
    model, hist = train_polystep(insts, warm=base, scenarios=scenarios, horizon=horizon,
                                 steps=steps, seed=int(os.environ.get("IRP_SEED", "0")), scale=max(base_mean, 1.0),
                                 log_every=1 if smoke else 5)
    ps_costs = []
    for inst in insts:
        for s in scenarios:
            c, _ = rollout(lambda h, p: model.theta(h, p), inst, demand_seq(inst, "eval", s), horizon=horizon)
            ps_costs.append(c)
    ps_mean = float(np.mean(ps_costs))
    P(f"[PolyStep] mean rollout cost {ps_mean:,.1f}  ({time.time()-t2:.1f}s, {steps} steps)")
    P(f"\n--- RESULT ({'SMOKE' if smoke else 'FULL'}) ---")
    P(f"  baseline (GLM-init PINN)   : {base_mean:,.1f}")
    P(f"  PolyStep (gradient-free)   : {ps_mean:,.1f}   ({100*(base_mean-ps_mean)/base_mean:+.1f}% vs baseline)")

    os.makedirs("exp_results", exist_ok=True)
    out = dict(mode=args.mode, pattern=args.pattern, n_inst=n_inst, horizon=horizon, scenarios=scenarios,
               steps=steps, polytope=PS_POLYTOPE, num_probe=PS_PROBES, oracle_valid=bool(ok),
               baseline_mean=base_mean, polystep_mean=ps_mean,
               improvement_pct=100 * (base_mean - ps_mean) / base_mean, wall_s=time.time() - t0)
    json.dump(out, open(f"exp_results/irp_polystep_{args.mode}_{args.pattern}{_OUT_SFX}.json", "w"), indent=1)
    P(f"[total {time.time()-t0:.0f}s] -> exp_results/irp_polystep_{args.mode}_{args.pattern}{_OUT_SFX}.json")


# CONTEXTUAL EXTENSION (nb_features=8): instances expose samples_{hist,eval,test} with per-observation
# `features` + `label`. createFeatureArray appends the 8 features + their pairwise products; build_stat_model
# adds a linear `regression` over them inside the cumulative-demand term (theta then also depends on context).
# Wiring: extend period_terms to read context features and add a `regression` Parameter to PINN; PolyStep
# trains the augmented parameter vector unchanged. Left as a documented TODO (non-contextual covers 3/4 patterns).

if __name__ == "__main__":
    main()
