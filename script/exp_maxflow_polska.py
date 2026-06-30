"""
MAX-FLOW POLSKA (cpaior23 Branch-and-Learn benchmark) -- predict-then-optimize with the
predicted parameters living in the CONSTRAINTS (edge capacities).

PROBLEM (from baselines/cpaior23_branch_and_learn/Max Flow/):
  - Fixed directed graph POLSKA: 12 nodes, 18 edges (all u<v -> DAG), source=0, sink=11.
  - Each instance gives, per edge, 8 features and a TRUE capacity. A LINEAR predictor
    (shared 8->1 map, exactly like Ridge_para_POLSKA) maps features -> 18 predicted capacities.
  - Deploy: route max-flow on the PREDICTED-capacity graph (augmenting paths).
  - Realize (Correction Function A, ported EXACTLY from test_corrA.py `corr_maxFlow`): along each
    predicted augmenting path the realized contribution uses the TRUE capacity of the path's
    PREDICTED-bottleneck edge; accumulate predicted flow `preFlow`; after augmenting, if any edge's
    preFlow exceeds its TRUE capacity, scale the whole realized flow by tau=min(true/preFlow) over
    violated edges. Realized = a flow VALUE to MAXIMIZE.
  - Regret = optimal_max_flow(TRUE caps) - realized_corr_flow(predicted).

WHY THE GRADIENT/SURROGATE CAMP IS N/A (structural):
  SPO+ / PFYL / IMLE / cvxpylayers all need an optModel with a FIXED feasible region and a PREDICTED
  OBJECTIVE COST VECTOR (regret/surrogates are c^T w over that region). Here the prediction
  parametrizes the CONSTRAINT (edge capacities), the feasible region itself moves with the
  prediction, and the realized objective is non-linear & non-differentiable in the prediction
  (augmenting-path bottleneck selection + tau rescale). There is no cost vector to consume. -> N/A.

WHO CAN RUN: two-stage (MSE on capacities), SFGE, PolyStep -- they only evaluate the realized
outcome of a deployed decision (a black box). PolyStep/SFGE optimize that realized value directly.

Data row format (confirmed from BAL.cpp parser, NOT guessed):
    each instance = 18 rows (one per edge); each row = [benchmarkId, f0..f7 (8 feats), realCap]
    -> col0 = id, cols 1..8 = features, col9 = TRUE capacity.
  (Note: Ridge_para_POLSKA aligns slightly better with col8 than col9 -- a quirk of the released
   warm-start params; we follow BAL.cpp's parser, which is the actual C++ trainer/evaluator, and
   col9 is what De_maxFlow / the corrA evaluation pipeline uses as the true capacity.)

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 .venv/bin/python exp_maxflow_polska.py
"""
from __future__ import annotations
import os, sys, time, json
os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, "polystep/src")
import numpy as np
import torch
import torch.nn as nn
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from pto.multiseed import summarize, md_table, write_json, write_md

P = lambda *a: print(*a, flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
# --- polytope x probe sweep (env-overridable; defaults preserve original behavior) ---
PS_POLYTOPE = os.environ.get("PS_POLYTOPE", "orthoplex")
PS_PROBES = int(os.environ.get("PS_PROBES", "1"))
OUT_TAG = os.environ.get("OUT_TAG", "")
_OUT_SFX = f"_{OUT_TAG}" if OUT_TAG else ""

# ------------------------------------------------------------------ constants / graph
DATA = "baselines/cpaior23_branch_and_learn/Max Flow/data"
NODE, EDGE, FEAT = 12, 18, 8
SRC, SINK = 0, NODE - 1
MAXAUG = 128            # Edmonds-Karp augmentation cap (<= V*E/2 = 108); loop breaks early
D_PARAMS = FEAT + 1     # linear predictor params (weight 8 + bias 1) -> for the 2*D*n*steps accounting

_arc = np.loadtxt(f"{DATA}/graph_POLSKA.txt", dtype=int)[1:, :]      # skip "source sink" line
assert _arc.shape == (EDGE, 2)
ARC_U = torch.tensor(_arc[:, 0], dtype=torch.long, device=dev)
ARC_V = torch.tensor(_arc[:, 1], dtype=torch.long, device=dev)
ARC_LIST = [(int(u), int(v)) for u, v in _arc]


# ================================================================== batched GPU max-flow
def _caps_to_mat(caps):
    """caps (B,EDGE) -> residual matrix (B,NODE,NODE) with forward edges filled, reverse=0."""
    B = caps.shape[0]
    M = torch.zeros(B, NODE, NODE, dtype=caps.dtype, device=caps.device)
    M[:, ARC_U, ARC_V] = caps
    return M


def _bfs(R):
    """Batched Ford-Fulkerson BFS that reproduces the reference FIFO + increasing-index parent tree.

    R (B,NODE,NODE) residual. Returns (parent (B,NODE) long, found (B,) bool).
    Simulates the FIFO queue by dequeue-time: at step dq each batch element processes the node with
    enqueue-time == dq, discovering unvisited positive-residual neighbours in increasing index order
    and stamping them with consecutive enqueue-times (== current enqueued count + rank). This yields
    the *exact* parent that the reference Python BFS would assign for every node on the s-t path.
    """
    B = R.shape[0]
    idx = torch.arange(NODE, device=R.device)
    T = torch.full((B, NODE), -1, dtype=torch.long, device=R.device)   # enqueue time
    T[:, SRC] = 0
    parent = torch.full((B, NODE), -1, dtype=torch.long, device=R.device)
    count = torch.ones(B, dtype=torch.long, device=R.device)           # nodes enqueued so far
    ar = torch.arange(B, device=R.device)
    for dq in range(NODE):
        u_oh = (T == dq)                                               # (B,NODE) one-hot dequeued node
        has_u = u_oh.any(1)
        Radj = torch.einsum('bn,bnm->bm', u_oh.to(R.dtype), R)         # R[b, u_b, :]
        u_idx = (u_oh.long() * idx).sum(1)                            # (B,)
        newly = (Radj > 0) & (T == -1) & has_u.unsqueeze(1)
        rank = torch.cumsum(newly.long(), dim=1) - newly.long()       # exclusive cumsum over index
        newT = count.unsqueeze(1) + rank
        T = torch.where(newly, newT, T)
        parent = torch.where(newly, u_idx.unsqueeze(1), parent)
        count = count + newly.long().sum(1)
    found = T[:, SINK] != -1
    return parent, found


def _walk_path(preG, parent, active):
    """Walk sink->source on `parent`; return (preDf, bn_u, bn_v, pathMask).

    preDf = predicted bottleneck value (min preG residual along path; ties -> closest to sink, via
    strict '<' which keeps the first occurrence encountered from the sink). bn_u,bn_v = that edge.
    pathMask (B,NODE,NODE) marks forward edges (parent[v]->v) on the path.
    """
    B = preG.shape[0]
    ar = torch.arange(B, device=preG.device)
    cur = torch.full((B,), SINK, dtype=torch.long, device=preG.device)
    minval = torch.full((B,), float('inf'), dtype=preG.dtype, device=preG.device)
    bn_u = torch.full((B,), -1, dtype=torch.long, device=preG.device)
    bn_v = torch.full((B,), -1, dtype=torch.long, device=preG.device)
    pathMask = torch.zeros(B, NODE, NODE, dtype=preG.dtype, device=preG.device)
    for _ in range(NODE):
        walking = active & (cur != SRC)
        p = parent.gather(1, cur.unsqueeze(1)).squeeze(1)
        valid = walking & (p >= 0)
        pc = p.clamp(min=0)
        preR = preG[ar, pc, cur]
        better = valid & (preR < minval)
        minval = torch.where(better, preR, minval)
        bn_u = torch.where(better, p, bn_u)
        bn_v = torch.where(better, cur, bn_v)
        cur_pm = pathMask[ar, pc, cur]
        pathMask[ar, pc, cur] = torch.where(valid, torch.ones_like(cur_pm), cur_pm)
        cur = torch.where(valid, p, cur)
    preDf = torch.where(active, minval, torch.zeros_like(minval))
    return preDf, bn_u, bn_v, pathMask


def batched_maxflow(caps):
    """Standard batched max-flow value (B,) on capacities caps (B,EDGE). = true optimum."""
    G = _caps_to_mat(caps)
    B = caps.shape[0]
    ar = torch.arange(B, device=caps.device)
    mf = torch.zeros(B, dtype=caps.dtype, device=caps.device)
    for _ in range(MAXAUG):
        parent, found = _bfs(G)
        if not bool(found.any()):
            break
        df, bn_u, bn_v, pathMask = _walk_path(G, parent, found)        # df = min residual bottleneck
        mf = mf + df
        df3 = df.view(B, 1, 1)
        G = G - df3 * pathMask + df3 * pathMask.transpose(1, 2)
    return mf


def batched_corr_maxflow(preCap, realCap):
    """Realized Correction-Function-A value (B,), EXACT batched port of test_corrA.Graph.corr_maxFlow.

    preCap, realCap : (B,EDGE).  Augment on the PREDICTED graph; realized increment df is the TRUE
    residual of the predicted-bottleneck edge; accumulate preFlow (of df) on forward path edges;
    finally rescale by tau = min over edges with preFlow>trueCap of trueCap/preFlow.
    """
    preG = _caps_to_mat(preCap)
    realG = _caps_to_mat(realCap)
    realG0 = realG.clone()                                             # original true caps (forward only)
    B = preCap.shape[0]
    preFlow = torch.zeros(B, NODE, NODE, dtype=preCap.dtype, device=preCap.device)
    mf = torch.zeros(B, dtype=preCap.dtype, device=preCap.device)
    ar = torch.arange(B, device=preCap.device)
    for _ in range(MAXAUG):
        parent, found = _bfs(preG)
        if not bool(found.any()):
            break
        preDf, bn_u, bn_v, pathMask = _walk_path(preG, parent, found)
        df = realG[ar, bn_u.clamp(min=0), bn_v.clamp(min=0)]
        df = torch.where(found, df, torch.zeros_like(df))
        mf = mf + df
        preDf3, df3 = preDf.view(B, 1, 1), df.view(B, 1, 1)
        preG = preG - preDf3 * pathMask + preDf3 * pathMask.transpose(1, 2)
        realG = realG - df3 * pathMask + df3 * pathMask.transpose(1, 2)
        preFlow = preFlow + df3 * pathMask
    # tau correction vs ORIGINAL true caps
    viol = preFlow > realG0
    ratio = torch.where(viol, realG0 / preFlow.clamp(min=1e-30),
                        torch.full_like(preFlow, float('inf')))
    tau = ratio.flatten(1).min(dim=1).values
    has = viol.flatten(1).any(dim=1)
    mf = torch.where(has, mf * tau, mf)
    return mf


# ================================================================== reference (numpy) ports
def _bfs_ref(graph, s, t, parent, N=NODE):
    visited = [False] * N
    q = [s]; visited[s] = True
    while q:
        u = q.pop(0)
        for ind in range(N):
            if (not visited[ind]) and graph[u][ind] > 0:
                q.append(ind); visited[ind] = True; parent[ind] = u
                if ind == t:
                    return True
    return False


def ref_corr_maxflow(preCap, realCap, N=NODE, s=SRC, t=SINK):
    """Faithful port of test_corrA.py Graph.corr_maxFlow (the value path only)."""
    preG = np.zeros((N, N)); realG = np.zeros((N, N))
    for i, (u, v) in enumerate(ARC_LIST):
        preG[u][v] = preCap[i]; realG[u][v] = realCap[i]
    realG0 = realG.copy()
    preFlow = np.zeros((N, N))
    parent = [-1] * N
    mf = 0.0
    while _bfs_ref(preG, s, t, parent, N):
        path_flow = float('inf'); pfi = t; ss = t
        while ss != s:
            if path_flow > preG[parent[ss]][ss]:
                path_flow = preG[parent[ss]][ss]; pfi = ss
            ss = parent[ss]
        preDf = path_flow
        df = realG[parent[pfi]][pfi]
        mf += df
        v = t
        while v != s:
            u = parent[v]
            preG[u][v] -= preDf; realG[u][v] -= df; preFlow[u][v] += df
            preG[v][u] += preDf; realG[v][u] += df
            v = parent[v]
    tauTemp = []
    for i in range(N):
        for j in range(N):
            if preFlow[i][j] > realG0[i][j]:
                tauTemp.append(realG0[i][j] / preFlow[i][j])
    if tauTemp:
        mf *= min(tauTemp)
    return mf


def ref_maxflow(cap, N=NODE, s=SRC, t=SINK):
    """Standard Ford-Fulkerson max-flow value (port of Graph.maxFlow without path bookkeeping)."""
    G = np.zeros((N, N))
    for i, (u, v) in enumerate(ARC_LIST):
        G[u][v] = cap[i]
    parent = [-1] * N
    mf = 0.0
    while _bfs_ref(G, s, t, parent, N):
        pf = float('inf'); ss = t
        while ss != s:
            pf = min(pf, G[parent[ss]][ss]); ss = parent[ss]
        mf += pf
        v = t
        while v != s:
            u = parent[v]; G[u][v] -= pf; G[v][u] += pf; v = parent[v]
    return mf


# ================================================================== validation
def validate(n=40, seed=0):
    P("=" * 78)
    P(f"VALIDATION of batched GPU max-flow vs reference Python (>= {n} random instances, tol 1e-3)")
    rng = np.random.default_rng(seed)
    # random true caps and predicted caps (predicted intentionally != true, some negative,
    # to exercise reverse edges + tau correction). float64 for an apples-to-apples comparison.
    real = rng.uniform(0, 100, size=(n, EDGE))
    pred = real + rng.normal(0, 40, size=(n, EDGE))            # noisy predictions, can go negative
    real_t = torch.tensor(real, dtype=torch.float64, device=dev)
    pred_t = torch.tensor(pred, dtype=torch.float64, device=dev)
    b_corr = batched_corr_maxflow(pred_t, real_t).cpu().numpy()
    b_mf = batched_maxflow(real_t).cpu().numpy()
    r_corr = np.array([ref_corr_maxflow(pred[i], real[i]) for i in range(n)])
    r_mf = np.array([ref_maxflow(real[i]) for i in range(n)])
    e_corr = np.max(np.abs(b_corr - r_corr))
    e_mf = np.max(np.abs(b_mf - r_mf))
    # cross-check the optimum against Gurobi LP (path formulation) on a handful, if available
    gurobi_err = None
    try:
        import gurobipy as gp
        from gurobipy import GRB
        c_data = np.loadtxt(f"{DATA}/POLSKA_C011.txt").tolist()
        G_data = np.loadtxt(f"{DATA}/POLSKA_G011.txt").tolist()
        allPathNum = len(c_data)
        errs = []
        for i in range(min(8, n)):
            m = gp.Model(); m.setParam('OutputFlag', 0)
            x = m.addVars(allPathNum, vtype=GRB.CONTINUOUS, name='x')
            m.setObjective(x.prod(c_data), GRB.MAXIMIZE)
            for e in range(EDGE):
                m.addConstr(x.prod(G_data[e]) <= float(real[i][e]))
            m.optimize()
            errs.append(abs(m.objVal - r_mf[i]))
        gurobi_err = float(np.max(errs))
    except Exception as ex:
        P(f"  (Gurobi cross-check skipped: {type(ex).__name__})")
    P(f"  max|batched_corr - ref_corr|   = {e_corr:.3e}")
    P(f"  max|batched_maxflow - ref_mf|  = {e_mf:.3e}")
    if gurobi_err is not None:
        P(f"  max|ref_maxflow - Gurobi LP|   = {gurobi_err:.3e}  (optimum sanity, 8 instances)")
    ok = (e_corr <= 1e-3) and (e_mf <= 1e-3) and (gurobi_err is None or gurobi_err <= 1e-3)
    P(f"  VALIDATION {'PASSED' if ok else 'FAILED'}")
    P("=" * 78)
    return ok


# ================================================================== data
def load_fold(k):
    def parse(path):
        d = np.loadtxt(path)
        n = d.shape[0] // EDGE
        d = d.reshape(n, EDGE, 10)
        X = d[:, :, 1:9].astype(np.float32)     # cols 1..8 features
        cap = d[:, :, 9].astype(np.float32)     # col 9 true capacity
        return X, cap
    Xtr, ctr = parse(f"{DATA}/train_POLSKA/train_POLSKA({k}).txt")
    Xte, cte = parse(f"{DATA}/test_POLSKA/test_POLSKA({k}).txt")
    # standardize features with TRAIN statistics (per feature, over all instance x edge rows)
    flat = Xtr.reshape(-1, FEAT)
    mu = flat.mean(0); sd = flat.std(0); sd[sd < 1e-8] = 1.0
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    return to(Xtr), to(ctr), to(Xte), to(cte)


# ================================================================== predictor + trainers
def make():
    return nn.Linear(FEAT, 1, bias=True).to(dev)     # shared 8->1 map, applied per edge (like Ridge_para)


def predict(m, X):
    """X (B,EDGE,FEAT) -> predicted caps (B,EDGE)."""
    return m(X).squeeze(-1)


def train_two_stage(Xtr, ctr, epochs=300, lr=1e-2):
    m = make(); opt = torch.optim.Adam(m.parameters(), lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((predict(m, Xtr) - ctr) ** 2).mean()
        loss.backward(); opt.step()
    return m


def train_polystep(Xtr, ctr, warm, scale, seed, steps=150, nb=256):
    m = make(); m.load_state_dict(warm.state_dict())
    pso = PolyStepOptimizer(m, polytope_type=PS_POLYTOPE, epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=0.4, probe_radius=0.8, num_probe=PS_PROBES, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    Ntr = Xtr.shape[0]
    g = torch.Generator(device=dev).manual_seed(1000 + seed)
    hold = {"idx": torch.arange(min(nb, Ntr), device=dev)}

    def closure(bp):
        idx = hold["idx"]
        Xb, cb = Xtr[idx], ctr[idx]                                   # (B,EDGE,FEAT),(B,EDGE)
        w = bp["weight"].reshape(-1, FEAT); b = bp["bias"].reshape(-1, 1); M = w.shape[0]
        pred = torch.einsum("mf,bef->mbe", w, Xb) + b.view(M, 1, 1)   # (M,B,EDGE)
        B = Xb.shape[0]
        real = cb.unsqueeze(0).expand(M, B, EDGE)
        rz = batched_corr_maxflow(pred.reshape(M * B, EDGE),
                                  real.reshape(M * B, EDGE)).reshape(M, B)
        return -(rz / scale).mean(-1)                                 # maximize realized -> min negative

    for _ in range(steps):
        hold["idx"] = torch.randint(0, Ntr, (min(nb, Ntr),), generator=g, device=dev)
        pso.step(closure)
    return m


def train_sfge(Xtr, ctr, warm, scale, seed, epochs=150, nb=256, n_samples=8, sigma=5.0, lr=5e-3):
    m = make(); m.load_state_dict(warm.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr)
    g = torch.Generator(device=dev).manual_seed(2000 + seed)
    Ntr = Xtr.shape[0]
    for _ in range(epochs):
        idx = torch.randint(0, Ntr, (min(nb, Ntr),), generator=g, device=dev)
        Xb, cb = Xtr[idx], ctr[idx]
        pred = predict(m, Xb)                                         # (B,EDGE)
        with torch.no_grad():
            eps = torch.randn(n_samples, *pred.shape, device=dev, generator=g)
            chat = pred.unsqueeze(0) + sigma * eps                    # (S,B,EDGE)
            S, B, E = chat.shape
            real = cb.unsqueeze(0).expand(S, B, E)
            r = batched_corr_maxflow(chat.reshape(S * B, E),
                                     real.reshape(S * B, E)).reshape(S, B) / scale
            adv = r - r.mean(0, keepdim=True)
        logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
        surrogate = -(adv * logp).mean()                             # maximize realized
        opt.zero_grad(); surrogate.backward(); opt.step()
    return m


# ================================================================== evaluation
def norm_regret(m, Xte, cte):
    with torch.no_grad():
        pred = predict(m, Xte)
        realized = batched_corr_maxflow(pred, cte)
        opt = batched_maxflow(cte)
        mask = opt > 1e-6
        reg = ((opt - realized) / opt.clamp(min=1e-6))[mask]
        return float(reg.mean().item())


def timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter(); out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


# ================================================================== main
def main():
    seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2]
    STEPS_PS, NB_PS = 150, 256
    EP_SFGE, NB_SFGE, S_SFGE = 150, 256, 8
    EP_TS = 300

    if not validate(n=40, seed=0):
        P("Aborting: batched solver does not match the reference. Fix before training.")
        sys.exit(1)

    P(f"\nMAX-FLOW POLSKA | predicted edge capacities in the CONSTRAINTS")
    P(f"folds(seeds)={seeds} | normalized realized-regret (lower better)")
    P("SPO+ / PFYL / IMLE / cvxpylayers : N/A (prediction parametrizes the constraint; no cost vector)\n")

    acc = {k: [] for k in ("two-stage", "SFGE", "PolyStep")}
    walls = {k: [] for k in ("two-stage", "SFGE", "PolyStep")}
    ntr_seen = None
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        Xtr, ctr, Xte, cte = load_fold(sd)
        ntr_seen = Xtr.shape[0]
        scale = float(batched_maxflow(ctr).mean().item())            # ~O(1) normalization for the closure

        ts, t_ts = timed(lambda: train_two_stage(Xtr, ctr, epochs=EP_TS))
        sf, t_sf = timed(lambda: train_sfge(Xtr, ctr, ts, scale, sd, epochs=EP_SFGE, nb=NB_SFGE, n_samples=S_SFGE))
        ps, t_ps = timed(lambda: train_polystep(Xtr, ctr, ts, scale, sd, steps=STEPS_PS, nb=NB_PS))

        r_ts, r_sf, r_ps = norm_regret(ts, Xte, cte), norm_regret(sf, Xte, cte), norm_regret(ps, Xte, cte)
        acc["two-stage"].append(r_ts); acc["SFGE"].append(r_sf); acc["PolyStep"].append(r_ps)
        walls["two-stage"].append(t_ts); walls["SFGE"].append(t_sf); walls["PolyStep"].append(t_ps)
        P(f"  fold {sd}: scale(optTOV)={scale:6.2f} | two-stage={r_ts:.4f}  SFGE={r_sf:.4f}  PolyStep={r_ps:.4f}")

    summ = {k: summarize(acc[k]) for k in acc}
    # cost instrumentation (faithful to repo convention: 2*D*n*steps batched forward-solves)
    ps_solves = 2 * D_PARAMS * NB_PS * STEPS_PS
    sfge_solves = S_SFGE * NB_SFGE * EP_SFGE
    cost = {
        "two-stage": {"wall_s": summarize(walls["two-stage"]), "forward_solves": 0, "solver_calls": 0,
                      "note": "Adam MSE on true capacities; no decision solves in training"},
        "SFGE": {"wall_s": summarize(walls["SFGE"]), "forward_solves": sfge_solves, "solver_calls": sfge_solves,
                 "note": "n_samples*n*epochs batched corr-maxflow evals; 0 exact-solver (Gurobi) calls"},
        "PolyStep": {"wall_s": summarize(walls["PolyStep"]), "forward_solves": ps_solves, "solver_calls": ps_solves,
                     "note": "2*D*n*steps batched corr-maxflow evals; 0 exact-solver (Gurobi) calls"},
        "SPO+/PFYL/IMLE/cvxpylayers": "N/A (predicted parameter lives in the constraint)",
    }

    best = min(summ, key=lambda k: summ[k]["mean"])
    gain = (summ["two-stage"]["mean"] - summ[best]["mean"]) / max(summ["two-stage"]["mean"], 1e-9) * 100
    P("\n" + "=" * 70)
    P(f"{'method':>12} | {'norm-regret (mean±std)':>24} | {'wall_s':>8} | {'fwd-solves':>12}")
    P("-" * 70)
    for k in ("two-stage", "SFGE", "PolyStep"):
        fs = cost[k]["forward_solves"]
        P(f"{k:>12} | {summ[k]['mean']:>10.4f} ± {summ[k]['std']:<9.4f} | "
          f"{cost[k]['wall_s']['mean']:>8.1f} | {fs:>12,}")
    P(f"{'SPO+/...':>12} | {'N/A (constraint)':>24} |")
    P("-" * 70)
    P(f"best = {best}  ({gain:+.0f}% normalized-regret cut vs two-stage)")
    P("=" * 70)

    out = {
        "problem": "max-flow POLSKA (cpaior23 Branch-and-Learn); predicted edge capacities in CONSTRAINTS",
        "graph": {"nodes": NODE, "edges": EDGE, "source": SRC, "sink": SINK},
        "data_format": "row=[id, f0..f7, trueCap]; features=cols1-8, true_cap=col9 (per BAL.cpp parser)",
        "objective": "Correction Function A (corr_maxFlow, ported EXACTLY); realized flow VALUE maximized",
        "metric": "normalized regret = mean (opt_maxflow(true) - realized_corr_flow(pred)) / opt",
        "applicable_methods": ["two-stage", "SFGE", "PolyStep"],
        "na_methods": ["SPO+", "PFYL", "IMLE", "cvxpylayers"],
        "seeds": seeds, "n_train": ntr_seen, "n_test": 179,
        "config": {"polystep_steps": STEPS_PS, "polystep_nb": NB_PS, "D_params": D_PARAMS,
                   "sfge_epochs": EP_SFGE, "sfge_nb": NB_SFGE, "sfge_samples": S_SFGE,
                   "two_stage_epochs": EP_TS, "predictor": "linear Linear(8,1) shared across edges"},
        "summary": summ, "cost": cost, "best": best, "regret_cut_vs_two_stage_pct": gain,
        "batched_maxflow_tractable": True,
        "tractability_note": ("Fully batched on GPU: the 12-node graph fits in (B,12,12) residual "
                              "tensors; BFS/bottleneck/augment are vectorized over N*n instances. "
                              "Augmenting-path count is small (<=Edmonds-Karp V*E/2), the loop breaks "
                              "early, and per-op tensors are tiny -- NOT compute-bound, well clear of "
                              "the track3_milp boundary."),
    }
    write_json(f"exp_results/maxflow_polska{_OUT_SFX}.json", out)

    L = ["# Max-Flow POLSKA -- predicted edge capacities in the CONSTRAINTS (cpaior23 B&L benchmark)", "",
         "Predicted parameters parametrize the **constraint** (edge capacities), so the feasible region "
         "moves with the prediction and the realized objective (augmenting-path bottleneck + tau rescale) "
         "is non-linear & non-differentiable in the prediction. **SPO+ / PFYL / IMLE / cvxpylayers are "
         "N/A** (no fixed feasible region, no predicted cost vector). Only two-stage / SFGE / PolyStep apply.",
         "",
         f"Graph: {NODE} nodes, {EDGE} edges (DAG, all u<v), source={SRC}, sink={SINK}. "
         f"Predictor: linear `Linear(8,1)` shared across edges (like `Ridge_para_POLSKA`). "
         f"Realized objective ported EXACTLY from `test_corrA.py corr_maxFlow` (Correction Function A); "
         f"validated batched-GPU vs reference Python to <=1e-3 before training.",
         f"Folds(seeds)={seeds}, n_train={ntr_seen}, n_test=179. Normalized regret = "
         "mean (opt_maxflow(true) - realized_corr_flow(pred)) / opt (lower better).", "",
         md_table(["method", "norm-regret (mean±std)", "wall_s", "fwd-solves", "exact-solver calls"],
                  [["two-stage", f"{summ['two-stage']['mean']:.4f}±{summ['two-stage']['std']:.4f}",
                    f"{cost['two-stage']['wall_s']['mean']:.1f}", "0", "0"],
                   ["SFGE", f"{summ['SFGE']['mean']:.4f}±{summ['SFGE']['std']:.4f}",
                    f"{cost['SFGE']['wall_s']['mean']:.1f}", f"{sfge_solves:,}", "0"],
                   ["PolyStep", f"{summ['PolyStep']['mean']:.4f}±{summ['PolyStep']['std']:.4f}",
                    f"{cost['PolyStep']['wall_s']['mean']:.1f}", f"{ps_solves:,}", "0"],
                   ["SPO+/PFYL/IMLE/cvxpylayers", "N/A (constraint)", "-", "-", "-"]]),
         "",
         f"Best: **{best}** ({gain:+.0f}% normalized-regret cut vs two-stage). "
         "PolyStep/SFGE make **0 exact-solver (Gurobi) calls**; their forward-solves are cheap, fully "
         "batched GPU augmenting-path max-flows.",
         "",
         "## Tractability of the batched max-flow on GPU",
         out["tractability_note"]]
    write_md(f"exp_results/maxflow_polska{_OUT_SFX}.md", "\n".join(L))
    P(f"\nwrote exp_results/maxflow_polska{_OUT_SFX}.{{json,md}}\nDONE")


if __name__ == "__main__":
    main()
