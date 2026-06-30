"""Batched, GPU forward solvers for predict-then-optimize downstream problems.

Every solver maps a batch of cost/consumption tensors to optimal/heuristic
decisions in one vectorized call. PolyStep uses these as black-box argmin
oracles; the same oracle is shared across methods for fair comparison.
"""
from __future__ import annotations
import os
import torch

# Batch-dimension chunk size for the batched forward solvers. Caps peak memory of
# the big per-instance intermediates (the (M, num_nodes) DAG tables and the
# (M, m, n) MDKP consumption tensors) so large probe-batches / constraint counts
# do not OOM. Numerically a no-op: the batch is processed in slices and the
# outputs are concatenated, identical to the unchunked computation. Env-tunable.
MDKP_CHUNK = int(os.environ.get("MDKP_CHUNK", "8192"))
KNAP_CHUNK = int(os.environ.get("KNAP_CHUNK", "4096"))  # batch-dim cap for knap1_dp DP tensor (OOM guard)


# ---------------------------------------------------------------------------
# Shortest path on a DAG (exact). Aligned to an arc list so it matches PyEPO.
# ---------------------------------------------------------------------------
def build_dag_solver(arcs, num_nodes, source, sink):
    """arcs: list[(u, v)] in cost-vector order (u < v, topologically sorted)."""
    out_by_node = [[] for _ in range(num_nodes)]
    for e, (u, v) in enumerate(arcs):
        out_by_node[u].append((v, e))
    E = len(arcs)

    def _solve(c):                                       # c (M,E) -> w (M,E) one-hot path
        M = c.shape[0]; dev = c.device; INF = float("inf")
        dist = torch.full((M, num_nodes), INF, device=dev); dist[:, source] = 0.0
        pe = torch.full((M, num_nodes), -1, dtype=torch.long, device=dev)
        pn = torch.full((M, num_nodes), -1, dtype=torch.long, device=dev)
        for u in range(num_nodes):
            for (v, e) in out_by_node[u]:
                nd = dist[:, u] + c[:, e]; better = nd < dist[:, v]
                dist[:, v] = torch.where(better, nd, dist[:, v])
                pe[:, v] = torch.where(better, torch.full_like(pe[:, v], e), pe[:, v])
                pn[:, v] = torch.where(better, torch.full_like(pn[:, v], u), pn[:, v])
        w = torch.zeros((M, E), device=dev)
        cur = torch.full((M,), sink, dtype=torch.long, device=dev)
        midx = torch.arange(M, device=dev)
        for _ in range(num_nodes):
            active = cur != source
            if not active.any(): break
            e = pe[midx, cur]
            w[midx[active], e[active]] = 1.0
            cur = torch.where(active, pn[midx, cur], cur)
        return w

    def solve_batch(c):                                  # chunked over batch dim (mem-bounded)
        M = c.shape[0]
        if M <= MDKP_CHUNK:
            return _solve(c)
        return torch.cat([_solve(c[s:s + MDKP_CHUNK]) for s in range(0, M, MDKP_CHUNK)], 0)
    return solve_batch


def grid_arcs(H, W):
    """East/south arcs of an HxW grid, node index r*W+c, source 0, sink HW-1."""
    arcs = []
    for r in range(H):
        for c in range(W):
            n = r * W + c
            if c + 1 < W: arcs.append((n, n + 1))
    for r in range(H):
        for c in range(W):
            n = r * W + c
            if r + 1 < H: arcs.append((n, n + W))
    return arcs, H * W, 0, H * W - 1


# ---------------------------------------------------------------------------
# 0/1 single-constraint knapsack (exact DP) with selection backtrack.
# ---------------------------------------------------------------------------
def knap1_dp(v, w, C, want_sel=True):
    """v,w: (M,n) values & INTEGER weights (>=1). C int. -> best (M,), sel (M,n)."""
    M, n = v.shape; dev = v.device
    if M > KNAP_CHUNK:
        # batch rows are independent -> chunk to bound the (chunk, n, C+1) DP tensor (numerically identical)
        bests, sels = [], []
        for s in range(0, M, KNAP_CHUNK):
            b, sl = knap1_dp(v[s:s + KNAP_CHUNK], w[s:s + KNAP_CHUNK], C, want_sel)
            bests.append(b)
            if want_sel: sels.append(sl)
        return torch.cat(bests, 0), (torch.cat(sels, 0) if want_sel else None)
    NEG = torch.finfo(v.dtype).min / 4
    dp = torch.zeros(M, C + 1, device=dev, dtype=v.dtype)
    keep = torch.zeros(M, n, C + 1, dtype=torch.bool, device=dev) if want_sel else None
    rng = torch.arange(C + 1, device=dev)
    for i in range(n):
        wi = w[:, i].long(); vi = v[:, i]
        src = rng[None, :] - wi[:, None]; valid = src >= 0
        cand = torch.where(valid, torch.gather(dp, 1, src.clamp(min=0)) + vi[:, None],
                           torch.full_like(dp, NEG))
        take = cand > dp
        if want_sel: keep[:, i, :] = take
        dp = torch.where(take, cand, dp)
    best = dp[:, C]
    if not want_sel:
        return best, None
    sel = torch.zeros(M, n, dtype=torch.bool, device=dev)
    c = torch.full((M,), C, dtype=torch.long, device=dev); midx = torch.arange(M, device=dev)
    for i in range(n - 1, -1, -1):
        t = keep[midx, i, c]; sel[:, i] = t
        c = torch.where(t, c - w[:, i].long(), c)
    return best, sel


# ---------------------------------------------------------------------------
# Multi-dimensional 0/1 knapsack (m resource constraints), batched greedy.
# Value-density heuristic; deployable, fully vectorized over the batch.
# ---------------------------------------------------------------------------
def knap1_repair(sel, v, w_true, C):
    """Single-constraint repair: drop lowest value/true-weight until sum w_true<=C."""
    sel = sel.float().clone(); density = v / (w_true + 1e-6)
    for _ in range(sel.shape[1]):
        over = (sel * w_true).sum(-1) > C
        if not over.any(): break
        d = torch.where(sel > 0.5, density, torch.full_like(density, float("inf")))
        drop = d.argmin(-1); midx = torch.arange(sel.shape[0], device=sel.device)
        sel[midx[over], drop[over]] = 0.0
    return (sel * v).sum(-1)


def mdkp_greedy(v, A, b):
    """v (M,n) values; A (M,m,n) consumption >=0; b (M,m) or (m,) capacities.
    Returns selection (M,n) bool by greedy value-per-normalized-resource.
    Chunked over the batch dim so the (M,m,n) intermediate stays bounded."""
    M, n = v.shape
    if M > MDKP_CHUNK:
        b_e = b.unsqueeze(0).expand(M, -1) if b.dim() == 1 else b
        return torch.cat([mdkp_greedy(v[s:s + MDKP_CHUNK], A[s:s + MDKP_CHUNK],
                                      b_e[s:s + MDKP_CHUNK]) for s in range(0, M, MDKP_CHUNK)], 0)
    m = A.shape[1]; dev = v.device
    if b.dim() == 1: b = b.unsqueeze(0).expand(M, -1)
    norm = (A / (b.unsqueeze(-1) + 1e-9)).sum(1)         # (M,n)
    score = v / (norm + 1e-9)
    order = score.argsort(dim=-1, descending=True)       # (M,n)
    sel = torch.zeros(M, n, dtype=torch.bool, device=dev)
    used = torch.zeros(M, m, device=dev); midx = torch.arange(M, device=dev)
    for k in range(n):
        idx = order[:, k]
        a_item = A.gather(2, idx[:, None, None].expand(M, m, 1)).squeeze(-1)  # (M,m)
        fits = ((used + a_item) <= b).all(-1)
        sel[midx, idx] = fits
        used = used + a_item * fits.unsqueeze(-1).float()
    return sel


def mdkp_repair(sel, v, A_true, b):
    """Drop lowest value-per-true-aggregate-resource taken item until feasible
    under A_true. sel (M,n), A_true (M,m,n), b (M,m) or (m,). -> realized value (M,).
    Chunked over the batch dim so the (M,m,n) intermediate stays bounded."""
    M, n = v.shape; dev = v.device
    if M > MDKP_CHUNK:
        b_e = b.unsqueeze(0).expand(M, -1) if b.dim() == 1 else b
        return torch.cat([mdkp_repair(sel[s:s + MDKP_CHUNK], v[s:s + MDKP_CHUNK],
                                      A_true[s:s + MDKP_CHUNK], b_e[s:s + MDKP_CHUNK])
                          for s in range(0, M, MDKP_CHUNK)], 0)
    if b.dim() == 1: b = b.unsqueeze(0).expand(M, -1)
    sel = sel.float().clone()
    norm = (A_true / (b.unsqueeze(-1) + 1e-9)).sum(1)
    density = v / (norm + 1e-9)
    for _ in range(n):
        used = torch.einsum("mn,mjn->mj", sel, A_true)   # (M,m)
        over = (used > b).any(-1)
        if not over.any(): break
        d = torch.where(sel > 0.5, density, torch.full_like(density, float("inf")))
        drop = d.argmin(-1); midx = torch.arange(M, device=dev)
        sel[midx[over], drop[over]] = 0.0
    return (sel * v).sum(-1)


# ---------------------------------------------------------------------------
# Variance-constrained Markowitz portfolio (SOCP), batched. Verified == Gurobi
# to 0.00% via Lagrangian bisection over the variance multiplier with the step
# adapted to the per-lambda Lipschitz constant 2*lambda*lambda_max(Sigma).
# ---------------------------------------------------------------------------
def proj_simplex(v):
    """Euclidean projection of each row of v onto {w>=0, sum w = 1}."""
    n = v.shape[-1]
    u, _ = torch.sort(v, dim=-1, descending=True)
    css = u.cumsum(-1) - 1.0
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    rho = ((u - css / ind) > 0).sum(-1, keepdim=True).clamp(min=1)
    theta = css.gather(-1, rho - 1) / rho
    return (v - theta).clamp(min=0)


def solve_portfolio_socp(r, Sigma, rho, bis=24, pg=40, hi0=1e7):
    """max r^T w s.t. sum w = 1, w >= 0, w^T Sigma w <= rho. r (M,n) -> w (M,n)."""
    M, n = r.shape
    lmax = float(torch.linalg.eigvalsh(Sigma)[-1])
    rscale = r.abs().mean() + 1e-9
    lo = torch.zeros(M, 1, device=r.device); hi = torch.full((M, 1), hi0, device=r.device)
    w = torch.full((M, n), 1.0 / n, device=r.device)
    for _ in range(bis):
        mid = (lo + hi) / 2
        lr = (1.0 / (2.0 * mid * lmax + rscale)).clamp(max=1.0)
        for _ in range(pg):
            w = proj_simplex(w + lr * (r - 2 * mid * (w @ Sigma)))
        risky = (w @ Sigma * w).sum(-1, keepdim=True) > rho
        lo = torch.where(risky, mid, lo); hi = torch.where(risky, hi, mid)
    return w


# ---------------------------------------------------------------------------
# Prediction-in-CONSTRAINTS forward solvers (exp #4). Both are LPs with an
# exact greedy/water-filling solution, fully batched. The PREDICTED quantity
# defines the feasible region -> SPO+/cvxpylayers/PFYL/IMLE cannot be formulated
# (no fixed S, no objective cost vector). Only two-stage / SFGE / PolyStep run.
# ---------------------------------------------------------------------------
def solve_fractional_knapsack(vhat, what, C):
    """max sum v_i x_i s.t. sum w_i x_i <= C, 0<=x<=1.  Predicted WEIGHT in the constraint.

    vhat,what: (M,n) values & strictly-positive weights; C: scalar or (M,). -> x (M,n) in [0,1].
    Exact fractional-knapsack greedy (sort by value/weight ratio, fill, one fractional boundary item),
    fully batched. == LP optimum (verified vs cvxpy in nv_fk_sanity)."""
    M, n = vhat.shape; dev = vhat.device
    ratio = vhat / (what + 1e-12)
    order = ratio.argsort(dim=-1, descending=True)
    w_sorted = what.gather(1, order)
    cum = w_sorted.cumsum(-1)
    Cc = (C.view(M, 1) if torch.is_tensor(C) else torch.full((M, 1), float(C), device=dev)).to(dev)
    prev = cum - w_sorted                                  # capacity used before this item
    take_full = cum <= Cc
    frac = ((Cc - prev).clamp(min=0) / (w_sorted + 1e-12)).clamp(max=1.0)
    x_sorted = torch.where(take_full, torch.ones_like(w_sorted), frac).clamp(0.0, 1.0)
    return torch.zeros_like(vhat).scatter(1, order, x_sorted)


def solve_set_multicover(req, A, c):
    """Weighted Set Multi-Cover: min sum_j c_j y_j  s.t.  A y >= req,  y in {0,1}^ns.

    Predicted per-element REQUIREMENT ``req`` (M,ne) sits in the constraint RHS (it parametrizes the
    feasible region), so SPO+/PFYL/IMLE/cvxpylayers cannot be formulated. A (ne,ns) is the fixed 0/1
    set-element incidence; c (ns,) fixed set costs. Returns y (M,ns) in {0,1}.

    Classic batched greedy multi-cover: repeatedly buy the set maximizing (#under-covered elements it
    hits)/cost until every element's requirement is met (or no buyable set helps). Each picked set
    covers each of its elements once (binary sets). Fully vectorized over the batch; <= ns iterations.
    Heuristic (Hn-approximate); the realized-cost vs Gurobi optimum gap is reported by the caller."""
    M, ne = req.shape
    ns = A.shape[1]; dev = req.device
    A = A.to(device=dev, dtype=req.dtype)                           # (ne,ns)
    rem = req.clamp(min=0).ceil()                                   # remaining requirement per element
    y = torch.zeros(M, ns, device=dev, dtype=req.dtype)
    midx = torch.arange(M, device=dev)
    for _ in range(ns):
        cov = (rem > 0).to(req.dtype)                              # (M,ne) still-needed elements
        gain = cov @ A                                             # (M,ns) #under-covered elems each set hits
        eff = gain / c.to(dev).unsqueeze(0)                       # cost-effectiveness
        eff = torch.where((y > 0.5) | (gain <= 0), torch.full_like(eff, -1.0), eff)
        pick = eff.argmax(-1)                                      # (M,) best set per row
        do = eff[midx, pick] > 0                                   # rows with a buyable, helpful set
        if not do.any():
            break
        y[midx[do], pick[do]] = 1.0
        a_pick = A[:, pick].t()                                    # (M,ne) coverage of picked set
        rem = torch.where(do.unsqueeze(-1), (rem - a_pick).clamp(min=0), rem)
    return y


def solve_newsvendor_cap(dhat, b, C):
    """Capacitated multi-item newsvendor allocation: max sum b_i q_i s.t. sum q_i <= C, 0<=q_i<=dhat_i.

    Predicted DEMAND dhat (M,n) caps each order (the box constraint q_i<=dhat_i is prediction-defined);
    b (n,) = per-item criticality/underage weight (fixed, known); C = shared budget (scalar).
    Exact greedy: fill highest-b items up to their predicted-demand cap until the budget binds. -> q (M,n).
    This is the budget-binding form of the newsvendor; realized over/under cost is evaluated on TRUE demand."""
    M, n = dhat.shape; dev = dhat.device
    order = b.argsort(descending=True)                    # static order by criticality (b fixed)
    d_sorted = dhat[:, order].clamp(min=0)
    cum = d_sorted.cumsum(-1)
    Cc = torch.full((M, 1), float(C), device=dev)
    prev = cum - d_sorted
    q_sorted = torch.minimum(torch.where(cum <= Cc, d_sorted, (Cc - prev).clamp(min=0)), d_sorted)
    q = torch.zeros_like(dhat)
    q[:, order] = q_sorted
    return q
