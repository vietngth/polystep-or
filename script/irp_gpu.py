"""GPU-batched CPCTSP solver + batched rollout for the DSIRP/IRP PolyStep case.

Author's guidance (An): "you should be able to run solvers on GPU as well to make PolyStep much
faster. Toni's experiment (reproducibility) can be done on CPU, but our case can leverage GPU."

PolyStep does a HUGE number of forward solves (probes x periods x scenarios). The IRP forward
problem is a Capacitated Prize-Collecting TSP (CPCTSP). Serial gurobipy throws away PolyStep's
entire advantage. KEY OBSERVATION: the routing geometry (distance matrix) is FIXED per instance,
so the optimal TSP tour length for every customer SUBSET can be precomputed ONCE (exact Held-Karp,
2^C subsets). Then each CPCTSP solve reduces to:

    max_S  sum_{c in S} theta_c  -  tour_cost[S]   s.t.  sum_{c in S} deliver_c <= v_cap

which is a batched (membership matmul -> mask infeasible -> argmax) over all 2^C subsets, fully on
GPU and EXACT (== gurobipy cpctsp, both optimal). For C=10 -> 1024 subsets, trivially batchable
over M=thousands of probes. The empty subset (visit nobody, value 0) is index 0 and always feasible.

This module exposes:
  * precompute_subsets(dist, C, device) -> (tour_cost[2^C], membership[C,2^C])
  * cpctsp_batched(theta, deliver, v_cap, tour_cost, membership) -> (visited[M,C], routing[M])
  * batched_rollout(...) -> per-probe total cost, advancing inventory with shared realized demand
  * verify_vs_gurobi(...) -> exactness check against exp_irp_polystep.cpctsp
"""
from __future__ import annotations
import numpy as np
import torch


# ---------------------------------------------------------------------------------------------
# 1. Precompute exact TSP tour cost for EVERY customer subset (Held-Karp), once per instance.
# ---------------------------------------------------------------------------------------------
def precompute_subsets(dist, C, device="cpu", dtype=torch.float64):
    """dist: (C+1,C+1) array, node 0 = depot, customers = nodes 1..C.
    Returns tour_cost:(2^C,) exact optimal tour length over {depot}+subset (0 for empty),
    membership:(C, 2^C) with membership[c, mask] = 1 iff customer c in mask."""
    nmask = 1 << C
    d = np.asarray(dist, dtype=np.float64)
    INF = np.inf
    # dp[mask, j] = min cost path depot(0) -> ... -> customer j (0-indexed), visiting exactly `mask`
    dp = np.full((nmask, C), INF)
    for j in range(C):
        dp[1 << j, j] = d[0, j + 1]
    for mask in range(1, nmask):
        for j in range(C):
            if not (mask >> j) & 1:
                continue
            base = dp[mask, j]
            if base == INF:
                continue
            for k in range(C):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                cand = base + d[j + 1, k + 1]
                if cand < dp[nm, k]:
                    dp[nm, k] = cand
    tour = np.zeros(nmask)
    for mask in range(1, nmask):
        best = INF
        for j in range(C):
            if (mask >> j) & 1 and dp[mask, j] < INF:
                v = dp[mask, j] + d[j + 1, 0]      # close the tour back to depot
                if v < best:
                    best = v
        tour[mask] = best
    membership = np.zeros((C, nmask), dtype=np.float64)
    for c in range(C):
        for mask in range(nmask):
            if (mask >> c) & 1:
                membership[c, mask] = 1.0
    return (torch.tensor(tour, dtype=dtype, device=device),
            torch.tensor(membership, dtype=dtype, device=device))


# ---------------------------------------------------------------------------------------------
# 2. Batched CPCTSP solve (the hot path PolyStep calls millions of times).
# ---------------------------------------------------------------------------------------------
def cpctsp_batched(theta, deliver, v_cap, tour_cost, membership):
    """theta:(M,C) prizes, deliver:(M,C) order-up-to quantities, v_cap: scalar.
    tour_cost:(2^C,), membership:(C,2^C). Returns visited:(M,C) {0,1}, routing:(M,)."""
    prize = theta @ membership                       # (M, 2^C)
    cap = deliver @ membership                       # (M, 2^C)
    value = prize - tour_cost.unsqueeze(0)           # (M, 2^C); empty subset (col 0) -> 0
    value = value.masked_fill(cap > v_cap + 1e-6, float("-inf"))
    best = value.argmax(dim=1)                       # (M,)
    visited = membership[:, best].t().contiguous()   # (M, C)
    routing = tour_cost[best]                         # (M,)
    return visited, routing


# ---------------------------------------------------------------------------------------------
# 3. Batched feature terms (period_terms), vectorized over M probes. The demand HISTORY (hence the
#    quantiles) is shared across probes -- it grows with the REALIZED demand, independent of policy;
#    only start_inv differs per probe. So qv is computed once per period, terms are batched over M.
# ---------------------------------------------------------------------------------------------
def batched_period_terms(start_inv, qv, holding, penalty, look_ahead):
    """start_inv:(M,C), qv:(C,|Q|) shared quantiles, holding/penalty:(C,). -> hold,pen each (M,C,NB_OBS)."""
    M, C = start_inv.shape
    k = torch.arange(1, look_ahead + 1, device=start_inv.device, dtype=start_inv.dtype)   # (K,)
    cum = k[None, :, None] * qv[:, None, :]                       # (C,K,|Q|) = k*q (shared)
    si = start_inv[:, :, None, None]                             # (M,C,1,1)
    hold = holding[None, :, None, None] * torch.clamp(si - cum[None], min=0.0)   # (M,C,K,|Q|)
    pen = penalty[None, :, None, None] * torch.clamp(cum[None] - si, min=0.0)
    nb = look_ahead * qv.shape[1]
    return hold.reshape(M, C, nb), pen.reshape(M, C, nb)


# ---------------------------------------------------------------------------------------------
# 4. Batched closed-loop rollout: M probes advanced in lock-step through the horizon, sharing the
#    realized demand `dseq`. Returns per-probe total cost (M,). Mirrors exp_irp_polystep.rollout.
# ---------------------------------------------------------------------------------------------
def batched_rollout(W_inv, W_pen, inst_t, dseq_t, horizon, look_ahead, quantiles, tour_cost, membership):
    """W_inv,W_pen:(M,NB_OBS) per-probe predictor weights. inst_t: dict of tensors (on device):
    start_inv0:(C,), max_inv:(C,), holding:(C,), penalty:(C,), v_cap: float, demand_hist: list[C] 1-D.
    dseq_t:(C,horizon) realized demand (shared across probes). Returns total:(M,) realized cost."""
    dev = W_inv.device
    M = W_inv.shape[0]
    C = inst_t["holding"].shape[0]
    start_inv = inst_t["start_inv0"].unsqueeze(0).expand(M, C).clone()    # (M,C)
    max_inv = inst_t["max_inv"]; holding = inst_t["holding"]; penalty = inst_t["penalty"]
    v_cap = inst_t["v_cap"]
    hist = [h.clone() for h in inst_t["demand_hist"]]                     # shared, grows with realized
    Q = torch.tensor(quantiles, device=dev, dtype=W_inv.dtype)
    tot = torch.zeros(M, device=dev, dtype=W_inv.dtype)
    for t in range(horizon):
        if t > 0:
            for c in range(C):
                hist[c] = torch.cat([hist[c], dseq_t[c, t - 1:t]])
        qv = torch.stack([torch.quantile(hist[c], Q) for c in range(C)])  # (C,|Q|) shared
        hold_t, pen_t = batched_period_terms(start_inv, qv, holding, penalty, look_ahead)  # (M,C,NB)
        theta = torch.einsum("mcn,mn->mc", hold_t, W_inv) + torch.einsum("mcn,mn->mc", pen_t, W_pen)  # (M,C)
        deliver = (max_inv.unsqueeze(0) - start_inv)                       # (M,C)
        visited, routing = cpctsp_batched(theta, deliver, float(v_cap), tour_cost, membership)
        q = deliver * visited                                             # order-up-to delivery
        nxt = start_inv + q - dseq_t[:, t].unsqueeze(0)                   # (M,C)
        short = torch.clamp(-nxt, min=0.0)
        carry = torch.clamp(nxt, min=0.0)
        tot = tot + (carry * holding[None]).sum(1) + (short * penalty[None]).sum(1) + routing
        start_inv = carry
    return tot


def inst_to_tensors(inst, device, dtype=torch.float64):
    """Pack an exp_irp_polystep instance dict into device tensors for batched_rollout."""
    C = inst["C"]
    return dict(
        start_inv0=torch.tensor(inst["start_inv"], dtype=dtype, device=device),
        max_inv=torch.tensor(inst["max_inv"], dtype=dtype, device=device),
        holding=torch.tensor(inst["holding"], dtype=dtype, device=device),
        penalty=torch.tensor(inst["penalty"], dtype=dtype, device=device),
        v_cap=float(inst["v_cap"]),
        demand_hist=[torch.tensor(h, dtype=dtype, device=device) for h in inst["demand_hist"]],
    )


# ---------------------------------------------------------------------------------------------
# 5. Exactness check against the gurobipy CPCTSP in exp_irp_polystep.
# ---------------------------------------------------------------------------------------------
def verify_vs_gurobi(inst, n_check=12, seed=0, device="cpu"):
    import exp_irp_polystep as B
    C = inst["C"]
    tour_cost, membership = precompute_subsets(inst["dist"], C, device=device)
    rng = np.random.default_rng(seed)
    max_abs = 0.0; nmism = 0
    for _ in range(n_check):
        theta_np = rng.normal(0, 50, size=C)                 # random prizes
        si = np.clip(inst["start_inv"] + rng.normal(0, 1, size=C), 0, None)
        deliver_np = inst["max_inv"] - si
        # gurobi reference
        gv, grout, _ = B.cpctsp(theta_np, si, inst)
        # gpu batched
        th = torch.tensor(theta_np[None], dtype=torch.float64, device=device)
        dl = torch.tensor(deliver_np[None], dtype=torch.float64, device=device)
        bv, brout = cpctsp_batched(th, dl, float(inst["v_cap"]), tour_cost, membership)
        # compare OBJECTIVE (subset choice may tie); objective = prize(visited) - routing
        def obj(visited, routing):
            return float((theta_np * np.asarray(visited)).sum() - routing)
        og = obj(gv, grout); ob = obj(bv[0].cpu().numpy(), float(brout[0]))
        max_abs = max(max_abs, abs(og - ob))
        if abs(og - ob) > 1e-4:
            nmism += 1
    return {"n": n_check, "max_abs_obj_diff": max_abs, "n_mismatch": nmism}
