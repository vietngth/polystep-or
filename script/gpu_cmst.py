"""Batched GPU CMST/MST surrogate for the DistrictNet districting problem.

DistrictNet scores a district D by the weight of a minimum spanning tree over the subgraph induced by
D with (predicted) edge weights theta, plus a size-violation penalty and a depot term:

    g(D; theta) = sum_{e in MST(D)} theta_e
                  + PENALTY * max(0, |D| - u, l - |D|)
                  + min_{v in D} theta_{depot,v}

which mirrors DistrictNet/src/district.jl::compute_cost_via_CMST (MST over the induced NON-depot
subgraph via Kruskal/Prim, size penalty on [l, u], and compute_depot_cost = the cheapest depot edge).
A disconnected district (its induced subgraph yields fewer than |D|-1 tree edges) is penalized like a
singleton: g = PENALTY * u. See DistrictNet/src/Solver/Kruskal.jl for the reference Kruskal + repair.

PolyStep is forward-only: it scores many probe points (probes x particles) with no gradient. The GPU
win is therefore to BATCH every probe evaluation into one solve. This module computes the MST weight
for a batch of edge-weight vectors that share ONE fixed topology, with a vectorized Boruvka on GPU
(union-find by pointer jumping, per-component minimum-edge reduction via scatter_reduce). It follows
the established repo pattern of a batched GPU solver paired with a clean CPU reference for validation
(see script/irp_gpu.py, which batches Held-Karp and self-checks with verify_vs_gurobi).

Determinism: Boruvka selects each component's minimum OUTGOING edge under a strict lexicographic order
(weight, edge_id). A strict total order on edges makes the selected set a forest (no equal-weight
cycle can survive), so the MST WEIGHT is unique and the batched result is reproducible. scatter_reduce
with amin is order-independent in value, and gather is deterministic, so the per-batch scalar is stable
across runs on a fixed device. Float64 is used throughout to keep the CPU/GPU comparison tight; the only
residual nondeterminism is the usual cross-device float summation order, which stays well under 1e-6.

Public API:
  * batched_mst_weight(edge_u, edge_v, weights, num_nodes) -> (mst_weight[B], n_edges[B])
  * cmst_surrogate_batched(...) -> per-probe districting surrogate + chosen partition
  * mst_weight_reference(...) / cmst_surrogate_reference(...) -> scipy/networkx CPU references
"""
from __future__ import annotations
import numpy as np
import torch

try:  # scipy is the primary CPU reference; networkx is an optional cross-check
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

PENALTY = 1000.0  # matches DistrictNet PENALITY (src/struct.jl); overridable per call


# =============================================================================================
# 1. Vectorized Boruvka: MST weight for a BATCH of weightings over one shared topology.
# =============================================================================================
def _flatten_roots(parent: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Pointer-jump the union-find forest (B, N) to idempotent roots (B, N)."""
    p = parent
    for _ in range(int(np.ceil(np.log2(max(n_nodes, 2)))) + 1):
        p = torch.gather(p, 1, p)
    return p


def batched_mst_weight(edge_u, edge_v, weights, num_nodes, tie_scale: float = None):
    """MST weight per batch element via Boruvka star-contraction, fully on `weights.device`.

    Args:
        edge_u, edge_v: (E,) long tensors, the (undirected) edge endpoints, shared across the batch.
        weights: (B, E) edge weights (one weighting per batch element). float dtype (float64 advised).
        num_nodes: number of graph vertices N. Vertices with no finite-weight incident edge stay
            isolated (this is how a district subgraph is scored: non-district edges are set to +inf).
        tie_scale: unused placeholder kept for API stability (tie-break is exact via edge id).

    Returns:
        mst_weight: (B,) total weight of the minimum spanning forest.
        n_edges: (B,) number of tree edges added (== #reachable_nodes - #components). If the finite
            subgraph on K nodes is connected, n_edges == K - 1; fewer means it is disconnected.
    """
    dev = weights.device
    dtype = weights.dtype
    B, E = weights.shape
    N = int(num_nodes)
    eu = edge_u.to(dev).long()
    ev = edge_v.to(dev).long()
    eid = torch.arange(E, device=dev, dtype=torch.long)

    parent = torch.arange(N, device=dev).unsqueeze(0).expand(B, N).clone()
    total = torch.zeros(B, device=dev, dtype=dtype)
    n_edges = torch.zeros(B, device=dev, dtype=torch.long)

    INF = torch.finfo(dtype).max
    euB = eu.unsqueeze(0).expand(B, E)
    evB = ev.unsqueeze(0).expand(B, E)

    for _ in range(N + 2):  # Boruvka needs O(log N) rounds; N+2 is a safe hard cap
        root = _flatten_roots(parent, N)                      # (B, N)
        cu = torch.gather(root, 1, euB)                       # (B, E) component of u
        cv = torch.gather(root, 1, evB)                       # (B, E) component of v
        # An edge is a valid MST candidate only if it crosses two components AND has finite weight.
        # Non-existent / masked-out edges (used to score a district subgraph) carry +inf and must NOT
        # connect their endpoints, so isolated / out-of-district nodes stay isolated (contribute 0).
        outgoing = (cu != cv) & torch.isfinite(weights)
        if not bool(outgoing.any()):
            break
        w = torch.where(outgoing, weights, torch.full_like(weights, INF))  # (B, E)

        # ---- per-component minimum outgoing weight (edge counts for BOTH endpoint comps) ----
        best_w = torch.full((B, N), INF, device=dev, dtype=dtype)
        best_w.scatter_reduce_(1, cu, w, reduce="amin", include_self=True)
        best_w.scatter_reduce_(1, cv, w, reduce="amin", include_self=True)

        # ---- among edges achieving that min weight, pick the smallest edge id (strict order) ----
        wu = torch.gather(best_w, 1, cu)                      # comp-min weight seen from u side
        wv = torch.gather(best_w, 1, cv)
        eidB = eid.unsqueeze(0).expand(B, E)
        big = E + 1
        cand_u = torch.where(outgoing & (w <= wu), eidB, torch.full_like(eidB, big))
        cand_v = torch.where(outgoing & (w <= wv), eidB, torch.full_like(eidB, big))
        best_eid = torch.full((B, N), big, device=dev, dtype=torch.long)
        best_eid.scatter_reduce_(1, cu, cand_u, reduce="amin", include_self=True)
        best_eid.scatter_reduce_(1, cv, cand_v, reduce="amin", include_self=True)

        has_edge = best_eid < big                             # (B, N) comp has an outgoing edge
        sel = torch.where(has_edge, best_eid, torch.zeros_like(best_eid))
        # the OTHER endpoint component reached by each comp's selected edge -> the hook proposal
        se_u = torch.gather(cu, 1, sel)                       # comp on u side of selected edge
        se_v = torch.gather(cv, 1, sel)
        comp_ids = torch.arange(N, device=dev).unsqueeze(0).expand(B, N)
        proposal = torch.where(se_u == comp_ids, se_v, se_u)  # neighbour component
        proposal = torch.where(has_edge, proposal, comp_ids)  # isolated comps point to themselves

        # ---- resolve mutual 2-cycles: in a pair (c<->d) keep the smaller id as root ----
        prop_of_prop = torch.gather(proposal, 1, proposal)
        mutual = (prop_of_prop == comp_ids) & (proposal > comp_ids)
        proposal = torch.where(mutual, comp_ids, proposal)
        hooking = proposal != comp_ids                        # comps that actually add their edge

        # ---- count each added tree edge exactly once (one per hooking component) ----
        add_w = torch.where(hooking, torch.gather(best_w, 1, comp_ids), torch.zeros_like(best_w))
        # best_w at a comp is its min outgoing weight; that IS the selected edge weight
        total = total + add_w.sum(dim=1)
        n_edges = n_edges + hooking.sum(dim=1)

        # ---- apply hooks, then flatten for the next round ----
        parent = torch.where(hooking, proposal, root)
        parent = _flatten_roots(parent, N)
        if not bool(hooking.any()):
            break

    return total, n_edges


# =============================================================================================
# 2. District-level CMST surrogate, batched over PolyStep probes AND candidate partitions.
# =============================================================================================
def build_partition_edge_masks(edge_u, edge_v, districts, depot_node):
    """For each district (list of block ids), a boolean edge mask of intra-district NON-depot edges.

    Returns (masks (R, E) bool, sizes (R,) int). An edge is intra-district iff both endpoints are in
    the district and neither is the depot (depot edges never enter the MST, per Kruskal.jl)."""
    eu = np.asarray(edge_u); ev = np.asarray(edge_v)
    R = len(districts)
    E = len(eu)
    masks = np.zeros((R, E), dtype=bool)
    sizes = np.zeros(R, dtype=np.int64)
    for j, D in enumerate(districts):
        Dset = set(int(x) for x in D)
        sizes[j] = len(Dset)
        in_u = np.array([u in Dset for u in eu])
        in_v = np.array([v in Dset for v in ev])
        masks[j] = in_u & in_v & (eu != depot_node) & (ev != depot_node)
    return masks, sizes


def cmst_surrogate_batched(edge_u, edge_v, theta, partitions_masks, partitions_sizes,
                           depot_edge_index, depot_theta, num_nodes, l, u,
                           penalty=PENALTY, max_size=None):
    """Batched districting surrogate over probes and candidate partitions.

    Args:
        edge_u, edge_v: (E,) long, shared topology (non-depot block edges).
        theta: (B, E) probe edge weights on CUDA/CPU (B = probes x particles).
        partitions_masks: list of (R_p, E) bool tensors, one per candidate partition.
        partitions_sizes: list of (R_p,) int tensors (district sizes).
        depot_edge_index: list (len P) of lists; for partition p, district j -> LongTensor of the
            edge ids (into theta's companion depot_theta) of that district's depot edges. May be None
            to drop the depot term.
        depot_theta: (B, E_depot) probe depot-edge weights, or None.
        num_nodes, l, u: graph size and [min, max] district size. max_size defaults to u.
        penalty: PENALTY constant.

    Returns:
        best_cost: (B,) minimum-over-partitions total surrogate per probe.
        best_part: (B,) index of the chosen candidate partition per probe.
        per_partition: (B, P) surrogate of each partition (for diagnostics).
    """
    dev = theta.device
    dtype = theta.dtype
    B, E = theta.shape
    max_size = u if max_size is None else max_size
    P = len(partitions_masks)
    per_partition = torch.empty(B, P, device=dev, dtype=dtype)

    for p in range(P):
        masks = partitions_masks[p].to(dev)           # (R, E) bool
        sizes = partitions_sizes[p].to(dev)           # (R,)
        R = masks.shape[0]
        # Stack (B x R) masked weightings: district j keeps only its intra-district edges (else +inf,
        # which batched_mst_weight treats as absent so out-of-district nodes stay isolated).
        INF = float("inf")
        th = theta.unsqueeze(1).expand(B, R, E)        # (B, R, E)
        mk = masks.unsqueeze(0).expand(B, R, E)        # (B, R, E)
        big_w = torch.where(mk, th, torch.full_like(th, INF)).reshape(B * R, E)
        mst, n_e = batched_mst_weight(edge_u, edge_v, big_w, num_nodes)
        mst = mst.reshape(B, R)
        n_e = n_e.reshape(B, R)

        need = (sizes - 1).clamp(min=0).unsqueeze(0)   # (1, R) edges for a connected district
        connected = n_e >= need                        # (B, R)
        singleton = (sizes <= 1).unsqueeze(0).expand(B, R)
        # size-violation penalty (constant in theta): PENALTY * max(0, |D|-u, l-|D|)
        viol = torch.clamp(torch.maximum(sizes - u, l - sizes), min=0).to(dtype)  # (R,)
        size_pen = (penalty * viol).unsqueeze(0).expand(B, R)

        # depot term: min over district nodes of that node's depot-edge weight
        depot_term = torch.zeros(B, R, device=dev, dtype=dtype)
        if depot_theta is not None and depot_edge_index is not None:
            for j in range(R):
                idx = depot_edge_index[p][j]
                if idx is None or len(idx) == 0:
                    continue
                depot_term[:, j] = depot_theta[:, idx.to(dev)].min(dim=1).values

        dist_cost = mst + size_pen + depot_term
        # disconnected or singleton district -> penalize like a singleton (PENALTY * max_size)
        bad = singleton | (~connected)
        dist_cost = torch.where(bad, torch.full_like(dist_cost, penalty * max_size), dist_cost)
        per_partition[:, p] = dist_cost.sum(dim=1)

    best_cost, best_part = per_partition.min(dim=1)
    return best_cost, best_part, per_partition


# =============================================================================================
# 3. CPU references (scipy MST + a direct Python mirror of compute_cost_via_CMST).
# =============================================================================================
def mst_weight_reference(edge_u, edge_v, weight_vec, num_nodes):
    """Single-weighting MST weight via scipy (undirected). Returns (weight, n_tree_edges).

    Only finite-weight edges participate; an isolated / unreachable node contributes no tree edge, so
    the returned n_tree_edges is (#reachable nodes - #components), matching batched_mst_weight."""
    eu = np.asarray(edge_u); ev = np.asarray(edge_v); w = np.asarray(weight_vec, dtype=np.float64)
    finite = np.isfinite(w)
    eu, ev, w = eu[finite], ev[finite], w[finite]
    if len(w) == 0:
        return 0.0, 0
    if not _HAVE_SCIPY:
        return _mst_weight_networkx(eu, ev, w, num_nodes)
    n = int(num_nodes)
    # symmetric CSR; scipy MST ignores explicit zeros, so shift weights to be strictly positive then
    # subtract the shift back out per selected edge count.
    shift = 0.0
    wpos = w
    if (w <= 0).any():
        shift = float(-w.min()) + 1.0
        wpos = w + shift
    rows = np.concatenate([eu, ev]); cols = np.concatenate([ev, eu])
    data = np.concatenate([wpos, wpos])
    M = csr_matrix((data, (rows, cols)), shape=(n, n))
    T = minimum_spanning_tree(M)
    n_tree = int(T.nnz)
    total = float(T.sum()) - shift * n_tree
    return total, n_tree


def _mst_weight_networkx(eu, ev, w, num_nodes):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(int(num_nodes)))
    for a, b, c in zip(eu, ev, w):
        if G.has_edge(int(a), int(b)):
            if c < G[int(a)][int(b)]["weight"]:
                G[int(a)][int(b)]["weight"] = float(c)
        else:
            G.add_edge(int(a), int(b), weight=float(c))
    T = nx.minimum_spanning_tree(G)
    return float(sum(d["weight"] for _, _, d in T.edges(data=True))), T.number_of_edges()


def cmst_surrogate_reference(edge_u, edge_v, theta_vec, districts, depot_node,
                             depot_edge_map, depot_theta_vec, num_nodes, l, u,
                             penalty=PENALTY, max_size=None):
    """Single-weighting districting surrogate, a direct Python mirror of compute_cost_via_CMST summed
    over the districts of ONE partition. Used as the ground truth for the batched scorer."""
    max_size = u if max_size is None else max_size
    eu = np.asarray(edge_u); ev = np.asarray(edge_v)
    theta = np.asarray(theta_vec, dtype=np.float64)
    total = 0.0
    for j, D in enumerate(districts):
        Dset = set(int(x) for x in D)
        size = len(Dset)
        if size <= 1:
            total += penalty * max_size
            continue
        mask = np.array([(eu[e] in Dset) and (ev[e] in Dset)
                         and eu[e] != depot_node and ev[e] != depot_node for e in range(len(eu))])
        sub_w = np.where(mask, theta, np.inf)
        mst, n_tree = mst_weight_reference(eu, ev, sub_w, num_nodes)
        if n_tree < size - 1:  # disconnected district (mirror Julia's < n-1 penalty)
            total += penalty * max_size
            continue
        cost = mst
        cost += penalty * max(0, size - u, l - size)
        if depot_theta_vec is not None and depot_edge_map is not None:
            idx = depot_edge_map[j]
            if idx is not None and len(idx) > 0:
                cost += float(np.asarray(depot_theta_vec)[np.asarray(idx)].min())
        total += cost
    return total
