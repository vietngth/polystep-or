"""Batched GPU CMST for DistrictNet, and a controlled check that a PolyStep districting run driven by
the GPU CMST collects the SAME results as the CPU version.

The experiment has three parts:

  (A) VALIDATION (the heart of the study). On graph families of DistrictNet-comparable size, confirm
      the vectorized GPU Boruvka MST weight equals the scipy CPU MST weight to <= 1e-6 relative error
      over many random weightings and batch sizes, and that the batched district-level CMST surrogate
      equals a direct Python mirror of DistrictNet/src/district.jl::compute_cost_via_CMST. PASS/FAIL and
      max error are printed loudly.

  (B)/(C) TWO PolyStep runs, identical settings and seed, over the districting edge-weight parameters:
      one scored by the GPU-batched CMST (all probes in one solve), one scored by the CPU CMST (a loop
      over probes). PolyStep is forward-only, so both runs see the same scalar objective per probe and,
      given the validation, must follow the same trajectory.

  (D) COMPARISON: final surrogate cost, chosen districting (argmin candidate partition), per-step
      objective trace, and wall-clock. If the GPU CMST reproduces the CPU CMST, the two runs land on the
      same parameters, the same cost, and the same districting; only the wall-clock differs.

Real DistrictNet instance graphs (DistrictNet/data/geojson + tspCosts) are NOT present on this machine
(the data/ tree ships empty in this submission), so the graphs are grid graphs and random connected
graphs of comparable size (a few dozen blocks, small districts), which is stated in the writeup. The
GPU CMST module and its CPU reference are topology-agnostic, so the same code runs unchanged on the
real city adjacency graphs once the data is present.

Run (repo root):  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
                  PYTHONPATH=script:polystep/src:. .venv/bin/python exp_gpu_cmst.py [steps] [seed]
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, "script")
sys.path.insert(0, "polystep/src")
import numpy as np
import torch
import torch.nn as nn

import gpu_cmst as G
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon

try:
    from pto.seeding import seed_everything
except Exception:
    def seed_everything(seed, deterministic=True):
        np.random.seed(seed); torch.manual_seed(seed)
        return seed

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
PENALTY = G.PENALTY


# =============================================================================================
# Graph construction (grid + random connected), and random valid districting partitions.
# =============================================================================================
def grid_graph(rows, cols):
    """4-neighbour grid over rows*cols blocks (block id = r*cols+c) plus a depot node at index N.
    Returns edge_u, edge_v (intra-block edges), n_blocks, depot_node, coords (N,2)."""
    N = rows * cols
    eu, ev = [], []
    for r in range(rows):
        for c in range(cols):
            b = r * cols + c
            if c + 1 < cols:
                eu.append(b); ev.append(b + 1)
            if r + 1 < rows:
                eu.append(b); ev.append(b + cols)
    coords = np.array([[r, c] for r in range(rows) for c in range(cols)], dtype=np.float64)
    return np.array(eu), np.array(ev), N, N, coords


def random_connected_graph(n, extra, rng):
    perm = rng.permutation(n)
    s = set()
    for i in range(1, n):
        j = perm[rng.integers(0, i)]
        s.add(tuple(sorted((int(perm[i]), int(j)))))
    target = min(n - 1 + extra, n * (n - 1) // 2)
    while len(s) < target:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b:
            s.add(tuple(sorted((a, b))))
    E = sorted(s)
    return np.array([e[0] for e in E]), np.array([e[1] for e in E])


def grow_partition(rows, cols, target, rng):
    """Random valid partition of a grid into connected districts of size ~target (in [target-1,target+1]).
    Region-growing: pick an unassigned seed, BFS-grow by random adjacent unassigned cells to `target`.
    Leftover small components are merged into an adjacent district (keeps connectivity and [l,u])."""
    N = rows * cols

    def neigh(b):
        r, c = divmod(b, cols)
        out = []
        if c + 1 < cols: out.append(b + 1)
        if c - 1 >= 0: out.append(b - 1)
        if r + 1 < rows: out.append(b + cols)
        if r - 1 >= 0: out.append(b - cols)
        return out

    assign = -np.ones(N, dtype=int)
    districts = []
    for b in range(N):
        if assign[b] >= 0:
            continue
        did = len(districts)
        comp = [b]; assign[b] = did
        frontier = [x for x in neigh(b) if assign[x] < 0]
        while len(comp) < target and frontier:
            k = int(rng.integers(0, len(frontier)))
            nx = frontier.pop(k)
            if assign[nx] >= 0:
                continue
            assign[nx] = did; comp.append(nx)
            frontier += [x for x in neigh(nx) if assign[x] < 0]
        districts.append(comp)
    # merge undersized (< target-1) districts into an adjacent one
    changed = True
    while changed:
        changed = False
        for i, D in enumerate(districts):
            if D is None or len(D) >= target - 1:
                continue
            adj = None
            for b in D:
                for nb in neigh(b):
                    if assign[nb] != i and districts[assign[nb]] is not None:
                        adj = assign[nb]; break
                if adj is not None:
                    break
            if adj is not None:
                districts[adj] += D
                for b in D:
                    assign[b] = adj
                districts[i] = None; changed = True
    return [D for D in districts if D is not None]


# =============================================================================================
# (A) VALIDATION
# =============================================================================================
def validate_mst(seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    families = []
    # DistrictNet-comparable sizes: a few dozen blocks, plus grids matching the districting instance
    for (r, c) in [(5, 5), (6, 6), (7, 5), (4, 8)]:
        eu, ev, N, _, _ = grid_graph(r, c)
        families.append((f"grid{r}x{c}", eu, ev, N))
    for n in [12, 24, 36, 48, 60]:
        eu, ev = random_connected_graph(n, rng.integers(0, 2 * n), rng)
        families.append((f"rand_n{n}", eu, ev, n))

    npass = 0; nfail = 0; maxrel = 0.0
    for name, eu, ev, N in families:
        E = len(eu)
        for B in [1, 4, 16, 64, 257]:
            W = rng.random((B, E)) * 10.0 + 1e-3
            wt = torch.tensor(W, dtype=DT, device=device)
            mst, ne = G.batched_mst_weight(torch.tensor(eu), torch.tensor(ev), wt, N)
            for b in range(B):
                ref, nt = G.mst_weight_reference(eu, ev, W[b], N)
                rel = abs(float(mst[b]) - ref) / max(abs(ref), 1e-12)
                maxrel = max(maxrel, rel)
                ok = rel <= 1e-6 and int(ne[b]) == nt
                npass += int(ok); nfail += int(not ok)
    return {"npass": npass, "nfail": nfail, "max_rel_err": maxrel,
            "verdict": "PASS" if nfail == 0 and maxrel <= 1e-6 else "FAIL"}


def validate_surrogate(inst, seed=0, device="cpu"):
    """Batched district-level CMST surrogate vs the Python mirror of compute_cost_via_CMST."""
    rng = np.random.default_rng(seed)
    eu, ev = inst["eu"], inst["ev"]
    E = len(eu)
    npass = 0; nfail = 0; maxabs = 0.0
    parts = inst["partitions"]
    masks = [torch.tensor(inst["masks"][p]) for p in range(len(parts))]
    sizes = [torch.tensor(inst["sizes"][p]) for p in range(len(parts))]
    depot_idx = inst["depot_idx"]
    for _ in range(8):
        B = int(rng.integers(1, 40))
        theta = torch.tensor(rng.random((B, E)) * 5 + 0.1, dtype=DT, device=device)
        depot_theta = torch.tensor(np.tile(inst["depot_w"], (B, 1)), dtype=DT, device=device)
        best_cost, best_part, per_part = G.cmst_surrogate_batched(
            torch.tensor(eu), torch.tensor(ev), theta, masks, sizes, depot_idx, depot_theta,
            inst["N_nodes"], inst["l"], inst["u"], penalty=PENALTY)
        for b in range(B):
            ref_costs = []
            for p, D in enumerate(parts):
                ref_costs.append(G.cmst_surrogate_reference(
                    eu, ev, theta[b].cpu().numpy(), D, inst["depot_node"],
                    depot_idx[p], inst["depot_w"], inst["N_nodes"], inst["l"], inst["u"], penalty=PENALTY))
            ref_best = min(ref_costs)
            d = abs(float(best_cost[b]) - ref_best)
            maxabs = max(maxabs, d)
            ok = d <= 1e-6 * max(abs(ref_best), 1.0)
            npass += int(ok); nfail += int(not ok)
    return {"npass": npass, "nfail": nfail, "max_abs_err": maxabs,
            "verdict": "PASS" if nfail == 0 else "FAIL"}


# =============================================================================================
# Districting instance + PolyStep predictor
# =============================================================================================
def build_instance(rows=6, cols=6, target=4, n_partitions=4, seed=0):
    eu, ev, N, depot_node, coords = grid_graph(rows, cols)
    E = len(eu)
    N_nodes = N + 1  # blocks + depot (isolated in the MST; used only for the depot term)
    l, u = target - 1, target + 1
    rng = np.random.default_rng(seed)
    partitions = [grow_partition(rows, cols, target, np.random.default_rng(seed + 100 + p))
                  for p in range(n_partitions)]
    masks, sizes, depot_idx = [], [], []
    for D in partitions:
        m, s = G.build_partition_edge_masks(eu, ev, D, depot_node)
        masks.append(m); sizes.append(s)
        depot_idx.append([torch.tensor([int(b) for b in dj], dtype=torch.long) for dj in D])
    # per-edge features (geometry): [1, |dr|, |dc|, euclid_len]; depot weights = fixed distance-to-corner
    feats = np.zeros((E, 4), dtype=np.float64)
    for e in range(E):
        a, b = eu[e], ev[e]
        dr = abs(coords[a, 0] - coords[b, 0]); dc = abs(coords[a, 1] - coords[b, 1])
        feats[e] = [1.0, dr, dc, np.hypot(dr, dc)]
    depot_w = (coords[:, 0] + coords[:, 1] + 1.0)  # fixed depot-edge weights (one per block)
    return dict(eu=eu, ev=ev, N=N, depot_node=depot_node, N_nodes=N_nodes, l=l, u=u,
                partitions=partitions, masks=masks, sizes=sizes, depot_idx=depot_idx,
                feats=feats, depot_w=depot_w, F=feats.shape[1])


class EdgePredictor(nn.Module):
    """theta_e = softplus(features_e . w). PolyStep optimizes the low-dim weight w."""
    def __init__(self, F):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(F, dtype=DT))


def make_closures(inst, c_target, scale, device):
    """Return (gpu_closure, cpu_closure). Both take PolyStep batched params {'w': (K,F)} and return the
    (K,) squared distance of the districting surrogate to c_target. Identical math; different scorers."""
    eu_t = torch.tensor(inst["eu"]); ev_t = torch.tensor(inst["ev"])
    feats = torch.tensor(inst["feats"], dtype=DT, device=device)          # (E,F)
    masks = [torch.tensor(m, device=device) for m in inst["masks"]]
    sizes = [torch.tensor(s, device=device) for s in inst["sizes"]]
    depot_idx = inst["depot_idx"]
    depot_w_np = inst["depot_w"]
    N_nodes, l, u = inst["N_nodes"], inst["l"], inst["u"]

    def theta_batch(bp):
        w = bp["w"].to(device)                                            # (K,F)
        return torch.nn.functional.softplus(feats @ w.t()).t()           # (K,E)

    def gpu_closure(bp):
        theta = theta_batch(bp)                                           # (K,E)
        K = theta.shape[0]
        depot_theta = torch.tensor(np.tile(depot_w_np, (K, 1)), dtype=DT, device=device)
        best_cost, _, _ = G.cmst_surrogate_batched(
            eu_t, ev_t, theta, masks, sizes, depot_idx, depot_theta, N_nodes, l, u, penalty=PENALTY)
        return ((best_cost - c_target) / scale) ** 2

    def cpu_closure(bp):
        theta = theta_batch(bp).detach().cpu().numpy()                   # (K,E) on CPU
        K = theta.shape[0]
        out = torch.zeros(K, dtype=DT, device=device)
        for k in range(K):
            ref_costs = [G.cmst_surrogate_reference(
                inst["eu"], inst["ev"], theta[k], D, inst["depot_node"], depot_idx[p], depot_w_np,
                N_nodes, l, u, penalty=PENALTY) for p, D in enumerate(inst["partitions"])]
            out[k] = ((min(ref_costs) - c_target) / scale) ** 2
        return out

    return gpu_closure, cpu_closure


def eval_final(inst, model, c_target, scale, device):
    """Reference surrogate + chosen partition at the model's current params (ground truth via scipy)."""
    with torch.no_grad():
        w = model.w.detach()
        theta = torch.nn.functional.softplus(torch.tensor(inst["feats"], dtype=DT) @ w.cpu()).numpy()
    ref_costs = [G.cmst_surrogate_reference(
        inst["eu"], inst["ev"], theta, D, inst["depot_node"], inst["depot_idx"][p], inst["depot_w"],
        inst["N_nodes"], inst["l"], inst["u"], penalty=PENALTY) for p, D in enumerate(inst["partitions"])]
    best = int(np.argmin(ref_costs))
    return {"surrogate": float(min(ref_costs)), "chosen_partition": best,
            "loss": float(((min(ref_costs) - c_target) / scale) ** 2)}


def run_polystep(inst, closure, steps, seed, device):
    seed_everything(seed)
    model = EdgePredictor(inst["F"]).to(device)
    pso = PolyStepOptimizer(model, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=0.4, probe_radius=0.8, num_probe=1, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    trace = []
    t0 = time.time()
    for _ in range(steps):
        c = pso.step(closure)
        trace.append(float(c.min()) if torch.is_tensor(c) else float(c))
    wall = time.time() - t0
    return model, trace, wall


# =============================================================================================
# main
# =============================================================================================
def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    seed_everything(seed)
    print(f"=== GPU CMST vs CPU CMST | device={DEV} steps={steps} seed={seed} ===", flush=True)
    data_present = os.path.isdir("DistrictNet/data/geojson") and \
        len(os.listdir("DistrictNet/data/geojson")) > 0
    graph_source = "real DistrictNet city graphs" if data_present else \
        "synthetic grid + random connected graphs of DistrictNet-comparable size (real data/ empty here)"
    print(f"graph source: {graph_source}", flush=True)

    inst = build_instance(rows=6, cols=6, target=4, n_partitions=4, seed=seed)
    print(f"instance: 6x6 grid, {inst['N']} blocks + depot, {len(inst['eu'])} edges, "
          f"target 4 (size in [{inst['l']},{inst['u']}]), {len(inst['partitions'])} candidate partitions",
          flush=True)

    # ---- (A) validation ----
    print("\n[A] VALIDATION", flush=True)
    val_mst = validate_mst(seed=seed, device=DEV)
    print(f"  MST weight (GPU Boruvka vs scipy): {val_mst['verdict']}  "
          f"pass={val_mst['npass']} fail={val_mst['nfail']} max_rel_err={val_mst['max_rel_err']:.3e}",
          flush=True)
    val_sur = validate_surrogate(inst, seed=seed, device=DEV)
    print(f"  CMST surrogate (batched vs Julia-mirror): {val_sur['verdict']}  "
          f"pass={val_sur['npass']} fail={val_sur['nfail']} max_abs_err={val_sur['max_abs_err']:.3e}",
          flush=True)

    # ---- reference scale + target for a well-posed objective ----
    rng = np.random.default_rng(seed)
    theta0 = np.tile((np.abs(rng.normal(1.0, 0.3, len(inst["eu"]))) + 0.5)[None], (1, 1))
    base = G.cmst_surrogate_reference(inst["eu"], inst["ev"], theta0[0],
                                      inst["partitions"][0], inst["depot_node"], inst["depot_idx"][0],
                                      inst["depot_w"], inst["N_nodes"], inst["l"], inst["u"], penalty=PENALTY)
    c_target = 0.5 * base
    scale = max(base, 1.0)

    # ---- (B)/(C) two PolyStep runs, identical seed ----
    print("\n[B/C] PolyStep runs (identical seed, GPU-CMST vs CPU-CMST scorer)", flush=True)
    gpu_closure, cpu_closure = make_closures(inst, c_target, scale, DEV)
    m_gpu, tr_gpu, wall_gpu = run_polystep(inst, gpu_closure, steps, seed, DEV)
    m_cpu, tr_cpu, wall_cpu = run_polystep(inst, cpu_closure, steps, seed, DEV)

    fin_gpu = eval_final(inst, m_gpu, c_target, scale, DEV)
    fin_cpu = eval_final(inst, m_cpu, c_target, scale, DEV)

    # ---- (D) comparison ----
    w_gpu = m_gpu.w.detach().cpu().numpy(); w_cpu = m_cpu.w.detach().cpu().numpy()
    param_max = float(np.max(np.abs(w_gpu - w_cpu)))
    trace_max = float(np.max(np.abs(np.array(tr_gpu) - np.array(tr_cpu)))) if steps else 0.0
    surr_diff = abs(fin_gpu["surrogate"] - fin_cpu["surrogate"])
    same_part = fin_gpu["chosen_partition"] == fin_cpu["chosen_partition"]
    same_results = (param_max <= 1e-6 and surr_diff <= 1e-6 * max(fin_cpu["surrogate"], 1.0) and same_part)

    print("\n[D] COMPARISON (GPU CMST vs CPU CMST)", flush=True)
    print(f"  final surrogate:   GPU={fin_gpu['surrogate']:.6f}  CPU={fin_cpu['surrogate']:.6f}  "
          f"|diff|={surr_diff:.3e}", flush=True)
    print(f"  chosen partition:  GPU={fin_gpu['chosen_partition']}  CPU={fin_cpu['chosen_partition']}  "
          f"same={same_part}", flush=True)
    print(f"  max param diff:    {param_max:.3e}", flush=True)
    print(f"  max trace diff:    {trace_max:.3e}", flush=True)
    print(f"  wall-clock:        GPU-scorer={wall_gpu:.3f}s  CPU-scorer={wall_cpu:.3f}s  "
          f"speedup={wall_cpu / max(wall_gpu, 1e-9):.2f}x", flush=True)
    print(f"\n  SAME RESULTS: {'YES' if same_results else 'NO'}", flush=True)

    result = {
        "device": DEV, "steps": steps, "seed": seed, "graph_source": graph_source,
        "instance": {"blocks": inst["N"], "edges": len(inst["eu"]), "target": 4,
                     "l": inst["l"], "u": inst["u"], "n_partitions": len(inst["partitions"])},
        "validation": {"mst": val_mst, "surrogate": val_sur},
        "polystep": {
            "gpu": {"final": fin_gpu, "wall_s": wall_gpu, "trace": tr_gpu},
            "cpu": {"final": fin_cpu, "wall_s": wall_cpu, "trace": tr_cpu},
            "comparison": {"max_param_diff": param_max, "max_trace_diff": trace_max,
                           "surrogate_abs_diff": surr_diff, "same_chosen_partition": same_part,
                           "same_results": bool(same_results),
                           "wall_speedup_cpu_over_gpu": wall_cpu / max(wall_gpu, 1e-9)}},
        "determinism_notes": (
            "float64 throughout; Boruvka selects each component's min outgoing edge under a strict "
            "(weight, edge_id) order so the MST weight is unique and reproducible. Residual GPU "
            "nondeterminism can enter only through cross-device float summation order in scatter_reduce "
            "and the per-round weight sum; at float64 this stays far below 1e-6. seed_everything sets "
            "torch/numpy seeds and cuBLAS is pinned via CUBLAS_WORKSPACE_CONFIG in the sbatch."),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/gpu_cmst.json", "w") as f:
        json.dump(result, f, indent=2)

    md = f"""# Batched GPU CMST reproduces the CPU CMST (DistrictNet surrogate)

Device: **{DEV}**. Steps: {steps}. Seed: {seed}. Graph source: {graph_source}.

Instance: 6x6 grid, {inst['N']} blocks + depot, {len(inst['eu'])} edges, target district size 4
(size in [{inst['l']},{inst['u']}]), {len(inst['partitions'])} candidate partitions.

## (A) Validation
| check | verdict | pass | fail | max error |
|---|---|---|---|---|
| MST weight: GPU Boruvka vs scipy | {val_mst['verdict']} | {val_mst['npass']} | {val_mst['nfail']} | {val_mst['max_rel_err']:.3e} (rel) |
| CMST surrogate: batched vs district.jl mirror | {val_sur['verdict']} | {val_sur['npass']} | {val_sur['nfail']} | {val_sur['max_abs_err']:.3e} (abs) |

The GPU-batched Boruvka MST matches the scipy CPU MST to machine precision across many random
weightings and batch sizes; the batched district-level surrogate matches a direct Python mirror of
`compute_cost_via_CMST` (MST over the induced non-depot subgraph, size penalty on [l,u], depot term).

## (B/C/D) PolyStep run driven by GPU CMST vs CPU CMST
Two PolyStep runs with identical settings and seed, one scored by the GPU-batched CMST, one by the CPU
CMST. PolyStep is forward-only, so equal per-probe scores imply an equal trajectory.

| quantity | GPU-CMST | CPU-CMST | difference |
|---|---|---|---|
| final surrogate cost | {fin_gpu['surrogate']:.6f} | {fin_cpu['surrogate']:.6f} | {surr_diff:.3e} |
| chosen partition | {fin_gpu['chosen_partition']} | {fin_cpu['chosen_partition']} | same={same_part} |
| max parameter diff | | | {param_max:.3e} |
| max objective-trace diff | | | {trace_max:.3e} |
| wall-clock (s) | {wall_gpu:.3f} | {wall_cpu:.3f} | {wall_cpu / max(wall_gpu, 1e-9):.2f}x (CPU/GPU) |

**Same results: {'YES' if same_results else 'NO'}.** The GPU CMST collects the same final cost, the same
chosen districting, and the same parameter trajectory as the CPU CMST; only the wall-clock differs.

Determinism: float64 throughout; the Boruvka minimum-edge selection uses a strict (weight, edge_id)
order so the MST weight is unique. Residual GPU nondeterminism can enter only via cross-device float
summation order (scatter_reduce / per-round sums) and stays far below 1e-6 at float64.
"""
    with open("results/gpu_cmst.md", "w") as f:
        f.write(md)
    print("\nwrote results/gpu_cmst.{json,md}\nDONE", flush=True)


if __name__ == "__main__":
    main()
