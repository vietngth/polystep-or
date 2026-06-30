"""
Axis 4b (lead result): PolyStep trains an INHERENTLY INTERPRETABLE but NON-DIFFERENTIABLE
predictor END-TO-END on decision regret — a capability the surrogate/gradient camp structurally
lacks.

The predictor is a hard axis-aligned DECISION TREE: each internal node hard-selects a feature
(argmax over per-node logits) and hard-thresholds it; each leaf emits a constant cost vector.
The forward map is piecewise constant, so d(cost)/d(params) = 0 almost everywhere — SPO+,
cvxpylayers, and every gradient-DFL method cannot train it (we verify the zero gradient below).
PolyStep is zeroth-order: it optimizes the tree's (continuous) thresholds / leaf values / feature
logits directly on the TRUE realized decision cost.

Comparison (knapsack ILP, PyEPO data + Gurobi-evaluated normalized regret):
  - tree + MSE          : the same interpretable tree fit by ordinary regression (sklearn) — two-stage,
                          decision-blind.
  - tree + SPO+ / DFL   : INAPPLICABLE (zero gradient; demonstrated numerically).
  - tree + PolyStep     : the same interpretable tree trained on decision regret (ours).
  - linear + SPO+       : reference point (a differentiable model the surrogate CAN train).
  - linear + PolyStep   : reference point.

Headline: tree+PolyStep < tree+MSE  (decision-focused training of an interpretable model helps),
and tree+PolyStep is unreachable by the surrogate camp at all.

Run: .venv/bin/python interpretable_predictor.py [deg] [seeds]
"""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "polystep/src")
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from pyepo import metric
from pto.capability import (setup_sp, setup_knap, setup_tsp, setup_port,
                            train_two_stage, train_dfl, train_polystep, dev, PF)

SETUPS = {"shortest_path": setup_sp, "knapsack": setup_knap, "tsp": setup_tsp, "portfolio": setup_port}


# ---------------------------------------------------------------------------
# Hard (non-differentiable) axis-aligned decision tree, batched over N particles.
# ---------------------------------------------------------------------------
def route(fl, thr, leaf, X):
    """fl (N,I,p) feature logits; thr (N,I) thresholds; leaf (N,L,dim) leaf costs;
    X (B,p) -> predicted costs (N,B,dim) via HARD routing (argmax + threshold)."""
    N, I, p = fl.shape
    L = leaf.shape[1]
    depth = int(round(np.log2(I + 1)))
    jsel = fl.argmax(-1)                                   # (N,I) feature picked at each node
    Xe = X.to(fl.dtype).unsqueeze(0).expand(N, -1, p)     # (N,B,p)
    B = X.shape[0]
    cur = torch.zeros(N, B, dtype=torch.long, device=fl.device)
    for _ in range(depth):
        j_cur = jsel.gather(1, cur)                        # (N,B) feature index at current node
        thr_cur = thr.gather(1, cur)                       # (N,B) threshold at current node
        xval = Xe.gather(2, j_cur.unsqueeze(-1)).squeeze(-1)
        go = (xval > thr_cur).long()                       # hard split: 1 = right child
        cur = 2 * cur + 1 + go
    leaf_idx = (cur - I).clamp(min=0, max=L - 1)           # (N,B)
    return leaf.gather(1, leaf_idx.unsqueeze(-1).expand(N, B, leaf.shape[-1]))


class HardTree(nn.Module):
    """Interpretable, non-differentiable predictor x -> cost vector."""
    def __init__(self, p, dim, depth, seed):
        super().__init__()
        I = 2 ** depth - 1; L = 2 ** depth
        g = torch.Generator().manual_seed(seed)
        self.I = I; self.L = L; self.depth = depth
        self.fl = nn.Parameter(torch.randn(I, p, generator=g) * 0.5)
        self.thr = nn.Parameter(torch.zeros(I))            # features are ~N(0,1) standardized
        self.leaf = nn.Parameter(torch.randn(L, dim, generator=g) * 0.5)

    def forward(self, x):                                   # x (B,p) -> (B,dim)
        out = route(self.fl.unsqueeze(0), self.thr.unsqueeze(0), self.leaf.unsqueeze(0),
                    torch.as_tensor(x, device=self.fl.device))
        return out.squeeze(0)


def fit_balanced_cart(X, C, depth):
    """Greedy CART filling a FULL binary tree of the given depth, multi-output, on (X, C).
    Returns (fl, thr, leaf) numpy arrays directly loadable into a HardTree. Same topology as
    the PolyStep-refined tree, so the MSE baseline and the refined model are apples-to-apples."""
    n, p = X.shape; dim = C.shape[1]
    I = 2 ** depth - 1; L = 2 ** depth
    fl = np.zeros((I, p), np.float32); thr = np.zeros(I, np.float32); leaf = np.zeros((L, dim), np.float32)
    node_idx = {0: np.arange(n)}
    def sse(idx):
        return float(((C[idx] - C[idx].mean(0)) ** 2).sum()) if len(idx) else 0.0
    for node in range(I):                                   # BFS over internal nodes
        idx = node_idx.get(node, np.array([], int))
        if len(idx) < 2:
            fl[node, 0] = 10.0; thr[node] = 0.0
            node_idx[2 * node + 1] = idx; node_idx[2 * node + 2] = np.array([], int); continue
        best = (np.inf, 0, 0.0, None, None)
        for j in range(p):
            xs = X[idx, j]; cands = np.quantile(xs, [0.25, 0.5, 0.75])
            for t in np.unique(cands):
                left = idx[xs <= t]; right = idx[xs > t]
                if len(left) == 0 or len(right) == 0: continue
                cost = sse(left) + sse(right)
                if cost < best[0]: best = (cost, j, float(t), left, right)
        _, j, t, left, right = best
        if left is None:                                   # no valid split: degenerate node
            fl[node, 0] = 10.0; thr[node] = 0.0
            node_idx[2 * node + 1] = idx; node_idx[2 * node + 2] = np.array([], int); continue
        fl[node, j] = 10.0; thr[node] = t                  # one-hot feature pick (argmax) + threshold
        node_idx[2 * node + 1] = left; node_idx[2 * node + 2] = right   # go=0 (x<=t) left, go=1 right
    for l in range(L):
        idx = node_idx.get(I + l, np.array([], int))
        leaf[l] = C[idx].mean(0) if len(idx) else C.mean(0)
    return fl, thr, leaf


def _load_cart(m, cfg, depth):
    X = cfg["Xtr"].cpu().numpy(); C = cfg["Cs"].cpu().numpy()   # standardized costs -> params stay O(1)
    fl, thr, leaf = fit_balanced_cart(X, C, depth)
    with torch.no_grad():
        m.fl.copy_(torch.tensor(fl, device=dev)); m.thr.copy_(torch.tensor(thr, device=dev))
        m.leaf.copy_(torch.tensor(leaf, device=dev))
    return m


def train_tree_mse(cfg, depth=3, seed=0):
    """The interpretable tree fit by ordinary regression (two-stage, decision-blind)."""
    return _load_cart(HardTree(PF, cfg["dim"], depth, seed).to(dev), cfg, depth)


def train_polystep_tree(cfg, depth=3, steps=250, seed=0):
    m = _load_cart(HardTree(PF, cfg["dim"], depth, seed).to(dev), cfg, depth)   # warm-start from MSE tree
    pso = PolyStepOptimizer(m, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=0.4, probe_radius=0.8, num_probe=1, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    X, Cs, solve, sgn = cfg["Xtr"], cfg["Cs"], cfg["ps_solve"], cfg["sign"]
    def closure(bp):
        pred = route(bp["fl"], bp["thr"], bp["leaf"], X)   # (N,B,dim)
        N, B, E = pred.shape
        w = solve(pred.reshape(N * B, E)).reshape(N, B, E)
        return sgn * (w * Cs.unsqueeze(0)).sum(-1).mean(-1)   # (N,) realized standardized cost
    for _ in range(steps):
        pso.step(closure)
    return m


def grad_is_zero_demo(cfg, depth=3, seed=0):
    """Numerically demonstrate the tree is untrainable by gradient/surrogate methods:
    the realized decision cost has zero gradient w.r.t. every tree parameter."""
    m = HardTree(PF, cfg["dim"], depth, seed).to(dev)
    X, Cs, solve, sgn = cfg["Xtr"], cfg["Cs"], cfg["ps_solve"], cfg["sign"]
    pred = m(X)                                             # (B,dim), through hard routing
    w = solve(pred)                                         # decision = argmin/argmax over pred
    loss = sgn * (w * Cs).sum(-1).mean()                   # realized decision cost (the true objective)
    if not loss.requires_grad:
        # the decision cost is piecewise-constant in the params: argmax/threshold routing and the
        # argmin solver both break the autograd graph -> NO gradient signal exists at all.
        return None
    g = torch.autograd.grad(loss, list(m.parameters()), allow_unused=True, retain_graph=True)
    return [0.0 if gi is None else float(gi.abs().sum()) for gi in g]


def run_problem(problem, deg, seeds, depth=3):
    setup = SETUPS[problem]
    rows = {k: [] for k in ("tree+MSE", "tree+PolyStep", "linear+SPO+", "linear+PolyStep")}
    for seed in seeds:
        cfg, _ = setup(seed, deg)
        ts = train_two_stage(cfg); cfg["warm"] = ts
        rows["tree+MSE"].append(metric.regret(train_tree_mse(cfg, depth, seed), cfg["om"], cfg["ld_te"]))
        rows["tree+PolyStep"].append(metric.regret(train_polystep_tree(cfg, depth, seed=seed),
                                                   cfg["om"], cfg["ld_te"]))
        try:
            rows["linear+SPO+"].append(metric.regret(train_dfl(cfg, "SPO+"), cfg["om"], cfg["ld_te"]))
        except Exception:
            rows["linear+SPO+"].append(float("nan"))
        rows["linear+PolyStep"].append(metric.regret(train_polystep(cfg), cfg["om"], cfg["ld_te"]))
    return {k: np.array(v) for k, v in rows.items()}


def main():
    problems = sys.argv[1].split(",") if len(sys.argv) > 1 else ["knapsack", "shortest_path", "portfolio"]
    deg = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2]
    DEPTH = 3
    print(f"Interpretable predictor on decision regret | deg={deg} | tree depth={DEPTH} | "
          f"{len(seeds)} seeds | normalized regret (lower better)\n")

    # one-off: prove the tree is untrainable by gradient methods
    cfg0, _ = SETUPS[problems[0]](seeds[0], deg)
    norms = grad_is_zero_demo(cfg0, DEPTH, seeds[0])
    if norms is None:
        print("[gradient check] realized decision cost has NO autograd connection to the tree "
              "parameters\n  (piecewise-constant: hard routing + argmin solver) -> gradient "
              "identically undefined.")
    else:
        print(f"[gradient check] sum|d(decision cost)/d(tree params)| = {norms}")
    print("  -> SPO+/cvxpylayers/gradient-DFL CANNOT train this predictor; PolyStep (zeroth-order) can.\n")

    hdr = f"{'problem':>14} | {'tree+MSE':>14} {'tree+PolyStep':>14} | {'lin+SPO+':>9} {'lin+PolyStep':>12} | tree refine"
    print(hdr); print("-" * len(hdr))
    for problem in problems:
        R = run_problem(problem, deg, seeds, DEPTH)
        tm, tp = R["tree+MSE"].mean(), R["tree+PolyStep"].mean()
        gain = (tm - tp) / max(tm, 1e-9) * 100
        print(f"{problem:>14} | {tm:>6.4f}+/-{R['tree+MSE'].std():<5.4f} "
              f"{tp:>6.4f}+/-{R['tree+PolyStep'].std():<5.4f} | "
              f"{np.nanmean(R['linear+SPO+']):>9.4f} {R['linear+PolyStep'].mean():>12.4f} | "
              f"{gain:+.0f}% {'(PolyStep)' if tp < tm else '(MSE)'}", flush=True)


if __name__ == "__main__":
    main()
