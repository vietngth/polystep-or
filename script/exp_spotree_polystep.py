"""
SPO-Tree (Elmachtoub, Liang, McNellis, ICML 2020) vs CART(+MSE) two-stage vs
PolyStep-refined tree, on the AUTHORS' 4x4-grid shortest-path benchmark.

Canonical NON-DIFFERENTIABLE PREDICTOR case: a decision tree's splits are
piecewise-constant, so gradient DFL (SPO+/DBB/PFYL/cvxpylayers) cannot train it.
PolyStep (gradient-free, realized-cost objective) CAN refine it: it treats the
tree's leaf cost-vectors as optimization variables and minimizes the realized
decision cost (a scalar, evaluated via the authors' Gurobi shortest-path solver).

Methodology: uses the AUTHORS' code (baselines/SPOTree, minimally Py3-ported) and
their dataset (archive.org spotree_shortestpathdata) on the paper-default setting:
  n_train=200, 4x4 grid, deg in {2,10}, eps=0, max_depth=3, min_weights_per_node=20,
  CART pruning on a 20% validation split, 10 dataset replications.

Methods compared (all share the AUTHORS' tree code + Gurobi optimizer):
  * SPOTree   : their greedy SPO tree            (SPO_weight_param=1.0)
  * CART(MSE) : their MSE/CART tree = two-stage   (SPO_weight_param=0.0)
  * PolyStep  : CART tree structure FIXED, leaf cost-vectors refined on realized
                regret by gradient-free PolyStep (no labels, forward solves only)
  * grad-DFL  : N/A  (zero gradient through the tree's argmin-of-piecewise-constant)

Writes exp_results/spotree.{json,md}.
"""
import sys, os, time, json, argparse, pickle
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "baselines/SPOTree/Applications/Shortest Path"))
sys.path.insert(0, os.path.join(HERE, "baselines/SPOTree/Algorithms"))
sys.path.insert(0, os.path.join(HERE, "polystep/src"))

from SPO_tree_greedy import SPOTree                       # authors' tree
from decision_problem_solver import find_opt_decision, get_num_decisions  # authors' Gurobi SP
from polystep.solver import PolyStep

DATA_DIR = os.environ.get(
    "SPOTREE_DATA",
    "/media/anindex/Data/project-cache/ot-or-project/spotree_data/ShortestPathData",
)


def load_data(n_train):
    fn = ("non_linear_bigdata10000_dim4.p" if n_train == 10000
          else "non_linear_data_dim4.p")
    path = os.path.join(DATA_DIR, fn)
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")   # authors' Py2 pickle


def fit_tree(spo_weight, tr_x, tr_cost, val_x, val_cost, depth, min_obs):
    t = time.time()
    tree = SPOTree(max_depth=depth, min_weights_per_node=min_obs, quant_discret=0.01,
                   debias_splits=False, SPO_weight_param=spo_weight, SPO_full_error=True)
    tree.fit(tr_x, tr_cost, verbose=False, feats_continuous=True)
    tree.prune(val_x, val_cost, verbose=False, one_SE_rule=False)
    return tree, time.time() - t


def realized_cost(tree, test_x, test_cost, leaf_decision=None):
    """Mean realized decision cost on the test set.
    If leaf_decision (dict leaf_node->decision) given, use refined decisions."""
    if leaf_decision is None:
        pred = tree.est_decision(test_x)
    else:
        _, (unq, unq_inds_vec) = tree.est_decision(test_x, return_loc=True)
        pred = np.zeros_like(test_cost, dtype=float)
        for i, n in enumerate(unq):
            pred[unq_inds_vec[i]] = leaf_decision[n]
    return np.array([np.sum(test_cost[i] * pred[i]) for i in range(len(pred))])


def extract_structure(base_tree):
    """Extract the pruned tree topology as a routing structure.

    NOTE: refining the leaf cost-vectors is a no-op here: the leaf model
    predicts the per-leaf MEAN cost, and argmin_path (mean_cost . path) ==
    argmin_path (sum_of_true_costs . path), so the two-stage (CART) leaf
    decision is ALREADY the in-leaf realized-cost optimum -- nothing to refine.
    The genuinely non-differentiable, refinable parameters are the SPLIT
    THRESHOLDS: they enter through hard comparisons (x < thr) -> routing ->
    argmin -> cost, which is piecewise-constant in thr (zero gradient
    everywhere). Gradient DFL cannot move them; PolyStep can."""
    mt = base_tree.tree
    alpha = base_tree.get_pruning_alpha()
    nodes = mt.tree

    def is_eff_leaf(n):
        return (alpha >= nodes[n].alpha_thresh) or bool(nodes[n].is_leaf)

    internal, node_var, node_thr0, left, right = [], {}, {}, {}, {}
    leaves = []
    stack = [0]
    while stack:
        n = stack.pop()
        if is_eff_leaf(n):
            leaves.append(n); continue
        internal.append(n)
        node_var[n] = int(nodes[n].split_var_ind)
        node_thr0[n] = float(nodes[n].split_val)
        left[n], right[n] = int(nodes[n].child_ind[0]), int(nodes[n].child_ind[1])
        stack.extend([left[n], right[n]])
    thr_pos = {n: i for i, n in enumerate(internal)}
    return dict(is_eff_leaf=is_eff_leaf, internal=internal, node_var=node_var,
                node_thr0=node_thr0, left=left, right=right, leaves=leaves,
                thr_pos=thr_pos)


def route(X, thr, st):
    """Assign each row of X to an effective-leaf node id, given threshold vector."""
    leaf = np.empty(X.shape[0], dtype=int)

    def rec(n, idx):
        if idx.size == 0:
            return
        if st["is_eff_leaf"](n):
            leaf[idx] = n; return
        v = thr[st["thr_pos"][n]]
        go_left = X[idx, st["node_var"][n]] < v
        rec(st["left"][n], idx[go_left])
        rec(st["right"][n], idx[~go_left])

    rec(0, np.arange(X.shape[0]))
    return leaf


def leaf_decisions(thr, st, tr_x, tr_cost, fallback):
    """Per-leaf decision from the TRAIN points routed to it under thresholds thr
    (decision = argmin path of the leaf's summed true cost; empty leaf -> fallback)."""
    la = route(tr_x, thr, st)
    leaves = st["leaves"]
    D = tr_cost.shape[1]
    Csum = np.zeros((len(leaves), D))
    nonempty = []
    for li, l in enumerate(leaves):
        m = la == l
        if m.any():
            Csum[li] = tr_cost[m].sum(0); nonempty.append(li)
    dec = {l: fallback[l] for l in leaves}
    if nonempty:
        w = find_opt_decision(Csum[nonempty])["weights"]
        for k, li in enumerate(nonempty):
            dec[leaves[li]] = w[k]
    return dec


def realized_routed(thr, st, x, cost, dec):
    la = route(x, thr, st)
    return np.array([cost[i] @ dec[la[i]] for i in range(x.shape[0])])


def refine_with_polystep(base_tree, tr_x, tr_cost, val_x, val_cost, D,
                         steps, n_particles, seed, min_obs=20):
    """Refine the (fixed-topology) tree's SPLIT THRESHOLDS with gradient-free
    PolyStep to minimize realized TRAIN cost; leaf decisions follow in closed
    form (in-leaf mean -> argmin via the authors' Gurobi solver, forward only).
    The authors' min_weights_per_node constraint is enforced as a penalty so
    refined splits stay valid (curbs boundary-hugging overfit). Select the best
    particle by realized VALIDATION cost. Returns structure + best thresholds."""
    st = extract_structure(base_tree)
    internal = st["internal"]
    K = len(internal)
    fallback = {n: base_tree.tree.tree[n].fitted_model.decision for n in st["leaves"]}
    if K == 0:                                   # degenerate: a stump, nothing to route
        return st, None, fallback
    thr0 = np.array([st["node_thr0"][n] for n in internal])
    leaves = st["leaves"]; L = len(leaves)
    ref = float(np.abs(tr_cost).sum())           # penalty scale ~ total train cost
    PEN = 5.0 * ref

    def train_eval(thr):
        """(realized train cost incl. validity penalty, leaf decisions)."""
        la = route(tr_x, thr, st)
        Csum = np.zeros((L, D)); cnt = np.zeros(L); nonempty = []
        for li, l in enumerate(leaves):
            m = la == l
            cnt[li] = m.sum()
            if m.any():
                Csum[li] = tr_cost[m].sum(0); nonempty.append(li)
        pen = PEN * int((cnt < min_obs).sum())
        return Csum, cnt, nonempty, pen

    class Obj:
        dim = K
        def __call__(self, X): return self.evaluate(X)
        def evaluate(self, X):
            orig = X.shape[:-1]
            Xf = X.reshape(-1, K).detach().cpu().numpy()
            P = Xf.shape[0]
            out = np.zeros(P)
            Csum_stack = np.zeros((P * L, D)); pens = np.zeros(P)
            for p in range(P):
                Csum, cnt, _, pen = train_eval(Xf[p])
                Csum_stack[p * L:(p + 1) * L] = Csum; pens[p] = pen
            w = find_opt_decision(Csum_stack)["weights"]
            realized = (w * Csum_stack).sum(1).reshape(P, L).sum(1) + pens
            return torch.tensor(realized, dtype=X.dtype).reshape(orig)

    sd = float(np.std(tr_x)) + 1e-6              # thresholds live in feature space
    solver = PolyStep.create(Obj(), dim=K, epsilon=0.25 * sd, step_radius=0.25 * sd,
                             probe_radius=0.5 * sd, num_probe=1,
                             max_iterations=steps, min_iterations=steps, compile=False)
    g = torch.Generator().manual_seed(seed)
    t0 = torch.tensor(thr0, dtype=torch.float32)
    X_init = t0.unsqueeze(0) + 0.2 * sd * torch.randn(n_particles, K, generator=g)
    X_init[0] = t0                                # exact warm start (the CART thresholds)
    state = solver.init_state(X_init)

    def val_cost_of(thr):
        Csum, cnt, nonempty, pen = train_eval(thr)
        if pen > 0:
            return float("inf")                  # reject invalid splits
        dec = dict(fallback)
        if nonempty:
            w = find_opt_decision(Csum[nonempty])["weights"]
            for k, li in enumerate(nonempty):
                dec[leaves[li]] = w[k]
        return float(realized_routed(thr, st, val_x, val_cost, dec).sum())

    best = (val_cost_of(thr0), thr0.copy())      # warm start is always a candidate
    for _ in range(steps):
        state = solver.step(state, generator=g)
        Xc = state.X.detach().cpu().numpy()
        for p in range(Xc.shape[0]):
            vc = val_cost_of(Xc[p])
            if vc < best[0]:
                best = (vc, Xc[p].copy())
    return st, best[1], fallback


def run(args):
    D = get_num_decisions()
    degs = [int(k) for k in args.degs.split("-")]
    data = load_data(args.n_train)
    valid_frac = 0.2
    rows = []
    for deg in degs:
        for rep in range(args.reps_st, args.reps_end):
            train_x, train_cost, test_x, test_cost = data[args.n_train][deg][args.eps][rep]
            n_valid = int(np.floor(args.n_train * valid_frac))
            val_x, val_cost = train_x[:n_valid], train_cost[:n_valid]
            tr_x, tr_cost = train_x[n_valid:], train_cost[n_valid:]

            opt = find_opt_decision(test_cost)["weights"]
            optc = np.array([np.sum(test_cost[i] * opt[i]) for i in range(opt.shape[0])])
            opt_sum = optc.sum()

            def nreg(realized):
                return float((realized.sum() - opt_sum) / opt_sum)

            spo_tree, t_spo = fit_tree(1.0, tr_x, tr_cost, val_x, val_cost,
                                       args.depth, args.min_obs)
            spo_real = realized_cost(spo_tree, test_x, test_cost)

            cart_tree, t_cart = fit_tree(0.0, tr_x, tr_cost, val_x, val_cost,
                                         args.depth, args.min_obs)
            cart_real = realized_cost(cart_tree, test_x, test_cost)

            t0 = time.time()
            st, thr_best, fallback = refine_with_polystep(
                cart_tree, tr_x, tr_cost, val_x, val_cost,
                D, args.steps, args.particles, args.seed + rep, min_obs=args.min_obs)
            if thr_best is None:                 # stump: no thresholds to refine
                ps_real = cart_real
            else:
                dec = leaf_decisions(thr_best, st, tr_x, tr_cost, fallback)
                ps_real = realized_routed(thr_best, st, test_x, test_cost, dec)
            t_ps = time.time() - t0
            L = len(st["leaves"])

            row = dict(deg=deg, rep=rep, n_leaves=int(L),
                       opt=float(optc.mean()),
                       SPOTree=nreg(spo_real), CART=nreg(cart_real),
                       PolyStep=nreg(ps_real),
                       t_spo=t_spo, t_cart=t_cart, t_ps=t_ps)
            rows.append(row)
            print(f"deg={deg} rep={rep} L={L} | SPOTree {row['SPOTree']:.4f}  "
                  f"CART {row['CART']:.4f}  PolyStep {row['PolyStep']:.4f}  "
                  f"(t_ps={t_ps:.1f}s)", flush=True)

    # aggregate
    summary = {}
    for deg in degs:
        dr = [r for r in rows if r["deg"] == deg]
        summary[str(deg)] = {m: dict(mean=float(np.mean([r[m] for r in dr])),
                                     std=float(np.std([r[m] for r in dr])))
                             for m in ("SPOTree", "CART", "PolyStep")}
    allr = rows
    summary["all"] = {m: dict(mean=float(np.mean([r[m] for r in allr])),
                              std=float(np.std([r[m] for r in allr])))
                      for m in ("SPOTree", "CART", "PolyStep")}

    out = dict(setting=dict(n_train=args.n_train, degs=degs, eps=args.eps,
                            max_depth=args.depth, min_weights_per_node=args.min_obs,
                            reps=[args.reps_st, args.reps_end], grid="4x4", D=D,
                            polystep_steps=args.steps, polystep_particles=args.particles),
               grad_dfl="N/A (zero gradient through tree's piecewise-constant argmin)",
               rows=rows, summary=summary)

    os.makedirs(os.path.join(HERE, "exp_results"), exist_ok=True)
    with open(os.path.join(HERE, "exp_results/spotree.json"), "w") as f:
        json.dump(out, f, indent=2)
    write_md(out)
    print("\nSUMMARY (normalized regret, lower=better):")
    for m in ("SPOTree", "CART", "PolyStep"):
        s = summary["all"][m]
        print(f"  {m:9s} {s['mean']:.4f} +/- {s['std']:.4f}")


def write_md(out):
    s = out["summary"]; cfg = out["setting"]
    L = []
    L.append("# SPO-Tree vs CART(two-stage) vs PolyStep-refined tree\n")
    L.append("Authors' code (Elmachtoub, Liang, McNellis, ICML 2020), 4x4-grid "
             "shortest path, authors' dataset. Non-differentiable predictor: "
             "**gradient DFL is N/A** (zero gradient through the tree's hard "
             "split routing + argmin); PolyStep refines the (fixed-topology) CART "
             "tree's SPLIT THRESHOLDS on realized regret, gradient-free, with the "
             "authors' min_weights_per_node constraint as a penalty. (Refining the "
             "leaf cost-vectors is a no-op: the leaf mean-cost decision already "
             "equals the in-leaf realized-cost optimum.)\n")
    L.append(f"Setting: n_train={cfg['n_train']}, deg={cfg['degs']}, eps={cfg['eps']}, "
             f"max_depth={cfg['max_depth']}, min_weights_per_node="
             f"{cfg['min_weights_per_node']}, reps={cfg['reps']}, "
             f"PolyStep(steps={cfg['polystep_steps']}, particles={cfg['polystep_particles']}).\n")
    L.append("Metric: normalized regret (realized-optimal)/optimal on 1000 test obs, "
             "lower is better.\n")
    L.append("| deg | SPOTree | CART(MSE/two-stage) | PolyStep(refined CART) |")
    L.append("|-----|---------|---------------------|------------------------|")
    for k in [d for d in s if d not in ("all",)]:
        r = s[k]
        L.append(f"| {k} | {r['SPOTree']['mean']:.4f} ± {r['SPOTree']['std']:.4f} | "
                 f"{r['CART']['mean']:.4f} ± {r['CART']['std']:.4f} | "
                 f"{r['PolyStep']['mean']:.4f} ± {r['PolyStep']['std']:.4f} |")
    r = s["all"]
    L.append(f"| **all** | **{r['SPOTree']['mean']:.4f}** | "
             f"**{r['CART']['mean']:.4f}** | **{r['PolyStep']['mean']:.4f}** |")
    with open(os.path.join(HERE, "exp_results/spotree.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--degs", type=str, default="2-10")
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--min_obs", type=int, default=20)
    ap.add_argument("--reps_st", type=int, default=0)
    ap.add_argument("--reps_end", type=int, default=10)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--particles", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args)
