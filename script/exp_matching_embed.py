"""
REAL min-cost PERFECT-MATCHING solver-embedding benchmark (Vlastelica/Pogancic et al., "Differentiation
of Blackbox Combinatorial Solvers", ICLR 2020; the MNIST 4x4 perfect-matching task, Table-4 analog),
PolyStep edition. Mirrors exp_warcraft.py.

Pipeline:  4x4 grid of MNIST digits (1x112x112 image)  ->  CNN  ->  16 per-VERTEX costs  ->  per-EDGE
weight (function of the two endpoint costs)  ->  min-cost PERFECT MATCHING (Blossom V, Kolmogorov 2009)
->  realized matching cost on the TRUE edge weights.  PolyStep trains the CNN GRADIENT-FREE on the
realized decision cost (solver-in-the-loop, no backprop through the solver) via subspace probing.
Metric = cost-match (predicted matching's TRUE cost == optimum) + regret. Baselines: two-stage (Adam MSE
to the true vertex digit-values) and blackbox-diff (Vlastelica's own perturb-and-resolve autograd.Function,
Hamming loss to the optimal-matching label = IMITATION). Everything instrumented: wall / #forward-solves.

PAPER COST.  In the paper each grid cell is an MNIST digit and an edge's TRUE weight is the two digits
"read as a two-digit number". We reproduce that exactly: TRUE edge weight w(u,v) = 10*d_lo + d_hi where
d_lo,d_hi are the digit values of the two endpoints in canonical (row-major) vertex order. NOTE: the naive
"edge weight = SUM of the two endpoint costs" is DEGENERATE for perfect matching (every perfect matching
covers all 16 vertices exactly once, so the total = sum of all vertex costs = constant -> nothing to
optimize). We therefore use the paper's asymmetric two-digit reading as the endpoint-cost combiner; it is
still a function of the two endpoint vertex costs, but non-degenerate. Set EDGE_COMBINE=sum to see the
degenerate variant.

DATA.  The notebook's Edmond imeji URL (item HrfrAxcoQ049qk4K) is dead (Edmond migrated imeji->Dataverse);
the migrated file (doi:10.17617/3.YJCQ5S, mnist_matching.tar.gz, file id 102057) is 4.9 GB and ships only
images + matching labels (NO per-digit values, which the realized-cost / regret metric needs). So we
generate the SAME paper-defined process from a local MNIST cache (true digits -> true edge weights ->
Blossom-V optimal matching labels). Schema produced is identical in spirit: full_images [N,1,112,112],
perfect_matching [N,24] 0/1, on edges_from_grid(4,'4-grid').

Run:  TQDM_DISABLE=1 .venv/bin/python exp_matching_embed.py [smoke|full]
        smoke -> CPU, tiny (no GPU; safe while a GPU job runs).  full -> GPU CNN + CPU Blossom solves.
"""
import os, sys, time, json, itertools, numpy as np, torch, torch.nn as nn

sys.path.insert(0, "polystep/src")
from polystep.epsilon import CosineEpsilon
from polystep import PolyStepOptimizer
from polystep.subspace import LinearSubspace
from polystep.transform import ParamLayout
from pto.forward import batched_predict

P = lambda *a: print(*a, flush=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"          # overridden to "cpu" for smoke (see run())
MNIST_ROOT = os.environ.get("MNIST_ROOT", "/media/anindex/Data/UniSSH/side-stuff/data")
GRID = 4                                                       # 4x4 grid -> 16 vertices
CELL = 28                                                     # MNIST cell size -> image 112x112
EDGE_COMBINE = os.environ.get("EDGE_COMBINE", "read2")        # "read2" (paper, non-degenerate) | "sum" (degenerate)

# --------------------------------------------------------------------------------------------------
# vendored Blossom V min-cost perfect matching (martius-lab/blackbox-backprop, perfect_matching.py)
# built from gitlab.tuebingen.mpg.de/mrolinek/blossom_python and persisted into data/mnist_matching/.
# --------------------------------------------------------------------------------------------------
_BLOSSOM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/mnist_matching/blossom_build")
SOLVER_BACKEND = "?"
try:
    if os.path.isdir(_BLOSSOM_DIR):
        sys.path.insert(0, _BLOSSOM_DIR)
    import blossom_v                                          # noqa: E402
    SOLVER_BACKEND = "blossom_v"
except Exception as _e:                                       # FALL BACK to networkx exact matcher (note in report)
    blossom_v = None
    SOLVER_BACKEND = "networkx"
    P(f"[solver] blossom_v unavailable ({type(_e).__name__}: {_e}); falling back to networkx exact matcher")


def _pm_blossom(edge_weights, num_vertices, edges):
    """Blossom V min-cost perfect matching -> 0/1 edge indicator (vendored from blackbox-backprop)."""
    edges = tuple(map(tuple, edges))
    pm = blossom_v.PerfectMatching(num_vertices, len(edges))
    for (v1, v2), w in zip(edges, edge_weights):
        pm.AddEdge(int(v1), int(v2), float(w))
    pm.Solve()
    edge_to_index = dict(zip(edges, itertools.count()))
    matched = [(v, pm.GetMatch(v)) for v in range(num_vertices) if v < pm.GetMatch(v)]
    sol = np.zeros(len(edges), dtype=np.float32)
    sol[[edge_to_index[e] for e in matched]] = 1.0
    return sol


def _pm_networkx(edge_weights, num_vertices, edges):
    """No-build exact fallback: min-cost perfect matching == max-weight perfect matching with weight=BIG-cost.
    maxcardinality=True first forces a perfect matching, then maximizes sum(BIG-cost) == minimizes sum(cost)."""
    import networkx as nx
    edge_weights = np.asarray(edge_weights, dtype=np.float64)
    BIG = float(np.abs(edge_weights).sum()) + 1.0             # > any achievable total cost -> cardinality dominates
    G = nx.Graph(); G.add_nodes_from(range(num_vertices))
    for (u, v), w in zip(edges, edge_weights):
        G.add_edge(int(u), int(v), weight=BIG - float(w))
    M = nx.max_weight_matching(G, maxcardinality=True)
    Mset = set(M) | {(b, a) for (a, b) in M}
    sol = np.zeros(len(edges), dtype=np.float32)
    for i, (u, v) in enumerate(edges):
        if (int(u), int(v)) in Mset:
            sol[i] = 1.0
    return sol


def min_cost_perfect_matching(edge_weights, num_vertices, edges):
    return (_pm_blossom if SOLVER_BACKEND == "blossom_v" else _pm_networkx)(edge_weights, num_vertices, edges)


# Vendored DBB autograd.Function (perfect_matching.py: MinCostPerfectMatchingSolver) -- correctly saves ctx.
# Modernized: passes edges/num_vertices through, batched serial solve (ray absent -> serial list-comp).
class _BlackboxPM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, lambda_val, num_vertices, edges):
        ctx.weights = weights.detach().cpu().numpy()
        ctx.lambda_val = float(lambda_val)
        ctx.num_vertices = num_vertices
        ctx.edges = edges
        ctx.sol = np.stack([min_cost_perfect_matching(w, num_vertices, edges) for w in ctx.weights])
        return torch.from_numpy(ctx.sol).float().to(weights.device)

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        g = grad_output.detach().cpu().numpy()
        wprime = np.maximum(ctx.weights + ctx.lambda_val * g, 0.0)         # perturb-and-resolve (Vlastelica)
        better = np.stack([min_cost_perfect_matching(w, ctx.num_vertices, ctx.edges) for w in wprime])
        grad = -(ctx.sol - better) / ctx.lambda_val
        return torch.from_numpy(grad).float().to(dev), None, None, None


# --------------------------------------------------------------------------------------------------
# graph: edges_from_grid(4, '4-grid') (verbatim from blackbox-differentiation .../data/utils.py)
# --------------------------------------------------------------------------------------------------
def _vidx(v, dim): x, y = v; return x * dim + y
def _nbr4(x, y, xm, ym):
    for dx, dy in [(1, 0), (0, 1), (0, -1), (-1, 0)]:
        xn, yn = x + dx, y + dy
        if 0 <= xn < xm and 0 <= yn < ym:
            yield xn, yn
def edges_from_grid(N):
    out = []
    for x, y in itertools.product(range(N), range(N)):
        out += [(x, y, *vn) for vn in _nbr4(x, y, N, N) if _vidx((x, y), N) < _vidx(vn, N)]
    flat = sorted(set(out))
    return np.asarray([(_vidx((a, b), N), _vidx((c, d), N)) for a, b, c, d in flat], dtype=np.int64)

EDGES = edges_from_grid(GRID)                                 # (E,2) vertex-index pairs, u<v ; E=24 for 4x4
EDGE_PAIRS = [(int(u), int(v)) for u, v in EDGES]
NV = GRID * GRID                                             # 16 vertices
E = len(EDGES)


def _edge_weights(cv, U, V):
    """per-vertex costs cv (...,16) -> per-EDGE weights (...,E). read2 = paper two-digit reading (lo=tens)."""
    lo = cv[..., U]; hi = cv[..., V]                          # U=min endpoint index (lo), V=max (hi)
    return (10.0 * lo + hi) if EDGE_COMBINE == "read2" else (lo + hi)


def solve_pm_batch(weights):
    """weights (M,E) tensor/ndarray -> (M,E) float32 ndarray of 0/1 matching indicators (CPU Blossom solves)."""
    w = weights.detach().cpu().numpy() if torch.is_tensor(weights) else np.asarray(weights)
    return np.stack([min_cost_perfect_matching(row, NV, EDGE_PAIRS) for row in w]).astype(np.float32)


# --------------------------------------------------------------------------------------------------
# dataset: paper-defined MNIST 4x4 perfect matching, generated from a local MNIST cache
# --------------------------------------------------------------------------------------------------
def _load_mnist():
    import torchvision
    for train in (True, False):
        try:
            ds = torchvision.datasets.MNIST(MNIST_ROOT, train=train, download=False)
            return ds.data.numpy().astype(np.float32), ds.targets.numpy().astype(np.int64)
        except Exception:
            continue
    ds = torchvision.datasets.MNIST("data/_mnist", train=True, download=True)   # last resort (small CPU dl)
    return ds.data.numpy().astype(np.float32), ds.targets.numpy().astype(np.int64)


def build_mnist_matching(n, seed):
    """-> dict(img [n,1,112,112] float, digits [n,16] int, true_w [n,E], label [n,E] 0/1, zstar [n])."""
    X, y = _load_mnist()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(y), size=(n, NV))                         # n*16 random MNIST cells
    digits = y[pick].astype(np.int64)                                   # (n,16) true vertex digit values
    img = np.zeros((n, 1, GRID * CELL, GRID * CELL), dtype=np.float32)
    for v in range(NV):
        x_, y_ = v // GRID, v % GRID
        img[:, 0, x_ * CELL:(x_ + 1) * CELL, y_ * CELL:(y_ + 1) * CELL] = X[pick[:, v]] / 255.0
    U, V = EDGES[:, 0], EDGES[:, 1]
    true_w = (10 * digits[:, U] + digits[:, V]).astype(np.float32) if EDGE_COMBINE == "read2" \
        else (digits[:, U] + digits[:, V]).astype(np.float32)            # (n,E) TRUE edge weights
    label = solve_pm_batch(true_w)                                      # (n,E) optimal matching (Blossom)
    zstar = (label * true_w).sum(1).astype(np.float32)                  # (n,) optimal cost
    return dict(img=img, digits=digits, true_w=true_w, label=label, zstar=zstar)


# --------------------------------------------------------------------------------------------------
# CNN: 1x112x112 image -> 16 per-vertex costs (GroupNorm -> batches cleanly under functional_call)
# --------------------------------------------------------------------------------------------------
def _gn(ch): return nn.GroupNorm(min(8, ch), ch)
class _Res(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1); self.b1 = _gn(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1); self.b2 = _gn(ch)
    def forward(self, x):
        h = torch.relu(self.b1(self.c1(x))); h = self.b2(self.c2(h)); return torch.relu(x + h)

class MatchCNN(nn.Module):
    """112x112 grid image -> per-vertex cost map adaptive-pooled to 4x4 -> 16 positive vertex costs."""
    def __init__(self, ch=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, ch, 7, stride=2, padding=3), _gn(ch), nn.ReLU(), nn.MaxPool2d(2))
        self.block = _Res(ch)
        self.head = nn.Conv2d(ch, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((GRID, GRID))
    def forward(self, x):                                                # x (B,1,112,112)
        h = self.pool(self.head(self.block(self.stem(x))))              # (B,1,4,4)
        return torch.nn.functional.softplus(h).reshape(x.shape[0], NV)  # (B,16) positive vertex costs


# --------------------------------------------------------------------------------------------------
# problem object (mirrors WarcraftSP / pto interface so train_* just work)
# --------------------------------------------------------------------------------------------------
class MatchingPM:
    spo_supported = False
    def __init__(self, n_train=2000, n_val=500, n_test=500, ch=32, mb=64, seed=0):
        self.ch = ch; self.mb = mb
        ns = {"train": (n_train, seed), "val": (n_val, seed + 1), "test": (n_test, seed + 2)}
        d = {k: build_mnist_matching(n, s) for k, (n, s) in ns.items()}
        T = lambda a: torch.tensor(a).to(DEV)
        self.img = {k: T(d[k]["img"]) for k in d}
        self.digit = {k: T(d[k]["digits"].astype(np.float32)) for k in d}        # true vertex costs (digits)
        self.true_w = {k: T(d[k]["true_w"]) for k in d}
        self.label = {k: T(d[k]["label"]) for k in d}
        self.zstar = {k: T(d[k]["zstar"]) for k in d}
        self.U = torch.tensor(EDGES[:, 0]).to(DEV); self.V = torch.tensor(EDGES[:, 1]).to(DEV)

    def predictor(self): return MatchCNN(self.ch).to(DEV)
    def mse_pairs(self, split): return self.img[split], self.digit[split]        # two-stage targets = true digits

    def polystep_closure(self, model, split="train", objective="cost"):
        """objective='cost'    -> minimize realized matching cost on TRUE weights (label-free; EXPERIENCE).
           objective='mse'     -> MSE(predicted vertex cost, TRUE digit value) == two-stage, GRADIENT-FREE, NO solver.
           objective='hamming' -> Hamming surrogate <w, 1-2w*> to the optimal-matching LABEL (=blackbox-diff's
           loss) optimized GRADIENT-FREE (no DBB trick) -> the MNIST-style imitation / decision loss."""
        IMG, DIG, TW, LAB, ZS = (self.img[split], self.digit[split], self.true_w[split],
                                 self.label[split], self.zstar[split])
        nfull = IMG.shape[0]; mb = min(self.mb, nfull)
        g = torch.Generator(device=DEV).manual_seed(0)
        def closure(bp):
            if mb < nfull:
                idx = torch.randint(0, nfull, (mb,), generator=g, device=DEV)
                img, dig, tw, lab, zs = IMG[idx], DIG[idx], TW[idx], LAB[idx], ZS[idx]
            else:
                img, dig, tw, lab, zs = IMG, DIG, TW, LAB, ZS
            cv = batched_predict(model, bp, img)                          # (N,B,16) per-vertex costs
            N, B = cv.shape[0], cv.shape[1]
            if objective == "mse":                                        # prediction-focused, NO solver
                return ((cv - dig.unsqueeze(0)) ** 2).flatten(1).mean(1) / (dig ** 2).mean()
            ew = _edge_weights(cv, self.U, self.V)                        # (N,B,E)
            m = torch.from_numpy(solve_pm_batch(ew.reshape(N * B, E))).to(DEV).reshape(N, B, E)
            if objective == "hamming":                                    # decision-focused imitation, gradient-free
                ham = (m * (1 - 2 * lab.unsqueeze(0))).sum(-1)            # (N,B): <w,1-2w*>
                return ham.sum(-1) / lab.sum()
            realized = (m * tw.unsqueeze(0)).sum(-1)                      # (N,B) realized cost on TRUE weights
            return realized.sum(-1) / zs.sum()                           # normalized realized cost (minimize)
        return closure

    @torch.no_grad()
    def _solve_pred(self, model, split):
        cv = model(self.img[split])                                      # (B,16)
        ew = _edge_weights(cv, self.U, self.V)                           # (B,E)
        m = torch.from_numpy(solve_pm_batch(ew)).to(DEV)
        return m, (m * self.true_w[split]).sum(1)                        # realized cost per instance

    @torch.no_grad()
    def fast_regret(self, model, split="val"):
        model.eval(); _, realized = self._solve_pred(model, split)
        return ((realized - self.zstar[split]).sum() / self.zstar[split].sum()).item()
    regret = fast_regret

    @torch.no_grad()
    def cost_match(self, model, split="test"):
        model.eval(); _, realized = self._solve_pred(model, split)
        return (realized <= self.zstar[split] + 1e-4).float().mean().item()   # predicted matching is also optimal

    @torch.no_grad()
    def match_acc(self, model, split="test"):
        model.eval(); m, _ = self._solve_pred(model, split)
        return (m == self.label[split]).all(1).float().mean().item()          # exact-matching agreement


def validate_solver(prob, k=64):
    """The vendored matcher must (a) reproduce the stored optimal-matching labels, and (b) agree with an
    independent exact reference (networkx) on the optimal COST -> confirms correctness of the embedded solver."""
    tw = prob.true_w["test"][:k]; lab = prob.label["test"][:k]; zs = prob.zstar["test"][:k]
    re_m = torch.from_numpy(solve_pm_batch(tw)).to(DEV)
    re_cost = (re_m * tw).sum(1)
    repro = (re_cost <= zs + 1e-4).float().mean().item()                 # re-solve cost == stored optimum
    # independent reference (networkx) on the SAME true weights
    twn = tw.detach().cpu().numpy()
    ref = np.stack([_pm_networkx(w, NV, EDGE_PAIRS) for w in twn])
    ref_cost = (torch.from_numpy(ref).to(DEV) * tw).sum(1)
    agree = (torch.abs(ref_cost - zs) <= 1e-4).float().mean().item()
    P(f"[solver check | backend={SOLVER_BACKEND}] re-solve==label-cost {repro*100:.1f}% | "
      f"networkx-ref==optimum {agree*100:.1f}% | mean opt cost {zs.mean():.2f}")
    return min(repro, agree)


# --------------------------------------------------------------------------------------------------
# trainers (mirror exp_warcraft.py)
# --------------------------------------------------------------------------------------------------
def train_two_stage_mb(prob, epochs=60, lr=1e-3, mb=64, seed=0):
    """two-stage / predict-then-optimize: Adam MSE of predicted vertex costs to the TRUE digit values."""
    torch.manual_seed(seed)
    model = prob.predictor(); model.train()
    opt = torch.optim.Adam(model.parameters(), lr)
    IMG, DIG = prob.img["train"], prob.digit["train"]; n = IMG.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, mb):
            idx = perm[i:i + mb]
            loss = ((model(IMG[idx]) - DIG[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_blackbox_diff(prob, lam=20.0, epochs=60, lr=1e-3, mb=64, seed=0, verbose=False):
    """Vlastelica blackbox-diff: Blossom-V-in-the-loop + perturb-and-resolve implicit gradient, supervised
    Hamming loss <w, 1-2w*> to the optimal-matching label (IMITATION). The paper's own method."""
    torch.manual_seed(seed)
    model = prob.predictor(); model.train()
    opt = torch.optim.Adam(model.parameters(), lr)
    IMG, LAB = prob.img["train"], prob.label["train"]; n = IMG.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV); tot = 0.0
        for i in range(0, n, mb):
            idx = perm[i:i + mb]
            cv = model(IMG[idx])
            ew = _edge_weights(cv, prob.U, prob.V)                       # (B,E) predicted edge weights
            w = _BlackboxPM.apply(ew, lam, NV, EDGE_PAIRS)
            loss = (w * (1 - 2 * LAB[idx])).sum(1).mean()               # <w, 1-2w*> Hamming surrogate
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if verbose and (ep % max(1, epochs // 6) == 0 or ep == epochs - 1):
            P(f"  [bb ep {ep:>3}] loss {tot/max(1,n//mb):+.3f}  val cost-match {prob.cost_match(model,'val')*100:5.1f}%")
            model.train()
    return model


def train_ps_subspace(prob, warm, q=128, steps=120, sr=11.0, eps0=4.0, seed=0, objective="cost"):
    """gradient-free PolyStep over a true small-q LinearSubspace (N=2*subspace_dim probes/step). Mirrors
    exp_warcraft.train_ps_subspace. objective in {cost (label-free), mse (prediction), hamming (imitation)}."""
    model = prob.predictor()
    if warm is not None:
        model.load_state_dict(warm.state_dict())
    layout = ParamLayout.from_module(model)
    sub = LinearSubspace.from_layout(layout, rank=4, max_subspace_dim=q, seed=seed)
    cfg = dict(polytope_type="orthoplex", subspace_particle_dim=8, num_probe=1, use_momentum=True,
               momentum_init=0.5, momentum_final=0.9, epsilon=CosineEpsilon(eps0, eps0 / 10),
               step_radius=sr, probe_radius=2.0)
    pso = PolyStepOptimizer(model, subspace=sub, seed=seed, **cfg)
    closure = prob.polystep_closure(model, "train", objective=objective)
    model.eval()                                                        # freeze (GN has no buffers; deterministic)
    best = (float("inf"), None)
    for s in range(steps):
        pso.step(closure)
        if s % 10 == 0 or s == steps - 1:
            rv = prob.fast_regret(model, "val")
            if rv < best[0]:
                best = (rv, {k: v.detach().clone() for k, v in model.state_dict().items()})
            if s % 40 == 0 or s == steps - 1:
                P(f"  [ps {objective} step {s:>4}] val cost-match {prob.cost_match(model,'val')*100:5.1f}% "
                  f"regret {rv:.4f} best {best[0]:.4f}")
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, int(sub.subspace_dim)


# --------------------------------------------------------------------------------------------------
def run(smoke):
    """Head-to-head: two-stage (MSE) vs blackbox-diff (Hamming, paper) vs PolyStep-cost (label-free) vs
    PolyStep-hamming (imitation, gradient-free). Smoke = CPU/tiny; full = GPU CNN + CPU Blossom solves."""
    global DEV
    if smoke:
        DEV = "cpu"                                                      # smoke MUST stay off the (busy) GPU
    nt, nval, ntest = (50, 50, 80) if smoke else (5000, 1000, 1000)
    ch = 16 if smoke else int(os.environ.get("PM_CH", "32"))
    mb = 16 if smoke else int(os.environ.get("PM_MB", "64"))
    q = 64 if smoke else int(os.environ.get("PM_Q", "512"))
    steps = 12 if smoke else int(os.environ.get("PM_STEPS", "300"))
    ep = 3 if smoke else int(os.environ.get("PM_EP", "60"))
    sr = float(os.environ.get("PM_SR", "11.0")) * (q / 128.0) ** 0.5    # scale step_radius with subspace size
    P(f"=== MNIST 4x4 perfect-matching solver-embedding | {'SMOKE(CPU)' if smoke else 'FULL'} | DEV={DEV} "
      f"| n_train={nt} mb={mb} ch={ch} q={q} steps={steps} combine={EDGE_COMBINE} backend={SOLVER_BACKEND} ===")
    t0 = time.time()
    prob = MatchingPM(n_train=nt, n_val=nval, n_test=ntest, ch=ch, mb=mb)
    D = sum(p.numel() for p in prob.predictor().parameters())
    P(f"MatchCNN live params D={D:,} | E={E} edges, NV={NV} vertices | load {time.time()-t0:.0f}s")
    ok = validate_solver(prob)
    if ok < 0.99: P("WARNING: solver did not reproduce optima / disagrees with reference")

    res = {}
    def report(tag, m, t, solves=0):
        cm, rg, ma = prob.cost_match(m), prob.regret(m), prob.match_acc(m)
        res[tag] = dict(cost_match=cm, regret=rg, match_acc=ma, wall_s=t, forward_solves=solves, gurobi=0)
        P(f"{tag:>16} | cost-match {cm*100:5.1f}% regret {rg:+.4f} match-acc {ma*100:5.1f}% | "
          f"{t:6.1f}s solves={solves:,}")
        return m

    _t = time.time(); m_ts = train_two_stage_mb(prob, epochs=ep, mb=mb); report("two-stage(MSE)", m_ts, time.time()-_t)
    _t = time.time(); m_bb = train_blackbox_diff(prob, lam=20.0, epochs=ep, mb=mb, verbose=smoke)
    bb_solves = 2 * (nt // mb + 1) * ep * mb                              # ~2 solves(fwd+bwd) per sample per epoch
    report("blackbox-diff", m_bb, time.time()-_t, bb_solves)
    _t = time.time(); m_pc, sd = train_ps_subspace(prob, warm=m_ts, q=q, steps=steps, sr=sr, objective="cost")
    report("PolyStep-cost", m_pc, time.time()-_t, 2 * sd * min(prob.mb, nt) * steps)
    _t = time.time(); m_ph, sd = train_ps_subspace(prob, warm=m_ts, q=q, steps=steps, sr=sr, objective="hamming")
    report("PolyStep-ham", m_ph, time.time()-_t, 2 * sd * min(prob.mb, nt) * steps)

    out = dict(mode="smoke" if smoke else "full", combine=EDGE_COMBINE, backend=SOLVER_BACKEND, D=D,
               n_train=nt, n_test=ntest, ch=ch, mb=mb, q=q, subspace_dim=sd, steps=steps, results=res)
    os.makedirs("exp_results", exist_ok=True)
    jf = f"exp_results/matching_embed_{'smoke' if smoke else 'full'}.json"
    json.dump(out, open(jf, "w"), indent=1)
    P(f"\n(paper ref: blackbox-diff >> two-stage on MNIST PM; two-digit-reading cost is the non-degenerate paper cost)")
    P(f"[total {time.time()-t0:.0f}s] -> {jf}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    run(smoke=(mode != "full"))
