"""
REAL Globe-TSP solver-embedding benchmark (Vlastelica/Paulus et al., "Differentiation of Blackbox
Combinatorial Solvers", ICLR 2020 -- the country-capitals / Table-3 analog), PolyStep edition.

Pipeline: k country-FLAG images -> a small CNN (one shared CNN applied per flag) -> a latent COORDINATE
per city -> induced Euclidean distance matrix [k,k] (a metric-TSP instance) -> TSP tour (Gurobi MIP with
lazy subtour-elimination = the paper's cutting-plane solver, embedded in the loop). The realized decision
cost is the sum of the TRUE geodesic distances along the PREDICTED tour; the optimum is the label tour's
true cost. Mirrors exp_warcraft.py: a "problem" object (TspGlobe) exposing predictor(), polystep_closure
(objective in {cost = realized true tour cost = EXPERIENCE/label-free, mse = regress to the true distance
matrix = PREDICTION, hamming = Hamming surrogate to the optimal-tour label = IMITATION}); train_two_stage_mb
(Adam MSE), train_blackbox_diff (vendored Vlastelica DBB perturb-and-resolve autograd.Function, Hamming
surrogate to the tour label, lam=20), train_ps_subspace (gradient-free PolyStep over a LinearSubspace, N=2q
probes/step; sr scales with sqrt(q)). Metric = tour-optimality / cost-match (predicted tour's TRUE cost ==
optimum) + normalized regret. Instrumented: wall_s / forward_solves / solver_calls / gurobi (exact count).

COORDS ROUTE (chosen): the CNN predicts a TSP_CDIM-D coordinate per city and the distance matrix is the
induced Euclidean (cdist) matrix -- i.e. we mirror the paper's capitals-coords->distances and guarantee a
metric instance. (The alternative, predicting the [k,k] matrix directly, is NOT used. TSP_CDIM defaults 3:
3-D + Euclidean = chordal distance on the globe, a faithful surrogate for the geodesic ground truth.)

DEVICE: defaults to CPU (TSP_DEV=cpu). Gurobi (the wall-clock bottleneck) runs on CPU regardless; the GPU
only accelerates the CNN forward, so set TSP_DEV=cuda for a FULL run *only when the GPU is free*.

Run (smoke, CPU only):  TSP_DEV=cpu TQDM_DISABLE=1 .venv/bin/python exp_tsp_embed.py smoke
Run (full, k=5 & k=10):           TSP_DEV=cuda .venv/bin/python exp_tsp_embed.py full
"""
import os, sys, time, json, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "polystep/src")
from polystep.epsilon import CosineEpsilon
from polystep import PolyStepOptimizer
from polystep.subspace import LinearSubspace
from polystep.transform import ParamLayout
from pto.forward import batched_predict

try:
    from gurobipy import GRB, Model, quicksum
except ImportError:
    print("GurobiPy missing, TSP module not available")

DEV = os.environ.get("TSP_DEV", "cpu")          # default CPU: safe + Gurobi (the bottleneck) is CPU anyway
if DEV == "cuda" and not torch.cuda.is_available():
    DEV = "cpu"
DDIR = os.environ.get("TSP_DDIR", "data/globe_tsp")
CDIM = int(os.environ.get("TSP_CDIM", "3"))     # latent coordinate dimension (3 = chordal-on-globe surrogate)
FLAG_H = int(os.environ.get("TSP_FLAGH", "20"))  # native Globe-TSP flag size is (20,40,3)
FLAG_W = int(os.environ.get("TSP_FLAGW", "40"))
P = lambda *a: print(*a, flush=True)

# ======================================================================================================
# VENDORED Gurobi TSP solver (from martius-lab/blackbox-backprop, travelling_salesman.py) -- the paper's
# own cutting-plane solver (lazy subtour elimination), plus a FIXED + modern-API DBB TspSolver.
# ======================================================================================================
_GUROBI_CALLS = [0]                              # exact instrumentation: every Gurobi MIP solve increments


def _subtour(n, edges):
    """Given selected edges, return the shortest subtour (for lazy subtour-elimination)."""
    visited = [False] * n
    cycles, lengths = [], []
    selected = [[] for _ in range(n)]
    for x, y in edges:
        selected[x].append(y)
    while True:
        current = visited.index(False)
        thiscycle = [current]
        while True:
            visited[current] = True
            neighbors = [x for x in selected[current] if not visited[x]]
            if len(neighbors) == 0:
                break
            current = neighbors[0]
            thiscycle.append(current)
        cycles.append(thiscycle)
        lengths.append(len(thiscycle))
        if sum(lengths) == n:
            break
    return cycles[lengths.index(min(lengths))]


def gurobi_tsp(distance_matrix):
    """Symmetric distance matrix -> {0,1} adjacency matrix of the optimal TSP tour (Gurobi MIP, lazy
    subtour-elimination cutting planes). Output is SYMMETRIC (both (i,j) and (j,i) set) -> 2 ones / edge."""
    _GUROBI_CALLS[0] += 1
    n = len(distance_matrix)
    m = Model()
    m.setParam("OutputFlag", False)
    m.setParam("Threads", 1)
    vars = {}
    for i in range(n):
        for j in range(i + 1):
            vars[i, j] = m.addVar(obj=0.0 if i == j else float(distance_matrix[i][j]),
                                  vtype=GRB.BINARY, name=f"e{i}_{j}")
            vars[j, i] = vars[i, j]
        m.update()
    for i in range(n):
        m.addConstr(quicksum(vars[i, j] for j in range(n)) == 2)
        vars[i, i].ub = 0
    m.update()
    m._vars = vars
    m.params.LazyConstraints = 1

    def subtourelim(model, where):
        if where == GRB.callback.MIPSOL:
            selected = []
            for i in range(n):
                sol = model.cbGetSolution([model._vars[i, j] for j in range(n)])
                selected += [(i, j) for j in range(n) if sol[j] > 0.5]
            tour = _subtour(n, selected)
            if len(tour) < n:
                expr = 0
                for a in range(len(tour)):
                    for b in range(a + 1, len(tour)):
                        expr += model._vars[tour[a], tour[b]]
                model.cbLazy(expr <= len(tour) - 1)

    m.optimize(subtourelim)
    solution = m.getAttr("x", vars)
    result = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if solution[i, j] > 0.5:
                result[i][j] = 1.0
    return result


class TspSolver(torch.autograd.Function):
    """Vlastelica DBB: forward = solve TSP; backward = implicit gradient via ONE perturbed re-solve at the
    incoming gradient (lam). FIXED vs upstream: forward now saves ctx.suggested_tours (upstream's backward
    asserted on it but forward never set it -> AssertionError). Modern static-method autograd API."""

    @staticmethod
    def forward(ctx, distance_matrices, lambda_val):
        ctx.distance_matrices = distance_matrices.detach().cpu().numpy()
        ctx.lambda_val = float(lambda_val)
        suggested = np.asarray([gurobi_tsp(d) for d in ctx.distance_matrices])
        ctx.suggested_tours = suggested                          # <-- THE BUG FIX (was missing upstream)
        return torch.from_numpy(suggested).float().to(distance_matrices.device)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.shape == ctx.suggested_tours.shape
        g = grad_output.detach().cpu().numpy()
        dist_prime = ctx.distance_matrices + ctx.lambda_val * g
        better = np.asarray([gurobi_tsp(d) for d in dist_prime])
        grad = -(ctx.suggested_tours - better) / ctx.lambda_val
        return torch.from_numpy(grad.astype(np.float32)).to(grad_output.device), None


def solve_tsp_batch(dist):
    """torch (M,k,k) distance matrices -> torch (M,k,k) {0,1} adjacency tours (loop Gurobi)."""
    arr = dist.detach().cpu().numpy()
    tours = [gurobi_tsp(arr[i]) for i in range(arr.shape[0])]
    return torch.from_numpy(np.asarray(tours)).to(dtype=dist.dtype, device=dist.device)


def tour_cost(adj, dist):
    """(M,k,k) adjacency x (M,k,k) distances -> (M,) realized tour cost. Robust to symmetric (2k ones) or
    directed (k ones) adjacency: a Hamiltonian tour has exactly k edges, so cost = sum * k / (#ones)."""
    s = (adj * dist).sum(dim=(-1, -2))
    e = adj.sum(dim=(-1, -2)).clamp(min=1)
    k = adj.shape[-1]
    return s * k / e


# ======================================================================================================
# Flag-CNN: one shared CNN per flag image -> CDIM-D city coordinate -> induced Euclidean distance matrix.
# GroupNorm (no running buffers) so it batches cleanly under PolyStep's functional_call probing. The 100
# flags live in a (frozen) buffer; the model input X is the (B,k) long index tensor of which flags/instance.
# ======================================================================================================
def _gn(ch): return nn.GroupNorm(min(8, ch), ch)


class FlagTSPNet(nn.Module):
    def __init__(self, flags, ch=32, cdim=CDIM):
        super().__init__()
        self.register_buffer("flags", flags)                     # (100,3,FLAG_H,FLAG_W), frozen input bank
        self.cnn = nn.Sequential(
            nn.Conv2d(3, ch, 3, stride=2, padding=1), _gn(ch), nn.ReLU(),     # H/2
            nn.Conv2d(ch, ch, 3, stride=2, padding=1), _gn(ch), nn.ReLU(),    # H/4
            nn.Conv2d(ch, ch, 3, padding=1), _gn(ch), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(ch, cdim)

    def forward(self, idx):                                       # idx (B,k) long -> (B,k,k) distances
        B, k = idx.shape
        f = self.flags[idx]                                      # (B,k,3,H,W)
        f = f.reshape(B * k, *f.shape[2:])
        h = self.cnn(f).flatten(1)                              # (B*k, ch)
        coords = self.head(h).reshape(B, k, -1)                 # (B,k,cdim)
        return torch.cdist(coords, coords)                      # (B,k,k) Euclidean, 0 diagonal, >=0


# ======================================================================================================
# Dataset loading (Globe-TSP). Per k-dir: countries.npy (100 (name, gps[lon,lat], flag (20,40,3) RGB));
# {train,val,test}_indices.npy [N,k] = which k flags per instance (the INPUT); {split}_distance_matrices.npy
# [N,k,k] = TRUE geodesic distances (LABEL/eval only, NOT fed to the model); {split}_tsp_tours.npy [N,k,k]
# = optimal-tour adjacency label (symmetric, 2 ones/edge). The dataset ships its own train/val/test splits.
# ======================================================================================================
def _load_flags(kdir):
    raw = np.load(os.path.join(kdir, "countries.npy"), allow_pickle=True)
    _names, _gps, flags = zip(*raw)                              # flags: 100 RGB images, native (20,40,3)
    out = []
    for fl in flags:
        a = np.asarray(fl).astype(np.float32) / 255.0           # (H,W,3)
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)
        if (t.shape[-2], t.shape[-1]) != (FLAG_H, FLAG_W):
            t = F.interpolate(t, size=(FLAG_H, FLAG_W), mode="bilinear", align_corners=False)
        out.append(t[0])
    return torch.stack(out).contiguous()                        # (100,3,FLAG_H,FLAG_W)


class TspGlobe:
    def __init__(self, k=5, n_train=10000, n_val=1000, n_test=1000, ch=32, mb=64):
        self.k, self.ch, self.mb = k, ch, mb
        kdir = os.path.join(DDIR, f"{k}_countries_from_100")
        self.flags = _load_flags(kdir).to(DEV)                  # (100,3,FLAG_H,FLAG_W)
        ns = {"train": n_train, "val": n_val, "test": n_test}

        def L(split, what, n):
            return np.load(os.path.join(kdir, f"{split}_{what}.npy"))[:n]
        self.idx = {s: torch.from_numpy(L(s, "indices", n).astype(np.int64)).to(DEV) for s, n in ns.items()}
        self.dist = {s: torch.from_numpy(L(s, "distance_matrices", n).astype(np.float32)).to(DEV)
                     for s, n in ns.items()}                    # TRUE distances (eval/label only)
        tr = {s: L(s, "tsp_tours", n).astype(np.float32) for s, n in ns.items()}
        self.tour = {s: torch.from_numpy(((t + t.transpose(0, 2, 1)) > 0).astype(np.float32)).to(DEV)
                     for s, t in tr.items()}                    # symmetrized tour-adjacency label
        # optimal cost per instance = label tour's TRUE cost (validated against the solver in validate_solver)
        self.zstar = {s: tour_cost(self.tour[s], self.dist[s]) for s in ns}

    def predictor(self):
        return FlagTSPNet(self.flags, ch=self.ch, cdim=CDIM).to(DEV)

    def mse_pairs(self, split):
        return self.idx[split], self.dist[split]

    def polystep_closure(self, model, split="train", objective="cost"):
        """objective='cost'    -> minimize realized TRUE tour cost of the predicted tour (label-free EXPERIENCE).
           objective='mse'     -> minimize normalized MSE(pred dist matrix, TRUE dist matrix) (PREDICTION; NO
                                  solver in loop) == two-stage's loss but gradient-free (Adam->PolyStep ablation).
           objective='hamming' -> minimize <w, 1-2w*> Hamming surrogate between solver tour w and the optimal
                                  tour label w* (IMITATION; identical to blackbox-diff's loss) gradient-free."""
        IDX, DT, ZS, TM = self.idx[split], self.dist[split], self.zstar[split], self.tour[split]
        nfull = IDX.shape[0]; mb = min(self.mb, nfull); k = self.k
        g = torch.Generator(device=DEV).manual_seed(0)

        def closure(bp):
            if mb < nfull:
                b = torch.randint(0, nfull, (mb,), generator=g, device=DEV)
                idx, dt, zs, tm = IDX[b], DT[b], ZS[b], TM[b]
            else:
                idx, dt, zs, tm = IDX, DT, ZS, TM
            pd = batched_predict(model, bp, idx)                # (N,mb,k,k) predicted distance matrices
            Nn, B = pd.shape[0], pd.shape[1]
            if objective == "mse":                              # prediction-focused, NO solver
                return ((pd - dt.unsqueeze(0)) ** 2).flatten(1).mean(1) / (dt ** 2).mean()
            adj = solve_tsp_batch(pd.reshape(Nn * B, k, k)).reshape(Nn, B, k, k)
            if objective == "hamming":                          # decision-focused imitation, gradient-free
                ham = (adj * (1 - 2 * tm.unsqueeze(0))).sum(dim=(-1, -2))   # (N,B)
                return ham.sum(-1) / tm.sum()
            realized = tour_cost(adj.reshape(Nn * B, k, k), dt.unsqueeze(0).expand(Nn, B, k, k).reshape(Nn * B, k, k))
            return realized.reshape(Nn, B).sum(-1) / zs.sum()   # normalized realized cost (minimize)
        return closure

    @torch.no_grad()
    def fast_regret(self, model, split="val"):
        model.eval()
        adj = solve_tsp_batch(model(self.idx[split]))
        realized = tour_cost(adj, self.dist[split])
        return ((realized - self.zstar[split]).sum() / self.zstar[split].sum()).item()
    regret = fast_regret

    @torch.no_grad()
    def cost_match(self, model, split="test"):
        model.eval()
        adj = solve_tsp_batch(model(self.idx[split]))
        realized = tour_cost(adj, self.dist[split])
        return (realized <= self.zstar[split] * (1 + 1e-4) + 1e-6).float().mean().item()


def validate_solver(prob, n=20, split="test"):
    """The vendored gurobi_tsp on the TRUE distance matrices must reproduce the dataset tour labels' cost
    (Gurobi optimum <= label cost, and the label IS optimal -> equal). Validates the solver + the schema."""
    dt = prob.dist[split][:n]; lab = prob.tour[split][:n]
    ours = tour_cost(solve_tsp_batch(dt), dt)
    data = tour_cost(lab, dt)
    gap = (ours - data)
    match = (torch.abs(ours - data) <= 1e-3 * data + 1e-4).float().mean().item()
    P(f"[solver check] n={n} | mean ours={ours.mean():.4f} label={data.mean():.4f} | "
      f"ours<=label on {(ours <= data + 1e-3).float().mean()*100:.0f}% | exact-match {match*100:.0f}% | "
      f"max|gap|={gap.abs().max():.4f}")
    return match


# ======================================================================================================
# Methods
# ======================================================================================================
def train_two_stage_mb(prob, epochs=30, lr=1e-3, mb=256, seed=0, verbose=False):
    """Two-stage (Adam, MSE of the induced Euclidean distance matrix to the TRUE distance matrix)."""
    torch.manual_seed(seed)
    model = prob.predictor(); model.train()
    opt = torch.optim.Adam(model.parameters(), lr)
    IDX, DT = prob.mse_pairs("train"); n = IDX.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, mb):
            b = perm[i:i + mb]
            loss = ((model(IDX[b]) - DT[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            P(f"  [ts ep {ep:>3}] mse {loss.item():.4f}  val cost-match {prob.cost_match(model,'val')*100:5.1f}%")
            model.train()
    return model


def train_blackbox_diff(prob, lam=20.0, epochs=30, lr=1e-3, mb=128, seed=0, verbose=False):
    """Vlastelica blackbox-diff (the paper's own method): solver-in-the-loop DBB autograd + Hamming surrogate
    <w, 1-2w*> to the TRUE optimal tour label. The fixed vendored TspSolver supplies the implicit gradient."""
    torch.manual_seed(seed)
    model = prob.predictor(); model.train()
    opt = torch.optim.Adam(model.parameters(), lr)
    IDX, TM = prob.idx["train"], prob.tour["train"]; n = IDX.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV); tot = 0.0
        for i in range(0, n, mb):
            b = perm[i:i + mb]
            dist = model(IDX[b])
            w = TspSolver.apply(dist, lam)
            loss = (w * (1 - 2 * TM[b])).sum(dim=(-1, -2)).mean()   # <w, 1-2w*> Hamming surrogate
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            P(f"  [bb ep {ep:>3}] loss {tot/max(1,n//mb):+.3f}  val cost-match {prob.cost_match(model,'val')*100:5.1f}%")
            model.train()
    return model


def train_ps_subspace(prob, warm, q=64, steps=120, sr=8.0, eps0=4.0, seed=0, objective="cost", verbose=True):
    """Gradient-free PolyStep over a TRUE small-q LinearSubspace (N=2q probes/step via max_subspace_dim=q).
    Mirrors exp_warcraft.train_ps_subspace. objective in {cost, mse, hamming} -> see polystep_closure."""
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
    model.eval()
    best = (float("inf"), None)
    for s in range(steps):
        pso.step(closure)
        if s % 10 == 0 or s == steps - 1:
            rv = prob.fast_regret(model, "val")
            if rv < best[0]:
                best = (rv, {k: v.detach().clone() for k, v in model.state_dict().items()})
            if verbose and (s % 40 == 0 or s == steps - 1):
                P(f"  [ps {objective} step {s:>4}] val cost-match {prob.cost_match(model,'val')*100:5.1f}%  "
                  f"regret {rv:.4f}  best {best[0]:.4f}")
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, int(sub.subspace_dim)


# ======================================================================================================
# Runner
# ======================================================================================================
def run_k(prob, k, smoke, ep_ts, ep_bb, q, steps, mb):
    P(f"\n=== Globe-TSP k={k} | {'SMOKE' if smoke else 'FULL'} | n_train={prob.idx['train'].shape[0]} "
      f"mb={mb} cdim={CDIM} q={q} steps={steps} ===")
    D = sum(p.numel() for p in prob.predictor().parameters())
    P(f"FlagTSPNet live params D={D:,} | flags {tuple(prob.flags.shape)}")
    ok = validate_solver(prob)
    if ok < 0.99:
        P("WARNING: vendored solver does not reproduce dataset optima")

    def stamp():
        return _GUROBI_CALLS[0]

    res = {}
    # two-stage (Adam MSE to true distance matrix)
    g0 = stamp(); _t = time.time()
    m_ts = train_two_stage_mb(prob, epochs=ep_ts, mb=mb, verbose=True); t_ts = time.time() - _t
    r_ts, cm_ts = prob.regret(m_ts, "test"), prob.cost_match(m_ts, "test")
    res["two_stage"] = dict(regret=r_ts, cost_match=cm_ts, wall_s=t_ts, gurobi=stamp() - g0)
    P(f"two-stage   : cost-match {cm_ts*100:5.1f}%  regret {r_ts:.4f}  ({t_ts:.0f}s)")
    # blackbox-diff (paper's method: DBB + Hamming to tour label)
    g0 = stamp(); _t = time.time()
    m_bb = train_blackbox_diff(prob, lam=20.0, epochs=ep_bb, mb=mb, verbose=True); t_bb = time.time() - _t
    r_bb, cm_bb = prob.regret(m_bb, "test"), prob.cost_match(m_bb, "test")
    res["blackbox_diff"] = dict(regret=r_bb, cost_match=cm_bb, wall_s=t_bb, gurobi=stamp() - g0)
    P(f"blackbox-diff: cost-match {cm_bb*100:5.1f}%  regret {r_bb:.4f}  ({t_bb:.0f}s)")
    # PolyStep gradient-free, objective=cost (label-free) and objective=hamming (imitation), warm from two-stage
    sr_tsp = float(os.environ.get("TSP_SR", "1.0")) * (q / 64.0) ** 0.5   # small: warm-start REFINE on the violent tour landscape (sr=8 detonated it)
    for obj in ["cost", "hamming"]:
        g0 = stamp(); _t = time.time()
        m_ps, sd = train_ps_subspace(prob, warm=m_ts, q=q, steps=steps, sr=sr_tsp,
                                     seed=0, objective=obj); t_ps = time.time() - _t
        r_ps, cm_ps = prob.regret(m_ps, "test"), prob.cost_match(m_ps, "test")
        solves = stamp() - g0
        res[f"polystep_{obj}"] = dict(regret=r_ps, cost_match=cm_ps, wall_s=t_ps, subspace_dim=sd, step_radius=round(sr_tsp, 3),
                                      forward_solves=solves, solver_calls=solves, gurobi=solves)
        P(f"PolyStep-{obj:<7}: cost-match {cm_ps*100:5.1f}%  regret {r_ps:.4f}  ({t_ps:.0f}s, sd={sd}, {solves:,} solves)")

    P(f"\n--- Globe-TSP k={k} RESULTS (test cost-match = predicted tour's TRUE cost == optimum) ---")
    P(f"{'method':>14} | {'regret':>8} {'cost-match':>10} | {'wall_s':>7} {'gurobi':>10}")
    name = dict(two_stage="two-stage", blackbox_diff="blackbox-diff",
                polystep_cost="PolyStep-cost", polystep_hamming="PolyStep-ham")
    for kk in ["two_stage", "blackbox_diff", "polystep_cost", "polystep_hamming"]:
        d = res[kk]
        P(f"{name[kk]:>14} | {d['regret']:>8.4f} {d['cost_match']*100:>9.1f}% | "
          f"{d['wall_s']:>6.0f}s {d['gurobi']:>10,}")
    return dict(k=k, D=D, n_train=prob.idx['train'].shape[0], cdim=CDIM, q=q, steps=steps, results=res)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    smoke = (mode != "full")
    os.makedirs("exp_results", exist_ok=True)
    t0 = time.time()
    if smoke:
        ks, splits = [5], dict(n_train=50, n_val=30, n_test=30)
        ep_ts, ep_bb, q, steps, mb = 3, 3, 16, 6, 8
    else:
        ks = [int(x) for x in os.environ.get("TSP_KS", "5,10").split(",")]
        splits = dict(n_train=int(os.environ.get("TSP_NTRAIN", "10000")),
                      n_val=int(os.environ.get("TSP_NVAL", "1000")),
                      n_test=int(os.environ.get("TSP_NTEST", "1000")))
        ep_ts = ep_bb = int(os.environ.get("TSP_EP", "40"))
        q = int(os.environ.get("TSP_Q", "64"))                 # subspace cap (note: pinned to rank-4 floor sd~295)
        steps = int(os.environ.get("TSP_STEPS", "120"))
        mb = int(os.environ.get("TSP_MB", "32"))
    P(f"=== Globe-TSP solver-embedding DFL | mode={mode} | DEV={DEV} | ks={ks} ===")
    out = {"mode": mode, "dev": DEV, "cdim": CDIM, "runs": []}
    for k in ks:
        prob = TspGlobe(k=k, ch=32, mb=mb, **splits)
        out["runs"].append(run_k(prob, k, smoke, ep_ts, ep_bb, q, steps, mb))
        json.dump(out, open(f"exp_results/tsp_embed_{mode}.json", "w"), indent=1)
    P(f"\n[total {time.time()-t0:.0f}s, {_GUROBI_CALLS[0]:,} Gurobi solves] -> exp_results/tsp_embed_{mode}.json")


if __name__ == "__main__":
    main()
