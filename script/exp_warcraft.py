"""
REAL Warcraft shortest-path benchmark (Vlastelica/Pogancic, ICLR 2020), PolyStep edition.

Pipeline: 96x96x3 Warcraft terrain image -> CNN -> 12x12 cell-cost grid -> 8-connected shortest path
(top-left -> bottom-right, vertex-weighted) -> realized path cost. PolyStep trains the CNN GRADIENT-FREE
on the realized decision cost (solver-in-the-loop, no backprop through the solver), via subspace probing
to handle the CNN's dimension. Metric = the paper's COST-MATCH accuracy (predicted path's TRUE cost ==
optimal). Baselines: two-stage (CNN regresses costs, MSE) and a no-solver direct-path head (reproduces
the ResNet-baseline failure). Everything instrumented: wall-clock / #forward-solves / #solver-calls.

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 .venv/bin/python exp_warcraft.py [smoke|full]
"""
import os, sys, time, json, math, numpy as np, torch, torch.nn as nn
import torchvision
sys.path.insert(0, "polystep/src")
from polystep.epsilon import CosineEpsilon
from polystep import PolyStepOptimizer
from polystep.subspace import LinearSubspace
from polystep.transform import ParamLayout
from pto import train_two_stage, train_polystep

# ---- polytope-robustness sweep knobs (mirror the cheap experiments; defaults keep the working config) ----
PS_POLYTOPE = os.environ.get("PS_POLYTOPE", "orthoplex")   # orthoplex | simplex | cube
PS_PROBES   = int(os.environ.get("PS_PROBES", "1"))
PS_PDIM     = int(os.environ.get("WC_PDIM", "8"))          # subspace particle dim (cube must use <=4: 2^pdim verts)
OUT_TAG     = os.environ.get("OUT_TAG", "")
_OUT_SFX    = f"_{OUT_TAG}" if OUT_TAG else ""


def train_ps_subspace(prob, warm, q=256, steps=200, sr=15.0, eps0=4.0, seed=0, objective="cost"):
    """PolyStep with a TRUE small-q subspace (N=2q probes/step) via LinearSubspace + max_subspace_dim=q.
    LinearSubspace needs a LARGER step_radius (its 1/sqrt(N) projection dilutes the full-space move).
    objective in {cost(label-free), mse(prediction-focused), hamming(decision-focused)} -> see polystep_closure."""
    model = prob.predictor()
    if warm is not None:
        model.load_state_dict(warm.state_dict())
    layout = ParamLayout.from_module(model)
    sub = LinearSubspace.from_layout(layout, rank=4, max_subspace_dim=q, seed=seed)
    cfg = dict(polytope_type=PS_POLYTOPE, subspace_particle_dim=PS_PDIM, num_probe=PS_PROBES, use_momentum=True,
               momentum_init=0.5, momentum_final=0.9, epsilon=CosineEpsilon(eps0, eps0 / 10),
               step_radius=sr, probe_radius=2.0)
    pso = PolyStepOptimizer(model, subspace=sub, seed=seed, **cfg)
    closure = prob.polystep_closure(model, "train", objective=objective)
    model.eval()    # freeze BN to the warm-start's running stats during functional_call probing (CombResNet)
    best = (float("inf"), None)
    for s in range(steps):
        pso.step(closure)
        if s % 10 == 0 or s == steps - 1:
            rv = prob.fast_regret(model, "val")
            if rv < best[0]:
                best = (rv, {k: v.detach().clone() for k, v in model.state_dict().items()})
            if s % 50 == 0 or s == steps - 1:
                P(f"  [step {s:>4}] val cost-match {prob.cost_match(model,'val')*100:5.1f}%  regret {rv:.4f}  best {best[0]:.4f}")
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, int(sub.subspace_dim)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DDIR = "data/warcraft/warcraft_shortest_path_oneskin/12x12"
P = lambda *a: print(*a, flush=True)

# ---- 8-connected, vertex-weighted shortest path: batched GPU Bellman-Ford + backtrack ----
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def _shift(d, dr, dc):
    """shifted[...,r,c] = d[...,r-dr,c-dc], padded with +inf (a neighbour stepping (dr,dc) INTO (r,c))."""
    B, H, W = d.shape
    out = torch.full_like(d, float("inf"))
    rs0, rs1 = max(0, dr), min(H, H + dr)        # dest rows that have a valid source
    cs0, cs1 = max(0, dc), min(W, W + dc)
    out[:, rs0:rs1, cs0:cs1] = d[:, rs0 - dr:rs1 - dr, cs0 - dc:cs1 - dc]
    return out

def solve_grid_sp(costs, iters=None):
    """costs (B,H,W) -> (path_mask (B,H,W){0,1}, path_cost (B,)). Cost of a path = sum of vertex costs
    on it (incl. both endpoints). Source=(0,0), sink=(H-1,W-1), 8-connected. Costs clamped > 0."""
    c = costs.clamp(min=1e-3)
    B, H, W = c.shape
    iters = iters or (H * W)
    INF = float("inf")
    dist = torch.full((B, H, W), INF, device=c.device); dist[:, 0, 0] = c[:, 0, 0]
    pred = torch.full((B, H, W), -1, dtype=torch.long, device=c.device)
    for _ in range(iters):
        best = c + _shift(dist, *OFFS[0]); arg = torch.zeros_like(pred)   # running min over 8 neighbours
        for k in range(1, 8):
            cand = c + _shift(dist, *OFFS[k])
            upd = cand < best
            best = torch.where(upd, cand, best)
            arg = torch.where(upd, torch.full_like(arg, k), arg)
        better = best < dist
        if not bool(better.any()):
            break
        dist = torch.where(better, best, dist)
        pred = torch.where(better, arg, pred)                  # arg in 0..7 = predecessor offset
    # backtrack from sink along predecessors
    mask = torch.zeros((B, H, W), device=c.device)
    r = torch.full((B,), H - 1, dtype=torch.long, device=c.device)
    cc = torch.full((B,), W - 1, dtype=torch.long, device=c.device)
    bidx = torch.arange(B, device=c.device)
    done = torch.zeros(B, dtype=torch.bool, device=c.device)
    offs_t = torch.tensor(OFFS, device=c.device)
    for _ in range(H * W + 2):
        mask[bidx, r, cc] = torch.where(done, mask[bidx, r, cc], torch.ones(B, device=c.device))
        at_src = (r == 0) & (cc == 0)
        done = done | at_src
        if bool(done.all()):
            break
        k = pred[bidx, r, cc].clamp(min=0)
        dr = offs_t[k, 0]; dc = offs_t[k, 1]
        r = torch.where(done, r, r - dr)
        cc = torch.where(done, cc, cc - dc)
    path_cost = (mask * c).sum(dim=(-1, -2))
    return mask, path_cost


# ---- CombResNet-style CNN: image -> 12x12 cost grid ----
def _gn(ch): return nn.GroupNorm(min(8, ch), ch)            # no running buffers -> batches cleanly

class _Res(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1); self.b1 = _gn(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1); self.b2 = _gn(ch)
    def forward(self, x):
        h = torch.relu(self.b1(self.c1(x))); h = self.b2(self.c2(h)); return torch.relu(x + h)

class CombResNet(nn.Module):
    """Truncated-ResNet-style stem + one residual block -> 1-channel cost map, adaptive-pooled to 12x12."""
    def __init__(self, grid=12, ch=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, ch, 7, stride=2, padding=3), _gn(ch),
                                  nn.ReLU(), nn.MaxPool2d(2))   # 96 -> 24
        self.block = _Res(ch)
        self.head = nn.Conv2d(ch, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
    def forward(self, x):                                       # x (B,3,96,96)
        h = self.pool(self.head(self.block(self.stem(x))))      # (B,1,12,12)
        return torch.nn.functional.softplus(h.squeeze(1))       # (B,12,12) positive costs


class CombResNet18(nn.Module):
    """The paper's EXACT Warcraft model (Vlastelica et al.): truncated ResNet18 (conv1->bn1->relu->maxpool
    ->layer1) -> AdaptiveMaxPool2d(grid) -> mean over channels. Dead layers (2,3,4,fc) stripped so the
    live param count (~150K) is what PolyStep's subspace covers. layer1 ends in ReLU -> outputs >=0."""
    def __init__(self, grid=12, in_ch=3):
        super().__init__()
        rn = torchvision.models.resnet18(weights=None)
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1, self.relu, self.maxpool, self.layer1 = rn.bn1, rn.relu, rn.maxpool, rn.layer1
        self.pool = nn.AdaptiveMaxPool2d((grid, grid))
    def forward(self, x):                                       # x (B,3,96,96)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)                                     # (B,64,24,24), >=0
        return self.pool(x).mean(dim=1)                        # (B,grid,grid) costs >=0


# ---- the "problem" object: mirrors pto TerrainSP so train_polystep / train_two_stage just work ----
class WarcraftSP:
    spo_supported = False
    def __init__(self, n_train=10000, n_test=1000, grid=12, ch=32, n_val=1000, mb=128, model_type="small"):
        def L(split, what): return np.load(f"{DDIR}/{split}_{what}.npy")
        def imgs(a): return torch.tensor(a[:, :, :, :3].astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()
        self.grid = grid; self.ch = ch; self.mb = mb; self.model_type = model_type
        ns = {"train": n_train, "val": n_val, "test": n_test}
        self.img = {k: imgs(L(k, "maps")[:n]).to(DEV) for k, n in ns.items()}
        self.cost = {k: torch.tensor(L(k, "vertex_weights")[:n].astype(np.float32)).to(DEV) for k, n in ns.items()}
        self.truemask = {k: torch.tensor(L(k, "shortest_paths")[:n].astype(np.float32)).to(DEV) for k, n in ns.items()}
        # optimal cost per instance via OUR solver on the TRUE costs (validated against dataset masks)
        self.zstar = {k: solve_grid_sp(self.cost[k])[1] for k in self.cost}

    def predictor(self):
        if self.model_type == "combresnet":
            return CombResNet18(self.grid).to(DEV)
        return CombResNet(self.grid, self.ch).to(DEV)
    def mse_pairs(self, split): return self.img[split], self.cost[split]

    def polystep_closure(self, model, split="train", objective="cost"):
        """objective='cost'    -> minimize realized path cost on TRUE costs (label-free; the original PolyStep).
           objective='mse'     -> minimize the PREDICTION-FOCUSED supervised loss: MSE(pred costs, TRUE cost
           labels) == two-stage's loss, but GRADIENT-FREE (no solver in loop). The Adam->PolyStep ablation.
           objective='hamming' -> minimize the benchmark's DECISION-FOCUSED supervised loss: the Hamming
           surrogate <w, 1-2w*> between the solver's path w and the TRUE optimal path w* (identical to
           blackbox-diff's loss, line ~242) but optimized GRADIENT-FREE (no DBB trick). MNIST-style label loss."""
        from pto.forward import batched_predict
        IMG, TC, ZS, TM = self.img[split], self.cost[split], self.zstar[split], self.truemask[split]
        nfull = IMG.shape[0]; mb = min(self.mb, nfull)
        g = torch.Generator(device=DEV).manual_seed(0)
        def closure(bp):
            if mb < nfull:                                      # minibatch the n instances (also a cost lever)
                idx = torch.randint(0, nfull, (mb,), generator=g, device=DEV)
                img, tc, zs, tm = IMG[idx], TC[idx], ZS[idx], TM[idx]
            else:
                img, tc, zs, tm = IMG, TC, ZS, TM
            pc = batched_predict(model, bp, img)                # (N,mb,12,12) predicted cost grids
            N, B = pc.shape[0], pc.shape[1]
            if objective == "mse":                              # prediction-focused supervised loss, NO solver
                return ((pc - tc.unsqueeze(0)) ** 2).flatten(1).mean(1) / (tc ** 2).mean()  # (N,) normalized MSE
            mask = solve_grid_sp(pc.reshape(N * B, self.grid, self.grid))[0].reshape(N, B, self.grid, self.grid)
            if objective == "hamming":                          # decision-focused supervised loss, gradient-free
                ham = (mask * (1 - 2 * tm.unsqueeze(0))).sum(dim=(-1, -2))   # (N,B): <w,1-2w*> = Hamming up to const
                return ham.sum(-1) / tm.sum()                               # normalize to O(1) like realized cost
            realized = (mask * tc.unsqueeze(0)).sum(dim=(-1, -2))           # (N,B)
            return realized.sum(-1) / zs.sum()                             # normalized realized cost (minimize)
        return closure

    @torch.no_grad()
    def fast_regret(self, model, split="val"):
        model.eval()                                            # BN uses warm-start running stats
        pc = model(self.img[split]); mask, _ = solve_grid_sp(pc)
        realized = (mask * self.cost[split]).sum(dim=(-1, -2))
        return ((realized - self.zstar[split]).sum() / self.zstar[split].sum()).item()
    regret = fast_regret

    @torch.no_grad()
    def cost_match(self, model, split="test"):
        model.eval()                                            # BN uses warm-start running stats
        pc = model(self.img[split]); mask, _ = solve_grid_sp(pc)
        realized = (mask * self.cost[split]).sum(dim=(-1, -2))
        opt = self.zstar[split]
        return (realized <= opt + 1e-4).float().mean().item()   # predicted path's TRUE cost == optimum


def validate_solver(prob):
    """Our solver on TRUE costs must reproduce the dataset's optimal path cost (<= dataset-mask cost)."""
    tc = prob.cost["test"]; dm = prob.truemask["test"]
    ours_cost = prob.zstar["test"]
    data_cost = (dm * tc).sum(dim=(-1, -2))
    gap = (ours_cost - data_cost)                               # ours is the min -> should be <= data_cost
    P(f"[solver check] ours vs dataset-mask cost: mean ours={ours_cost.mean():.3f} data={data_cost.mean():.3f} "
      f"| ours<=data on {(ours_cost <= data_cost + 1e-3).float().mean()*100:.1f}% | maxover={gap.clamp(min=0).max():.4f}")
    return float((ours_cost <= data_cost + 1e-3).float().mean())


# ---- blackbox-differentiation of the shortest-path solver (Vlastelica et al., the paper's own method) ----
class _BlackboxSP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, costs, lam):
        w, _ = solve_grid_sp(costs)
        ctx.save_for_backward(costs, w)
        ctx.lam = float(lam)
        return w
    @staticmethod
    def backward(ctx, grad_w):
        costs, w = ctx.saved_tensors
        wp, _ = solve_grid_sp(costs + ctx.lam * grad_w)        # perturbed re-solve
        return -(w - wp) / ctx.lam, None


def train_blackbox_diff(prob, lam=20.0, epochs=80, lr=1e-3, mb=256, seed=0, verbose=False):
    """Vlastelica blackbox-diff: solver-in-the-loop + implicit gradient, supervised Hamming loss to true paths."""
    torch.manual_seed(seed)
    model = prob.predictor(); model.train()
    opt = torch.optim.Adam(model.parameters(), lr)
    IMG, TM = prob.img["train"], prob.truemask["train"]; n = IMG.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV); tot = 0.0
        for i in range(0, n, mb):
            idx = perm[i:i + mb]
            costs = model(IMG[idx])
            w = _BlackboxSP.apply(costs, lam)
            loss = (w * (1 - 2 * TM[idx])).sum(dim=(-1, -2)).mean()   # <w, 1-2w*> Hamming surrogate
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if verbose and (ep % max(1, epochs // 8) == 0 or ep == epochs - 1):
            P(f"  [bb ep {ep:>3}] loss {tot/max(1,n//mb):+.3f}  val cost-match {prob.cost_match(model,'val')*100:5.1f}%")
            model.train()
    return model


def train_two_stage_mb(prob, epochs=80, lr=1e-3, mb=256, seed=0):
    """Minibatched two-stage (Adam MSE on true costs) — pto.train_two_stage is full-batch (OOMs on CombResNet)."""
    torch.manual_seed(seed)
    model = prob.predictor(); model.train()
    opt = torch.optim.Adam(model.parameters(), lr)
    IMG, COST = prob.img["train"], prob.cost["train"]; n = IMG.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, mb):
            idx = perm[i:i + mb]
            loss = ((model(IMG[idx]) - COST[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def run_combresnet(smoke):
    """Head-to-head on the paper's EXACT CombResNet18: two-stage vs blackbox-diff (paper's method) vs PolyStep."""
    nt, ntest, nval = (200, 200, 200) if smoke else (10000, 1000, 1000)
    mb = 64 if smoke else 256
    P(f"=== Warcraft CombResNet18 head-to-head | {'SMOKE' if smoke else 'FULL'} | n_train={nt} mb={mb} ===")
    t0 = time.time()
    prob = WarcraftSP(model_type="combresnet", n_train=nt, n_test=ntest, n_val=nval, mb=mb)
    D = sum(p.numel() for p in prob.predictor().parameters())
    P(f"CombResNet18 live params D={D:,} | load {time.time()-t0:.0f}s")
    ok = validate_solver(prob)
    if ok < 0.99: P("WARNING: solver does not reproduce dataset optima")
    ep_ts = 5 if smoke else 80
    ep_bb = 5 if smoke else 80
    q = 256 if smoke else 1024
    steps = 20 if smoke else 400
    # two-stage (minibatched MSE)
    _t = time.time(); m_ts = train_two_stage_mb(prob, epochs=ep_ts, mb=mb); t_ts = time.time() - _t
    r_ts, cm_ts = prob.regret(m_ts), prob.cost_match(m_ts)
    P(f"two-stage: cost-match {cm_ts*100:.1f}% regret {r_ts:.4f} ({t_ts:.0f}s)")
    # blackbox-diff (paper's method)
    _t = time.time(); m_bb = train_blackbox_diff(prob, lam=20.0, epochs=ep_bb, mb=mb, seed=0, verbose=True); t_bb = time.time() - _t
    r_bb, cm_bb = prob.regret(m_bb), prob.cost_match(m_bb)
    P(f"blackbox-diff: cost-match {cm_bb*100:.1f}% regret {r_bb:.4f} ({t_bb:.0f}s)")
    # PolyStep (gradient-free, subspace, warm from two-stage)
    _t = time.time(); m_ps, sd = train_ps_subspace(prob, warm=m_ts, q=q, steps=steps, seed=0); t_ps = time.time() - _t
    r_ps, cm_ps = prob.regret(m_ps), prob.cost_match(m_ps)
    ps_solves = 2 * sd * min(prob.mb, nt) * steps
    P("\n--- COMBRESNET RESULTS (test cost-match = predicted path's TRUE cost == optimum) ---")
    P(f"{'method':>14} | {'regret':>8} {'cost-match':>10} | {'wall_s':>7} {'solves':>14} {'gurobi':>7}")
    P(f"{'two-stage':>14} | {r_ts:>8.4f} {cm_ts*100:>9.1f}% | {t_ts:>6.0f}s {0:>14,} {0:>7}")
    P(f"{'blackbox-diff':>14} | {r_bb:>8.4f} {cm_bb*100:>9.1f}% | {t_bb:>6.0f}s {0:>14,} {0:>7}")
    P(f"{'PolyStep':>14} | {r_ps:>8.4f} {cm_ps*100:>9.1f}% | {t_ps:>6.0f}s {ps_solves:>14,} {0:>7}")
    P(f"(paper ref: blackbox-diff CombResNet ~95%+ on 12x12; D={D:,}, subspace q={q}->sub_dim={sd}, N={2*sd})")
    out = dict(mode="combresnet", smoke=smoke, D=D, n_train=nt, n_test=ntest, subspace_q=q, subspace_dim=sd, steps=steps,
               two_stage=dict(regret=r_ts, cost_match=cm_ts, wall_s=t_ts),
               blackbox_diff=dict(regret=r_bb, cost_match=cm_bb, wall_s=t_bb),
               polystep=dict(regret=r_ps, cost_match=cm_ps, wall_s=t_ps, forward_solves=ps_solves, solver_calls=ps_solves, gurobi=0))
    json.dump(out, open("exp_results/warcraft_combresnet.json", "w"), indent=1)
    P(f"[total {time.time()-t0:.0f}s] -> exp_results/warcraft_combresnet.json")


def run_sweep():
    """cost-match vs subspace dim q: does subspace-PolyStep recover the full-space gain? (intrinsic dim)"""
    P("=== Warcraft 12x12: cost-match vs subspace dim q (scaling / intrinsic-dimension characterization) ===")
    nt = 500
    prob = WarcraftSP(n_train=nt, n_test=500, n_val=300, ch=16, mb=128)
    D = sum(p.numel() for p in prob.predictor().parameters())
    validate_solver(prob)
    m_ts = train_two_stage(prob, epochs=40)
    P(f"two-stage baseline: cost-match {prob.cost_match(m_ts)*100:.1f}% regret {prob.regret(m_ts):.4f} | D={D}")
    STEPS = 60
    P(f"{'subspace q':>11} {'N=2q':>7} | {'cost-match':>10} {'regret':>8} | {'wall_s':>7} {'solves':>14}")
    rows = []
    for q in [256, 512, 1024, 2048, 4096]:
        _t = time.time(); m, sd = train_ps_subspace(prob, m_ts, q=q, steps=STEPS, seed=0); dt = time.time() - _t
        cm, r = prob.cost_match(m), prob.regret(m); N = 2 * sd; solves = N * min(prob.mb, nt) * STEPS
        P(f"{q:>11} {N:>7} | {cm*100:>9.1f}% {r:>8.4f} | {dt:>6.0f}s {solves:>14,}")
        rows.append(dict(q=q, subspace_dim=sd, N=N, cost_match=cm, regret=r, wall_s=dt, solves=solves))
        json.dump(rows, open("exp_results/warcraft_qsweep.json", "w"), indent=1)
    cfg = dict(polytope_type="orthoplex", num_probe=1, use_momentum=True, momentum_init=0.5, momentum_final=0.9,
               epsilon=CosineEpsilon(3.0, 0.3), step_radius=2.0, probe_radius=2.0)
    _t = time.time(); m = train_polystep(prob, cfg, steps=STEPS, warm=m_ts, subspace_rank=0, seed=0); dt = time.time() - _t
    cm, r = prob.cost_match(m), prob.regret(m)
    P(f"{'FULL-SPACE':>11} {2*D:>7} | {cm*100:>9.1f}% {r:>8.4f} | {dt:>6.0f}s  (N=2D reference)")
    rows.append(dict(q="full", N=2 * D, cost_match=cm, regret=r, wall_s=dt))
    json.dump(rows, open("exp_results/warcraft_qsweep.json", "w"), indent=1)
    P("-> exp_results/warcraft_qsweep.json")


def run_combresnet_sup(smoke):
    """ABLATION: can GRADIENT-FREE PolyStep recover the Adam baselines using the SAME objectives?
    GroupNorm CombResNet (batches cleanly -> no BN / no warm-start confound), ALL methods FROM SCRATCH on
    the IDENTICAL model. 2x3 grid: optimizer {Adam, PolyStep-gradient-free} x objective {MSE cost-labels
    (=two-stage, prediction-focused) / Hamming path-labels (=blackbox-diff, decision-focused) / realized
    cost (label-free)}. Adam covers the two supervised cells; PolyStep covers all three, swept over q."""
    nt, ntest, nval = (300, 300, 300) if smoke else (10000, 1000, 1000)
    mb = 64 if smoke else int(os.environ.get("WC_MB", "128"))
    ch = int(os.environ.get("WC_CH", "64"))
    qs = [512] if smoke else [int(x) for x in os.environ.get("WC_QS", "2048,4096").split(",")]
    objs = ["cost", "mse", "hamming"]
    steps = 30 if smoke else int(os.environ.get("WC_STEPS", "400"))
    ep = 5 if smoke else 80
    P(f"=== Warcraft GroupNorm-CombResNet(ch={ch}) Adam->PolyStep ablation | {'SMOKE' if smoke else 'FULL'} "
      f"| n_train={nt} mb={mb} q={qs} steps={steps} ===")
    t0 = time.time()
    prob = WarcraftSP(model_type="small", ch=ch, n_train=nt, n_test=ntest, n_val=nval, mb=mb)
    D = sum(p.numel() for p in prob.predictor().parameters())
    P(f"model live params D={D:,} | load {time.time()-t0:.0f}s"); validate_solver(prob)
    out = dict(mode="combresnet_sup", smoke=smoke, model=f"GroupNorm CombResNet ch={ch}", D=D,
               n_train=nt, n_test=ntest, steps=steps, qs=qs, results={})
    JF = "exp_results/warcraft_combresnet_sup.json"
    if os.path.exists(JF):                       # merge: keep prior cells (e.g. an earlier q) on a partial rerun
        try: out["results"] = json.load(open(JF)).get("results", {})
        except Exception: pass
    def rec(key, **kw):
        out["results"][key] = kw; json.dump(out, open(JF, "w"), indent=1)
    # ---- Adam references (gradient-based), from scratch ----
    _t = time.time(); m = train_two_stage_mb(prob, epochs=ep, mb=mb); dt = time.time() - _t
    cm, r = prob.cost_match(m), prob.regret(m)
    P(f"[Adam] two-stage     (MSE cost-labels, PRED)      cost-match {cm*100:5.1f}%  regret {r:.4f}  ({dt:.0f}s)")
    rec("adam_mse", optimizer="adam", objective="mse", cost_match=cm, regret=r, wall_s=dt)
    _t = time.time(); m = train_blackbox_diff(prob, lam=20.0, epochs=ep, mb=mb, seed=0, verbose=True); dt = time.time() - _t
    cm, r = prob.cost_match(m), prob.regret(m)
    P(f"[Adam] blackbox-diff (Hamming, DECISION) =BENCH    cost-match {cm*100:5.1f}%  regret {r:.4f}  ({dt:.0f}s)")
    rec("adam_hamming", optimizer="adam+dbb", objective="hamming", cost_match=cm, regret=r, wall_s=dt)
    # ---- PolyStep (gradient-free), FROM SCRATCH, each objective x q ----
    label = {"cost": "PolyStep-cost (realized, LABEL-FREE)",
             "mse": "PolyStep-MSE  (cost-labels, PRED)  ",
             "hamming": "PolyStep-Ham (path-labels, DECISION)"}
    sr_base = float(os.environ.get("WC_SR", "15.0"))
    for q in qs:
        sr_q = sr_base * (q / 2048.0) ** 0.5         # scale step_radius with subspace size (LinearSubspace dilutes ~1/sqrt(N))
        for obj in objs:
            _t = time.time()
            m, sd = train_ps_subspace(prob, warm=None, q=q, steps=steps, sr=sr_q, seed=0, objective=obj)
            dt = time.time() - _t
            cm, r = prob.cost_match(m), prob.regret(m)
            solves = (0 if obj == "mse" else 2 * sd * min(prob.mb, nt) * steps)
            P(f"[GF q={q:>4} sr={sr_q:.1f}] {label[obj]}  cost-match {cm*100:5.1f}%  regret {r:.4f}  "
              f"({dt:.0f}s, sd={sd}, {solves:,} solves)")
            rec(f"polystep_{obj}_q{q}", optimizer="polystep_gradfree", objective=obj, q=q, subspace_dim=sd,
                step_radius=round(sr_q, 2), cost_match=cm, regret=r, wall_s=dt, forward_solves=solves, gurobi=0)
    P(f"\n[total {time.time()-t0:.0f}s] -> exp_results/warcraft_combresnet_sup.json")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "sweep":
        return run_sweep()
    if mode == "combresnet":
        return run_combresnet(smoke=(len(sys.argv) < 3 or sys.argv[2] != "full"))
    if mode == "combresnet_sup":
        return run_combresnet_sup(smoke=(len(sys.argv) < 3 or sys.argv[2] != "full"))
    smoke = (mode == "smoke")
    nt, ntest = (300, 200) if smoke else (10000, 1000)
    P(f"=== Warcraft 12x12 shortest path | mode={mode} | n_train={nt} n_test={ntest} "
      f"| polytope={PS_POLYTOPE} num_probe={PS_PROBES} pdim={PS_PDIM} seed={int(os.environ.get('WC_SEED','0'))} "
      f"tag={OUT_TAG or '(none)'} ===")
    t0 = time.time()
    prob = WarcraftSP(n_train=nt, n_test=ntest, n_val=200 if smoke else 1000,
                      ch=16, mb=128 if smoke else 256)   # ch=16: the config where q=512 hit 62% in the sweep
    D = sum(p.numel() for p in prob.predictor().parameters())
    P(f"CNN params D={D:,}  | load+optcost {time.time()-t0:.0f}s")
    ok = validate_solver(prob)
    if ok < 0.99:
        P("WARNING: solver does not reproduce dataset optima — fix before trusting results.");
    # two-stage (Adam, MSE on costs)
    (m_ts), t_ts = (lambda: train_two_stage(prob, epochs=20 if smoke else 50)), None
    _t = time.time(); m_ts = train_two_stage(prob, epochs=20 if smoke else 50); t_ts = time.time() - _t
    r_ts, cm_ts = prob.regret(m_ts), prob.cost_match(m_ts)
    # PolyStep with a TRUE small-q subspace (the fix: N=2q via max_subspace_dim, not 2D)
    q = 128 if smoke else 512
    steps = 100 if smoke else int(os.environ.get("WC_MAIN_STEPS", "400"))
    ps_seed = int(os.environ.get("WC_SEED", "0"))
    _t = time.time(); m_ps, sub_dim = train_ps_subspace(prob, warm=m_ts, q=q, steps=steps, seed=ps_seed)
    t_ps = time.time() - _t
    r_ps, cm_ps = prob.regret(m_ps), prob.cost_match(m_ps)
    N = 2 * sub_dim
    ps_solves = N * min(prob.mb, nt) * steps                   # N=2*subspace_dim probe-solves/step x minibatch
    P(f"(subspace q={q} -> subspace_dim={sub_dim}, N={N} probes/step vs full-space N~{2*D})")
    P("\n--- RESULTS (cost-match accuracy = predicted path's TRUE cost == optimum) ---")
    P(f"{'method':>12} | {'regret':>8} {'cost-match':>10} | {'wall_s':>7} {'solves':>14} {'gurobi':>7}")
    P(f"{'two-stage':>12} | {r_ts:>8.4f} {cm_ts*100:>9.1f}% | {t_ts:>6.1f}s {0:>14,} {0:>7}")
    P(f"{'PolyStep':>12} | {r_ps:>8.4f} {cm_ps*100:>9.1f}% | {t_ps:>6.1f}s {ps_solves:>14,} {0:>7}")
    P(f"\n(paper ref: blackbox-diff CombResNet ~95%+ perfect-match on 12x12; plain ResNet baseline far lower)")
    out = dict(mode=mode, D=D, n_train=nt, n_test=ntest, subspace_q=q, subspace_dim=sub_dim, N_probes=N, steps=steps,
               polytope=PS_POLYTOPE, num_probe=PS_PROBES, particle_dim=PS_PDIM, seed=ps_seed,
               two_stage=dict(regret=r_ts, cost_match=cm_ts, wall_s=t_ts, solves=0),
               polystep=dict(regret=r_ps, cost_match=cm_ps, wall_s=t_ps, forward_solves=ps_solves, solver_calls=ps_solves, gurobi=0))
    json.dump(out, open(f"exp_results/warcraft_{mode}{_OUT_SFX}.json", "w"), indent=1)
    P(f"[total {time.time()-t0:.0f}s] -> exp_results/warcraft_{mode}{_OUT_SFX}.json")


if __name__ == "__main__":
    main()
