"""Track 1: per-OPTIMIZATION-CATEGORY capability map.

For each benchmark (grouped by category: LP / ILP / convex-nonlinear) train the FULL
PyEPO DFL suite + two-stage + PolyStep on the SAME PyEPO data and regret metric, and
tabulate normalized regret. PolyStep uses the verified batched GPU forward solver.

Run:  .venv/bin/python -m pto.capability <problems> <degs> <seeds>
      e.g.  .venv/bin/python -m pto.capability sp,knap,tsp,port 4 42,43,44
"""
from __future__ import annotations
import sys, itertools, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "polystep/src")
from torch.utils.data import DataLoader
from pyepo.data import shortestpath, knapsack, tsp, portfolio
from pyepo.model.grb import shortestPathModel, knapsackModel, tspMTZModel, portfolioModel
from pyepo.data.dataset import optDataset
from pyepo import metric
import pyepo.func as F
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from pto.solvers import build_dag_solver, knap1_dp, solve_portfolio_socp

dev = "cuda"; PF = 5

# ---- DFL adapter table: name -> (builder(optmodel,ds), kind, fwd_args) ----
DFL = {
    "SPO+":  (lambda om, ds: F.SPOPlus(om),                                        "loss", ["pred", "c", "w", "z"]),
    "DBB":   (lambda om, ds: F.blackboxOpt(om, lambd=10),                          "opt",  None),
    "NID":   (lambda om, ds: F.negativeIdentity(om),                              "opt",  None),
    "DPO":   (lambda om, ds: F.perturbedOpt(om, n_samples=3, sigma=1.0),           "opt",  None),
    "IMLE":  (lambda om, ds: F.implicitMLE(om, n_samples=3, sigma=1.0, lambd=10),  "opt",  None),
    "PFYL":  (lambda om, ds: F.perturbedFenchelYoung(om, n_samples=3, sigma=1.0),  "loss", ["pred", "w"]),
    "NCE":   (lambda om, ds: F.noiseContrastiveEstimation(om, dataset=ds, solve_ratio=0.05), "loss", ["pred", "w"]),
    "LTR":   (lambda om, ds: F.pairwiseLearningToRank(om, dataset=ds, solve_ratio=0.05), "loss", ["pred", "c"]),
    "ptLTR": (lambda om, ds: F.pairwiseLearningToRank(om, dataset=ds, solve_ratio=0.05), "loss", ["pred", "c"]),
    "lsLTR": (lambda om, ds: F.listwiseLearningToRank(om, dataset=ds, solve_ratio=0.05), "loss", ["pred", "c"]),
    "ptwLTR":(lambda om, ds: F.pointwiseLearningToRank(om, dataset=ds, solve_ratio=0.05), "loss", ["pred", "c"]),
    "PG":    (lambda om, ds: F.perturbationGradient(om, sigma=0.1),                "loss", ["pred", "c"]),
}
METHODS = ["two-stage"] + list(DFL) + ["PolyStep"]


def _adam(model, lr=1e-2): return torch.optim.Adam(model.parameters(), lr)

def train_two_stage(cfg, epochs=40):
    m = cfg["make"](); opt = _adam(m)
    for _ in range(epochs):
        for xb, cb, wb, zb in cfg["ld_tr"]:
            xb, cb = xb.float().to(dev), cb.float().to(dev)
            opt.zero_grad(); ((m(xb) - cb) ** 2).mean().backward(); opt.step()
    return m

def train_dfl(cfg, name, epochs=30):
    build, kind, fwd = DFL[name]
    om = cfg["om"]; sense = om.modelSense
    m = cfg["make"](); opt = _adam(m); loss_mod = build(om, cfg["ds_tr"])
    for _ in range(epochs):
        for xb, cb, wb, zb in cfg["ld_tr"]:
            xb, cb, wb, zb = [t.float().to(dev) for t in (xb, cb, wb, zb)]
            pred = m(xb)
            if kind == "opt":
                loss = sense * (loss_mod(pred) * cb).sum(-1).mean()
            else:
                pick = {"pred": pred, "c": cb, "w": wb, "z": zb}
                out = loss_mod(*[pick[a] for a in fwd]); loss = out.mean() if out.dim() > 0 else out
            opt.zero_grad(); loss.backward(); opt.step()
    return m

def train_polystep(cfg, steps=150):
    steps = cfg.get("ps_steps", steps)
    m = cfg["make"]()
    with torch.no_grad():
        if cfg.get("warm") is not None: m.weight.copy_(cfg["warm"].weight)
    # polytope / probe-point / radii are cfg-overridable (defaults reproduce the original orthoplex run)
    pso = PolyStepOptimizer(m, polytope_type=cfg.get("ps_polytope", "orthoplex"),
                            epsilon=CosineEpsilon(cfg.get("ps_eps0", 0.5), cfg.get("ps_eps1", 0.05)),
                            step_radius=cfg.get("ps_step_radius", 0.4),
                            probe_radius=cfg.get("ps_probe_radius", 0.8),
                            num_probe=cfg.get("ps_num_probe", 1), seed=cfg["seed"],
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    X, Cs, solve, sgn = cfg["Xtr"], cfg["Cs"], cfg["ps_solve"], cfg["sign"]
    def closure(bp):
        pred = torch.einsum("nef,bf->nbe", bp["weight"], X); N, nb, E = pred.shape
        w = solve(pred.reshape(N * nb, E)).reshape(N, nb, E)
        return sgn * (w * Cs.unsqueeze(0)).sum(-1).mean(-1)
    for _ in range(steps): pso.step(closure)
    return m


def train_sfge(cfg, epochs=120, n_samples=8, sigma=0.5, lr=1e-2):
    """SFGE (Silvestri et al., JAIR 2026): the closest GRADIENT-FREE rival. Places a Gaussian
    over the predicted parameters, samples, solves the (black-box) forward problem, and uses a
    score-function/REINFORCE estimator on the realized decision loss with a per-batch baseline
    (variance reduction). Warm-started identically to PolyStep for a fair head-to-head."""
    m = cfg["make"]()
    if cfg.get("warm") is not None:
        with torch.no_grad(): m.weight.copy_(cfg["warm"].weight)
    opt = _adam(m, lr)
    X, Cs, solve, sgn = cfg["Xtr"], cfg["Cs"], cfg["ps_solve"], cfg["sign"]
    for _ in range(epochs):
        pred = m(X)                                                  # (B, D), differentiable in theta
        with torch.no_grad():
            eps = torch.randn(n_samples, *pred.shape, device=dev)
            chat = pred.unsqueeze(0) + sigma * eps                   # (S, B, D) sampled predictions
            S, B, D = chat.shape
            w = solve(chat.reshape(S * B, D)).reshape(S, B, D)
            r = sgn * (w * Cs.unsqueeze(0)).sum(-1)                  # (S,B) per-sample loss (minimize)
            adv = r - r.mean(0, keepdim=True)                        # baseline-subtracted advantage
        logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)   # grad flows via pred
        surrogate = (adv * logp).mean()                             # REINFORCE: d/dtheta = E[adv * dlogp]
        opt.zero_grad(); surrogate.backward(); opt.step()
    return m


# ---------------- per-problem setup: returns a cfg dict ----------------
def _common(om, x, c, dim, ps_solve, objective, standardize, seed, n_train):
    xtr, ctr, xte, cte = x[:n_train], c[:n_train], x[n_train:], c[n_train:]
    ds_tr = optDataset(om, xtr, ctr); ds_te = optDataset(om, xte, cte)
    Ctr = torch.tensor(ctr, dtype=torch.float32, device=dev)
    shift = Ctr.mean() if standardize == "affine" else 0.0
    cfg = dict(om=om, dim=dim, seed=seed, ps_solve=ps_solve,
               make=lambda: nn.Linear(PF, dim, bias=False).to(dev),
               ld_tr=DataLoader(ds_tr, batch_size=128, shuffle=True),
               ld_te=DataLoader(ds_te, batch_size=256), ds_tr=ds_tr,
               Xtr=torch.tensor(xtr, dtype=torch.float32, device=dev),
               Cs=(Ctr - shift) / Ctr.std(),
               sign=(1.0 if objective == "min" else -1.0))
    # stash raw test tensors so callers can compute a cheap GPU-solver regret proxy (no Gurobi)
    cfg["Xte"] = torch.tensor(xte, dtype=torch.float32, device=dev)
    cfg["Cte"] = torch.tensor(cte, dtype=torch.float32, device=dev)
    return cfg

def setup_sp(seed, deg, n_train=900, n_test=200, H=5, W=5):
    om = shortestPathModel((H, W)); arcs = list(om.arcs)
    sb = build_dag_solver(arcs, H * W, 0, H * W - 1)
    x, c = shortestpath.genData(n_train + n_test, PF, (H, W), deg=deg, noise_width=0, seed=seed)
    return _common(om, x, c, len(arcs), sb, "min", "affine", seed, n_train), "LP"

def setup_knap(seed, deg, n_train=900, n_test=200, NIT=16):
    W_np, _, _ = knapsack.genData(2, PF, NIT, dim=1, deg=1, seed=1)
    weights = W_np[0].astype(int); CAP = int(weights.sum() * 0.5)
    om = knapsackModel(weights=W_np.astype(int), capacity=[CAP])
    Wt = torch.tensor(weights, dtype=torch.float32, device=dev)
    sb = lambda v: knap1_dp(v, Wt.expand(v.shape[0], -1), CAP)[1].float()
    _, x, c = knapsack.genData(n_train + n_test, PF, NIT, dim=1, deg=deg, noise_width=0, seed=seed)
    return _common(om, x, c, NIT, sb, "max", "scale", seed, n_train), "ILP"

def setup_tsp(seed, deg, n_train=280, n_test=120, N=8):
    om = tspMTZModel(num_nodes=N); ei = {e: i for i, e in enumerate(om.edges)}; E = om.num_cost
    rows = []
    for perm in itertools.permutations(range(1, N)):
        if perm[0] > perm[-1]: continue
        cyc = [0] + list(perm); v = torch.zeros(E)
        for a, b in zip(cyc, cyc[1:] + [0]): v[ei[(min(a, b), max(a, b))]] = 1.0
        rows.append(v)
    T = torch.stack(rows).to(dev)
    sb = lambda c: T[(c @ T.T).argmin(1)]
    x, c = tsp.genData(n_train + n_test, PF, N, deg=deg, noise_width=0, seed=seed)
    return _common(om, x, c, E, sb, "min", "affine", seed, n_train), "ILP"

def setup_port(seed, deg, n_train=200, n_test=400, NA=20):
    cov, x, r = portfolio.genData(n_train + n_test, PF, NA, deg=deg, noise_level=1, seed=seed)
    om = portfolioModel(num_assets=NA, covariance=cov, gamma=2.25); rho = float(om.risk_level)
    Sig = torch.tensor(cov, dtype=torch.float32, device=dev)
    sb = lambda rhat: solve_portfolio_socp(rhat, Sig, rho, bis=16, pg=25)
    cfg = _common(om, x, r, NA, sb, "max", "scale", seed, n_train); cfg["ps_steps"] = 80
    return cfg, "nonlin"

SETUPS = {"sp": setup_sp, "knap": setup_knap, "tsp": setup_tsp, "port": setup_port}


def run(problems, degs, seeds, methods=None):
    dfl_run = [m for m in DFL if methods is None or m in methods]
    cols = ["two-stage"] + dfl_run + ["PolyStep"]
    print(f"{'problem':>6} {'cat':>7} {'deg':>4} | " + " ".join(f"{m:>9}" for m in cols), flush=True)
    for pname in problems:
        for deg in degs:
            acc = {m: [] for m in cols}
            for seed in seeds:
                cfg, cat = SETUPS[pname](seed, deg)
                ts = train_two_stage(cfg); acc["two-stage"].append(metric.regret(ts, cfg["om"], cfg["ld_te"]))
                for name in dfl_run:
                    try: acc[name].append(metric.regret(train_dfl(cfg, name), cfg["om"], cfg["ld_te"]))
                    except Exception: acc[name].append(float("nan"))
                cfg["warm"] = ts
                acc["PolyStep"].append(metric.regret(train_polystep(cfg), cfg["om"], cfg["ld_te"]))
            row = " ".join(f"{np.nanmean(acc[m]):>9.4f}" for m in cols)
            print(f"{pname:>6} {cat:>7} {deg:>4} | {row}", flush=True)


if __name__ == "__main__":
    probs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["knap"]
    degs = [int(d) for d in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4]
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [42]
    methods = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    run(probs, degs, seeds, methods)
