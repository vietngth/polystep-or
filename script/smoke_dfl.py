"""Stage 1 smoke test: train the FULL PyEPO DFL suite + two-stage + PolyStep on the
PyEPO knapsack (ILP), confirm each produces a finite, sensible normalized regret.
Validates the train_dfl adapter table before scaling to the Track-1 capability map.
"""
import sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "polystep/src")
from torch.utils.data import DataLoader
from pyepo.data import knapsack
from pyepo.model.grb import knapsackModel
from pyepo.data.dataset import optDataset
from pyepo import metric
import pyepo.func as F
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from pto.solvers import knap1_dp

dev = "cuda"; NIT = 16; PF = 5
torch.manual_seed(0)
W_np, _, _ = knapsack.genData(2, PF, NIT, dim=1, deg=1, seed=1)
weights = W_np[0].astype(int); CAP = int(weights.sum() * 0.5)
optmodel = knapsackModel(weights=W_np.astype(int), capacity=[CAP])
SENSE = optmodel.modelSense                                            # -1 (MAXIMIZE)
Wt = torch.tensor(weights, dtype=torch.float32, device=dev)

def solve_batch(values):
    _, sel = knap1_dp(values, Wt.expand(values.shape[0], -1), CAP); return sel.float()

# ---- DFL adapter table: name -> (builder(optmodel,ds), kind, fwd_args) ----
# kind 'loss' returns scalar; kind 'opt' returns a solution -> wrap as sense*(sol*c)
DFL = {
    "SPO+":  (lambda om, ds: F.SPOPlus(om),                                   "loss", ["pred", "c", "w", "z"]),
    "DBB":   (lambda om, ds: F.blackboxOpt(om, lambd=10),                     "opt",  None),
    "NID":   (lambda om, ds: F.negativeIdentity(om),                         "opt",  None),
    "DPO":   (lambda om, ds: F.perturbedOpt(om, n_samples=5, sigma=1.0),      "opt",  None),
    "IMLE":  (lambda om, ds: F.implicitMLE(om, n_samples=5, sigma=1.0, lambd=10), "opt", None),
    "PFYL":  (lambda om, ds: F.perturbedFenchelYoung(om, n_samples=5, sigma=1.0), "loss", ["pred", "w"]),
    "NCE":   (lambda om, ds: F.noiseContrastiveEstimation(om, dataset=ds, solve_ratio=0.05), "loss", ["pred", "w"]),
    "CMAP":  (lambda om, ds: F.contrastiveMAP(om, dataset=ds, solve_ratio=0.05), "loss", ["pred", "w"]),
    "ptLTR": (lambda om, ds: F.pairwiseLearningToRank(om, dataset=ds, solve_ratio=0.05), "loss", ["pred", "c"]),
    "lsLTR": (lambda om, ds: F.listwiseLearningToRank(om, dataset=ds, solve_ratio=0.05), "loss", ["pred", "c"]),
    "PG":    (lambda om, ds: F.perturbationGradient(om, sigma=0.1),           "loss", ["pred", "c"]),
}

def make_lin(): return nn.Linear(PF, NIT, bias=False).to(dev)

def train_dfl(name, ld_tr, ds_tr, epochs=30, lr=1e-2):
    build, kind, fwd = DFL[name]
    model = make_lin(); opt = torch.optim.Adam(model.parameters(), lr)
    loss_mod = build(optmodel, ds_tr)
    for _ in range(epochs):
        for xb, cb, wb, zb in ld_tr:
            xb, cb, wb, zb = [t.float().to(dev) for t in (xb, cb, wb, zb)]
            pred = model(xb)
            if kind == "opt":
                sol = loss_mod(pred); loss = SENSE * (sol * cb).sum(-1).mean()
            else:
                pick = {"pred": pred, "c": cb, "w": wb, "z": zb}
                out = loss_mod(*[pick[a] for a in fwd])
                loss = out.mean() if out.dim() > 0 else out
            opt.zero_grad(); loss.backward(); opt.step()
    return model

# data
_, x, c = knapsack.genData(1100, PF, NIT, dim=1, deg=4, noise_width=0, seed=42)
xtr, ctr, xte, cte = x[:900], c[:900], x[900:], c[900:]
ds_tr = optDataset(optmodel, xtr, ctr); ds_te = optDataset(optmodel, xte, cte)
ld_tr = DataLoader(ds_tr, batch_size=128, shuffle=True)
ld_te = DataLoader(ds_te, batch_size=256)

print("Stage-1 smoke: PyEPO knapsack (ILP, deg=4) | normalized regret per method", flush=True)
# two-stage
m = make_lin(); opt = torch.optim.Adam(m.parameters(), 1e-2)
Xtr = torch.tensor(xtr, dtype=torch.float32, device=dev); Ctr = torch.tensor(ctr, dtype=torch.float32, device=dev)
for _ in range(40):
    for xb, cb, wb, zb in ld_tr:
        xb, cb = xb.float().to(dev), cb.float().to(dev)
        opt.zero_grad(); ((m(xb) - cb) ** 2).mean().backward(); opt.step()
print(f"  {'two-stage':>8}: {metric.regret(m, optmodel, ld_te):.4f}", flush=True)
# DFL suite
for name in DFL:
    t0 = time.time()
    try:
        mdl = train_dfl(name, ld_tr, ds_tr)
        r = metric.regret(mdl, optmodel, ld_te)
        print(f"  {name:>8}: {r:.4f}   ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  {name:>8}: ERROR {type(e).__name__}: {str(e)[:80]}", flush=True)
# PolyStep
Cs = Ctr / Ctr.std(); m_ps = make_lin()
with torch.no_grad(): m_ps.weight.copy_(m.weight)
pso = PolyStepOptimizer(m_ps, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                        step_radius=0.4, probe_radius=0.8, num_probe=1, seed=0,
                        use_momentum=True, momentum_init=0.5, momentum_final=0.9)
def closure(bp):
    vhat = torch.einsum("nef,bf->nbe", bp["weight"], Xtr); N, nb, E = vhat.shape
    sel = solve_batch(vhat.reshape(N * nb, E)).reshape(N, nb, E)
    return -(sel * Cs.unsqueeze(0)).sum(-1).mean(-1)
for _ in range(150): pso.step(closure)
print(f"  {'PolyStep':>8}: {metric.regret(m_ps, optmodel, ld_te):.4f}", flush=True)
