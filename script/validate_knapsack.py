"""Phase 2 -- FAIR comparison on an established SPO+ benchmark: PyEPO knapsack.

Value (objective) prediction; known integer weights; single capacity. Uses PyEPO's
data generator, authoritative SPOPlus loss, and regret metric (Gurobi-evaluated).
PolyStep trains gradient-free with our exact batched knapsack DP (verified == Gurobi).
"""
import sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "polystep/src")
from torch.utils.data import DataLoader
from pyepo.data import knapsack
from pyepo.model.grb import knapsackModel
from pyepo.data.dataset import optDataset
from pyepo.func import SPOPlus
from pyepo import metric
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from pto.solvers import knap1_dp

dev = "cuda"; NIT = 16; PF = 5
torch.manual_seed(0)

# fixed known weights + capacity (shared across instances)
W_np, _, _ = knapsack.genData(2, PF, NIT, dim=1, deg=1, seed=1)        # weights (1, NIT)
weights = W_np[0].astype(int)
CAP = int(weights.sum() * 0.5)
optmodel = knapsackModel(weights=W_np.astype(int), capacity=[CAP])
Wt = torch.tensor(weights, dtype=torch.float32, device=dev)

def solve_batch(values):                                              # (M,NIT) -> sel (M,NIT)
    M = values.shape[0]
    _, sel = knap1_dp(values, Wt.expand(M, -1), CAP)
    return sel.float()

# verify our DP == Gurobi
vv = np.random.rand(128, NIT)
sel = solve_batch(torch.tensor(vv, dtype=torch.float32, device=dev))
ours = (sel.cpu().numpy() * vv).sum(1); gur = []
for i in range(128):
    optmodel.setObj(vv[i]); _, o = optmodel.solve(); gur.append(o)
print(f"knapsack DP vs Gurobi: max abs diff = {np.abs(ours - np.array(gur)).max():.2e}  (cap={CAP})")


def make_linear(): return nn.Linear(PF, NIT, bias=False).to(dev)

def run(deg, seed=42, n_data=1100, n_train=900, epochs=40, ps_steps=150):
    _, x, c = knapsack.genData(n_data, PF, NIT, dim=1, deg=deg, noise_width=0, seed=seed)
    xtr, ctr, xte, cte = x[:n_train], c[:n_train], x[n_train:], c[n_train:]
    ds_tr = optDataset(optmodel, xtr, ctr); ds_te = optDataset(optmodel, xte, cte)
    ld_tr = DataLoader(ds_tr, batch_size=128, shuffle=True)
    ld_te = DataLoader(ds_te, batch_size=256)
    Xtr = torch.tensor(xtr, dtype=torch.float32, device=dev)
    Ctr = torch.tensor(ctr, dtype=torch.float32, device=dev)

    # two-stage MSE
    m_ts = make_linear(); opt = torch.optim.Adam(m_ts.parameters(), 1e-2)
    for _ in range(epochs):
        for xb, cb, w_, z_ in ld_tr:
            xb, cb = xb.float().to(dev), cb.float().to(dev)
            opt.zero_grad(); ((m_ts(xb) - cb) ** 2).mean().backward(); opt.step()
    r_ts = metric.regret(m_ts, optmodel, ld_te)

    # SPO+ (PyEPO authoritative)
    m_spo = make_linear(); opt = torch.optim.Adam(m_spo.parameters(), 1e-2)
    spop = SPOPlus(optmodel, processes=1)
    for _ in range(epochs):
        for xb, cb, w_, z_ in ld_tr:
            xb, cb, w_, z_ = [t.float().to(dev) for t in (xb, cb, w_, z_)]
            opt.zero_grad(); spop(m_spo(xb), cb, w_, z_).mean().backward(); opt.step()
    r_spo = metric.regret(m_spo, optmodel, ld_te)

    # PolyStep: maximize realized value via our batched DP (warm from two-stage)
    Cs = Ctr / Ctr.std()                                              # scale-only (argmax-invariant)
    m_ps = make_linear()
    with torch.no_grad(): m_ps.weight.copy_(m_ts.weight)
    pso = PolyStepOptimizer(m_ps, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=0.4, probe_radius=0.8, num_probe=1, seed=0,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    def closure(bp):
        vhat = torch.einsum("nef,bf->nbe", bp["weight"], Xtr); N, nb, E = vhat.shape
        sel = solve_batch(vhat.reshape(N * nb, E)).reshape(N, nb, E)
        return -(sel * Cs.unsqueeze(0)).sum(-1).mean(-1)              # maximize realized value
    t0 = time.time()
    for _ in range(ps_steps): pso.step(closure)
    dt = time.time() - t0
    r_ps = metric.regret(m_ps, optmodel, ld_te)
    return r_ts, r_spo, r_ps, dt


print("\nPyEPO knapsack (dim=1, 16 items) | normalized regret (PyEPO metric, Gurobi), 3 seeds")
print(f"{'deg':>4} | {'two-stage':>16} {'SPO+ (PyEPO)':>16} {'PolyStep':>16}", flush=True)
for deg in [1, 2, 4, 6]:
    R = np.array([run(deg, seed=s)[:3] for s in (42, 43, 44)])
    m, sd = R.mean(0), R.std(0)
    print(f"{deg:>4} | {m[0]:>7.4f}±{sd[0]:<7.4f} {m[1]:>7.4f}±{sd[1]:<7.4f} {m[2]:>7.4f}±{sd[2]:<7.4f}", flush=True)
