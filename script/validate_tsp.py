"""Phase 2 (3rd fair benchmark): PyEPO TSP (symmetric, n=8) -- established SPO+ benchmark.

n=8 has only 7!/2 = 2520 distinct tours, so the forward solve is EXACT by brute-force over
all tours (one batched matmul). PyEPO data + authoritative SPOPlus + regret metric (Gurobi).
All tours have n edges, so edge-cost argmin is invariant to affine scaling -> we standardize.
"""
import sys, time, itertools, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "polystep/src")
from torch.utils.data import DataLoader
from pyepo.data import tsp
from pyepo.model.grb import tspMTZModel
from pyepo.data.dataset import optDataset
from pyepo.func import SPOPlus
from pyepo import metric
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon

dev = "cuda"; N = 8; PF = 5
torch.manual_seed(0)
optmodel = tspMTZModel(num_nodes=N)
edge_index = {e: i for i, e in enumerate(optmodel.edges)}              # (i,j) i<j -> col
E = optmodel.num_cost

# enumerate all distinct tours -> incidence matrix (T, E)
def build_incidence():
    rows = []
    for perm in itertools.permutations(range(1, N)):
        if perm[0] > perm[-1]:                                         # dedupe reversal
            continue
        cyc = [0] + list(perm)
        v = torch.zeros(E)
        for a, b in zip(cyc, cyc[1:] + [0]):
            v[edge_index[(min(a, b), max(a, b))]] = 1.0
        rows.append(v)
    return torch.stack(rows).to(dev)                                   # (2520, E)
T_inc = build_incidence()
print(f"TSP n={N}: {E} edges, {T_inc.shape[0]} tours")

def solve_batch(c):                                                    # c (M,E) -> w (M,E)
    best = (c @ T_inc.T).argmin(1)
    return T_inc[best]

# verify brute-force == Gurobi
vv = np.random.rand(64, E)
w = solve_batch(torch.tensor(vv, dtype=torch.float32, device=dev))
ours = (w.cpu().numpy() * vv).sum(1); gur = []
for i in range(64):
    optmodel.setObj(vv[i]); _, o = optmodel.solve(); gur.append(o)
print(f"TSP brute-force vs Gurobi: max abs diff = {np.abs(ours - np.array(gur)).max():.2e}")


def make_lin(): return nn.Linear(PF, E, bias=False).to(dev)

def run(deg, seed, n_train=280, n_data=400, epochs=20, ps_steps=120):
    x, c = tsp.genData(n_data, PF, N, deg=deg, noise_width=0, seed=seed)
    xtr, ctr, xte, cte = x[:n_train], c[:n_train], x[n_train:], c[n_train:]
    ld_tr = DataLoader(optDataset(optmodel, xtr, ctr), batch_size=128, shuffle=True)
    ld_te = DataLoader(optDataset(optmodel, xte, cte), batch_size=256)
    Xtr = torch.tensor(xtr, dtype=torch.float32, device=dev)
    Ctr = torch.tensor(ctr, dtype=torch.float32, device=dev)
    # two-stage MSE
    m_ts = make_lin(); opt = torch.optim.Adam(m_ts.parameters(), 1e-2)
    for _ in range(epochs):
        for xb, cb, w_, z_ in ld_tr:
            xb, cb = xb.float().to(dev), cb.float().to(dev)
            opt.zero_grad(); ((m_ts(xb) - cb) ** 2).mean().backward(); opt.step()
    r_ts = metric.regret(m_ts, optmodel, ld_te)
    # SPO+
    m_sp = make_lin(); opt = torch.optim.Adam(m_sp.parameters(), 1e-2); spo = SPOPlus(optmodel, processes=1)
    for _ in range(epochs):
        for xb, cb, w_, z_ in ld_tr:
            xb, cb, w_, z_ = [t.float().to(dev) for t in (xb, cb, w_, z_)]
            opt.zero_grad(); spo(m_sp(xb), cb, w_, z_).mean().backward(); opt.step()
    r_sp = metric.regret(m_sp, optmodel, ld_te)
    # PolyStep (standardized costs; all tours have N edges -> affine-invariant)
    Cs = (Ctr - Ctr.mean()) / Ctr.std()
    m_ps = make_lin(); m_ps.load_state_dict(m_ts.state_dict())
    pso = PolyStepOptimizer(m_ps, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=0.4, probe_radius=0.8, num_probe=1, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    def closure(bp):
        chat = torch.einsum("nef,bf->nbe", bp["weight"], Xtr); Nn, nb, e = chat.shape
        w = solve_batch(chat.reshape(Nn * nb, e)).reshape(Nn, nb, e)
        return (w * Cs.unsqueeze(0)).sum(-1).mean(-1)
    for _ in range(ps_steps): pso.step(closure)
    return r_ts, r_sp, metric.regret(m_ps, optmodel, ld_te)


print("\nPyEPO TSP (n=8) | normalized regret (PyEPO metric, Gurobi), 2 seeds", flush=True)
print(f"{'deg':>4} | {'two-stage':>16} {'SPO+ (PyEPO)':>16} {'PolyStep':>16}", flush=True)
for deg in [1, 2, 4]:
    R = np.array([run(deg, s) for s in (42, 43)]); m, sd = R.mean(0), R.std(0)
    print(f"{deg:>4} | {m[0]:>7.4f}±{sd[0]:<7.4f} {m[1]:>7.4f}±{sd[1]:<7.4f} {m[2]:>7.4f}±{sd[2]:<7.4f}", flush=True)
