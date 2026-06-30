"""VALIDATION against the standard library (PyEPO 2.2.2).

Same data (PyEPO genData), same downstream model (PyEPO shortestPathModel / Gurobi),
same metric (PyEPO normalized regret). Compares:
  - two-stage MSE        (linear + Adam)
  - SPO+                 (PyEPO's authoritative SPOPlus autograd loss + Adam)
  - PolyStep-Regret      (gradient-free, trained with our batched GPU DAG solver)

Confirms our hand-rolled POC numbers hold under the reference implementation, and
that PolyStep trained with the GPU-batched solver is competitive with PyEPO's SPO+
as measured by PyEPO's own regret metric.
"""
import sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "polystep/src")
from torch.utils.data import DataLoader
from pyepo.data import shortestpath
from pyepo.model.grb import shortestPathModel
from pyepo.data.dataset import optDataset
from pyepo.func import SPOPlus
from pyepo import metric
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon

dev = "cuda"; GRID = (5, 5); P_FEAT = 5
torch.manual_seed(0)

# ---- generic batched DAG shortest-path solver aligned to PyEPO's arc order ----
def build_solver(arcs, num_nodes, source, sink):
    out_by_node = [[] for _ in range(num_nodes)]
    for e, (u, v) in enumerate(arcs):
        out_by_node[u].append((v, e))
    E = len(arcs)
    def solve_batch(c):                       # c (M,E) -> w (M,E) one-hot path
        M = c.shape[0]; INF = float("inf")
        dist = torch.full((M, num_nodes), INF, device=c.device); dist[:, source] = 0.0
        pe = torch.full((M, num_nodes), -1, dtype=torch.long, device=c.device)
        pn = torch.full((M, num_nodes), -1, dtype=torch.long, device=c.device)
        for u in range(num_nodes):
            for (v, e) in out_by_node[u]:
                nd = dist[:, u] + c[:, e]; better = nd < dist[:, v]
                dist[:, v] = torch.where(better, nd, dist[:, v])
                pe[:, v] = torch.where(better, torch.full_like(pe[:, v], e), pe[:, v])
                pn[:, v] = torch.where(better, torch.full_like(pn[:, v], u), pn[:, v])
        w = torch.zeros((M, E), device=c.device)
        cur = torch.full((M,), sink, dtype=torch.long, device=c.device)
        midx = torch.arange(M, device=c.device)
        for _ in range(num_nodes):
            active = cur != source
            if not active.any(): break
            e = pe[midx, cur]
            w[midx[active], e[active]] = 1.0
            cur = torch.where(active, pn[midx, cur], cur)
        return w
    return solve_batch

optmodel = shortestPathModel(GRID)
NN = GRID[0] * GRID[1]
solve_batch = build_solver(optmodel.arcs, NN, 0, NN - 1)

# validate our solver vs PyEPO's Gurobi on random costs
cc = np.random.rand(64, 40)
wb = solve_batch(torch.tensor(cc, device=dev, dtype=torch.float32))
ours = (wb.cpu().numpy() * cc).sum(1)
gur = []
for i in range(64):
    optmodel.setObj(cc[i]); _, z = optmodel.solve(); gur.append(z)
print(f"solver check vs PyEPO/Gurobi: max|ours-gurobi| = {np.abs(ours - np.array(gur)).max():.2e}")


def make_linear():
    return nn.Linear(P_FEAT, 40, bias=False).to(dev)

def eval_regret(model, loader):
    return metric.regret(model, optmodel, loader)

def run(deg, n_data=1100, n_train=900, epochs=40, ps_steps=150):
    x, c = shortestpath.genData(n_data, P_FEAT, GRID, deg=deg, noise_width=0, seed=42)
    xtr, ctr, xte, cte = x[:n_train], c[:n_train], x[n_train:], c[n_train:]
    ds_tr = optDataset(optmodel, xtr, ctr); ds_te = optDataset(optmodel, xte, cte)
    ld_tr = DataLoader(ds_tr, batch_size=128, shuffle=True)
    ld_te = DataLoader(ds_te, batch_size=256, shuffle=False)
    Xtr = torch.tensor(xtr, dtype=torch.float32, device=dev)
    Ctr = torch.tensor(ctr, dtype=torch.float32, device=dev)

    # two-stage MSE
    m_ts = make_linear(); opt = torch.optim.Adam(m_ts.parameters(), lr=1e-2)
    for _ in range(epochs):
        for xb, cb, wb_, zb in ld_tr:
            xb, cb = xb.float().to(dev), cb.float().to(dev)
            opt.zero_grad(); loss = ((m_ts(xb) - cb) ** 2).mean(); loss.backward(); opt.step()
    r_ts = eval_regret(m_ts, ld_te)

    # SPO+ (PyEPO authoritative)
    m_spo = make_linear(); opt = torch.optim.Adam(m_spo.parameters(), lr=1e-2)
    spop = SPOPlus(optmodel, processes=1)
    for _ in range(epochs):
        for xb, cb, wb_, zb in ld_tr:
            xb, cb = xb.float().to(dev), cb.float().to(dev)
            wb_, zb = wb_.float().to(dev), zb.float().to(dev)
            opt.zero_grad(); loss = spop(m_spo(xb), cb, wb_, zb).mean()
            loss.backward(); opt.step()
    r_spo = eval_regret(m_spo, ld_te)

    # PolyStep: minimize realized cost via our GPU-batched solver (warm-start from two-stage)
    mu, sd = Ctr.mean(), Ctr.std(); Ctr_s = (Ctr - mu) / sd
    m_ps = make_linear()
    with torch.no_grad(): m_ps.weight.copy_(m_ts.weight)
    pso = PolyStepOptimizer(m_ps, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=0.4, probe_radius=0.8, num_probe=1, seed=0,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    def closure(bp):
        chat = torch.einsum("nef,bf->nbe", bp["weight"], Xtr); N, nb, E = chat.shape
        w = solve_batch(chat.reshape(N * nb, E)).reshape(N, nb, E)
        return (w * Ctr_s.unsqueeze(0)).sum(-1).mean(-1)
    t0 = time.time()
    for _ in range(ps_steps): pso.step(closure)
    dt = time.time() - t0
    r_ps = eval_regret(m_ps, ld_te)
    return r_ts, r_spo, r_ps, dt

print("\nPyEPO 5x5 shortest path | normalized regret (PyEPO metric, Gurobi solves)")
print(f"{'deg':>4} | {'two-stage':>10} {'SPO+ (PyEPO)':>13} {'PolyStep':>10} | {'PS train s':>10}")
print("-" * 60)
for deg in [1, 4, 6, 8]:
    r_ts, r_spo, r_ps, dt = run(deg)
    print(f"{deg:>4} | {r_ts:>10.4f} {r_spo:>13.4f} {r_ps:>10.4f} | {dt:>10.1f}")
