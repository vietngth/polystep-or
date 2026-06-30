"""Track 3: MILP boundary + warm-start demo (honest compute boundary).

Multi-dimensional knapsack as an EXACT Gurobi MILP (un-batchable). Shows:
  (1) PolyStep WORKS on the MILP (solver-agnostic) -- regret decreases;
  (2) it is THROUGHPUT-BOUND -- one step needs P*V*K*batch sequential Gurobi solves;
  (3) Gurobi warm-starting (MIPStart from a base solution) speeds the near-identical
      candidate solves; quote solves/sec cold vs warm and the wall-clock vs a batched DP.
"""
import sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "polystep/src")
from pyepo.data import knapsack
from pyepo.model.grb import knapsackModel
from pyepo.data.dataset import optDataset
from pyepo import metric
from torch.utils.data import DataLoader
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon

dev = "cuda"; NIT = 16; PF = 5; MDIM = 3
np.random.seed(0); torch.manual_seed(0)

W, x0, c0 = knapsack.genData(2, PF, NIT, dim=MDIM, deg=1, seed=1)
weights = W.astype(int); CAP = (weights.sum(1) * 0.4).astype(int).tolist()
optmodel = knapsackModel(weights=weights, capacity=CAP)
SENSE = optmodel.modelSense
gm = optmodel._model
print(f"MILP: multi-dim knapsack, {NIT} binary vars, {MDIM} constraints", flush=True)

def solve_cold(c):
    optmodel.setObj(c); sol, _ = optmodel.solve(); return np.asarray(sol)

_xvars = optmodel._model.getVars()   # raw Gurobi vars, ordered like the cost vector
def solve_warm(c, base):
    for v, b in zip(_xvars, base):
        v.Start = float(b)
    optmodel.setObj(c); sol, _ = optmodel.solve(); return np.asarray(sol)

# ---- (2)/(3) throughput: cold vs warm on near-identical perturbed objectives ----
base_c = np.random.rand(NIT)
base_sol = solve_cold(base_c)
N = 300
t0 = time.time()
for _ in range(N): solve_cold(base_c + 0.05 * np.random.randn(NIT))
cold = N / (time.time() - t0)
t0 = time.time()
for _ in range(N): solve_warm(base_c + 0.05 * np.random.randn(NIT), base_sol)
warm = N / (time.time() - t0)
print(f"throughput: cold {cold:.0f} solves/s | warm {warm:.0f} solves/s | warm speedup {warm/cold:.2f}x", flush=True)

# ---- (1) PolyStep actually trains on the MILP (small config; Gurobi closure) ----
_, x, c = knapsack.genData(400, PF, NIT, dim=MDIM, deg=4, noise_width=0, seed=42)
xtr, ctr, xte, cte = x[:300], c[:300], x[300:], c[300:]
ds_te = optDataset(optmodel, xte, cte); ld_te = DataLoader(ds_te, batch_size=128)
Xtr = torch.tensor(xtr, dtype=torch.float32, device=dev)
Ctr = torch.tensor(ctr, dtype=torch.float32, device=dev)

# strong-ish two-stage warm start
m_ts = nn.Linear(PF, NIT, bias=False).to(dev); opt = torch.optim.Adam(m_ts.parameters(), 1e-2)
ds_tr = optDataset(optmodel, xtr, ctr); ld_tr = DataLoader(ds_tr, batch_size=128, shuffle=True)
for _ in range(40):
    for xb, cb, wb, zb in ld_tr:
        xb, cb = xb.float().to(dev), cb.float().to(dev)
        opt.zero_grad(); ((m_ts(xb) - cb) ** 2).mean().backward(); opt.step()
r_ts = metric.regret(m_ts, optmodel, ld_te)

m_ps = nn.Linear(PF, NIT, bias=False).to(dev); m_ps.load_state_dict(m_ts.state_dict())
Cs = Ctr / Ctr.std()
pso = PolyStepOptimizer(m_ps, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                        step_radius=0.4, probe_radius=0.8, num_probe=1, seed=0,
                        use_momentum=True, momentum_init=0.5, momentum_final=0.9)
SUB = 16   # subsample training instances per closure (throughput control)
def closure(bp):
    Wc = bp["weight"]; Ncand = Wc.shape[0]
    idx = torch.randperm(Xtr.shape[0])[:SUB]
    vhat = torch.einsum("nef,bf->nbe", Wc, Xtr[idx]).detach().cpu().numpy()   # (Ncand, SUB, NIT)
    cs = Cs[idx].cpu().numpy()
    out = np.empty(Ncand)
    for n in range(Ncand):
        tot = 0.0
        for b in range(SUB):
            sol = solve_cold(vhat[n, b]); tot += float((sol * cs[b]).sum())
        out[n] = -tot / SUB                                                   # maximize value
    return torch.tensor(out, dtype=torch.float32, device=dev)

STEPS = 20
t0 = time.time()
for s in range(STEPS): pso.step(closure)
dt = (time.time() - t0)
r_ps = metric.regret(m_ps, optmodel, ld_te)
print(f"PolyStep on MILP: two-stage regret {r_ts:.4f} -> PolyStep {r_ps:.4f}  "
      f"(works={'YES' if r_ps < r_ts else 'no'})", flush=True)
print(f"wall-clock: {dt/STEPS:.1f} s/step ({STEPS} steps, {SUB} instances/closure); "
      f"a full P*V*batch run would be ~{cold:.0f} solves/s bound", flush=True)
print(f"contrast: the batched-DP knapsack does the same solves at ~80M/s on GPU "
      f"(this MILP: {cold:.0f}/s) -> ~{80e6/cold:.0e}x throughput gap (the un-batchable-solver boundary)", flush=True)
