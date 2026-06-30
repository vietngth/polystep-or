"""Phase 2 (2nd fair benchmark): PyEPO portfolio (variance-constrained Markowitz).
max r^T w  s.t. sum w = 1, w>=0, w^T Sigma w <= rho.  SPO+ applies (linear objective).
Batched forward solver = Lagrangian bisection over the variance multiplier; verified vs Gurobi.
"""
import sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "polystep/src")
from torch.utils.data import DataLoader
from pyepo.data import portfolio
from pyepo.model.grb import portfolioModel
from pyepo.data.dataset import optDataset
from pyepo.func import SPOPlus
from pyepo import metric
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon

dev = "cuda"; NA = 20; PF = 5
torch.manual_seed(0)

# generate covariance + data; build model
cov, x0, r0 = portfolio.genData(1100, PF, NA, deg=1, noise_level=1, seed=42)
print("genData shapes:", "cov", np.shape(cov), "x", np.shape(x0), "r", np.shape(r0))
optmodel = portfolioModel(num_assets=NA, covariance=cov, gamma=2.25)
RHO = float(optmodel.risk_level)
Sig = torch.tensor(cov, dtype=torch.float32, device=dev)
print("risk_level rho =", RHO)


def proj_simplex(v):
    n = v.shape[-1]
    u, _ = torch.sort(v, dim=-1, descending=True)
    css = u.cumsum(-1) - 1.0
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    rho = ((u - css / ind) > 0).sum(-1, keepdim=True).clamp(min=1)
    theta = css.gather(-1, rho - 1) / rho
    return (v - theta).clamp(min=0)


_LMAX = float(torch.linalg.eigvalsh(Sig)[-1])                    # spectral norm of Sigma

def solve_portfolio(r, rho=None, bis=34, pg=80, hi0=1e7):
    """max r^T w s.t. simplex, w^T Sigma w <= rho. Bisection over the variance multiplier;
    step adapted to the per-lambda Lipschitz constant 2*lambda*lambda_max(Sigma)."""
    if rho is None: rho = RHO
    M, n = r.shape
    lmax = float(torch.linalg.eigvalsh(Sig)[-1])
    rscale = r.abs().mean() + 1e-9
    lo = torch.zeros(M, 1, device=r.device); hi = torch.full((M, 1), hi0, device=r.device)
    w = torch.full((M, n), 1.0 / n, device=r.device)
    for _ in range(bis):
        mid = (lo + hi) / 2
        lr = (1.0 / (2.0 * mid * lmax + rscale)).clamp(max=1.0)    # (M,1) adaptive step
        for _ in range(pg):
            w = proj_simplex(w + lr * (r - 2 * mid * (w @ Sig)))
        var = (w @ Sig * w).sum(-1, keepdim=True)
        risky = var > rho
        lo = torch.where(risky, mid, lo); hi = torch.where(risky, hi, mid)
    return w


# verify vs Gurobi on objective value
rr = r0[:64].astype(float)
w_ours = solve_portfolio(torch.tensor(rr, dtype=torch.float32, device=dev))
obj_ours = (w_ours.cpu().numpy() * rr).sum(1)
obj_gur = []
for i in range(64):
    optmodel.setObj(rr[i]); _, o = optmodel.solve(); obj_gur.append(o)
obj_gur = np.array(obj_gur)
rel = np.abs(obj_ours - obj_gur) / (np.abs(obj_gur) + 1e-9)
print(f"portfolio solver vs Gurobi: mean rel obj gap = {rel.mean()*100:.2f}%  max = {rel.max()*100:.2f}%")

if rel.mean() < 0.03:                                            # proceed only if solver is faithful
    def make_lin(): return nn.Linear(PF, NA, bias=False).to(dev)
    def run(deg, seed):
        cov_, x, r = portfolio.genData(1100, PF, NA, deg=deg, noise_level=1, seed=seed)
        om = portfolioModel(num_assets=NA, covariance=cov_, gamma=2.25)
        global Sig, RHO; Sig = torch.tensor(cov_, dtype=torch.float32, device=dev); RHO = float(om.risk_level)
        xtr, rtr, xte, rte = x[:300], r[:300], x[300:1100], r[300:1100]
        ld_tr = DataLoader(optDataset(om, xtr, rtr), batch_size=128, shuffle=True)
        ld_te = DataLoader(optDataset(om, xte, rte), batch_size=256)
        Xtr = torch.tensor(xtr, dtype=torch.float32, device=dev); Rtr = torch.tensor(rtr, dtype=torch.float32, device=dev)
        m_ts = make_lin(); opt = torch.optim.Adam(m_ts.parameters(), 1e-2)
        for _ in range(40):
            for xb, cb, w_, z_ in ld_tr:
                xb, cb = xb.float().to(dev), cb.float().to(dev)
                opt.zero_grad(); ((m_ts(xb) - cb) ** 2).mean().backward(); opt.step()
        r_ts = metric.regret(m_ts, om, ld_te)
        m_sp = make_lin(); opt = torch.optim.Adam(m_sp.parameters(), 1e-2); spo = SPOPlus(om, processes=1)
        for _ in range(50):
            for xb, cb, w_, z_ in ld_tr:
                xb, cb, w_, z_ = [t.float().to(dev) for t in (xb, cb, w_, z_)]
                opt.zero_grad(); spo(m_sp(xb), cb, w_, z_).mean().backward(); opt.step()
        r_sp = metric.regret(m_sp, om, ld_te)
        Rs = Rtr / Rtr.std()
        m_ps = make_lin();  m_ps.load_state_dict(m_ts.state_dict())
        pso = PolyStepOptimizer(m_ps, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                                step_radius=0.4, probe_radius=0.8, num_probe=1, seed=seed,
                                use_momentum=True, momentum_init=0.5, momentum_final=0.9)
        def closure(bp):
            rhat = torch.einsum("nef,bf->nbe", bp["weight"], Xtr); N, nb, E = rhat.shape
            w = solve_portfolio(rhat.reshape(N * nb, E), bis=20, pg=35).reshape(N, nb, E)
            return -(w * Rs.unsqueeze(0)).sum(-1).mean(-1)
        for _ in range(120): pso.step(closure)
        return r_ts, r_sp, metric.regret(m_ps, om, ld_te)
    print("\nPyEPO portfolio (variance-constrained, 20 assets) | normalized regret, 3 seeds")
    print(f"{'deg':>4} | {'two-stage':>16} {'SPO+ (PyEPO)':>16} {'PolyStep':>16}", flush=True)
    for deg in [1, 4, 8, 16]:
        R = np.array([run(deg, s) for s in (42, 43, 44)]); m, sd = R.mean(0), R.std(0)
        print(f"{deg:>4} | {m[0]:>7.4f}±{sd[0]:<7.4f} {m[1]:>7.4f}±{sd[1]:<7.4f} {m[2]:>7.4f}±{sd[2]:<7.4f}", flush=True)
else:
    print("solver gap too large -- skipping portfolio run (would need a tighter batched SOCP solver)")
