"""Track 2b (real data): predict-then-optimize on REAL market returns.

Basket of S&P large-caps; per-period features (trailing return + volatility per asset) ->
predict next-period returns -> variance-constrained Markowitz (PyEPO portfolioModel, SOCP).
Chronological train/test split (no look-ahead). Compares two-stage, SPO+, DPO, IMLE, PFYL,
PolyStep on PyEPO normalized regret + realized out-of-sample return.
"""
import sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "polystep/src")
import yfinance as yf
from torch.utils.data import DataLoader
from pyepo.model.grb import portfolioModel
from pyepo.data.dataset import optDataset
from pyepo import metric
import pyepo.func as F
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from pto.solvers import solve_portfolio_socp

dev = "cuda"
TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "PFE", "KO", "WMT", "CAT", "BA", "NEE", "PG", "VZ"]
NA = len(TICKERS); K = 5   # weekly periods

# ---- real data: weekly returns + features ----
px = yf.download(TICKERS, start="2010-01-01", end="2024-01-01", progress=False, auto_adjust=True)["Close"]
px = px[TICKERS].dropna()
dret = np.log(px).diff().dropna().values                       # daily log returns (T, NA)
T = (dret.shape[0] // K) * K
wret = dret[:T].reshape(-1, K, NA).sum(1)                       # weekly returns (P, NA)
P = wret.shape[0]
# features at period t = [prev-week return, prev-week abs-return (vol proxy)] per asset
X = np.concatenate([wret[:-1], np.abs(wret[:-1])], axis=1)      # (P-1, 2*NA)
C = wret[1:]                                                    # next-week returns (P-1, NA)
PF = X.shape[1]
n_train = int(0.7 * len(X))
xtr, ctr, xte, cte = X[:n_train], C[:n_train], X[n_train:], C[n_train:]
print(f"real data: {NA} assets, {len(X)} weekly periods ({n_train} train / {len(xte)} test), {PF} features", flush=True)

# covariance from TRAIN returns; build PyEPO model
cov = np.cov(ctr.T) + 1e-6 * np.eye(NA)
optmodel = portfolioModel(num_assets=NA, covariance=cov, gamma=2.25)
RHO = float(optmodel.risk_level); SENSE = optmodel.modelSense
Sig = torch.tensor(cov, dtype=torch.float32, device=dev)
def ps_solve(r): return solve_portfolio_socp(r, Sig, RHO, bis=24, pg=50)

# cvxpylayers KKT-layer baseline (paper d's own method): differentiable SOCP layer
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
_wv = cp.Variable(NA); _rp = cp.Parameter(NA)
_kkt_prob = cp.Problem(cp.Maximize(_rp @ _wv),
                       [cp.sum(_wv) == 1, _wv >= 0, cp.quad_form(_wv, cp.psd_wrap(cov)) <= RHO])
_kkt_layer = CvxpyLayer(_kkt_prob, parameters=[_rp], variables=[_wv])

# verify our SOCP solver == Gurobi on a few test return vectors
gap = []
for i in range(min(40, len(cte))):
    w = ps_solve(torch.tensor(cte[i:i+1], dtype=torch.float32, device=dev))
    o_ours = float((w.cpu().numpy()[0] * cte[i]).sum())
    optmodel.setObj(cte[i]); _, o_g = optmodel.solve(); gap.append(abs(o_ours - o_g) / (abs(o_g) + 1e-9))
print(f"portfolio SOCP solver vs Gurobi (real data): mean rel gap {np.mean(gap)*100:.2f}%", flush=True)

ds_tr = optDataset(optmodel, xtr, ctr); ds_te = optDataset(optmodel, xte, cte)
ld_tr = DataLoader(ds_tr, batch_size=64, shuffle=True); ld_te = DataLoader(ds_te, batch_size=128)
Xtr = torch.tensor(xtr, dtype=torch.float32, device=dev); Ctr = torch.tensor(ctr, dtype=torch.float32, device=dev)

def make(): return nn.Linear(PF, NA, bias=False).to(dev)

DFL = {"SPO+": (lambda: F.SPOPlus(optmodel), "loss", ["pred", "c", "w", "z"]),
       "DPO":  (lambda: F.perturbedOpt(optmodel, n_samples=3, sigma=1.0), "opt", None),
       "IMLE": (lambda: F.implicitMLE(optmodel, n_samples=3, sigma=1.0, lambd=10), "opt", None),
       "PFYL": (lambda: F.perturbedFenchelYoung(optmodel, n_samples=3, sigma=1.0), "loss", ["pred", "w"])}

def realized_return(m):
    with torch.no_grad(): rhat = m(torch.tensor(xte, dtype=torch.float32, device=dev))
    w = ps_solve(rhat).cpu().numpy()
    return float((w * cte).sum(1).mean())   # mean weekly out-of-sample return

def train_two_stage():
    m = make(); opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(60):
        for xb, cb, wb, zb in ld_tr:
            xb, cb = xb.float().to(dev), cb.float().to(dev)
            opt.zero_grad(); ((m(xb) - cb) ** 2).mean().backward(); opt.step()
    return m

def train_dfl(name):
    build, kind, fwd = DFL[name]; m = make(); opt = torch.optim.Adam(m.parameters(), 1e-2); lm = build()
    for _ in range(40):
        for xb, cb, wb, zb in ld_tr:
            xb, cb, wb, zb = [t.float().to(dev) for t in (xb, cb, wb, zb)]
            pred = m(xb)
            loss = SENSE * (lm(pred) * cb).sum(-1).mean() if kind == "opt" else \
                   lm(*[{"pred": pred, "c": cb, "w": wb, "z": zb}[a] for a in fwd]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m

def train_cvxpylayer(epochs=40, lr=1e-2):
    m = make(); opt = torch.optim.Adam(m.parameters(), lr)
    for _ in range(epochs):
        pred = m(Xtr); (wsol,) = _kkt_layer(pred.cpu().double())
        loss = -(wsol.to(dev).float() * Ctr).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m

def train_polystep(warm):
    m = make(); m.load_state_dict(warm.state_dict())
    Cs = Ctr / Ctr.std()
    pso = PolyStepOptimizer(m, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=0.4, probe_radius=0.8, num_probe=1, seed=0,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    def closure(bp):
        rhat = torch.einsum("nef,bf->nbe", bp["weight"], Xtr); N, nb, E = rhat.shape
        w = ps_solve(rhat.reshape(N * nb, E)).reshape(N, nb, E)
        return -(w * Cs.unsqueeze(0)).sum(-1).mean(-1)
    for _ in range(120): pso.step(closure)
    return m

print(f"\n{'method':>10} | {'norm regret':>11} | {'OOS weekly ret':>14}", flush=True)
ts = train_two_stage()
print(f"{'two-stage':>10} | {metric.regret(ts, optmodel, ld_te):>11.4f} | {realized_return(ts)*100:>13.3f}%", flush=True)
for name in DFL:
    try:
        m = train_dfl(name)
        print(f"{name:>10} | {metric.regret(m, optmodel, ld_te):>11.4f} | {realized_return(m)*100:>13.3f}%", flush=True)
    except Exception as e:
        print(f"{name:>10} | ERROR {str(e)[:50]}", flush=True)
try:
    mk = train_cvxpylayer()
    print(f"{'cvxpylayer':>10} | {metric.regret(mk, optmodel, ld_te):>11.4f} | {realized_return(mk)*100:>13.3f}%", flush=True)
except Exception as e:
    print(f"{'cvxpylayer':>10} | ERROR {str(e)[:50]}", flush=True)
mp = train_polystep(ts)
print(f"{'PolyStep':>10} | {metric.regret(mp, optmodel, ld_te):>11.4f} | {realized_return(mp)*100:>13.3f}%", flush=True)
