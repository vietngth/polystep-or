"""Phase 1 problems: regimes where PolyStep's edge over SPO+/two-stage is real.

  Newsvendor (asymmetric cost)  -- costly under/over, decision-optimal = critical
                                    fractile (a quantile), NOT the MSE mean.
  Portfolio mean-variance (QP)  -- NONLINEAR objective; SPO+ is derived for linear
                                    objectives, so it does not directly apply.

Both: linear predictor, gradient-free PolyStep minimizing the TRUE decision cost,
vs a strong Adam two-stage MSE. Portfolio uses a batched simplex-projected-gradient
QP forward solver (GPU).
"""
from __future__ import annotations
import sys, math, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "polystep/src")
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon

dev = "cuda"


# --------------------------------------------------------------------------
# batched Euclidean projection onto the probability simplex {w>=0, sum w = 1}
# --------------------------------------------------------------------------
def proj_simplex(v):
    n = v.shape[-1]
    u, _ = torch.sort(v, dim=-1, descending=True)
    css = u.cumsum(-1) - 1.0
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = (u - css / ind) > 0
    rho = cond.sum(-1, keepdim=True).clamp(min=1)
    theta = css.gather(-1, rho - 1) / rho
    return (v - theta).clamp(min=0)


def portfolio_solve(rhat, Sigma, gamma, iters=80, lr=0.1):
    """max rhat^T w - gamma w^T Sigma w  s.t. w in simplex. rhat (M,n) -> w (M,n)."""
    M, n = rhat.shape
    w = torch.full((M, n), 1.0 / n, device=rhat.device, dtype=rhat.dtype)
    for _ in range(iters):
        grad = rhat - 2 * gamma * (w @ Sigma)
        w = proj_simplex(w + lr * grad)
    return w


def gen_features(n, p, sd):
    g = torch.Generator(device=dev).manual_seed(sd)
    return torch.randn(n, p, generator=g, device=dev)


# ==========================================================================
# Newsvendor: n items, asymmetric holding (h) / stockout (b) costs.
# ==========================================================================
def newsvendor(deg=4, n_item=20, p=5, h=1.0, b=9.0, n_train=256, n_val=256,
               n_test=2000, steps=200, seeds=(0, 1, 2)):
    cuts, costs_ts, costs_ps = [], [], []
    for s in seeds:
        g = torch.Generator(device=dev).manual_seed(s)
        Bstar = (torch.rand(n_item, p, generator=g, device=dev) < 0.5).float()
        def demand(X):
            raw = (Bstar @ X.T).T / math.sqrt(p)
            d = ((raw + 3.0).pow(deg)); return d / d.mean() * 10.0     # positive demand ~10
        Xtr, Xva, Xte = (gen_features(nn_, p, 10*s+i) for i, nn_ in
                         enumerate((n_train, n_val, n_test), 1))
        dtr, dva, dte = demand(Xtr), demand(Xva), demand(Xte)
        def cost(q, d):                                                # newsvendor cost
            return (h * (q - d).clamp(min=0) + b * (d - q).clamp(min=0)).mean()
        def nv_regret(m, X, d):
            with torch.no_grad(): q = m(X).clamp(min=0)
            return cost(q, d).item()
        # strong Adam two-stage MSE (predicts the MEAN demand)
        m_ts = nn.Linear(p, n_item, bias=True).to(dev); opt = torch.optim.Adam(m_ts.parameters(), 1e-2)
        for _ in range(200):
            opt.zero_grad(); ((m_ts(Xtr) - dtr) ** 2).mean().backward(); opt.step()
        costs_ts.append(nv_regret(m_ts, Xte, dte))
        # PolyStep on TRUE newsvendor cost (learns the critical-fractile quantile)
        m_ps = nn.Linear(p, n_item, bias=True).to(dev); m_ps.load_state_dict(m_ts.state_dict())
        pso = PolyStepOptimizer(m_ps, polytope_type="orthoplex", epsilon=CosineEpsilon(0.6, 0.05),
                                step_radius=0.5, probe_radius=1.0, num_probe=1, seed=s,
                                use_momentum=True, momentum_init=0.5, momentum_final=0.9)
        def closure(bp):
            q = (torch.einsum("nof,bf->nbo", bp["weight"], Xtr) + bp["bias"].unsqueeze(1)).clamp(min=0)
            d = dtr.unsqueeze(0)
            return (h * (q - d).clamp(min=0) + b * (d - q).clamp(min=0)).mean(dim=(1, 2))
        best = (float("inf"), None)
        for st in range(steps):
            pso.step(closure)
            if st % 10 == 0 or st == steps - 1:
                rv = nv_regret(m_ps, Xva, dva)
                if rv < best[0]: best = (rv, {k: v.clone() for k, v in m_ps.state_dict().items()})
        m_ps.load_state_dict(best[1]); costs_ps.append(nv_regret(m_ps, Xte, dte))
    a, c = np.mean(costs_ts), np.mean(costs_ps)
    return a, c, (a - c) / a * 100


# ==========================================================================
# Portfolio mean-variance (quadratic objective -> SPO+ N/A).
# ==========================================================================
def portfolio(deg=4, n_asset=15, p=5, gamma=2.0, n_train=256, n_val=256,
              n_test=2000, steps=150, seeds=(0, 1, 2)):
    regs_ts, regs_ps = [], []
    for s in seeds:
        g = torch.Generator(device=dev).manual_seed(s)
        Bstar = (torch.rand(n_asset, p, generator=g, device=dev) < 0.5).float()
        L = torch.randn(n_asset, n_asset, generator=g, device=dev) / math.sqrt(n_asset)
        Sigma = (L @ L.T + 0.5 * torch.eye(n_asset, device=dev))      # known PSD covariance
        def returns(X):
            raw = (Bstar @ X.T).T / math.sqrt(p)
            return ((raw + 3.0).pow(deg)) / 50.0                       # positive-ish returns
        Xtr, Xva, Xte = (gen_features(nn_, p, 10*s+i) for i, nn_ in
                         enumerate((n_train, n_val, n_test), 1))
        rtr, rva, rte = returns(Xtr), returns(Xva), returns(Xte)
        def value(w, r): return (r * w).sum(-1) - gamma * (w @ Sigma * w).sum(-1)
        def opt_value(r): return value(portfolio_solve(r, Sigma, gamma), r)
        def pf_regret(m, X, r):
            with torch.no_grad(): rhat = m(X)
            w = portfolio_solve(rhat, Sigma, gamma)
            vstar = opt_value(r)
            return ((vstar - value(w, r)).sum() / vstar.sum()).item()
        # strong Adam two-stage MSE (predicts returns)
        m_ts = nn.Linear(p, n_asset, bias=True).to(dev); opt = torch.optim.Adam(m_ts.parameters(), 1e-2)
        for _ in range(200):
            opt.zero_grad(); ((m_ts(Xtr) - rtr) ** 2).mean().backward(); opt.step()
        regs_ts.append(pf_regret(m_ts, Xte, rte))
        # PolyStep on TRUE risk-adjusted regret (nonlinear objective)
        m_ps = nn.Linear(p, n_asset, bias=True).to(dev); m_ps.load_state_dict(m_ts.state_dict())
        vstar_tr = opt_value(rtr)
        pso = PolyStepOptimizer(m_ps, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                                step_radius=0.4, probe_radius=0.8, num_probe=1, seed=s,
                                use_momentum=True, momentum_init=0.5, momentum_final=0.9)
        def closure(bp):
            rhat = torch.einsum("nof,bf->nbo", bp["weight"], Xtr) + bp["bias"].unsqueeze(1)
            N, nb, na = rhat.shape
            w = portfolio_solve(rhat.reshape(N * nb, na), Sigma, gamma).reshape(N, nb, na)
            val = (rtr.unsqueeze(0) * w).sum(-1) - gamma * (w @ Sigma * w).sum(-1)
            return (vstar_tr.unsqueeze(0) - val).mean(-1)
        best = (float("inf"), None)
        for st in range(steps):
            pso.step(closure)
            if st % 10 == 0 or st == steps - 1:
                rv = pf_regret(m_ps, Xva, rva)
                if rv < best[0]: best = (rv, {k: v.clone() for k, v in m_ps.state_dict().items()})
        m_ps.load_state_dict(best[1]); regs_ps.append(pf_regret(m_ps, Xte, rte))
    a, c = np.mean(regs_ts), np.mean(regs_ps)
    return a, c, (a - c) / a * 100 if a > 1e-9 else 0


if __name__ == "__main__":
    print("PHASE 1 -- regimes where SPO+ can't go and decision-focus pays off (3 seeds)\n")
    print("Newsvendor (asymmetric cost h=1,b=9; decision-opt = 0.9 quantile, not MSE mean):")
    print(f"{'deg':>4} | {'two-stage cost':>15} {'PolyStep cost':>14} | {'cut':>6}")
    for deg in [2, 4, 6]:
        a, c, cut = newsvendor(deg=deg)
        print(f"{deg:>4} | {a:>15.4f} {c:>14.4f} | {cut:>+5.0f}%", flush=True)
    print("\nPortfolio mean-variance (QUADRATIC objective; SPO+ = N/A), normalized regret:")
    print(f"{'deg':>4} | {'two-stage':>15} {'PolyStep':>14} | {'cut':>6}")
    for deg in [2, 4, 6]:
        a, c, cut = portfolio(deg=deg)
        print(f"{deg:>4} | {a:>15.4f} {c:>14.4f} | {cut:>+5.0f}%", flush=True)
