"""Predict-then-optimize problems in two settings:

  (A) OBJECTIVE prediction  (SPO+ applies):
        ShortestPath     -- HxW grid, edge costs (PyEPO-backed)
        TerrainSP (#4)   -- image -> per-cell costs via CNN, shortest path
  (B) CONSTRAINT prediction (SPO+ = N/A):
        MDKPConsumption  -- multi-dim knapsack, predicted resource consumption

Common interface:
  .predictor()                         -> fresh nn.Module
  .polystep_closure(model, split)      -> closure(bp)->(N,) training loss
  .fast_regret(model, split)           -> normalized regret via our GPU solver (val-select)
  .regret(model, split)                -> headline normalized regret (PyEPO/Gurobi if available)
  .spo_supported, .spo_batch(...)      -> SPO+ hooks where applicable
"""
from __future__ import annotations
import math, numpy as np, torch, torch.nn as nn
from .solvers import (build_dag_solver, grid_arcs, mdkp_greedy, mdkp_repair,
                      knap1_dp, knap1_repair)
from .forward import batched_predict

DEV = "cuda"


def make_mlp(p, out, hidden):
    if hidden and hidden > 0:
        return nn.Sequential(nn.Linear(p, hidden), nn.ReLU(),
                             nn.Linear(hidden, out, bias=False)).to(DEV)
    return nn.Linear(p, out, bias=False).to(DEV)


# ===========================================================================
# (A1) Shortest path -- objective prediction, PyEPO-backed
# ===========================================================================
class ShortestPath:
    spo_supported = True

    def __init__(self, H=5, W=5, p_feat=5, deg=6, n_train=800, n_val=200,
                 n_test=200, hidden=0, seed=42):
        from pyepo.data import shortestpath
        from pyepo.model.grb import shortestPathModel
        from pyepo.data.dataset import optDataset
        self.H, self.W, self.p_feat, self.hidden = H, W, p_feat, hidden
        self.optmodel = shortestPathModel((H, W))
        arcs = list(self.optmodel.arcs)                  # PyEPO's cost-vector arc order
        self.E = len(arcs); self.solve_batch = build_dag_solver(arcs, H * W, 0, H * W - 1)
        ntot = n_train + n_val + n_test
        x, c = shortestpath.genData(ntot, p_feat, (H, W), deg=deg, noise_width=0, seed=seed)
        sl = {"train": slice(0, n_train), "val": slice(n_train, n_train + n_val),
              "test": slice(n_train + n_val, ntot)}
        self.X = {k: torch.tensor(x[v], dtype=torch.float32, device=DEV) for k, v in sl.items()}
        self.C = {k: torch.tensor(c[v], dtype=torch.float32, device=DEV) for k, v in sl.items()}
        self.x_np = {k: x[v] for k, v in sl.items()}; self.c_np = {k: c[v] for k, v in sl.items()}
        self._optDataset = optDataset
        self.zstar = {k: (self.solve_batch(self.C[k]) * self.C[k]).sum(-1) for k in self.C}
        mu, sd = self.C["train"].mean(), self.C["train"].std()
        self.Cs = {k: (self.C[k] - mu) / sd for k in self.C}

    def predictor(self): return make_mlp(self.p_feat, self.E, self.hidden)

    def mse_pairs(self, split): return self.X[split], self.C[split]

    def polystep_closure(self, model, split="train"):
        X, Cs = self.X[split], self.Cs[split]; sb = self.solve_batch
        def closure(bp):
            chat = batched_predict(model, bp, X); N, nb, E = chat.shape
            w = sb(chat.reshape(N * nb, E)).reshape(N, nb, E)
            return (w * Cs.unsqueeze(0)).sum(-1).mean(-1)
        return closure

    def fast_regret(self, model, split="test"):
        with torch.no_grad(): chat = model(self.X[split])
        real = (self.solve_batch(chat) * self.C[split]).sum(-1)
        return ((real - self.zstar[split]).sum() / self.zstar[split].sum()).item()

    def regret(self, model, split="test"):
        from pyepo import metric
        from torch.utils.data import DataLoader
        ds = self._optDataset(self.optmodel, self.x_np[split], self.c_np[split])
        return metric.regret(model, self.optmodel, DataLoader(ds, batch_size=256))

    # SPO+ training data for one split (PyEPO autograd loss)
    def spo_dataset(self, split):
        from torch.utils.data import DataLoader
        ds = self._optDataset(self.optmodel, self.x_np[split], self.c_np[split])
        return DataLoader(ds, batch_size=128, shuffle=(split == "train"))


# ===========================================================================
# (A2) Terrain shortest path -- IMAGE -> per-cell costs via CNN (#4)
# Warcraft-style: each cell has a hidden terrain type with a base cost; an HxW
# tile image is rendered; a CNN predicts the HxW cost map; monotone shortest
# path (edge cost = head-cell cost). SPO+ applies (cost/objective prediction).
# ===========================================================================
class TerrainSP:
    spo_supported = True

    def __init__(self, H=12, W=12, ps=4, n_type=5, p_noise=0.25,
                 n_train=800, n_val=200, n_test=400, seed=0):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.H, self.W, self.ps = H, W, ps
        self.arcs, NN, s, t = grid_arcs(H, W)
        self.E = len(self.arcs); self.solve_batch = build_dag_solver(self.arcs, NN, s, t)
        # edge -> head cell index (cost of an arc (u,v) is the cost of cell v)
        self.head = torch.tensor([v for (u, v) in self.arcs], device=DEV)
        self.type_cost = torch.linspace(1.0, 9.0, n_type, device=DEV)        # base costs
        # type color signatures (n_type, 3)
        self.color = torch.rand(n_type, 3, generator=g, device=DEV)
        def gen(n, sd):
            gg = torch.Generator(device=DEV).manual_seed(sd)
            types = torch.randint(0, n_type, (n, H, W), generator=gg, device=DEV)
            cellcost = self.type_cost[types]                                  # (n,H,W) true cell cost
            img = self.color[types].permute(0, 3, 1, 2)                       # (n,3,H,W)
            img = img.repeat_interleave(ps, 2).repeat_interleave(ps, 3)       # (n,3,H*ps,W*ps)
            img = img + p_noise * torch.randn(img.shape, generator=gg, device=DEV)
            edgecost = cellcost.reshape(n, -1)[:, self.head]                  # (n,E) true edge cost
            return img, edgecost, cellcost.reshape(n, -1)
        self.img, self.C, self.cell = {}, {}, {}
        for k, (nn_, sd) in {"train": (n_train, 10*seed+1), "val": (n_val, 10*seed+2),
                             "test": (n_test, 10*seed+3)}.items():
            self.img[k], self.C[k], self.cell[k] = gen(nn_, sd)
        self.zstar = {k: (self.solve_batch(self.C[k]) * self.C[k]).sum(-1) for k in self.C}

    def predictor(self):
        H, W = self.H, self.W
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((H, W)),                # downsample to grid resolution
            nn.Conv2d(16, 1, 1), nn.Flatten(),           # -> (batch, H*W) cell costs
        ).to(DEV)

    def mse_pairs(self, split): return self.img[split], self.cell[split]

    def _edge_from_cells(self, cellcost):                 # (..., H*W) -> (..., E)
        return cellcost[..., self.head]

    def polystep_closure(self, model, split="train"):
        IMG = self.img[split]; zsum = self.zstar[split].sum(); C = self.C[split]
        sb = self.solve_batch; head = self.head
        def closure(bp):
            cells = batched_predict(model, bp, IMG)        # (N, batch, H*W)
            chat = cells[..., head]                        # (N, batch, E)
            N, nb, E = chat.shape
            w = sb(chat.reshape(N * nb, E)).reshape(N, nb, E)
            return (w * C.unsqueeze(0)).sum(-1).sum(-1) / zsum   # normalized realized cost
        return closure

    def fast_regret(self, model, split="test"):
        with torch.no_grad(): cells = model(self.img[split])
        chat = self._edge_from_cells(cells)
        real = (self.solve_batch(chat) * self.C[split]).sum(-1)
        return ((real - self.zstar[split]).sum() / self.zstar[split].sum()).item()

    regret = fast_regret  # no Gurobi needed; our DAG solver is exact


# ===========================================================================
# (B1) Single-constraint knapsack, predicted WEIGHTS (SPO+ = N/A) -- exact DP.
# The headline clear-advantage problem: binding capacity + exact solver make the
# overflow->repair asymmetry sharp, so decision-aware (conservative) prediction
# beats MSE. Predictor has a bias so a conservative margin is expressible.
# ===========================================================================
class KnapsackWeights:
    spo_supported = False

    def __init__(self, n_item=20, p_feat=5, C=40, wmax=14, deg=6,
                 n_train=256, n_val=256, n_test=2000, hidden=0, seed=0):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.n, self.C, self.wmax, self.p_feat, self.hidden = n_item, C, wmax, p_feat, hidden
        self.Bstar = (torch.rand(n_item, p_feat, generator=g, device=DEV) < 0.5).float()
        self.v = torch.randint(2, 12, (n_item,), generator=g, device=DEV).float()
        def gen(n, sd):
            gg = torch.Generator(device=DEV).manual_seed(sd)
            X = torch.randn(n, p_feat, generator=gg, device=DEV)
            raw = (self.Bstar @ X.T).T / math.sqrt(p_feat)
            w = ((raw + 3.0).pow(deg)); w = (w / w.mean() * 6.0).round().clamp(1, wmax)
            return X, w
        self.X, self.W = {}, {}
        for k, (nn_, sd) in {"train": (n_train, 10*seed+1), "val": (n_val, 10*seed+2),
                             "test": (n_test, 10*seed+3)}.items():
            self.X[k], self.W[k] = gen(nn_, sd)
        self.Vstar = {k: knap1_dp(self.v.expand(self.X[k].shape[0], -1), self.W[k], C,
                                  want_sel=False)[0] for k in self.X}

    def predictor(self):
        return (nn.Sequential(nn.Linear(self.p_feat, self.hidden), nn.ReLU(),
                              nn.Linear(self.hidden, self.n)).to(DEV)
                if self.hidden else nn.Linear(self.p_feat, self.n, bias=True).to(DEV))

    def mse_pairs(self, split): return self.X[split], self.W[split]

    def _deploy(self, pred_w, w_true):
        M = pred_w.shape[0]; V = self.v.expand(M, -1)
        pw = pred_w.round().clamp(1, self.wmax)
        _, sel = knap1_dp(V, pw, self.C)
        return knap1_repair(sel, V, w_true, self.C)

    def polystep_closure(self, model, split="train"):
        X = self.X[split]; Vstar = self.Vstar[split]; W = self.W[split]; nb = X.shape[0]
        def closure(bp):
            pred = batched_predict(model, bp, X); N, _, od = pred.shape
            Wt = W.unsqueeze(0).expand(N, nb, self.n).reshape(N * nb, self.n)
            real = self._deploy(pred.reshape(N * nb, od), Wt).reshape(N, nb)
            return (Vstar.unsqueeze(0) - real).mean(-1)
        return closure

    def fast_regret(self, model, split="test"):
        with torch.no_grad(): pred = model(self.X[split])
        real = self._deploy(pred, self.W[split])
        return ((self.Vstar[split] - real).sum() / self.Vstar[split].sum()).item()

    regret = fast_regret


# ===========================================================================
# (B2) Multi-dimensional knapsack, predicted CONSUMPTION (SPO+ = N/A)
# ===========================================================================
class MDKPConsumption:
    spo_supported = False

    def __init__(self, n_item=40, m_res=5, p_feat=5, deg=6, fill=0.35,
                 n_train=256, n_val=256, n_test=2000, hidden=0, seed=0):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.n, self.m, self.p_feat, self.hidden = n_item, m_res, p_feat, hidden
        self.out_dim = m_res * n_item
        self.Bstar = (torch.rand(self.out_dim, p_feat, generator=g, device=DEV) < 0.5).float()
        self.v = torch.randint(2, 12, (n_item,), generator=g, device=DEV).float()
        def gen(n, sd):
            gg = torch.Generator(device=DEV).manual_seed(sd)
            X = torch.randn(n, p_feat, generator=gg, device=DEV)
            raw = (self.Bstar @ X.T).T / math.sqrt(p_feat)
            A = ((raw + 3.0).pow(deg)); A = (A / A.mean() * 3.0 + 0.5)
            return X, A.reshape(n, m_res, n_item)
        self.X, self.A = {}, {}
        for k, (nn_, sd) in {"train": (n_train, 10*seed+1), "val": (n_val, 10*seed+2),
                             "test": (n_test, 10*seed+3)}.items():
            self.X[k], self.A[k] = gen(nn_, sd)
        self.b = self.A["train"].sum(-1).mean(0) * fill
        self.Vstar = {k: (mdkp_greedy(self.v.expand(self.X[k].shape[0], -1), self.A[k], self.b).float()
                          * self.v).sum(-1) for k in self.X}

    def predictor(self):
        return (nn.Sequential(nn.Linear(self.p_feat, self.hidden), nn.ReLU(),
                              nn.Linear(self.hidden, self.out_dim)).to(DEV)
                if self.hidden else nn.Linear(self.p_feat, self.out_dim, bias=True).to(DEV))

    def mse_pairs(self, split):
        A = self.A[split]; return self.X[split], A.reshape(A.shape[0], -1)

    def _deploy(self, pred_flat, A_true):
        """pred_flat (M,out_dim) predicted consumption; A_true (M,m,n) -> realized (M,)."""
        M = pred_flat.shape[0]
        Apred = pred_flat.reshape(M, self.m, self.n).clamp(min=0.1)
        V = self.v.expand(M, -1)
        sel = mdkp_greedy(V, Apred, self.b)
        return mdkp_repair(sel, V, A_true, self.b)

    def polystep_closure(self, model, split="train"):
        X = self.X[split]; Vstar = self.Vstar[split]; A = self.A[split]; nb = X.shape[0]
        def closure(bp):
            pred = batched_predict(model, bp, X); N, _, od = pred.shape
            At = A.unsqueeze(0).expand(N, nb, self.m, self.n).reshape(N * nb, self.m, self.n)
            real = self._deploy(pred.reshape(N * nb, od), At).reshape(N, nb)
            return (Vstar.unsqueeze(0) - real).mean(-1)
        return closure

    def fast_regret(self, model, split="test"):
        with torch.no_grad(): pred = model(self.X[split])
        real = self._deploy(pred, self.A[split])
        return ((self.Vstar[split] - real).sum() / self.Vstar[split].sum()).item()

    regret = fast_regret
