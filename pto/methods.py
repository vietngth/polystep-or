"""Trainers: two-stage MSE, SPO+ (PyEPO), PolyStep-Regret (gradient-free)."""
from __future__ import annotations
import sys, torch
sys.path.insert(0, "polystep/src")
from polystep import PolyStepOptimizer
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout

DEV = "cuda"


def train_two_stage(problem, epochs=60, lr=1e-2):
    model = problem.predictor()
    Xin, Y = problem.mse_pairs("train")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); ((model(Xin) - Y) ** 2).mean().backward(); opt.step()
    return model


def train_spo_plus(problem, epochs=40, lr=1e-2):
    from pyepo.func import SPOPlus
    assert problem.spo_supported and hasattr(problem, "optmodel")
    model = problem.predictor()
    spop = SPOPlus(problem.optmodel, processes=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, cb, wb, zb in problem.spo_dataset("train"):
            xb, cb, wb, zb = [t.float().to(DEV) for t in (xb, cb, wb, zb)]
            opt.zero_grad(); spop(model(xb), cb, wb, zb).mean().backward(); opt.step()
    return model


def train_polystep(problem, cfg, steps=150, warm=None, val_select=True,
                   subspace_rank=0, seed=0):
    model = problem.predictor()
    if warm is not None:
        model.load_state_dict(warm.state_dict())
    sub = None
    if subspace_rank > 0:
        layout = ParamLayout.from_module(model)
        sub = HybridSubspace.from_layout(layout, rank=subspace_rank)
    pso = PolyStepOptimizer(model, subspace=sub, seed=seed, **cfg)
    closure = problem.polystep_closure(model, "train")
    best = (float("inf"), None)
    for s in range(steps):
        pso.step(closure)
        if val_select and (s % 10 == 0 or s == steps - 1):
            rv = problem.fast_regret(model, "val")
            if rv < best[0]:
                best = (rv, {k: v.detach().clone() for k, v in model.state_dict().items()})
    if val_select and best[1] is not None:
        model.load_state_dict(best[1])
    return model
