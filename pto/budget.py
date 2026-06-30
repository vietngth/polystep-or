"""Uniform compute accounting for the solve-budget Pareto (#2) and the matched-budget ablation (#5).

The honest "cost" of a DFL trainer has THREE distinct axes that must not be conflated:
  (a) expensive solver invocations  -- SPO+ calls Gurobi per instance per step; gradient-free ~0;
  (b) wall-clock training time       -- what the user actually pays;
  (c) batched-GPU forward-solves     -- PolyStep pays the MOST here, but each is cheap & batched.

This module measures (b)/(c) exactly by wrapping the shared batched solver, and gives analytic
cross-checks for all three. The forward-solve counter reuses the ``fwd_evals`` idiom from
``polystep/experiments/runners/ablation_ot_vs_softmax.py`` (count the leading dim per closure call).
"""
from __future__ import annotations
import time
import torch


class SolveCounter:
    """Wrap a batched forward solver ``solve(c)->w`` to count usage exactly.

    ``calls``     = number of times the solver was invoked (closure calls / minibatch steps);
    ``instances`` = total optimization instances solved (sum of leading dims) == forward-solves.
    Use ``wrap(cfg)`` to return a cfg whose ``ps_solve`` is this counter (PolyStep & SFGE go through it).
    """

    def __init__(self, solve):
        self._solve = solve
        self.calls = 0
        self.instances = 0

    def __call__(self, c):
        self.calls += 1
        self.instances += int(c.shape[0])
        return self._solve(c)

    def reset(self):
        self.calls = 0
        self.instances = 0
        return self

    def wrap(self, cfg):
        cfg = dict(cfg)
        cfg["ps_solve"] = self
        return cfg


class Timer:
    """Context manager for wall-clock with CUDA sync, so GPU-async work is fully counted."""

    def __init__(self, cuda_sync=True):
        self.cuda_sync = cuda_sync and torch.cuda.is_available()
        self.seconds = 0.0

    def __enter__(self):
        if self.cuda_sync:
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda_sync:
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self._t0
        return False


def gpu_peak_mb(reset=True):
    """Peak CUDA memory (MiB) since the last reset; call reset before the region you measure."""
    if not torch.cuda.is_available():
        return 0.0
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# ---- analytic cross-checks (well-defined per method; primary measure is SolveCounter/Timer) ----
ORTHOPLEX_VERTS = lambda particle_dim=2: 2 * particle_dim  # geometry.POLYTOPE_NUM_VERTICES_MAP


def spoplus_gurobi_solves(epochs, n_train):
    """SPO+ solves the (c-2*chat) program once per training instance per epoch (+ n_train precompute)."""
    return epochs * n_train + n_train


def sfge_forward_solves(epochs, n_samples, n_train):
    """SFGE solves n_samples sampled predictions for each of n_train instances, once per epoch."""
    return epochs * n_samples * n_train


def two_stage_solves(*_a, **_k):
    return 0
