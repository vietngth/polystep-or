"""Determinism utilities for the PolyStep-frontier experiments.

Ported and generalized from ``dfl-ablation/dfl.py`` (the collaborator's hardened, reproducible
seed-stream machinery) so the main-repo cfg-style experiments get the same guarantees. Needed for
the >=20-seed bias-variance study and to make every grid bit-reproducible.

Run scripts with ``CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python ...`` so the deterministic
cuBLAS path is available (otherwise ``torch.use_deterministic_algorithms`` warns/errors on GEMM).
"""
from __future__ import annotations
import hashlib
import random

import numpy as np
import torch

_SEED_MOD = 2**31 - 1


def stream_seed(seed: int, stream: str) -> int:
    """Derive a stable positive seed for one named stochastic stream.

    Unlike the dfl-ablation version (fixed offset dict) this accepts ANY stream name via a stable
    blake2b hash, so callers can name streams freely ('model', 'spo', 'sfge', 'polystep', 'data',
    'imle', ...) without editing a table. Deterministic across processes (unlike ``hash()``).
    """
    h = hashlib.blake2b(stream.encode("utf-8"), digest_size=8).digest()
    offset = int.from_bytes(h, "big")
    return (int(seed) * 1_000_003 + offset) % _SEED_MOD


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch (incl. CUDA) RNGs for a reproducible run."""
    seed = int(seed) % _SEED_MOD
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return seed


def clone_state_dict(model_or_state):
    """Clone a model/state-dict so a trainer cannot mutate a shared baseline (e.g. the warm start)."""
    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    return {k: v.detach().clone() for k, v in state.items()}


def device_generator(seed: int, device: str):
    """A torch.Generator on ``device`` seeded deterministically (for sampling RNG, e.g. SFGE eps)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) % _SEED_MOD)
    return gen
