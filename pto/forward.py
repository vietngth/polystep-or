"""Batched forward over PolyStep candidate parameter sets.

PolyStep hands the closure a dict ``{param_name: (N, *shape)}`` of N candidate
parameter sets. We need predictions ``(N, batch, out)``. Linear / 2-layer-MLP
use fast einsum; anything else (CNN) falls back to a functional_call loop, which
keeps our data-dependent batched solver out of vmap (where it would break).
"""
from __future__ import annotations
import torch, torch.nn as nn
from torch.func import functional_call


def _is_linear(m): return isinstance(m, nn.Linear)
def _is_mlp(m):
    return (isinstance(m, nn.Sequential) and len(m) >= 3
            and isinstance(m[0], nn.Linear) and isinstance(m[-1], nn.Linear))


def batched_predict(model, bp, X):
    """model: the nn.Module; bp: {name:(N,*shape)}; X:(batch,p) -> (N,batch,out)."""
    if _is_linear(model):
        out = torch.einsum("nop,bp->nbo", bp["weight"], X)
        if "bias" in bp: out = out + bp["bias"].unsqueeze(1)
        return out
    if _is_mlp(model):
        h = torch.einsum("nhp,bp->nbh", bp["0.weight"], X)
        if "0.bias" in bp: h = h + bp["0.bias"].unsqueeze(1)
        h = torch.relu(h)
        out = torch.einsum("noh,nbh->nbo", bp[f"{len(model)-1}.weight"], h)
        if f"{len(model)-1}.bias" in bp: out = out + bp[f"{len(model)-1}.bias"].unsqueeze(1)
        return out
    # general fallback (CNN): loop functional_call over candidates
    names = [n for n, _ in model.named_parameters()]
    bufs = dict(model.named_buffers())
    N = bp[names[0]].shape[0]
    outs = [functional_call(model, {**{k: bp[k][i] for k in names}, **bufs}, (X,))
            for i in range(N)]
    return torch.stack(outs)
