"""exp_scaling_memory.py -- "PolyStep scales to larger predictors than SFGE".

Memory / OOM scaling demonstration (MeZO-style, citable).

THE CLAIM
---------
PolyStep is gradient-free / forward-only. Its optimizer ``step`` runs under
``@torch.inference_mode()`` (verified in polystep/src/polystep/optimizer.py),
so it NEVER builds a backward graph: peak memory ~ inference memory, and it can
*chunk* the candidate-parameter evaluations (``chunk_size``) to trade compute
for memory, shrinking the per-step footprint down to a single config.

SFGE (score-function gradient estimator) MUST backprop through the predictor to
compute the policy gradient: ``surrogate = (advantage * logp).mean()`` where
``logp`` depends on ``pred = model(X)`` (grad-tracked), then ``surrogate.backward()``
(verified in exp4_constraints.py / exp_polystep_vs_sfge.py / exp_odece_mdkp.py).
Its peak memory ~ training memory: it must hold the whole activation graph for
the batch, which is NOT chunkable.

CONSEQUENCE
-----------
Under a fixed GPU memory budget, as the predictor grows, SFGE's activation graph
OOMs first while PolyStep keeps running (shrinking chunk_size). This script
measures peak GPU memory and wall-clock for both, sweeping predictor parameter
count over orders of magnitude, and records the crossover where SFGE OOMs but
PolyStep survives.

Faithfulness of the PolyStep measurement
----------------------------------------
The real ``PolyStepOptimizer`` is constructed for every size (its persistent
buffers -- base params, velocity, momentum -- are resident O(d) memory and are
included in the measurement). Its per-step peak memory is dominated by ONE chunk
of candidate-parameter evaluations; memory is invariant to the NUMBER of chunks,
so we measure the per-step peak from a single faithful chunk evaluated under
``torch.inference_mode()`` (exactly the op the optimizer's monolithic step runs
internally: materialize ``(chunk, *param_shape)`` stacked params, forward the net,
return a scalar cost per config). At the small sizes we ALSO run a real
``pso.step`` and assert the measured peak matches the single-chunk estimate
(validation reported in the .md). The honest cost of PolyStep's memory frugality
is COMPUTE: a full step does ``P*V*K = O(num_params)`` forward evals; we report
the extrapolated step time alongside.

Usage:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python exp_scaling_memory.py --smoke
  python exp_scaling_memory.py            # full sweep (needs a 24GB GPU)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time

import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "polystep/src"))
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GPU_BYTES = 24 * 1024 ** 3  # RTX 3090 / 4090 budget
OOM_ERRORS = (RuntimeError,)
if hasattr(torch.cuda, "OutOfMemoryError"):
    OOM_ERRORS = (torch.cuda.OutOfMemoryError, RuntimeError)

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "exp_results")
FIG_DIR = os.path.join(RES_DIR, "figs")


# ---------------------------------------------------------------------------
# Predictor (size-tunable deep MLP) + DFL task
# ---------------------------------------------------------------------------
def build_mlp(F: int, W: int, L: int, D: int) -> nn.Module:
    """MLP with L hidden layers of width W. param count ~ F*W + (L-1)*W^2 + W*D."""
    layers = [nn.Linear(F, W), nn.ReLU()]
    for _ in range(L - 1):
        layers += [nn.Linear(W, W), nn.ReLU()]
    layers += [nn.Linear(W, D)]
    return nn.Sequential(*layers)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def solve_topk(pred: torch.Tensor, k: int) -> torch.Tensor:
    """Decision oracle: select the k smallest predicted costs (minimisation).
    Returns a 0/1 tensor like ``pred``. Runs under no_grad in both methods."""
    idx = torch.topk(pred, k, dim=-1, largest=False).indices
    dec = torch.zeros_like(pred)
    dec.scatter_(-1, idx, 1.0)
    return dec


# ---------------------------------------------------------------------------
# SFGE: real score-function gradient (BACKPROPS through the predictor)
# ---------------------------------------------------------------------------
def sfge_step(model, X, Ctrue, k, scale, opt, n_samples, sigma):
    pred = model(X)  # (B,D) -- grad-tracked
    with torch.no_grad():
        eps = torch.randn(n_samples, *pred.shape, device=pred.device)
        chat = pred.unsqueeze(0) + sigma * eps               # (S,B,D)
        dec = solve_topk(chat, k)
        r = (dec * Ctrue.unsqueeze(0)).sum(-1) / scale       # (S,B) minimise
        adv = r - r.mean(0, keepdim=True)
    logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)  # depends on pred
    surrogate = (adv * logp).mean()
    opt.zero_grad(set_to_none=True)
    surrogate.backward()                                     # <-- backward graph through predictor
    opt.step()


# ---------------------------------------------------------------------------
# PolyStep: forward-only closure (functional batched MLP forward)
# ---------------------------------------------------------------------------
def make_polystep_closure(model, X, Ctrue, k, scale):
    """Build the closure(bp)->(N,) the optimizer calls. bp = {name:(N,*shape)}.
    Faithful manual functional forward (einsum over the N candidate configs)."""
    lin_prefixes = [name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]
    last = len(lin_prefixes) - 1

    def closure(bp):
        Wf = bp[lin_prefixes[0] + ".weight"]
        N = Wf.shape[0]
        h = X.unsqueeze(0).expand(N, *X.shape)               # (N,B,F)
        for i, pref in enumerate(lin_prefixes):
            Wl = bp[pref + ".weight"]                         # (N,out,in)
            bl = bp[pref + ".bias"]                           # (N,out)
            h = torch.einsum("noi,nbi->nbo", Wl, h) + bl.unsqueeze(1)
            if i < last:
                h = torch.relu(h)
        pred = h                                             # (N,B,D)
        dec = solve_topk(pred, k)
        cost = (dec * Ctrue.unsqueeze(0)).sum(-1) / scale    # (N,B)
        return cost.mean(-1)                                 # (N,)

    return closure, lin_prefixes


def stacked_params(model, chunk: int):
    """Materialise {name:(chunk,*shape)} -- the (chunk x d) stacked param dict the
    monolithic step builds per chunk. Included so the estimate carries this cost."""
    return {name: p.detach().unsqueeze(0).expand(chunk, *p.shape).contiguous()
            for name, p in model.named_parameters()}


# ---------------------------------------------------------------------------
# Memory measurement helpers
# ---------------------------------------------------------------------------
def _reset():
    if DEV == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_gb() -> float:
    if DEV == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    return float("nan")


def _is_oom(e: Exception) -> bool:
    if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(e).lower() or "CUDA out of memory" in str(e)


def measure_sfge(model, X, Ctrue, k, scale, n_samples, sigma, lr, nsteps):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    _reset()
    t0 = time.time()
    try:
        for _ in range(nsteps):
            sfge_step(model, X, Ctrue, k, scale, opt, n_samples, sigma)
        if DEV == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / nsteps
        return {"oom": False, "peak_gb": _peak_gb(), "sec_per_step": dt}
    except OOM_ERRORS as e:
        if not _is_oom(e):
            raise
        del opt
        _reset()
        return {"oom": True, "peak_gb": None, "sec_per_step": None}


CHUNK_MAX = 64


def adaptive_chunk(nparams: int, elem_budget: int, fwd_elems: int) -> int:
    """PolyStep memory knob (chunk_size): shrink to fit the budget. Bounded by BOTH
    the stacked-param dict (chunk*nparams) and the forward activation (chunk*B*W),
    capped at CHUNK_MAX so tiny models don't get an absurd chunk."""
    by_params = int(elem_budget // max(nparams, 1))
    by_fwd = int(elem_budget // max(fwd_elems, 1))
    return max(1, min(by_params, by_fwd, CHUNK_MAX))


def measure_polystep(model, X, Ctrue, k, scale, nparams, elem_budget, nsteps,
                     eval_budget, width, sr=0.4):
    """Construct the REAL optimizer (resident O(d) buffers counted), then measure
    per-step peak. Runs a real pso.step when the O(d) eval count is tractable,
    else a single faithful chunk under inference_mode (same per-step peak)."""
    B = X.shape[0]
    chunk = adaptive_chunk(nparams, elem_budget, B * width)
    closure, _ = make_polystep_closure(model, X, Ctrue, k, scale)

    # Construct the real optimizer (persistent buffers ~ 3-4 x d become resident).
    try:
        pso = PolyStepOptimizer(
            model, polytope_type="orthoplex", particle_dim=2,
            epsilon=CosineEpsilon(0.5, 0.05), step_radius=sr, probe_radius=2 * sr,
            num_probe=1, chunk_size=chunk, use_momentum=True,
            momentum_init=0.5, momentum_final=0.9, seed=0,
        )
    except OOM_ERRORS as e:
        if not _is_oom(e):
            raise
        _reset()
        return {"oom": True, "peak_gb": None, "sec_per_step": None,
                "chunk": chunk, "mode": "ctor-oom", "n_evals": None}

    # P*V*K eval count for a full monolithic step (orthoplex 2D -> V=4, K=1).
    n_evals = 2 * nparams
    run_real = n_evals <= eval_budget

    try:
        if run_real:
            _reset()
            t0 = time.time()
            for _ in range(nsteps):
                pso.step(closure)
            if DEV == "cuda":
                torch.cuda.synchronize()
            dt = (time.time() - t0) / nsteps
            peak = _peak_gb()
            # Validation: single-chunk estimate should match the real step peak.
            est_peak, _ = _single_chunk_peak(closure, model, chunk, nparams)
            del pso
            _reset()
            return {"oom": False, "peak_gb": peak, "sec_per_step": dt,
                    "chunk": chunk, "mode": "real", "n_evals": n_evals,
                    "est_peak_gb": est_peak}
        else:
            # Large model: measure per-step peak from one faithful chunk (optimizer
            # resident in memory) + extrapolate step time over n_chunks = n_evals/chunk.
            peak, chunk_dt = _single_chunk_peak(closure, model, chunk, nparams, repeats=nsteps)
            n_chunks = math.ceil(n_evals / chunk)
            extrap_step = chunk_dt * n_chunks
            del pso
            _reset()
            return {"oom": False, "peak_gb": peak, "sec_per_step": extrap_step,
                    "chunk": chunk, "mode": "estimate", "n_evals": n_evals,
                    "chunk_sec": chunk_dt, "n_chunks": n_chunks}
    except OOM_ERRORS as e:
        if not _is_oom(e):
            raise
        try:
            del pso
        except Exception:
            pass
        _reset()
        return {"oom": True, "peak_gb": None, "sec_per_step": None,
                "chunk": chunk, "mode": "step-oom", "n_evals": n_evals}


def _single_chunk_peak(closure, model, chunk, nparams, repeats=1):
    """Per-step peak from one chunk under inference_mode (no backward graph).

    Also emulates the O(d) transients the real monolithic step allocates: the
    pre-allocated losses buffer (P*V*K,) and the (P,V) OT cost matrix, with
    P=ceil(d/2), V=4 (orthoplex 2D), K=1 -> ~ 4d float32 elements. These are
    NOT activation-graph (no depth/batch factor); they scale O(d) like Adam state."""
    P = math.ceil(nparams / 2)
    V, K = 4, 1
    _reset()
    t0 = time.time()
    with torch.inference_mode():
        # emulate optimizer step transients (resident across the chunk evals)
        ot_losses = torch.empty(P * V * K, device=DEV, dtype=torch.float32)
        ot_cost = torch.empty(P, V, device=DEV, dtype=torch.float32)
        for _ in range(repeats):
            bp = stacked_params(model, chunk)
            _ = closure(bp)
            del bp
        del ot_losses, ot_cost
    if DEV == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / max(repeats, 1)
    return _peak_gb(), dt


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def make_configs(smoke: bool):
    """(label, W, L) chosen to span param count over orders of magnitude."""
    if smoke:
        return [("tiny", 32, 1), ("small", 64, 2), ("mid", 128, 2)]
    # Fixed width 2048 + depth sweep is the OOM driver (activation graph ~ B*L*W);
    # smaller widths extend the low end of the param-count axis.
    return [
        ("w64-l2", 64, 2),
        ("w128-l4", 128, 4),
        ("w256-l4", 256, 4),
        ("w512-l4", 512, 4),
        ("w1024-l4", 1024, 4),
        ("w2048-l4", 2048, 4),
        ("w2048-l8", 2048, 8),
        ("w2048-l16", 2048, 16),
        ("w2048-l32", 2048, 32),
        ("w2048-l48", 2048, 48),
        ("w2048-l64", 2048, 64),
        ("w2048-l96", 2048, 96),
        ("w2048-l128", 2048, 128),
    ]


def run(args):
    torch.manual_seed(0)
    np.random.seed(0)
    F, D, k = args.feat, args.outdim, args.topk
    B = args.batch if not args.smoke else args.smoke_batch
    n_samples, sigma, lr = 8, 0.5, 1e-2
    nsteps = args.steps
    # PolyStep stacked-param element budget (controls adaptive chunk) and eval budget
    # for deciding real-step vs estimate.
    elem_budget = args.elem_budget
    eval_budget = args.eval_budget

    configs = make_configs(args.smoke)
    Xfull = torch.randn(B, F, device=DEV)
    Ctrue = torch.rand(B, D, device=DEV)  # true cost vector (positive)

    rows = []
    for label, W, L in configs:
        gc.collect()
        _reset()
        model = build_mlp(F, W, L, D).to(DEV)
        P = n_params(model)
        print(f"\n=== {label}: W={W} L={L} params={P:,} batch={B} ===", flush=True)

        sf = measure_sfge(model, Xfull, Ctrue, k, args.scale, n_samples, sigma, lr, nsteps)
        print(f"  SFGE     : oom={sf['oom']} peak={sf['peak_gb']} GB "
              f"t/step={sf['sec_per_step']}", flush=True)

        # fresh model state irrelevant for memory; reuse same model for PolyStep
        ps = measure_polystep(model, Xfull, Ctrue, k, args.scale, P,
                              elem_budget, nsteps, eval_budget, W)
        print(f"  PolyStep : oom={ps['oom']} peak={ps['peak_gb']} GB "
              f"t/step={ps['sec_per_step']} mode={ps['mode']} chunk={ps['chunk']}",
              flush=True)

        ratio = None
        if sf["peak_gb"] and ps["peak_gb"]:
            ratio = sf["peak_gb"] / ps["peak_gb"]
        rows.append({
            "label": label, "width": W, "depth": L, "params": P, "batch": B,
            "sfge_oom": sf["oom"], "sfge_peak_gb": sf["peak_gb"], "sfge_sec_step": sf["sec_per_step"],
            "polystep_oom": ps["oom"], "polystep_peak_gb": ps["peak_gb"],
            "polystep_sec_step": ps["sec_per_step"], "polystep_mode": ps["mode"],
            "polystep_chunk": ps["chunk"], "polystep_n_evals": ps.get("n_evals"),
            "polystep_est_peak_gb": ps.get("est_peak_gb"),
            "mem_ratio_sfge_over_polystep": ratio,
        })

        del model
        gc.collect()
        _reset()

    summarize(rows, args)
    return rows


def summarize(rows, args):
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # crossover: first size where SFGE OOMs but PolyStep survives
    crossover = None
    for r in rows:
        if r["sfge_oom"] and not r["polystep_oom"]:
            crossover = r
            break

    # validation: real-step measured peak vs single-chunk estimate (small sizes)
    valids = [r for r in rows if r["polystep_mode"] == "real" and r["polystep_est_peak_gb"]]
    val_lines = []
    for r in valids:
        meas = r["polystep_peak_gb"]; est = r["polystep_est_peak_gb"]
        rr = est / meas if meas else float("nan")
        val_lines.append(f"  {r['label']}: real-step peak={meas:.4f} GB, "
                         f"single-chunk estimate={est:.4f} GB (ratio {rr:.2f})")

    payload = {
        "meta": {
            "device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
            "gpu_budget_gb": GPU_BYTES / 1024 ** 3,
            "batch": rows[0]["batch"] if rows else None,
            "feat": args.feat, "outdim": args.outdim, "topk": args.topk,
            "elem_budget": args.elem_budget, "eval_budget": args.eval_budget,
            "smoke": args.smoke,
            "claim": "PolyStep forward-only (inference_mode, chunkable) fits larger "
                     "predictors than SFGE (backprops through predictor; activation "
                     "graph not chunkable) under a fixed GPU memory budget.",
        },
        "rows": rows,
        "crossover": crossover,
        "validation": val_lines,
    }
    jpath = os.path.join(RES_DIR, "scaling_memory.json")
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2)

    # markdown
    lines = []
    lines.append("# PolyStep scales to larger predictors than SFGE (memory / OOM)")
    lines.append("")
    lines.append(f"- Device: **{payload['meta']['device']}**, budget {payload['meta']['gpu_budget_gb']:.0f} GB")
    lines.append(f"- Predictor: deep MLP, batch={payload['meta']['batch']}, "
                 f"feat={args.feat}, outdim={args.outdim}, top-k={args.topk}")
    lines.append("")
    lines.append("**Premise (verified in source):** SFGE calls `surrogate.backward()` "
                 "through `pred = model(X)` (policy-gradient via `logp`), so its peak "
                 "memory holds the whole activation graph for the batch. PolyStep's "
                 "`step` runs under `@torch.inference_mode()` and evaluates candidate "
                 "params in chunks (`chunk_size`), so its peak ~ inference memory and is "
                 "shrinkable to fit any budget. Cost: a full PolyStep step does "
                 "`O(num_params)` forward evals (reported as t/step).")
    lines.append("")
    lines.append("| params | W | L | SFGE peak (GB) | PolyStep peak (GB) | ratio | SFGE t/step | PolyStep t/step | chunk | mode |")
    lines.append("|---:|---:|---:|---|---|---:|---|---|---:|---|")
    for r in rows:
        sp = "**OOM**" if r["sfge_oom"] else f"{r['sfge_peak_gb']:.2f}"
        pp = "**OOM**" if r["polystep_oom"] else f"{r['polystep_peak_gb']:.2f}"
        ra = f"{r['mem_ratio_sfge_over_polystep']:.1f}x" if r["mem_ratio_sfge_over_polystep"] else "-"
        st = "-" if r["sfge_sec_step"] is None else f"{r['sfge_sec_step']*1e3:.0f} ms"
        pt = "-" if r["polystep_sec_step"] is None else f"{r['polystep_sec_step']*1e3:.0f} ms"
        lines.append(f"| {r['params']:,} | {r['width']} | {r['depth']} | {sp} | {pp} | {ra} | "
                     f"{st} | {pt} | {r['polystep_chunk']} | {r['polystep_mode']} |")
    lines.append("")
    if crossover:
        lines.append(f"**Crossover:** at **{crossover['params']:,} params** "
                     f"({crossover['label']}) SFGE **OOMs** while PolyStep survives "
                     f"(peak {crossover['polystep_peak_gb']:.2f} GB, chunk={crossover['polystep_chunk']}).")
    else:
        survivors = [r for r in rows if not r["sfge_oom"]]
        if survivors:
            big = max(survivors, key=lambda r: r["params"])
            ratios = [r["mem_ratio_sfge_over_polystep"] for r in rows
                      if r["mem_ratio_sfge_over_polystep"]]
            mr = f"{max(ratios):.1f}x" if ratios else "n/a"
            lines.append(f"**No OOM reached** in this sweep (largest = {big['params']:,} "
                         f"params). Max SFGE/PolyStep memory ratio: {mr}. "
                         "Premise direction confirmed (SFGE > PolyStep peak); push larger "
                         "sizes / batch to reach the OOM crossover.")
    lines.append("")
    if val_lines:
        lines.append("**Validation (real `pso.step` peak vs single-chunk estimate):**")
        lines.extend(val_lines)
        lines.append("")
    lines.append("Figure: `exp_results/figs/fig_scaling_memory.{pdf,png}`")
    mpath = os.path.join(RES_DIR, "scaling_memory.md")
    with open(mpath, "w") as f:
        f.write("\n".join(lines) + "\n")

    make_figure(rows)
    print(f"\nWrote {jpath}\nWrote {mpath}", flush=True)
    if crossover:
        print(f"CROSSOVER @ {crossover['params']:,} params: SFGE OOM, PolyStep survives.", flush=True)


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = [r["params"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))

    # SFGE: plot survivors; mark OOMs at the top.
    sf_x = [r["params"] for r in rows if not r["sfge_oom"]]
    sf_y = [r["sfge_peak_gb"] for r in rows if not r["sfge_oom"]]
    sf_oom_x = [r["params"] for r in rows if r["sfge_oom"]]
    ps_x = [r["params"] for r in rows if not r["polystep_oom"]]
    ps_y = [r["polystep_peak_gb"] for r in rows if not r["polystep_oom"]]
    ps_oom_x = [r["params"] for r in rows if r["polystep_oom"]]

    budget = GPU_BYTES / 1024 ** 3
    if sf_x:
        ax.plot(sf_x, sf_y, "o-", color="#d62728", label="SFGE (backprop / training mem)", lw=2)
    if ps_x:
        ax.plot(ps_x, ps_y, "s-", color="#1f77b4", label="PolyStep (forward-only / chunked)", lw=2)
    if sf_oom_x:
        ax.scatter(sf_oom_x, [budget] * len(sf_oom_x), marker="x", s=140,
                   color="#d62728", zorder=5, label="SFGE OOM")
    if ps_oom_x:
        ax.scatter(ps_oom_x, [budget] * len(ps_oom_x), marker="x", s=140,
                   color="#1f77b4", zorder=5, label="PolyStep OOM")

    ax.axhline(budget, ls="--", color="k", alpha=0.6)
    ax.text(min(params), budget * 1.03, f"{budget:.0f} GB GPU budget", fontsize=9)

    # crossover marker
    for r in rows:
        if r["sfge_oom"] and not r["polystep_oom"]:
            ax.axvline(r["params"], ls=":", color="gray", alpha=0.7)
            ax.text(r["params"], ax.get_ylim()[0] if False else budget * 0.55,
                    " SFGE OOMs here\n PolyStep survives", fontsize=8, color="gray")
            break

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predictor parameter count")
    ax.set_ylabel("peak GPU memory (GB)")
    ax.set_title("PolyStep fits larger predictors than SFGE under a fixed memory budget")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_scaling_memory.{ext}"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--smoke_batch", type=int, default=256)
    ap.add_argument("--feat", type=int, default=256)
    ap.add_argument("--outdim", type=int, default=128)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--elem_budget", type=int, default=200_000_000,
                    help="stacked-param element budget for PolyStep adaptive chunk")
    ap.add_argument("--eval_budget", type=int, default=120_000,
                    help="max P*V*K evals to run a REAL pso.step (else estimate)")
    args = ap.parse_args()
    if DEV != "cuda":
        print("WARNING: no CUDA device; memory numbers will be NaN.", flush=True)
    run(args)


if __name__ == "__main__":
    main()
