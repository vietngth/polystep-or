"""PolyStep vs SFGE characterization: probe SFGE's three known weaknesses, HONESTLY.

SFGE (score-function gradient-free, Silvestri et al. JAIR 2026) is PolyStep's closest
gradient-free rival. Prior bias-variance work (exp_results/bias_variance.md) found the two
statistically TIED on across-seed regret. This experiment isolates the THREE places where SFGE
is *theoretically* weaker and tests whether any of them is an EMPIRICAL PolyStep advantage --
designed as fair tests, reporting ties/losses as readily as wins. Forward-solve counts (the
honest compute axis for gradient-free DFL) are recorded throughout via pto.budget.SolveCounter.

(A) VARIANCE vs DIMENSION. Sweep knapsack n_items. Headline = the variance of the proposed
    UPDATE at a FIXED warm-start point: draw the estimator N times WITHOUT updating, measure the
    variance of the proposed step. SFGE = REINFORCE score-function gradient (famously high-var,
    grows with dim); PolyStep = orthoplex barycenter step (randomness only from probe rotation).
    Scale-invariant metric = CV^2 = trace(Cov(step)) / ||mean step||^2. Hypothesis to TEST (not
    assume): SFGE step-variance grows faster with dimension. Wilcoxon on across-seed regret-std.

(B) HP SENSITIVITY (the most likely PolyStep win). SFGE: sweep exploration sigma; PolyStep:
    sweep probe_radius. Report each method's "usable HP range" = HP values within 10% of its own
    best regret. SFGE expected to have a narrow optimal sigma band (too small -> ~0 gradient on a
    piecewise-constant objective; too large -> biased); PolyStep expected flatter.

(C) SAMPLE EFFICIENCY (camp-level, labelled honestly). Sweep n_train for two-stage / SPO+ / IMLE
    / SFGE / PolyStep on knap + sp. Framing: structure-exploiting camp (SPO+/IMLE) vs structure-
    free camp (SFGE/PolyStep). This is NOT a PolyStep-over-SFGE differentiator -- both are
    structure-free -- and is reported as such; the SFGE-vs-PolyStep head-to-head is incidental.

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp_polystep_vs_sfge.py [--smoke]
"""
from __future__ import annotations
import os, sys, time, argparse, math
sys.path.insert(0, "polystep/src")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyepo import metric
from pto.capability import (setup_knap, setup_sp, train_two_stage, train_sfge, train_dfl,
                            _adam, dev)
from pto.sweep_lr import train_spoplus_lr, train_polystep_sr
from pto.budget import SolveCounter, spoplus_gurobi_solves
from pto.seeding import seed_everything
from pto.multiseed import (summarize, wilcoxon_pair, md_table, write_json, write_md, fmt_mean_std)
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon

# ---- colorblind-safe palette (Wong 2011) ----
C_SFGE = "#0072B2"   # blue
C_PS   = "#D55E00"   # vermillion
C_TS   = "#999999"   # grey
C_SPO  = "#009E73"   # green
C_IMLE = "#CC79A7"   # purple
FIGDIR = "exp_results/figs"

# per-problem sweep-best hyperparameters (LR_SWEEP_RESULTS.md / bias_variance BEST_HP)
BEST = {
    "sp":   {"sfge_lr": 1e-1, "ps_sr": 0.2, "spo_lr": 3e-3, "sigma": 0.5},
    "knap": {"sfge_lr": 3e-2, "ps_sr": 0.8, "spo_lr": 3e-2, "sigma": 0.5},
}
SETUP = {"sp": setup_sp, "knap": setup_knap}
PLABEL = {"sp": "shortest_path (LP)", "knap": "knapsack (ILP)"}


# ===========================================================================
# shared helpers
# ===========================================================================
def make_cfg(problem, seed, deg, n_train, **kw):
    """Build a capability cfg and wrap its forward solver in a SolveCounter."""
    cfg, cat = SETUP[problem](seed, deg, n_train=n_train, **kw)
    raw = cfg["ps_solve"]
    counter = SolveCounter(raw)
    cfg["ps_solve"] = counter
    return cfg, counter, raw


def fast_regret(model, cfg, raw_solve):
    """Cheap GPU-solver normalized-regret proxy on the test split (no Gurobi). Monotone training
    progress signal; used ONLY to define solves-to-target. Headline numbers use pyepo regret."""
    with torch.no_grad():
        Xte, Cte, sgn = cfg["Xte"], cfg["Cte"], cfg["sign"]
        w = raw_solve(model(Xte)); wo = raw_solve(Cte)
        achieved = sgn * (w * Cte).sum(-1)
        opt = sgn * (wo * Cte).sum(-1)
        return float(((achieved - opt).sum() / opt.abs().sum().clamp(min=1e-6)).item())


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _set_warm(model, warm):
    with torch.no_grad():
        if warm is not None:
            model.weight.copy_(warm.weight)


# ---- instrumented trainers that checkpoint (cum_forward_solves, fast_regret) ----
def sfge_train_traj(cfg, counter, raw, sigma, lr, epochs, n_samples=8, ckpt_every=10):
    m = cfg["make"](); _set_warm(m, cfg.get("warm"))
    opt = _adam(m, lr); X, Cs, solve, sgn = cfg["Xtr"], cfg["Cs"], counter, cfg["sign"]
    traj = []
    counter.reset()
    for e in range(epochs):
        pred = m(X)
        with torch.no_grad():
            eps = torch.randn(n_samples, *pred.shape, device=dev)
            chat = pred.unsqueeze(0) + sigma * eps
            S, B, D = chat.shape
            w = solve(chat.reshape(S * B, D)).reshape(S, B, D)
            r = sgn * (w * Cs.unsqueeze(0)).sum(-1)
            adv = r - r.mean(0, keepdim=True)
        logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
        surrogate = (adv * logp).mean()
        opt.zero_grad(); surrogate.backward(); opt.step()
        if e % ckpt_every == 0 or e == epochs - 1:
            traj.append((counter.instances, fast_regret(m, cfg, raw)))
    return m, traj, counter.instances


def polystep_train_traj(cfg, counter, raw, sr, steps, ckpt_every=10):
    m = cfg["make"](); _set_warm(m, cfg.get("warm"))
    pso = PolyStepOptimizer(m, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=sr, probe_radius=2 * sr, num_probe=1, seed=cfg["seed"],
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    X, Cs, solve, sgn = cfg["Xtr"], cfg["Cs"], counter, cfg["sign"]
    def closure(bp):
        pred = torch.einsum("nef,bf->nbe", bp["weight"], X); N, nb, E = pred.shape
        w = solve(pred.reshape(N * nb, E)).reshape(N, nb, E)
        return sgn * (w * Cs.unsqueeze(0)).sum(-1).mean(-1)
    traj = []
    counter.reset()
    for s in range(steps):
        pso.step(closure)
        if s % ckpt_every == 0 or s == steps - 1:
            traj.append((counter.instances, fast_regret(m, cfg, raw)))
    return m, traj, counter.instances


def solves_to_target(traj, target):
    """First cumulative forward-solve count at which fast_regret <= target (None if never)."""
    for solves, reg in traj:
        if reg <= target:
            return solves
    return None


# ---- step-variance at a FIXED warm-start point (the (A) headline) ----
def sfge_step_draws(cfg, warm, raw, sigma, n_samples, n_draws):
    """N independent SFGE score-function gradient estimates at the warm point (no update)."""
    X, Cs, sgn = cfg["Xtr"], cfg["Cs"], cfg["sign"]
    draws = []
    for _ in range(n_draws):
        m = cfg["make"](); _set_warm(m, warm)
        pred = m(X)
        with torch.no_grad():
            eps = torch.randn(n_samples, *pred.shape, device=dev)
            chat = pred.unsqueeze(0) + sigma * eps
            S, B, D = chat.shape
            w = raw(chat.reshape(S * B, D)).reshape(S, B, D)
            r = sgn * (w * Cs.unsqueeze(0)).sum(-1)
            adv = r - r.mean(0, keepdim=True)
        logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
        surrogate = (adv * logp).mean()
        g = torch.autograd.grad(surrogate, list(m.parameters()))
        draws.append(torch.cat([gi.reshape(-1) for gi in g]).detach())
    return torch.stack(draws)  # (n_draws, P)


def polystep_step_draws(cfg, warm, sr, n_draws):
    """N independent PolyStep single-step deltas at the warm point (only randomness: probe seed)."""
    X, Cs, sgn = cfg["Xtr"], cfg["Cs"], cfg["sign"]
    def closure_factory(solve):
        def closure(bp):
            pred = torch.einsum("nef,bf->nbe", bp["weight"], X); N, nb, E = pred.shape
            w = solve(pred.reshape(N * nb, E)).reshape(N, nb, E)
            return sgn * (w * Cs.unsqueeze(0)).sum(-1).mean(-1)
        return closure
    draws = []
    for d in range(n_draws):
        m = cfg["make"](); _set_warm(m, warm)
        before = flat_params(m).clone()
        pso = PolyStepOptimizer(m, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                                step_radius=sr, probe_radius=2 * sr, num_probe=1,
                                seed=1000 + d, use_momentum=True,
                                momentum_init=0.5, momentum_final=0.9)
        pso.step(closure_factory(cfg["ps_solve"]))
        draws.append((flat_params(m) - before).detach())
    return torch.stack(draws)  # (n_draws, P)


def step_var_metrics(draws):
    """trace(Cov), ||mean||, CV^2 = trace(Cov)/||mean||^2 (scale-invariant)."""
    mean = draws.mean(0)
    trace_cov = float(draws.var(0, unbiased=True).sum().item())
    mean_norm = float(mean.norm().item())
    cv2 = trace_cov / (mean_norm ** 2 + 1e-12)
    return {"trace_cov": trace_cov, "mean_norm": mean_norm, "cv2": cv2}


# ===========================================================================
# (A) variance vs dimension
# ===========================================================================
def exp_A(dims, seeds, deg, n_train, n_test, sfge_epochs, ps_steps, n_var_draws):
    print(f"\n[A] variance vs dimension | knap n_items={dims} seeds={seeds}", flush=True)
    hp = BEST["knap"]
    per_dim = {}
    for nit in dims:
        reg = {"SFGE": [], "PolyStep": []}
        solves_final = {"SFGE": [], "PolyStep": []}
        s2t = {"SFGE": [], "PolyStep": []}
        sv = {"SFGE": [], "PolyStep": []}   # cv2 per seed
        sv_full = {"SFGE": [], "PolyStep": []}
        for seed in seeds:
            seed_everything(seed)
            cfg, counter, raw = make_cfg("knap", seed, deg, n_train, n_test=n_test, NIT=nit)
            ts = train_two_stage(cfg); cfg["warm"] = ts
            # train (instrumented)
            m_sf, tr_sf, fs_sf = sfge_train_traj(cfg, counter, raw, hp["sigma"], hp["sfge_lr"], sfge_epochs)
            m_ps, tr_ps, fs_ps = polystep_train_traj(cfg, counter, raw, hp["ps_sr"], ps_steps)
            r_sf = float(metric.regret(m_sf, cfg["om"], cfg["ld_te"]))
            r_ps = float(metric.regret(m_ps, cfg["om"], cfg["ld_te"]))
            reg["SFGE"].append(r_sf); reg["PolyStep"].append(r_ps)
            solves_final["SFGE"].append(fs_sf); solves_final["PolyStep"].append(fs_ps)
            # solves-to-target (shared target = 1.05 x worst final fast-regret of the two)
            tgt = 1.05 * max(tr_sf[-1][1], tr_ps[-1][1])
            s2t["SFGE"].append(solves_to_target(tr_sf, tgt))
            s2t["PolyStep"].append(solves_to_target(tr_ps, tgt))
            # step-variance at the warm point
            ds_sf = sfge_step_draws(cfg, ts, raw, hp["sigma"], 8, n_var_draws)
            ds_ps = polystep_step_draws(cfg, ts, hp["ps_sr"], n_var_draws)
            msf = step_var_metrics(ds_sf); mps = step_var_metrics(ds_ps)
            sv["SFGE"].append(msf["cv2"]); sv["PolyStep"].append(mps["cv2"])
            sv_full["SFGE"].append(msf); sv_full["PolyStep"].append(mps)
            print(f"  n_items={nit:>3} seed={seed}: reg SFGE={r_sf:.4f} PS={r_ps:.4f} | "
                  f"cv2 SFGE={msf['cv2']:.3g} PS={mps['cv2']:.3g} | "
                  f"solves SFGE={fs_sf} PS={fs_ps}", flush=True)
        per_dim[nit] = {
            "regret": {m: summarize(reg[m]) for m in reg},
            "solves_final": {m: summarize([float(x) for x in solves_final[m]]) for m in reg},
            "solves_to_target": {m: summarize([float(x) for x in s2t[m] if x is not None]) for m in reg},
            "cv2": {m: summarize(sv[m]) for m in reg},
            "cv2_detail": {m: sv_full[m] for m in reg},
        }
    # Wilcoxon across the dimension grid on across-seed regret-std and on cv2-mean
    sfge_std = [per_dim[d]["regret"]["SFGE"]["std"] for d in dims]
    ps_std = [per_dim[d]["regret"]["PolyStep"]["std"] for d in dims]
    sfge_cv2 = [per_dim[d]["cv2"]["SFGE"]["mean"] for d in dims]
    ps_cv2 = [per_dim[d]["cv2"]["PolyStep"]["mean"] for d in dims]
    return {
        "dims": dims, "per_dim": per_dim,
        "wilcoxon_PS_std_lt_SFGE": wilcoxon_pair(ps_std, sfge_std),
        "wilcoxon_PS_cv2_lt_SFGE": wilcoxon_pair(ps_cv2, sfge_cv2),
    }


# ===========================================================================
# (B) HP sensitivity
# ===========================================================================
def usable_range(hp_values, regrets):
    """HP values within 10% of the best (min) regret. Returns dict with band and decade-width."""
    finite = [(h, r) for h, r in zip(hp_values, regrets) if np.isfinite(r)]
    if not finite:
        return {"best_hp": None, "best_regret": None, "usable": [], "n_usable": 0, "decades": 0.0}
    best = min(finite, key=lambda t: t[1])
    thr = best[1] * 1.10 if best[1] >= 0 else best[1] * 0.90
    usable = [h for h, r in finite if r <= thr]
    decades = (math.log10(max(usable) / min(usable)) if usable and min(usable) > 0 else 0.0)
    return {"best_hp": best[0], "best_regret": best[1], "usable": usable,
            "n_usable": len(usable), "decades": decades}


def sfge_regret_at_sigma(cfg, sigma, lr, epochs):
    cc = dict(cfg)
    return float(metric.regret(train_sfge(cc, epochs=epochs, sigma=sigma, lr=lr),
                               cfg["om"], cfg["ld_te"]))


def polystep_regret_at_pr(cfg, probe_radius, sr, steps):
    """PolyStep regret holding step_radius=sr, varying probe_radius."""
    m = cfg["make"](); _set_warm(m, cfg.get("warm"))
    pso = PolyStepOptimizer(m, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=sr, probe_radius=probe_radius, num_probe=1,
                            seed=cfg["seed"], use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    X, Cs, solve, sgn = cfg["Xtr"], cfg["Cs"], cfg["ps_solve"], cfg["sign"]
    def closure(bp):
        pred = torch.einsum("nef,bf->nbe", bp["weight"], X); N, nb, E = pred.shape
        w = solve(pred.reshape(N * nb, E)).reshape(N, nb, E)
        return sgn * (w * Cs.unsqueeze(0)).sum(-1).mean(-1)
    for _ in range(steps):
        pso.step(closure)
    return float(metric.regret(m, cfg["om"], cfg["ld_te"]))


def exp_B(problems, sigmas, probe_radii, seeds, deg, n_train, sfge_epochs, ps_steps):
    print(f"\n[B] HP sensitivity | problems={problems} sigmas={sigmas} probe_radii={probe_radii}", flush=True)
    out = {}
    for p in problems:
        hp = BEST[p]
        sfge_curve = {s: [] for s in sigmas}
        ps_curve = {pr: [] for pr in probe_radii}
        for seed in seeds:
            seed_everything(seed)
            cfg, counter, raw = make_cfg(p, seed, deg, n_train)
            ts = train_two_stage(cfg); cfg["warm"] = ts
            for s in sigmas:
                sfge_curve[s].append(sfge_regret_at_sigma(cfg, s, hp["sfge_lr"], sfge_epochs))
            for pr in probe_radii:
                ps_curve[pr].append(polystep_regret_at_pr(cfg, pr, hp["ps_sr"], ps_steps))
            print(f"  [{p}] seed {seed} done", flush=True)
        sfge_mean = [float(np.nanmean(sfge_curve[s])) for s in sigmas]
        ps_mean = [float(np.nanmean(ps_curve[pr])) for pr in probe_radii]
        out[p] = {
            "sigmas": sigmas, "probe_radii": probe_radii,
            "sfge_curve": {str(s): summarize(sfge_curve[s]) for s in sigmas},
            "ps_curve": {str(pr): summarize(ps_curve[pr]) for pr in probe_radii},
            "sfge_mean": sfge_mean, "ps_mean": ps_mean,
            "sfge_usable": usable_range(sigmas, sfge_mean),
            "ps_usable": usable_range(probe_radii, ps_mean),
        }
        su = out[p]["sfge_usable"]; pu = out[p]["ps_usable"]
        print(f"  [{p}] SFGE usable sigma: {su['n_usable']}/{len(sigmas)} ({su['decades']:.2f} dec); "
              f"PolyStep usable probe_radius: {pu['n_usable']}/{len(probe_radii)} ({pu['decades']:.2f} dec)",
              flush=True)
    return out


# ===========================================================================
# (C) sample efficiency
# ===========================================================================
def exp_C(problems, ntrains, seeds, deg, sfge_epochs, ps_steps):
    print(f"\n[C] sample efficiency | problems={problems} n_train={ntrains} seeds={seeds}", flush=True)
    methods = ["two-stage", "SPO+", "IMLE", "SFGE", "PolyStep"]
    out = {}
    for p in problems:
        hp = BEST[p]
        curve = {n: {m: [] for m in methods} for n in ntrains}
        gf_solves = {n: {"SFGE": [], "PolyStep": []} for n in ntrains}
        for n in ntrains:
            for seed in seeds:
                seed_everything(seed)
                cfg, counter, raw = make_cfg(p, seed, deg, n)
                ts = train_two_stage(cfg); cfg["warm"] = ts
                curve[n]["two-stage"].append(float(metric.regret(ts, cfg["om"], cfg["ld_te"])))
                try:
                    curve[n]["SPO+"].append(float(metric.regret(
                        train_spoplus_lr(cfg, ts, hp["spo_lr"]), cfg["om"], cfg["ld_te"])))
                except Exception:
                    curve[n]["SPO+"].append(float("nan"))
                try:
                    curve[n]["IMLE"].append(float(metric.regret(
                        train_dfl(cfg, "IMLE"), cfg["om"], cfg["ld_te"])))
                except Exception:
                    curve[n]["IMLE"].append(float("nan"))
                m_sf, _, fs_sf = sfge_train_traj(cfg, counter, raw, hp["sigma"], hp["sfge_lr"], sfge_epochs)
                curve[n]["SFGE"].append(float(metric.regret(m_sf, cfg["om"], cfg["ld_te"])))
                gf_solves[n]["SFGE"].append(fs_sf)
                m_ps, _, fs_ps = polystep_train_traj(cfg, counter, raw, hp["ps_sr"], ps_steps)
                curve[n]["PolyStep"].append(float(metric.regret(m_ps, cfg["om"], cfg["ld_te"])))
                gf_solves[n]["PolyStep"].append(fs_ps)
            print(f"  [{p}] n={n}: " + " ".join(
                f"{m}={np.nanmean(curve[n][m]):.4f}" for m in methods), flush=True)
        out[p] = {
            "ntrains": ntrains, "methods": methods,
            "curve": {n: {m: summarize(curve[n][m]) for m in methods} for n in ntrains},
            "gf_solves": {n: {m: summarize([float(x) for x in gf_solves[n][m]]) for m in ("SFGE", "PolyStep")}
                          for n in ntrains},
            "wilcoxon_PS_lt_SFGE": {n: wilcoxon_pair(curve[n]["PolyStep"], curve[n]["SFGE"]) for n in ntrains},
        }
    return out


# ===========================================================================
# figures
# ===========================================================================
def fig_var_vs_dim(A, path):
    dims = A["dims"]; pd = A["per_dim"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, c in (("SFGE", C_SFGE), ("PolyStep", C_PS)):
        std = [pd[d]["regret"][m]["std"] for d in dims]
        ax[0].plot(dims, std, "-o", color=c, label=m)
        cv2m = [pd[d]["cv2"][m]["mean"] for d in dims]
        cv2s = [pd[d]["cv2"][m]["std"] for d in dims]
        ax[1].errorbar(dims, cv2m, yerr=cv2s, fmt="-o", color=c, label=m, capsize=3)
    ax[0].set_xlabel("knapsack n_items (output dimension)"); ax[0].set_ylabel("across-seed regret std")
    ax[0].set_title("(A1) Across-seed regret variance"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_xlabel("knapsack n_items (output dimension)")
    ax[1].set_ylabel("step CV$^2$ = trace(Cov)/||mean||$^2$")
    ax[1].set_title("(A2) Update-direction variance @ fixed warm point")
    ax[1].set_yscale("log"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_hp_sensitivity(B, path):
    probs = list(B.keys())
    fig, axes = plt.subplots(1, len(probs), figsize=(5.5 * len(probs), 4.2), squeeze=False)
    for j, p in enumerate(probs):
        ax = axes[0][j]
        sig = B[p]["sigmas"]; pr = B[p]["probe_radii"]
        ax.plot(sig, B[p]["sfge_mean"], "-o", color=C_SFGE, label="SFGE (vs sigma)")
        ax.plot(pr, B[p]["ps_mean"], "-s", color=C_PS, label="PolyStep (vs probe_radius)")
        ax.set_xscale("log"); ax.set_xlabel("HP value (sigma / probe_radius, log)")
        ax.set_ylabel("normalized regret"); ax.set_title(f"(B) {PLABEL[p]}")
        ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_sample_efficiency(C, path):
    probs = list(C.keys())
    fig, axes = plt.subplots(1, len(probs), figsize=(5.5 * len(probs), 4.2), squeeze=False)
    cols = {"two-stage": C_TS, "SPO+": C_SPO, "IMLE": C_IMLE, "SFGE": C_SFGE, "PolyStep": C_PS}
    for j, p in enumerate(probs):
        ax = axes[0][j]; ns = C[p]["ntrains"]
        for m in C[p]["methods"]:
            ys = [C[p]["curve"][n][m]["mean"] for n in ns]
            ax.plot(ns, ys, "-o", color=cols[m], label=m)
        ax.set_xscale("log"); ax.set_xlabel("n_train (log)"); ax.set_ylabel("normalized regret")
        ax.set_title(f"(C) {PLABEL[p]}"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_forwardsolves(A, path):
    dims = A["dims"]; pd = A["per_dim"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(dims)); w = 0.38
    for k, (m, c) in enumerate((("SFGE", C_SFGE), ("PolyStep", C_PS))):
        ys = [pd[d]["solves_to_target"][m]["mean"] for d in dims]
        ax.bar(x + (k - 0.5) * w, ys, w, color=c, label=m)
    ax.set_xticks(x); ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("knapsack n_items"); ax.set_ylabel("forward-solves to reach target regret")
    ax.set_title("(D) Forward-solves to target regret (GPU-solver proxy)")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ===========================================================================
# markdown
# ===========================================================================
def to_markdown(A, B, C, meta):
    L = ["# PolyStep vs SFGE characterization (probing SFGE's three weaknesses)", "",
         f"deg={meta['deg']}, smoke={meta['smoke']}. Forward-solve counts via `SolveCounter`. "
         "HONEST framing: prior bias-variance work found PolyStep ~= SFGE; a tie on any axis is a "
         "legitimate reported outcome.", ""]

    # (A)
    L += ["## (A) Variance vs dimension (knapsack n_items)", "",
          "Headline = variance of the proposed update at a FIXED warm-start point. "
          "`CV^2 = trace(Cov(step)) / ||mean step||^2` (scale-invariant). "
          "Hypothesis tested: SFGE step-variance grows faster with dimension than PolyStep's.", ""]
    dims = A["dims"]
    rows = []
    for d in dims:
        pdd = A["per_dim"][d]
        rows.append([d,
                     fmt_mean_std(pdd["regret"]["SFGE"]), fmt_mean_std(pdd["regret"]["PolyStep"]),
                     f"{pdd['cv2']['SFGE']['mean']:.3g}", f"{pdd['cv2']['PolyStep']['mean']:.3g}",
                     f"{pdd['solves_final']['SFGE']['mean']:.0f}",
                     f"{pdd['solves_final']['PolyStep']['mean']:.0f}"])
    L.append(md_table(["n_items", "SFGE regret", "PolyStep regret", "SFGE CV^2", "PolyStep CV^2",
                       "SFGE solves", "PS solves"], rows))
    L += ["",
          f"- across-dim Wilcoxon (PolyStep regret-std < SFGE regret-std): "
          f"p = {A['wilcoxon_PS_std_lt_SFGE']}",
          f"- across-dim Wilcoxon (PolyStep CV^2 < SFGE CV^2): p = {A['wilcoxon_PS_cv2_lt_SFGE']}", ""]

    # (B)
    L += ["## (B) HP sensitivity (usable HP range = within 10% of own best)", ""]
    rows = []
    for p in B:
        su = B[p]["sfge_usable"]; pu = B[p]["ps_usable"]
        rows.append([PLABEL[p],
                     f"{su['n_usable']}/{len(B[p]['sigmas'])}", f"{su['decades']:.2f}",
                     str(su["best_hp"]),
                     f"{pu['n_usable']}/{len(B[p]['probe_radii'])}", f"{pu['decades']:.2f}",
                     str(pu["best_hp"])])
    L.append(md_table(["problem", "SFGE usable sigmas", "SFGE width (dec)", "SFGE best sigma",
                       "PS usable probe_r", "PS width (dec)", "PS best probe_r"], rows))
    for p in B:
        L += ["", f"### {PLABEL[p]} curves",
              md_table(["sigma"] + [str(s) for s in B[p]["sigmas"]],
                       [["SFGE regret"] + [f"{v:.4f}" for v in B[p]["sfge_mean"]]]), "",
              md_table(["probe_radius"] + [str(s) for s in B[p]["probe_radii"]],
                       [["PolyStep regret"] + [f"{v:.4f}" for v in B[p]["ps_mean"]]])]
    L.append("")

    # (C)
    L += ["## (C) Sample efficiency (structure-exploiting vs structure-free camps)", "",
          "NOTE: SFGE and PolyStep are BOTH structure-free, so this is NOT a PolyStep-over-SFGE "
          "differentiator; the head-to-head is reported for completeness only.", ""]
    for p in C:
        ns = C[p]["ntrains"]; ms = C[p]["methods"]
        rows = [[m] + [fmt_mean_std(C[p]["curve"][n][m]) for n in ns] for m in ms]
        L += [f"### {PLABEL[p]}", md_table(["method"] + [f"n={n}" for n in ns], rows), ""]
        L.append("GF forward-solves: " + ", ".join(
            f"n={n}: SFGE={C[p]['gf_solves'][n]['SFGE']['mean']:.0f} "
            f"PS={C[p]['gf_solves'][n]['PolyStep']['mean']:.0f}" for n in ns))
        L.append("")
    tg = "_smoke" if meta["smoke"] else ""
    L += ["## Figures", f"- exp_results/figs/fig_var_vs_dim{tg}.png",
          f"- exp_results/figs/fig_hp_sensitivity{tg}.png",
          f"- exp_results/figs/fig_sample_efficiency{tg}.png",
          f"- exp_results/figs/fig_forwardsolves{tg}.png"]
    return "\n".join(L)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--deg", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    t0 = time.time()

    if args.smoke:
        A_dims = [10, 20]; A_seeds = [0, 1]; A_ntr, A_nte = 200, 100
        A_sfge_ep, A_ps_steps, A_nvar = 40, 40, 12
        B_probs = ["knap"]; B_sig = [0.05, 0.5, 2.0]; B_pr = [0.2, 0.8, 3.2]
        B_seeds = [0, 1]; B_ntr = 200; B_sfge_ep, B_ps_steps = 50, 50
        C_probs = ["knap"]; C_ntr = [50, 100]; C_seeds = [0, 1]
        C_sfge_ep, C_ps_steps = 50, 50
        tag = "_smoke"
    else:
        A_dims = [10, 20, 50, 100, 200]; A_seeds = [0, 1, 2, 3, 4]; A_ntr, A_nte = 400, 200
        A_sfge_ep, A_ps_steps, A_nvar = 120, 150, 50
        B_probs = ["sp", "knap"]; B_sig = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
        B_pr = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        B_seeds = [0, 1, 2]; B_ntr = 900; B_sfge_ep, B_ps_steps = 120, 150
        C_probs = ["knap", "sp"]; C_ntr = [50, 100, 200, 500, 1000]; C_seeds = [0, 1, 2, 3, 4]
        C_sfge_ep, C_ps_steps = 120, 150
        tag = ""

    A = exp_A(A_dims, A_seeds, args.deg, A_ntr, A_nte, A_sfge_ep, A_ps_steps, A_nvar)
    B = exp_B(B_probs, B_sig, B_pr, B_seeds, args.deg, B_ntr, B_sfge_ep, B_ps_steps)
    C = exp_C(C_probs, C_ntr, C_seeds, args.deg, C_sfge_ep, C_ps_steps)

    meta = {"deg": args.deg, "smoke": args.smoke, "elapsed_s": time.time() - t0}
    payload = {"meta": meta, "A_variance_vs_dim": A, "B_hp_sensitivity": B, "C_sample_efficiency": C}
    write_json(f"exp_results/polystep_vs_sfge{tag}.json", payload)
    write_md(f"exp_results/polystep_vs_sfge{tag}.md", to_markdown(A, B, C, meta))

    fig_var_vs_dim(A, f"{FIGDIR}/fig_var_vs_dim{tag}.png")
    fig_hp_sensitivity(B, f"{FIGDIR}/fig_hp_sensitivity{tag}.png")
    fig_sample_efficiency(C, f"{FIGDIR}/fig_sample_efficiency{tag}.png")
    fig_forwardsolves(A, f"{FIGDIR}/fig_forwardsolves{tag}.png")

    print(f"\nwrote exp_results/polystep_vs_sfge{tag}.{{json,md}} + 4 figures "
          f"({meta['elapsed_s']:.0f}s)\nDONE", flush=True)


if __name__ == "__main__":
    main()
