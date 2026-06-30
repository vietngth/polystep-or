"""Experiment 1b -- HKM model-capacity arm (completes the phase-diagram story).

Hu-Kallus-Mao predict that on the linear program, decision-focused learning helps when the model is
misspecified-but-simple, and that a sufficiently FLEXIBLE model restores the plug-in (two-stage)
estimator. We test this directly on shortest path (the row where HKM is rigorous) by sweeping the
predictor capacity from linear to a small MLP and measuring whether the decision-focused advantage
over two-stage SHRINKS as capacity rises.

Methods: two-stage, SPO+, SFGE, PolyStep (each at the shortest-path best hyperparameter; all
decision-focused methods warm-started from the same two-stage model of the SAME architecture).
Predictor capacity: linear, MLP-8, MLP-32. All trainers handle MLP via pto.forward.batched_predict
and state_dict warm starts.

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp1_capacity.py [degs] [seeds]
"""
from __future__ import annotations
import sys
sys.path.insert(0, "polystep/src")
import numpy as np
import torch
import torch.nn as nn
from pyepo import metric
import pyepo.func as F
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from pto.capability import setup_sp, dev, PF
from pto.forward import batched_predict
from pto.seeding import seed_everything, device_generator
from pto.multiseed import summarize, md_table, write_json, write_md

CAPS = ["linear", "mlp8", "mlp32"]
# shortest-path best hyperparameters (from the LR sweep)
HP = {"spo_lr": 3e-3, "sfge_lr": 1e-1, "ps_sr": 0.2}


def make_predictor(kind, dim):
    if kind == "linear":
        return nn.Linear(PF, dim, bias=True).to(dev)
    h = int(kind[3:])
    return nn.Sequential(nn.Linear(PF, h), nn.ReLU(), nn.Linear(h, dim)).to(dev)


def train_two_stage(cfg, make, epochs=80):
    m = make(); opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(epochs):
        for xb, cb, wb, zb in cfg["ld_tr"]:
            xb, cb = xb.float().to(dev), cb.float().to(dev)
            opt.zero_grad(); ((m(xb) - cb) ** 2).mean().backward(); opt.step()
    return m


def train_spoplus(cfg, make, warm, lr=HP["spo_lr"], epochs=100):
    m = make(); m.load_state_dict(warm.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr); spop = F.SPOPlus(cfg["om"])
    for _ in range(epochs):
        for xb, cb, wb, zb in cfg["ld_tr"]:
            xb, cb, wb, zb = [t.float().to(dev) for t in (xb, cb, wb, zb)]
            opt.zero_grad(); spop(m(xb), cb, wb, zb).mean().backward(); opt.step()
    return m


def train_sfge(cfg, make, warm, lr=HP["sfge_lr"], epochs=120, n_samples=8, sigma=0.5, seed=0):
    m = make(); m.load_state_dict(warm.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr); g = device_generator(seed, dev)
    X, Cs, solve, sgn = cfg["Xtr"], cfg["Cs"], cfg["ps_solve"], cfg["sign"]
    for _ in range(epochs):
        pred = m(X)
        with torch.no_grad():
            eps = torch.randn(n_samples, *pred.shape, device=dev, generator=g)
            chat = pred.unsqueeze(0) + sigma * eps; S, B, D = chat.shape
            w = solve(chat.reshape(S * B, D)).reshape(S, B, D)
            r = sgn * (w * Cs.unsqueeze(0)).sum(-1); adv = r - r.mean(0, keepdim=True)
        logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
        surr = (adv * logp).mean(); opt.zero_grad(); surr.backward(); opt.step()
    return m


def train_polystep(cfg, make, warm, sr=HP["ps_sr"], steps=150, seed=0):
    m = make(); m.load_state_dict(warm.state_dict())
    pso = PolyStepOptimizer(m, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=sr, probe_radius=2 * sr, num_probe=1, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    X, Cs, solve, sgn = cfg["Xtr"], cfg["Cs"], cfg["ps_solve"], cfg["sign"]
    def closure(bp):
        pred = batched_predict(m, bp, X); N, B, E = pred.shape
        w = solve(pred.reshape(N * B, E)).reshape(N, B, E)
        return sgn * (w * Cs.unsqueeze(0)).sum(-1).mean(-1)
    for _ in range(steps):
        pso.step(closure)
    return m


def run(degs, seeds):
    results = {}
    for cap in CAPS:
        for deg in degs:
            acc = {m: [] for m in ("two-stage", "SPO+", "SFGE", "PolyStep")}
            for seed in seeds:
                seed_everything(seed)
                cfg, _ = setup_sp(seed, deg)
                mk = lambda: make_predictor(cap, cfg["dim"])
                ts = train_two_stage(cfg, mk)
                acc["two-stage"].append(metric.regret(ts, cfg["om"], cfg["ld_te"]))
                try:
                    acc["SPO+"].append(metric.regret(train_spoplus(cfg, mk, ts), cfg["om"], cfg["ld_te"]))
                except Exception:
                    acc["SPO+"].append(float("nan"))
                acc["SFGE"].append(metric.regret(train_sfge(cfg, mk, ts, seed=seed), cfg["om"], cfg["ld_te"]))
                acc["PolyStep"].append(metric.regret(train_polystep(cfg, mk, ts, seed=seed), cfg["om"], cfg["ld_te"]))
            summ = {m: summarize(acc[m]) for m in acc}
            best_gf = min(("SFGE", "PolyStep"), key=lambda m: summ[m]["mean"])
            adv = (summ["two-stage"]["mean"] - summ[best_gf]["mean"]) / max(summ["two-stage"]["mean"], 1e-9)
            results[f"{cap}|{deg}"] = {"cap": cap, "deg": deg, "summary": summ,
                                       "best_gradfree": best_gf, "adv_vs_two_stage": adv}
            print(f"  {cap:>6} deg={deg}: two-stage={summ['two-stage']['mean']:.4f} "
                  f"best_gf({best_gf})={summ[best_gf]['mean']:.4f}  adv={adv:+.1%}", flush=True)
    return results


def main():
    degs = [int(d) for d in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2, 4, 6, 8]
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3, 4]
    print(f"CAPACITY ARM (shortest path) | caps={CAPS} degs={degs} seeds={seeds}", flush=True)
    results = run(degs, seeds)
    write_json("exp_results/capacity.json", {"caps": CAPS, "degs": degs, "seeds": seeds, "results": results})
    # markdown: advantage of best gradient-free over two-stage, capacity x degree
    L = ["# Experiment 1b -- HKM model-capacity arm (shortest path, LP)", "",
         f"seeds={seeds}. Decision-focused advantage of the best gradient-free method over two-stage, "
         "as predictor capacity rises. HKM predicts the advantage shrinks as the model becomes flexible "
         "enough to fit the misspecified truth.", ""]
    headers = ["capacity"] + [f"deg={d}" for d in degs]
    rows = []
    for cap in CAPS:
        rows.append([cap] + [f"{results[f'{cap}|{d}']['adv_vs_two_stage']:+.0%}" for d in degs])
    L.append(md_table(headers, rows))
    L += ["", "Two-stage normalized regret (should fall as capacity rises if the MLP fits the truth):"]
    rows2 = [[cap] + [f"{results[f'{cap}|{d}']['summary']['two-stage']['mean']:.4f}" for d in degs] for cap in CAPS]
    L.append(md_table(headers, rows2))
    write_md("exp_results/capacity.md", "\n".join(L))
    print("\nwrote exp_results/capacity.{json,md}\nDONE", flush=True)


if __name__ == "__main__":
    main()
