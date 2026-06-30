"""Experiment #5 -- Ablation: the OT-structured step vs generic gradient-free.

Shows the OT-structured step -- NOT mere gradient-freeness -- is what helps, so PolyStep is not
interchangeable with any zeroth-order optimizer. Everything (closure, batched forward solver,
warm start, forward-solve budget) is held IDENTICAL; only the optimizer / update rule changes:

  PolyStep-OT       : entropic OT (Sinkhorn) barycentric step      [PolyStep default in full-space]
  PolyStep-softmax  : one-sided softmax barycentric step           [drop the column marginal]
  PolyStep-greedy   : min-cost-greedy hard selection               [hard-selection ablation]
  PolyStep-topk     : top-k-mean hard selection                    [hard-selection ablation]
  CMA-ES            : covariance-adaptation evolution strategy      [generic ZO via pycma]
  SPSA              : simultaneous-perturbation finite difference   [generic ZO]
  SFGE              : score-function / REINFORCE                    [the gradient-free sibling]

NB our linear predictors are FULL-SPACE (P particles >> V=2*particle_dim vertices), so OT != softmax
here (the column-marginal is active) -- the OT-vs-softmax ablation is meaningful, unlike the
few-particle subspace regime where they coincide. All optimizers reuse the SAME closure(bp)->(N,)
contract and are compared at a MATCHED #forward-solve budget (PolyStep's solve count sets the budget).

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp5_ot_ablation.py [problems] [deg] [seeds]
"""
from __future__ import annotations
import sys
sys.path.insert(0, "polystep/src")
import numpy as np
import torch
from pyepo import metric
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from pto.capability import SETUPS, train_two_stage, train_sfge, dev
from pto.budget import SolveCounter
from pto.seeding import seed_everything
from pto.multiseed import summarize, md_table, write_json, write_md, fmt_mean_std

CATLABEL = {"sp": "shortest_path (LP)", "knap": "knapsack (ILP)", "tsp": "tsp (ILP)", "port": "portfolio (SOCP)"}
PS_SOLVERS = {"PolyStep-OT": "sinkhorn", "PolyStep-softmax": "softmax",
              "PolyStep-greedy": "min_cost_greedy", "PolyStep-topk": "top_k_mean"}


def make_flat_eval(cfg, solve):
    """Shared per-candidate objective: weight batch (N,dim,PF) -> (N,) loss to MINIMIZE."""
    X, Cs, sgn = cfg["Xtr"], cfg["Cs"], cfg["sign"]
    def eval_W(Wb):
        pred = torch.einsum("nef,bf->nbe", Wb, X); N, nb, E = pred.shape
        w = solve(pred.reshape(N * nb, E)).reshape(N, nb, E)
        return sgn * (w * Cs.unsqueeze(0)).sum(-1).mean(-1)
    return eval_W


def train_polystep_solver(cfg, warm, solver, solve, steps=150, sr=0.4, seed=0):
    m = cfg["make"]()
    with torch.no_grad():
        m.weight.copy_(warm.weight)
    pso = PolyStepOptimizer(m, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=sr, probe_radius=2 * sr, num_probe=1, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9, solver=solver)
    X, Cs, sgn = cfg["Xtr"], cfg["Cs"], cfg["sign"]
    def closure(bp):
        pred = torch.einsum("nef,bf->nbe", bp["weight"], X); N, nb, E = pred.shape
        w = solve(pred.reshape(N * nb, E)).reshape(N, nb, E)
        return sgn * (w * Cs.unsqueeze(0)).sum(-1).mean(-1)
    for _ in range(steps):
        pso.step(closure)
    return m


def cma_train(cfg, warm, solve, max_solves, sigma0=0.3, seed=0):
    import cma
    W0 = warm.weight.detach(); dim, PF = W0.shape; nt = int(cfg["Xtr"].shape[0])
    es = cma.CMAEvolutionStrategy(W0.flatten().cpu().numpy().astype(float), sigma0,
                                  {"seed": seed + 1, "verbose": -9})
    evalW = make_flat_eval(cfg, solve); solves = 0
    while solves < max_solves and not es.stop():
        cands = es.ask()
        Wb = torch.tensor(np.stack(cands), dtype=torch.float32, device=dev).reshape(len(cands), dim, PF)
        es.tell(cands, evalW(Wb).detach().cpu().numpy().tolist())
        solves += len(cands) * nt
    best = torch.tensor(es.result.xbest, dtype=torch.float32, device=dev).reshape(dim, PF)
    m = cfg["make"]()
    with torch.no_grad():
        m.weight.copy_(best)
    return m


def spsa_train(cfg, warm, solve, max_solves, a=0.1, c=0.15, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed + 1)
    W = warm.weight.detach().clone(); nt = int(cfg["Xtr"].shape[0])
    evalW = make_flat_eval(cfg, solve)
    steps = max(1, max_solves // (2 * nt)); solves = 0; t = 0
    while solves < max_solves:
        t += 1
        delta = torch.randint(0, 2, W.shape, generator=g, device=dev).float() * 2 - 1
        ct = c / t ** 0.101; at = a / t ** 0.602
        Lp = evalW((W + ct * delta).unsqueeze(0))[0]
        Lm = evalW((W - ct * delta).unsqueeze(0))[0]
        W = W - at * (Lp - Lm) / (2 * ct) * (1.0 / delta)
        solves += 2 * nt
    m = cfg["make"]()
    with torch.no_grad():
        m.weight.copy_(W)
    return m


def run_cell(problem, deg, seeds, cold=False):
    """cold=True: all optimizers start from the SAME random init (the paper's from-scratch ablation
    protocol), so the update rule must do the work. cold=False: shared two-stage warm start (near the
    optimum -> the rule barely matters, methods tie -- an honest secondary view)."""
    methods = list(PS_SOLVERS) + ["CMA-ES", "SPSA", "SFGE"]
    acc = {m: [] for m in methods}; solves = {m: [] for m in methods}
    for seed in seeds:
        seed_everything(seed)
        cfg, _ = SETUPS[problem](seed, deg)
        ts = cfg["make"]() if cold else train_two_stage(cfg)      # shared init for every optimizer
        cfg["warm"] = ts
        # PolyStep-OT first, with a solve counter, to set the matched budget
        for name, solver in PS_SOLVERS.items():
            sc = SolveCounter(cfg["ps_solve"])
            try:
                m = train_polystep_solver(cfg, ts, solver, sc, steps=150, seed=seed)
                acc[name].append(metric.regret(m, cfg["om"], cfg["ld_te"]))
            except Exception as e:
                acc[name].append(float("nan")); print(f"      {name} failed: {e}", flush=True)
            solves[name].append(sc.instances)
        budget = int(np.median([s for s in solves["PolyStep-OT"] if s])) or 200000
        for name, fn in (("CMA-ES", cma_train), ("SPSA", spsa_train)):
            sc = SolveCounter(cfg["ps_solve"])
            m = fn(cfg, ts, sc, budget, seed=seed)
            acc[name].append(metric.regret(m, cfg["om"], cfg["ld_te"])); solves[name].append(sc.instances)
        scf = SolveCounter(cfg["ps_solve"]); cc = dict(cfg); cc["ps_solve"] = scf; cc["warm"] = ts
        acc["SFGE"].append(metric.regret(train_sfge(cc), cfg["om"], cfg["ld_te"])); solves["SFGE"].append(scf.instances)
        print(f"    seed {seed}: " + "  ".join(f"{m}={acc[m][-1]:.4f}" for m in methods), flush=True)
    return {"regret": {m: summarize(acc[m]) for m in methods},
            "solves": {m: summarize(solves[m]) for m in methods}, "methods": methods}


def main():
    problems = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sp", "knap"]
    deg = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2, 3, 4]
    cold = (len(sys.argv) > 4 and sys.argv[4].lower() == "cold")
    print(f"OT-STEP ABLATION | problems={problems} deg={deg} seeds={seeds} init={'cold' if cold else 'warm'}", flush=True)
    results = {}
    for p in problems:
        print(f"[ablation] {p}", flush=True)
        results[p] = run_cell(p, deg, seeds, cold=cold)
    payload = {"problems": problems, "deg": deg, "seeds": seeds, "init": "cold" if cold else "warm", "results": results}
    write_json("exp_results/ot_ablation.json", payload)
    write_md("exp_results/ot_ablation.md", to_markdown(results, problems, deg, seeds))
    print("\nwrote exp_results/ot_ablation.{json,md}\nDONE", flush=True)


def to_markdown(results, problems, deg, seeds):
    L = ["# Experiment #5 -- OT-structured step vs generic gradient-free", "",
         f"deg={deg}, seeds={seeds}. Identical closure / solver / warm-start; only the optimizer changes, "
         "at a matched #forward-solve budget. Normalized regret (lower better). Full-space linear models "
         "(P>>V), so OT != softmax here.", ""]
    for p in problems:
        r = results[p]; methods = r["methods"]
        L.append(f"## {CATLABEL.get(p, p)}")
        rows = [[m, fmt_mean_std(r["regret"][m]), f"{r['solves'][m]['mean']:.0f}"] for m in methods]
        L.append(md_table(["optimizer", "regret (mean±std)", "fwd_solves"], rows)); L.append("")
    L.append("**Reading:** if PolyStep-OT/softmax (smooth barycentric step) beat the hard-selection "
             "ablations (greedy/top-k) and the generic ZO methods (CMA-ES, SPSA) at matched budget, the "
             "OT-structured update -- not gradient-freeness alone -- is the source of the advantage.")
    return "\n".join(L)


if __name__ == "__main__":
    main()
