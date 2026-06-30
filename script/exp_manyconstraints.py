"""
MANY-CONSTRAINT sweep WITH cost instrumentation (runtime / #forward-passes / #solver-calls).

How does PolyStep perform — and at what compute cost — as the number of HARD constraints in the inner
solver grows? Two regimes:

  (1) OBJECTIVE prediction (SPO+ APPLIES): grid shortest path; #flow-constraints = #nodes grows with
      the grid {5x5..16x16}. PolyStep vs SPO+ vs two-stage. PolyStep target = parity on regret, with a
      very different cost profile (0 exact-solver calls vs SPO+'s per-instance Gurobi calls).
  (2) CONSTRAINT prediction (SPO+ = N/A): multi-dim knapsack, predicted consumption in m_res
      constraints {2..40}. PolyStep vs two-stage (greedy+repair batched GPU solver).

Per method we record: regret, wall-clock(s), #NN-forward-passes, #solver(oracle)-calls.
Accounting (faithful to exp_results/pareto.md):
  PolyStep : solves = NN_fwd = (2*D)*n_train*steps    (D = predictor params; orthoplex 2*d_p*ceil(D/d_p)=2D, K=1)
  SPO+     : gurobi  = epochs*n_train + n_train ;  NN_fwd = epochs*n_train
  two-stage: solves  = 0 ;                          NN_fwd = epochs*n_train

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 .venv/bin/python exp_manyconstraints.py
"""
import os, sys, time, json, numpy as np, torch
os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, "polystep/src")
from polystep.epsilon import CosineEpsilon
from pto import (ShortestPath, MDKPConsumption, train_two_stage, train_spo_plus, train_polystep)

P = lambda *a: print(*a, flush=True)
# chunk_size caps PolyStep's per-step probe batch (N) handed to the closure so the
# closure's (N*nb,*) tensors stay memory-bounded at large grids / m_res (numerically
# identical: probes are evaluated in slices, cost matrix reassembled the same way).
PS_CHUNK = int(os.environ.get("PS_CHUNK", "1024"))
LIN = dict(polytope_type="orthoplex", num_probe=1, use_momentum=True, momentum_init=0.5,
           momentum_final=0.9, chunk_size=PS_CHUNK)
def ms(xs): return float(np.mean(xs)), float(np.std(xs))
def ndim(prob): return int(sum(p.numel() for p in prob.predictor().parameters()))


def timed(fn):
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter(); out = fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def objective_prediction(seeds=(0, 1, 2)):
    P("\n=== (1) OBJECTIVE prediction: grid shortest path, #flow-constraints=#nodes grows (SPO+ applies) ===")
    P(f"{'grid':>7} {'#cons':>6} {'D':>5} | {'two-stage reg':>13} {'SPO+ reg':>10} {'PolyStep reg':>12} "
      f"| {'PS s':>6} {'SPO+ s':>7} | {'PS solves':>11} {'SPO+ gurobi':>11}")
    cfg = dict(LIN, epsilon=CosineEpsilon(0.5, 0.05), step_radius=0.4, probe_radius=0.8)
    EP_SPO, ST_PS, EP_TS, NTR = 40, 120, 60, 800
    rows = []
    for (H, W) in [(5, 5), (8, 8), (12, 12), (16, 16)]:
        ts, sp, ps, tps, tsp = [], [], [], [], []
        D = 0
        for s in seeds:
            prob = ShortestPath(H=H, W=W, deg=6, n_train=NTR, n_val=200, n_test=200, seed=40 + s)
            D = ndim(prob)
            m_ts = train_two_stage(prob, epochs=EP_TS); ts.append(prob.regret(m_ts))
            (m_sp, dt_sp) = timed(lambda: train_spo_plus(prob, epochs=EP_SPO)); sp.append(prob.regret(m_sp)); tsp.append(dt_sp)
            (m_ps, dt_ps) = timed(lambda: train_polystep(prob, cfg, steps=ST_PS, warm=m_ts, seed=s)); ps.append(prob.regret(m_ps)); tps.append(dt_ps)
        a, b, c = ms(ts), ms(sp), ms(ps)
        ps_solves = 2 * D * NTR * ST_PS                  # batched forward-solves (oracle evals)
        spo_gurobi = EP_SPO * NTR + NTR                  # per-instance Gurobi calls
        P(f"{H}x{W:<4} {H*W:>6} {D:>5} | {a[0]:>7.4f}±{a[1]:<5.4f} {b[0]:>6.4f}±{b[1]:<3.4f} {c[0]:>6.4f}±{c[1]:<5.4f} "
          f"| {np.mean(tps):>5.1f} {np.mean(tsp):>6.1f} | {ps_solves:>11,} {spo_gurobi:>11,}")
        rows.append(dict(grid=f"{H}x{W}", n_cons=H * W, D=D, two_stage=a, spo=b, polystep=c,
                         ps_wall_s=float(np.mean(tps)), spo_wall_s=float(np.mean(tsp)),
                         ps_forward_solves=ps_solves, ps_solver_calls=ps_solves,
                         spo_gurobi_calls=spo_gurobi, two_stage_solver_calls=0))
    return rows


def constraint_prediction(seeds=(0, 1, 2)):
    P("\n=== (2) CONSTRAINT prediction: MDKP, predicted consumption in m_res constraints (SPO+ = N/A) ===")
    P(f"{'m_res':>6} {'D':>6} | {'two-stage reg':>13} {'PolyStep reg':>12} {'cut':>6} "
      f"| {'PS s':>6} | {'PS solves':>13} {'SPO+':>5}")
    cfg = dict(LIN, epsilon=CosineEpsilon(1.0, 0.08), step_radius=1.0, probe_radius=2.0)
    ST_PS, NTR = 150, 256
    rows = []
    MRES = [int(x) for x in os.environ.get("MC_MRES", "2,5,10,20,40").split(",")]   # constraint-count sweep (env-overridable)
    for m_res in MRES:
        ts, ps, tps = [], [], []
        D = 0
        for s in seeds:
            prob = MDKPConsumption(n_item=40, m_res=m_res, deg=8, fill=0.2,
                                   n_train=NTR, n_val=256, n_test=1500, seed=s)
            D = ndim(prob)
            m_ts = train_two_stage(prob, epochs=60); ts.append(prob.regret(m_ts))
            (m_ps, dt) = timed(lambda: train_polystep(prob, cfg, steps=ST_PS, warm=m_ts, seed=s))
            ps.append(prob.regret(m_ps)); tps.append(dt)
        a, b = ms(ts), ms(ps); cut = (a[0] - b[0]) / a[0] * 100 if a[0] > 1e-6 else 0.0
        ps_solves = 2 * D * NTR * ST_PS
        P(f"{m_res:>6} {D:>6} | {a[0]:>7.4f}±{a[1]:<5.4f} {b[0]:>6.4f}±{b[1]:<5.4f} {cut:>+5.0f}% "
          f"| {np.mean(tps):>5.1f} | {ps_solves:>13,} {'N/A':>5}")
        rows.append(dict(m_res=m_res, n_cons=m_res, D=D, two_stage=a, polystep=b, cut_pct=cut,
                         ps_wall_s=float(np.mean(tps)), ps_forward_solves=ps_solves,
                         ps_solver_calls=ps_solves, spo="N/A"))
    return rows


if __name__ == "__main__":
    SEEDS = tuple(int(x) for x in os.environ.get("MC_SEEDS", "0,1,2").split(","))   # seed count (env-overridable)
    t0 = time.time()
    r1 = objective_prediction(SEEDS)
    r2 = constraint_prediction(SEEDS)
    json.dump({"objective": r1, "constraint": r2}, open("exp_results/many_constraints.json", "w"), indent=1)
    P(f"\n[done in {time.time()-t0:.0f}s] -> exp_results/many_constraints.json")
    P("Note: PolyStep makes 0 exact-solver (Gurobi) calls; its solves are cheap batched-GPU oracle evals.")
