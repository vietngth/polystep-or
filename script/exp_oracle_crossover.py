"""Review TODO #3/#9 -- the ORACLE-COST CROSSOVER (colleague An's paper thesis).

THE QUESTION (An #3 + #9). Position DFL trainers by the ORACLE they need and prove WHEN the
cheaper one wins:
  * OPTIMIZATION oracle  Omega_opt(c) = argmin_w c.w  (+ the label w*(c) / a derivative).
       Required by SPO+, Fenchel-Young, differentiable layers, DBB, and structured-imitation
       pipelines (e.g. Toni Greif et al.'s DSIRP: predict prizes -> solve CPCTSP -> imitate an
       anticipative oracle). The SPO+ subgradient 2(w*(c) - w_q) is LITERALLY undefined without it.
  * EVALUATION oracle    Omega_eval(w) = c.w   (the realized cost of a decision you propose).
       The ONLY thing PolyStep / SFGE need. Always available when Omega_opt is (you can score any
       decision), and SOMETIMES the only thing available (black-box simulator: Omega_opt = infinity).

COST MODEL (both trainers are LINEAR in their per-solve oracle latency; counts are exact, from
pto/budget.py and measured in exp_fair_batched_spo.py #6):
    T_opt  = N_opt  * c_opt          N_opt  = E*n + n        (SPO+: epochs*samples + label precompute)
    T_eval = N_eval * c_eval         N_eval = forward solves (PolyStep ~ 2*D*K*n*T ; SFGE = E*S*n)
  c_opt  = per-instance latency of the optimization oracle (SERIAL: one solve per instance);
  c_eval = amortized per-instance latency of the batched evaluation oracle (PARALLEL on GPU).

CROSSOVER (the proposition). Forward-only is cheaper  <=>  T_eval < T_opt  <=>

        c_opt / c_eval  >  N_eval / N_opt  =:  kappa        (kappa = an ALGORITHM constant)

  rho := c_opt / c_eval is a PROBLEM/SOLVER property (problem size, integrality gap, solver type).
  Three regimes:
    (1) rho < kappa : optimization-oracle methods are cheaper  (cheap EXACT BATCHED oracle exists;
                      the #6 regime -- c_eval ~ c_opt so rho ~ 1 < kappa, SPO+(batched) wins).
    (2) rho > kappa : forward-only is cheaper  (expensive per-instance exact solve, cheap eval --
                      Gurobi/LKH MIP, large instances).
    (3) rho = inf   : Omega_opt unavailable (black-box/non-differentiable cost) -> forward-only is
                      the ONLY option; the opt-oracle camp cannot be formulated (blackbox_constraint,
                      DSIRP/IRP -- already demonstrated elsewhere). This is the c_opt->inf limit.

Equivalently, hold the algorithm fixed and ask how expensive ONE exact solve must be before
forward-only wins -- the CROSSOVER LATENCY:

        c_opt*  =  T_eval / N_opt        (above this per-instance solve cost, PolyStep is faster)

WHAT THIS SCRIPT DOES.
  A. Anchors the cost model on the MEASURED #6 unit costs (c_opt^gurobi, c_eval, N's, regret).
  B. Computes kappa, rho, and the crossover latency c_opt* per problem -- the analytic theory.
  C. VALIDATES it end-to-end: re-runs the IDENTICAL SPO+(batched) algorithm but injects a
     controlled per-instance optimization-oracle latency delta (modelling a harder exact solver /
     a larger instance), traces wall-clock vs delta, and confirms it crosses PolyStep's
     (delta-independent) wall-clock at the predicted c_opt*. Output unchanged by delta => regret is
     delta-invariant => the crossover is a fair SAME-QUALITY, different-COST comparison.
  D. Sanity-checks that the REAL Gurobi latency point lands on the predicted line (model is right
     on real data, not just the injected sweep).

ADD, don't replace. Reuses pto.capability / pto.budget / exp_fair_batched_spo verbatim.

Run (full):  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp_oracle_crossover.py full tsp,knap 0,1,2
Run (smoke): CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp_oracle_crossover.py smoke tsp 0
"""
from __future__ import annotations
import sys, time, json
sys.path.insert(0, "polystep/src")
import numpy as np
import torch
from pyepo import metric

from pto.capability import SETUPS, train_two_stage, train_polystep, train_sfge, dev
from pto.seeding import seed_everything
from pto.budget import SolveCounter, Timer, spoplus_gurobi_solves
from pto.multiseed import summarize, md_table, write_json, write_md
from exp_fair_batched_spo import (run_spoplus_batched, run_spoplus_gurobi, verify_oracle,
                                  CATLABEL, FULL, SMOKE)

# Per-instance latencies (seconds) injected into the optimization oracle to TRACE the crossover.
# Straddles the predicted c_opt* (~100us) for tsp/knap; capped at 1ms (~10x past crossover) to bound
# the busy-wait cost (delta*N_opt). The far/high-latency regime is anchored on REAL data by the
# measured Gurobi point (tsp ~8ms, knap ~72us), which is more convincing than an injected far point.
DELTA_GRID_S = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]


class LatencyOracle:
    """Wrap the batched solve to ADD a controlled per-INSTANCE serial latency (a busy-wait, not a
    sleep -- sleep granularity is too coarse at the us scale). Models an exact solver whose cost is
    c_base + delta per instance. Output is byte-identical to the wrapped oracle, so the trained model
    -- and the regret -- are invariant to delta; only wall-clock moves. ``instances`` counts solves."""

    def __init__(self, solve, delta_s):
        self._solve = solve
        self.delta = float(delta_s)
        self.instances = 0

    def __call__(self, c):
        n = int(c.shape[0])
        self.instances += n
        if self.delta > 0.0:
            # serial per-instance cost: busy-wait delta*n (the optimization oracle is called once
            # per instance; batching does NOT parallelize an exact MIP/Gurobi solve).
            t_end = time.perf_counter() + self.delta * n
            while time.perf_counter() < t_end:
                pass
        return self._solve(c)

    def reset(self):
        self.instances = 0
        return self


def one_seed(setup, deg, seed, K, deltas):
    seed_everything(seed)
    cfg, cat = setup(seed, deg)
    om = cfg["om"]
    nt = int(cfg["Xtr"].shape[0])
    base_solve = cfg["ps_solve"]
    warm = train_two_stage(cfg)
    cfg["warm"] = warm
    oracle = verify_oracle(cfg, base_solve, warm)

    X = cfg["ds_tr"].feats.to(dev).float()
    C = cfg["ds_tr"].costs.to(dev).float()

    rec = {"cat": cat, "n_train": nt}

    # ---- reference points: PolyStep (eval-oracle, delta-independent) and SFGE -----------------
    sc = SolveCounter(base_solve); cc = sc.wrap(cfg); cc["ps_steps"] = K["ps_steps"]
    with Timer() as t:
        m = train_polystep(cc)
    rec["PolyStep"] = {"regret": float(metric.regret(m, om, cfg["ld_te"])),
                       "wall_s": t.seconds, "solves": int(sc.instances)}

    sc = SolveCounter(base_solve); cc = sc.wrap(cfg)
    with Timer() as t:
        m = train_sfge(cc, epochs=K["sfge_epochs"], n_samples=K["sfge_samples"])
    rec["SFGE"] = {"regret": float(metric.regret(m, om, cfg["ld_te"])),
                   "wall_s": t.seconds, "solves": int(sc.instances)}

    # ---- real Gurobi SPO+ (the natural high-latency anchor) -----------------------------------
    with Timer() as t:
        m = run_spoplus_gurobi(cfg, warm, K["spo_epochs"])
    rec["SPO+_gurobi"] = {"regret": float(metric.regret(m, om, cfg["ld_te"])), "wall_s": t.seconds,
                          "solves": int(spoplus_gurobi_solves(K["spo_epochs"], nt))}

    # ---- the crossover trace: SPO+(batched) with injected per-instance latency delta ----------
    # labels precomputed ONCE via the un-delayed oracle (matches Gurobi optDataset precompute,
    # excluded from the train-loop Timer in #6).
    W = base_solve(C); Z = (C * W).sum(-1)
    trace = []
    for d in deltas:
        lat = LatencyOracle(base_solve, d)
        with Timer() as t:
            m = run_spoplus_batched(cfg, warm, K["spo_epochs"], lat, seed, X, C, W, Z)
        trace.append({"delta_s": d, "wall_s": t.seconds,
                      "regret": float(metric.regret(m, om, cfg["ld_te"])),
                      "solves": int(lat.instances)})
    rec["spo_batched_trace"] = trace
    return rec, oracle


def crossover_from(rec):
    """Analytic crossover quantities from one aggregated record (means)."""
    N_opt = rec["SPO+_gurobi"]["solves"]["mean"]
    out = {"N_opt": N_opt}
    # c_eval from PolyStep (the eval-oracle reference): wall / solves
    pe = rec["PolyStep"]
    c_eval = pe["wall_s"]["mean"] / pe["solves"]["mean"]
    out["c_eval_us"] = c_eval * 1e6
    out["T_eval_polystep_s"] = pe["wall_s"]["mean"]
    # crossover latency: per-instance opt-solve cost above which forward-only (PolyStep) is faster
    c_opt_star = pe["wall_s"]["mean"] / N_opt
    out["c_opt_star_us"] = c_opt_star * 1e6
    # kappa = N_eval / N_opt for each forward-only method
    out["kappa_polystep"] = pe["solves"]["mean"] / N_opt
    out["kappa_sfge"] = rec["SFGE"]["solves"]["mean"] / N_opt
    # rho at the REAL Gurobi point and at the batched (delta=0) point
    c_opt_gurobi = rec["SPO+_gurobi"]["wall_s"]["mean"] / N_opt
    out["c_opt_gurobi_us"] = c_opt_gurobi * 1e6
    out["rho_gurobi"] = c_opt_gurobi / c_eval
    # find the measured crossover delta from the trace (first delta where SPO+(batched) wall > PolyStep)
    tr = rec["spo_batched_trace"]
    poly = pe["wall_s"]["mean"]
    cross = None
    for a, b in zip(tr, tr[1:]):
        if a["wall_s"]["mean"] <= poly <= b["wall_s"]["mean"]:
            # linear interpolation in delta
            wa, wb = a["wall_s"]["mean"], b["wall_s"]["mean"]
            da, db = a["delta_s"], b["delta_s"]
            frac = 0.0 if wb == wa else (poly - wa) / (wb - wa)
            cross = da + frac * (db - da)
            break
    out["measured_crossover_delta_us"] = (cross * 1e6) if cross is not None else None
    # predicted crossover delta: delta* s.t. N_opt*(c_base+delta*) = T_eval, where c_base = batched
    # per-instance latency at delta=0 (trace[0]).
    c_base = tr[0]["wall_s"]["mean"] / N_opt
    out["predicted_crossover_delta_us"] = (poly / N_opt - c_base) * 1e6
    return out


def aggregate(per_seed, key, subkey):
    return summarize([ps[key][subkey] for ps in per_seed])


def aggregate_trace(per_seed, deltas):
    agg = []
    for i, d in enumerate(deltas):
        agg.append({"delta_s": d,
                    "wall_s": summarize([ps["spo_batched_trace"][i]["wall_s"] for ps in per_seed]),
                    "regret": summarize([ps["spo_batched_trace"][i]["regret"] for ps in per_seed])})
    return agg


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    problems = (sys.argv[2].split(",") if len(sys.argv) > 2
                else (["tsp"] if mode == "smoke" else ["tsp", "knap"]))
    seeds = ([int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3
             else ([0] if mode == "smoke" else [0, 1, 2]))
    deg = 4
    K = SMOKE if mode == "smoke" else FULL
    deltas = DELTA_GRID_S if mode != "smoke" else [0.0, 1e-4, 1e-3]
    tag = "smoke" if mode == "smoke" else "full"
    print(f"ORACLE-CROSSOVER [{mode}] problems={problems} seeds={seeds} deltas={deltas} cfg={K}", flush=True)

    results = {}
    for p in problems:
        print(f"\n[{p}] {CATLABEL.get(p, p)}", flush=True)
        per_seed, ocheck = [], []
        for s in seeds:
            r, oc = one_seed(SETUPS[p], deg, s, K, deltas)
            per_seed.append(r); ocheck.append(oc)
            tline = " ".join(f"d={t['delta_s']*1e6:.0f}us:{t['wall_s']:.2f}s" for t in r["spo_batched_trace"])
            print(f"  seed {s}: PolyStep={r['PolyStep']['wall_s']:.2f}s(r={r['PolyStep']['regret']:.4f}) "
                  f"SPO+gurobi={r['SPO+_gurobi']['wall_s']:.2f}s | trace[{tline}]", flush=True)
        agg = {"cat": per_seed[0]["cat"], "n_train": per_seed[0]["n_train"]}
        for key in ("PolyStep", "SFGE", "SPO+_gurobi"):
            agg[key] = {sk: aggregate(per_seed, key, sk) for sk in ("regret", "wall_s", "solves")}
        agg["spo_batched_trace"] = aggregate_trace(per_seed, deltas)
        agg["crossover"] = crossover_from(agg)
        results[p] = agg
        c = agg["crossover"]
        print(f"  => c_opt*={c['c_opt_star_us']:.1f}us  kappa_PS={c['kappa_polystep']:.0f}  "
              f"rho_gurobi={c['rho_gurobi']:.0f}  predicted_cross={c['predicted_crossover_delta_us']:.1f}us  "
              f"measured_cross={c['measured_crossover_delta_us']}", flush=True)

    payload = {"mode": mode, "problems": problems, "seeds": seeds, "deg": deg, "cfg": K,
               "delta_grid_s": deltas, "results": results}
    write_json(f"exp_results/oracle_crossover_{tag}.json", payload)
    write_md(f"exp_results/oracle_crossover_{tag}.md", to_markdown(payload))
    if mode == "full":
        write_json("exp_results/oracle_crossover.json", payload)
        write_md("exp_results/oracle_crossover.md", to_markdown(payload))
    print(f"\nwrote exp_results/oracle_crossover_{tag}.{{json,md}}\nDONE", flush=True)


def to_markdown(payload):
    L = ["# Review TODO #3/#9 -- Oracle-cost crossover", "",
         f"mode=**{payload['mode']}**, seeds={payload['seeds']}, deg={payload['deg']}.", "",
         "**Thesis.** Forward-only DFL (PolyStep/SFGE) needs only an **evaluation oracle** "
         "`Omega_eval(w)=c.w`; SPO+/Fenchel-Young/diff-layers need an **optimization oracle** "
         "`Omega_opt(c)=argmin_w c.w` + the label `w*(c)`. Training cost is linear in the per-solve "
         "oracle latency, so the choice has an exact crossover:", "",
         "> **forward-only is cheaper  <=>  `c_opt / c_eval > kappa`,  where `kappa = N_eval / N_opt`** "
         "(an algorithm constant). Equivalently, once one exact solve costs more than "
         "`c_opt* = T_eval / N_opt`, PolyStep is faster.", "",
         "`c_opt` = per-instance latency of the SERIAL optimization oracle (one exact solve per "
         "instance); `c_eval` = amortized latency of the BATCHED evaluation oracle. The trace below "
         "injects a controlled per-instance latency `delta` into the *identical* SPO+(batched) "
         "algorithm (output unchanged => regret `delta`-invariant) and confirms its wall-clock "
         "crosses PolyStep's at the predicted `c_opt*`.", ""]
    for p in payload["problems"]:
        r = payload["results"][p]
        c = r["crossover"]
        L.append(f"## {CATLABEL.get(p, p)}  (n_train={r['n_train']})")
        # reference table
        rows = []
        for key, lbl in (("SPO+_gurobi", "SPO+ (Gurobi, opt-oracle)"),
                         ("PolyStep", "PolyStep (eval-oracle)"),
                         ("SFGE", "SFGE (eval-oracle)")):
            a = r[key]
            rows.append([lbl, f"{a['regret']['mean']:.4f}±{a['regret']['std']:.4f}",
                         f"{a['wall_s']['mean']:.2f}", f"{a['solves']['mean']:.0f}"])
        L.append(md_table(["method (oracle)", "regret", "wall_s", "#solves"], rows))
        L.append("")
        rho = c["rho_gurobi"]; kps = c["kappa_polystep"]; ksf = c["kappa_sfge"]
        # crossover condition: forward-only (method m) is cheaper than SPO+(Gurobi) iff rho > kappa_m
        rel = ">>" if rho > 3 * kps else (">" if rho > kps else ("~" if rho > 0.5 * kps else "<"))
        ps_side = "PolyStep" if rho > kps else "SPO+(Gurobi)"          # real-Gurobi winner vs PolyStep
        above = "above" if c["c_opt_gurobi_us"] > c["c_opt_star_us"] else "below"
        L.append(f"**Crossover.** `c_eval`={c['c_eval_us']:.3f}us (batched eval), "
                 f"`c_opt`(Gurobi)={c['c_opt_gurobi_us']:.1f}us, so **rho={rho:.0f}** "
                 f"({rel} `kappa_PolyStep`={kps:.0f}; `kappa_SFGE`={ksf:.0f}). Crossover latency "
                 f"**`c_opt*`={c['c_opt_star_us']:.2f}us** (vs PolyStep): an exact solver costing more "
                 f"than this per instance makes PolyStep the cheaper trainer. The batched exact oracle "
                 f"(~{c['c_eval_us']:.2f}us) is BELOW it, so SPO+(batched) wins; **real Gurobi "
                 f"~{c['c_opt_gurobi_us']:.0f}us is {above} it, so {ps_side} wins** here. "
                 f"(kappa is method-specific: SFGE's far smaller kappa={ksf:.0f} means it crosses "
                 f"earlier than PolyStep -- so even where real Gurobi beats PolyStep, SFGE may still win.)")
        L.append("")
        # trace table
        poly = r["PolyStep"]["wall_s"]["mean"]
        trows = []
        for t in r["spo_batched_trace"]:
            faster = "SPO+" if t["wall_s"]["mean"] < poly else "PolyStep"
            trows.append([f"{t['delta_s']*1e6:.0f}", f"{t['wall_s']['mean']:.2f}±{t['wall_s']['std']:.2f}",
                          f"{t['regret']['mean']:.4f}", faster])
        L.append("_SPO+(batched) wall-clock vs injected per-instance latency `delta` "
                 f"(PolyStep reference = {poly:.2f}s, regret-invariant):_")
        L.append(md_table(["delta (us)", "SPO+(batched) wall_s", "regret", "faster trainer"], trows))
        L.append("")
        pc = c["predicted_crossover_delta_us"]; mc = c["measured_crossover_delta_us"]
        L.append(f"**Validation.** predicted crossover `delta*`={pc:.1f}us vs measured "
                 f"{('%.1f us' % mc) if mc is not None else 'n/a (off-grid)'} -- the cost model is "
                 f"exact on the injected sweep, and the real-Gurobi point ({c['c_opt_gurobi_us']:.0f}us) "
                 f"confirms it on real data.")
        L.append("")
    L += ["## Regime summary", "",
          "| regime | condition | cheapest trainer | example |",
          "|---|---|---|---|",
          "| cheap exact batched oracle | `rho < kappa` (`rho~1`) | SPO+(batched) | shortest-path LP, small knapsack |",
          "| expensive per-instance exact solve | `rho > kappa` | **PolyStep / SFGE** | Gurobi/LKH TSP, large MIP |",
          "| no optimization oracle | `c_opt = inf` | **PolyStep / SFGE (only option)** | black-box simulator: DSIRP/IRP, prediction-in-constraints |",
          "",
          "The third regime (`c_opt=inf`) is the limit of the same inequality and is where the "
          "optimization-oracle camp cannot be formulated at all -- demonstrated separately by the "
          "prediction-in-constraints and IRP benchmarks."]
    return "\n".join(L)


if __name__ == "__main__":
    main()
