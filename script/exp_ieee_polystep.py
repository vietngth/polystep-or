"""
IEEE-CIS Predict+Optimize (Renewable Energy Scheduling)  x  PolyStep
====================================================================
Benchmark target: Bergmeir et al. (2023/2025), "Comparison and Evaluation of Methods
for a Predict+Optimize Problem in Renewable Energy Scheduling", arXiv:2212.10723
(IEEE-CIS 3rd Technical Challenge).  We REUSE the authors' optimisation engine + cost
model rather than re-implementing it (methodology rule).

REUSED CODE (co-author MA&RE optimiser, Python 3.9 + Gurobi):
    baselines/mare_optimiser/  (github.com/resmaeilbeigi/IEEE_CIS_3rd_Technical_Challenge_Optimiser)
      codes/python/Optimizer.py  -> the MILP scheduling model (battery + activities + linearised
                                    peak-demand charge); create_objective() IS the competition cost.
      codes/python/Instance.py   -> data loader. load_scenario() reads *_submission.csv FORECASTS;
                                    load_real_data() reads All_data.csv ACTUALS  (= realised-cost oracle).
      COMPETITION DATASET FILES/ -> phase-1 & phase-2 instances + AEMO prices + forecast scenarios
                                    + phase-1 ACTUALS (All_data.csv).   NO IEEE DataPort login needed.

PIPELINE (one phase-1 instance):
    features (calendar)  --phi_w-->  per-period net-load forecast multipliers
        -> overwrite scenario.base_load / scenario.solar_load        (the predictor)
        -> MILP solve (THEIR Optimizer, forecast mode)               (non-diff combinatorial oracle)
        -> battery+activity schedule
        -> realised cost = THEIR Optimizer with use_real_data=True, fixsol=True on that schedule
           (objective re-evaluated on the revealed Oct-2020 actuals)  <-- the PolyStep objective (scalar).

TWO-STAGE baseline = identity predictor (use the bundled forecast as-is).
POLYSTEP = gradient-free; minimise realised cost directly in predictor-parameter space (orthoplex step).
This mirrors the paper's headline finding (the best point forecast is NOT the best for downstream cost;
a decision-focused / quantile-shifted forecast wins) -- here learned label-free on realised cost.

Run:  .venv/bin/python exp_ieee_polystep.py smoke   (1 instance, 2 steps, short MILP time limit)
      .venv/bin/python exp_ieee_polystep.py full
Needs:  GRB_LICENSE_FILE pointing at a valid Gurobi licence (WLS academic verified, model ~158k vars).
"""
from __future__ import annotations
import os, sys, shutil, math, time, json, argparse
import numpy as np

# --- locate reused engine + bundled data -----------------------------------------------------------
ROOT   = os.path.dirname(os.path.abspath(__file__))
REPO   = os.path.join(ROOT, "baselines", "mare_optimiser")
ENGINE = os.path.join(REPO, "codes", "python")
DATA   = os.path.join(REPO, "COMPETITION DATASET FILES")
sys.path.insert(0, ENGINE)
sys.path.insert(0, os.path.join(ROOT, "polystep", "src"))

# writable working dir for the engine (storage rule: keep artefacts off the system disk)
CACHE = os.environ.get("IEEE_CACHE",
                       "/media/anindex/Data/project-cache/ot-or-project/ieee_cis/IEEE-CIS Predict+Optimize")
P = lambda *a: print(*a, flush=True)


def setup_layout():
    os.makedirs(os.path.join(CACHE, "output"), exist_ok=True)
    link = os.path.join(CACHE, "COMPETITION DATASET FILES")
    if not os.path.exists(link):
        os.symlink(DATA, link)
    ss = os.path.join(CACHE, "startsol")
    if not os.path.isdir(ss):                       # startsol must be writable (engine overwrites it)
        shutil.copytree(os.path.join(REPO, "startsol"), ss)
    if not os.path.isdir(DATA):
        raise SystemExit(f"[BLOCKED] bundled competition data not found at {DATA}")


import Setting as Smod
Smod.Setting._get_main_dir = lambda self: CACHE
from Setting import Setting
import Util
from Data import Data
from Optimizer import Optimizer
from Solution import Solution

SMALL0 = 5            # alpha-sorted phase_1: large_0..4 = idx 0..4 ; small_0..4 = idx 5..9
RUNTIME = float(os.environ.get("IEEE_RUNTIME", "8"))   # MILP time limit (s) per forecast solve
GAP     = float(os.environ.get("IEEE_GAP", "0.02"))
ALPHA   = 0.30        # max +-30% forecast rescale the predictor can apply


def _setting(real_data: bool, fixsol: bool, runtime: float):
    st = Setting()
    st.phase = 1; st.use_utc_time = False
    st.start_date = "20-10-01"; st.end_date = "20-10-31"
    st.use_real_data = real_data
    st.use_multiple_scenarios = False
    st.solver.runtime = runtime
    st.solver.gap = GAP
    st.solver.setstart = False
    st.solver.fixsol = fixsol
    st.algorithm = 0
    st.dataset_keys = ["phase_1"]
    st.input_dir   = Util.joinpath(st.main_dir, "COMPETITION DATASET FILES")
    st.startsol_dir = Util.joinpath(st.main_dir, "startsol")
    return st


# --- calendar feature basis (per 15-min slot) ------------------------------------------------------
def calendar_basis(inst):
    """B in R^{T x 2}: [sin, cos] of hour-of-day -> lets the predictor shift the forecast by time-of-day."""
    T = len(inst.planning_horizon)
    rows = []
    for t in inst.planning_horizon:
        a = inst.time.slots[t].interval.a
        h = a.hour + a.minute / 60.0
        rows.append([math.sin(2 * math.pi * h / 24.0), math.cos(2 * math.pi * h / 24.0)])
    return np.asarray(rows, dtype=np.float64)            # (T,2)


def apply_predictor(inst, w):
    """Decision-focused predictor: multiply the FORECAST base_load & solar_load by per-period factors.
    w = [w_load_sin, w_load_cos, w_solar_sin, w_solar_cos, b_load, b_solar] (R^6)."""
    B = calendar_basis(inst)                              # (T,2)
    w = np.asarray(w, dtype=np.float64)
    f_load  = 1.0 + ALPHA * np.tanh(B @ w[0:2] + w[4])
    f_solar = 1.0 + ALPHA * np.tanh(B @ w[2:4] + w[5])
    base  = np.asarray(inst.scenarios[0].base_load,  dtype=np.float64) * f_load
    solar = np.asarray(inst.scenarios[0].solar_load, dtype=np.float64) * f_solar
    inst.scenarios[0].base_load  = base.tolist()
    inst.scenarios[0].solar_load = solar.tolist()


# --- the competition cost (closed form = THEIR create_objective, single scenario) ------------------
def schedule_cost(opt, scen):
    """Evaluate THEIR objective (energy + 0.005*peak^2 + once-off penalty - revenue) for the schedule
    fixed in `opt`, under load/solar/price arrays in `scen`. Mirrors Optimizer.create_objective() with
    one scenario, but uses the EXACT quadratic peak charge (what the official Optim_eval scores), so it
    is robust to any realised peak (the engine's fixsol path truncates the peak index range)."""
    sph = opt.slots_per_hour
    vget = lambda a, t: (opt.V_VAR[a, t].x if (a, t) in opt.V_VAR else 0.0)
    L = []
    for t in opt.slot_indices:
        v = scen.base_load[t] - scen.solar_load[t]
        for b in opt.batteries:
            bat = opt.batteries[b]
            v += (opt.X_VAR[b, t].x - bat.efficiency * opt.Y_VAR[b, t].x) * (
                bat.max_power / math.sqrt(bat.efficiency))
        for a in opt.activities_o:
            act = opt.activities[a]
            v += vget(a, t) * act.load_per_room * (act.small_rooms + act.large_rooms)
        for a in opt.activities_r:
            act = opt.activities[a]
            v += vget(a, opt.map_time(t)) * act.load_per_room * (act.small_rooms + act.large_rooms)
        L.append(v)
    energy = sum(L[i] * scen.price[t] / (sph * 1000) for i, t in enumerate(opt.slot_indices))
    peak = 0.005 * (max(abs(x) for x in L) ** 2)
    onceoff = (sum(opt.activities[a].penalty for a in opt.activities_o if opt.U_VAR[a].x > 0.5)
               - sum(opt.activities[a].revenue for a in opt.activities_o if opt.W_VAR[a].x > 0.5))
    return energy + peak + onceoff


# --- the predict -> optimise -> realised-cost closure ----------------------------------------------
_REAL_CACHE = {}
def _real_scenario(idx):
    if idx not in _REAL_CACHE:
        ste = _setting(real_data=True, fixsol=False, runtime=1.0)
        inste = Data(ste).get_instance_by_index("phase_1", idx)   # builds actuals scenario from All_data.csv
        _REAL_CACHE[idx] = inste.scenarios[0]
    return _REAL_CACHE[idx]


def realized_cost(w, idx=SMALL0, runtime=None):
    runtime = RUNTIME if runtime is None else runtime
    # (1) FORECAST mode: predictor adjusts the forecast, THEIR MILP schedules against it
    stf = _setting(real_data=False, fixsol=False, runtime=runtime)
    instf = Data(stf).get_instance_by_index("phase_1", idx)
    apply_predictor(instf, w)
    opt = Optimizer(instf); opt.formulate(); opt.model.setParam("OutputFlag", 0)
    info = opt.solve()
    if info.STATUS == 3 or opt.model.SolCount == 0:
        return float("inf"), None
    forecast_obj = schedule_cost(opt, instf.scenarios[0])         # cost on the (predicted) forecast
    # (2) REALISED cost: same schedule scored on the revealed Oct-2020 ACTUALS (All_data.csv)
    realized = schedule_cost(opt, _real_scenario(idx))
    return float(realized), float(forecast_obj)


# --- PolyStep --------------------------------------------------------------------------------------
import torch, torch.nn as nn
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon


class Predictor(nn.Module):
    def __init__(self, d=6):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))            # zeros => identity predictor (two-stage)


def train_polystep(idx, steps, seed, runtime, scale):
    model = Predictor()
    pso = PolyStepOptimizer(model, polytope_type="orthoplex",
                            epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=0.4, probe_radius=0.8, num_probe=1,
                            seed=seed, use_momentum=True, momentum_init=0.5, momentum_final=0.9)

    def closure(bp):
        W = bp["w"]                                       # (K, d)
        K = W.shape[0]
        out = torch.zeros(K)
        for k in range(K):
            rc, _ = realized_cost(W[k].detach().cpu().numpy(), idx=idx, runtime=runtime)
            out[k] = rc / scale
        return out

    best = (float("inf"), None); hist = []
    for s in range(steps):
        pso.step(closure)
        cur, _ = realized_cost(model.w.detach().cpu().numpy(), idx=idx, runtime=runtime)
        if cur < best[0]:
            best = (cur, model.w.detach().clone())
        hist.append((s, cur))
        P(f"  [step {s}] realised cost {cur:,.1f}  best {best[0]:,.1f}")
    if best[1] is not None:
        with torch.no_grad():
            model.w.copy_(best[1])
    return model, best[0], hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--idx", type=int, default=SMALL0)
    args = ap.parse_args()
    smoke = args.mode == "smoke"
    steps   = 1 if smoke else 25
    runtime = max(RUNTIME, 15.0) if smoke else max(RUNTIME, 60.0)   # MILP needs >=~15s to find an incumbent
    seed = 0
    setup_layout()
    P(f"=== IEEE-CIS Predict+Optimize x PolyStep | {'SMOKE' if smoke else 'FULL'} | "
      f"phase1 idx={args.idx} steps={steps} milp_runtime={runtime}s ===")
    P(f"reused engine: {ENGINE}")
    P(f"bundled data : {DATA}  (no IEEE DataPort login)")

    t0 = time.time()
    # TWO-STAGE baseline (identity predictor; uses bundled forecast as-is)
    base_rc, base_fc = realized_cost(np.zeros(6), idx=args.idx, runtime=runtime)
    P(f"[two-stage] forecast-opt cost {base_fc:,.1f}  ->  REALISED cost {base_rc:,.1f}")
    scale = max(1.0, abs(base_rc))

    # POLYSTEP (decision-focused, gradient-free on realised cost)
    _, ps_rc, hist = train_polystep(args.idx, steps=steps, seed=seed, runtime=runtime, scale=scale)
    impr = 100.0 * (base_rc - ps_rc) / base_rc if math.isfinite(base_rc) and base_rc else float("nan")
    P("--------------------------------------------------------------")
    P(f"  two-stage realised cost : {base_rc:,.1f}")
    P(f"  PolyStep  realised cost : {ps_rc:,.1f}   ({impr:+.2f}% vs two-stage)")
    P(f"[total {time.time()-t0:.0f}s]")

    os.makedirs(os.path.join(ROOT, "exp_results"), exist_ok=True)
    out = dict(mode=args.mode, idx=args.idx, steps=steps, milp_runtime=runtime,
               two_stage_realized=base_rc, two_stage_forecast=base_fc,
               polystep_realized=ps_rc, improvement_pct=impr, history=hist)
    with open(os.path.join(ROOT, "exp_results", f"ieee_polystep_{args.mode}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
