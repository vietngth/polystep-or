"""ICON energy-cost-aware scheduling benchmark (Mandi et al., AAAI 2020 / JAIR 2024 DFL survey)
with PolyStep added as a gradient-free DFL trainer.

USES THE AUTHORS' CODE at baselines/predopt-benchmarks/Energy for everything except PolyStep:
  * data            : Trainer.data_utils.EnergyDataModule + Trainer.get_energy (real SEMO 2013 prices)
  * scheduling ILP  : Trainer.comb_solver.SolveICON (Gurobi, BINARY x -> exact ILP, relax=False)
  * regret metric   : Trainer.utils.regret_fn (normalized regret, lower is better)
  * baselines       : Trainer.PO_models.baseline_mse (two-stage MSE), SPO (SPO+), DBB (blackbox-diff)
                      driven by pytorch_lightning (their training loop), same as their testenergy.py

PAPER-DEFAULT SETTING (JAIR 2024 benchmark, matches the repo):
  3 machines x {10,15,20} tasks (instances 1/2/3), 1 resource, q=30min -> T=48 half-hour slots,
  real Irish SEMO price data (Trainer/prices2013.dat, label SMPEP2), predictor = nn.Linear(nfeat,1),
  data split 550 train / 100 valid / 142 test days, batch_size=64, max_epochs=20, Gurobi exact ILP,
  10 seeds. lr per method from config.json.

THE ONLY ADDED METHOD = PolyStep (polystep/src): predict prices -> their ICON ILP (SolveICON) ->
realized energy cost (sol . y_true) as the scalar objective PolyStep minimizes directly. No labels of
the optimal schedule are used in PolyStep's objective (it consumes only the realized scalar cost).

Run:  ICON_INSTANCES=1 ICON_SEEDS=0 ICON_EPOCHS=20 .venv/bin/python exp_icon_polystep.py        # smoke
      ICON_INSTANCES=1,2,3 ICON_SEEDS=0,1,2,3,4,5,6,7,8,9 .venv/bin/python exp_icon_polystep.py  # full
"""
from __future__ import annotations
import os, sys, json, time, types

import numpy as np
import torch
import torch.nn as nn

# ------------------------------------------------------------------ paths + qpth stub
ROOT = os.path.dirname(os.path.abspath(__file__))
ENERGY = os.path.join(ROOT, "baselines", "predopt-benchmarks", "Energy")
sys.path.insert(0, os.path.join(ROOT, "polystep", "src"))
sys.path.insert(0, ENERGY)
# Their comb_solver.py / PO_models.py import qpth, cvxpy, cvxpylayers at module top, but those are ONLY
# used by DCOL/QPTL/IntOpt-style models we do NOT run (we run two-stage MSE, SPO+, DBB, PolyStep, all
# Gurobi-only via SolveICON). qpth also forces an incompatible numpy downgrade. Stub any that are not
# importable so the rest of their code loads cleanly (real packages are used when present, e.g. locally).
def _stub(name, submods=()):
    try:
        __import__(name)
        return
    except Exception:
        m = types.ModuleType(name); sys.modules[name] = m
        for sub, attrs in submods:
            sm = types.ModuleType(f"{name}.{sub}")
            for a in attrs:
                setattr(sm, a, object)
            setattr(m, sub, sm); sys.modules[f"{name}.{sub}"] = sm

_stub("qpth", [("qp", ["QPFunction"])])
_stub("cvxpy")
_stub("cvxpylayers", [("torch", ["CvxpyLayer"])])

os.chdir(ENERGY)  # their code uses relative paths (Trainer/prices2013.dat, SchedulingInstances/...)

# pandas >=3.0 removed read_csv(delim_whitespace=); their get_energy.py still uses it -> compat shim.
import pandas as _pd
_orig_read_csv = _pd.read_csv
def _read_csv_compat(*a, **k):
    if k.pop("delim_whitespace", False):
        k.setdefault("sep", r"\s+")
    return _orig_read_csv(*a, **k)
_pd.read_csv = _read_csv_compat

import pytorch_lightning as pl
from Trainer.comb_solver import data_reading, SolveICON
from Trainer.data_utils import EnergyDataModule
from Trainer.utils import regret_fn, batch_solve
from Trainer.PO_models import baseline_mse, SPO, DBB

from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon

OUT = os.path.join(ROOT, "exp_results")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ config (paper-default)
INSTANCES = [int(s) for s in os.environ.get("ICON_INSTANCES", "1").split(",")]
SEEDS = [int(s) for s in os.environ.get("ICON_SEEDS", "0").split(",")]
EPOCHS = int(os.environ.get("ICON_EPOCHS", "20"))          # paper-default max_epochs
BATCH = int(os.environ.get("ICON_BATCH", "64"))            # paper-default batch_size
METHODS = os.environ.get("ICON_METHODS", "two-stage,SPO,DBB,PolyStep").split(",")
# PolyStep objective is over a subset of training days for tractability (each day = 1 Gurobi solve);
# the reported regret is ALWAYS on the full test set, identical to the baselines.
PS_DAYS = int(os.environ.get("ICON_PS_DAYS", "60"))
PS_STEPS = int(os.environ.get("ICON_PS_STEPS", "60"))
PS_VAL_DAYS = int(os.environ.get("ICON_PS_VAL_DAYS", "40"))
# per-method learning rates from their config.json (instance-indexed); fall back to 0.5
LR = {"two-stage": {1: 0.5, 2: 0.5, 3: 0.5},
      "SPO":       {1: 1.0, 2: 0.5, 3: 0.5},
      "DBB":       {1: 0.01, 2: 0.5, 3: 0.5}}
DBB_LAMBDA = {1: 0.1, 2: 1.0, 3: 1.0}

DEV = "cpu"  # tiny linear model; bottleneck is the (CPU) Gurobi ILP, so CPU avoids device churn


def seed_all(seed):
    pl.seed_everything(seed, verbose=False)
    torch.manual_seed(seed); np.random.seed(seed)


# ------------------------------------------------------------------ helpers
def build_data(param, seed):
    # all data prep + solving happens in EnergyDataModule.__init__ (no separate setup needed)
    dm = EnergyDataModule(param=param, batch_size=BATCH, num_workers=0, seed=seed, relax=False)
    return dm


def test_regret(net, dm, solver):
    """Normalized regret on the full test set, using the authors' regret_fn."""
    X = torch.from_numpy(dm.test_df.X).float()
    y = torch.from_numpy(dm.test_df.y).float()
    sol = torch.from_numpy(dm.test_df.sol).float()
    with torch.no_grad():
        y_hat = net(X).squeeze(-1)
    return float(regret_fn(solver, y_hat, y, sol))


# ------------------------------------------------------------------ baselines (authors' LightningModules)
def train_lightning(modelcls, param, dm, seed, lr, **kw):
    seed_all(seed)
    model = modelcls(param=param, lr=lr, max_epochs=EPOCHS, seed=seed, **kw)
    trainer = pl.Trainer(max_epochs=EPOCHS, min_epochs=1, accelerator="cpu", devices=1,
                         logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False)
    trainer.fit(model, datamodule=dm)
    return model


# ------------------------------------------------------------------ PolyStep (the ONLY added method)
def train_polystep(param, dm, warm_net, seed, nfeat):
    """Gradient-free PolyStep over the linear predictor's weights.

    closure(bp): for each candidate parameter set, predict prices for PS_DAYS training days, solve the
    ICON ILP (authors' SolveICON) per day, and return the mean realized energy cost (sol . y_true).
    """
    solver = SolveICON(relax=False, **param)
    solver.make_model()

    Xtr = torch.from_numpy(dm.train_df.X[:PS_DAYS]).float().to(DEV)
    ytr = torch.from_numpy(dm.train_df.y[:PS_DAYS]).float().to(DEV)
    Xval = torch.from_numpy(dm.train_df.X[PS_DAYS:PS_DAYS + PS_VAL_DAYS]).float().to(DEV)
    yval = torch.from_numpy(dm.train_df.y[PS_DAYS:PS_DAYS + PS_VAL_DAYS]).float().to(DEV)

    model = nn.Linear(nfeat, 1).to(DEV)
    model.load_state_dict(warm_net.state_dict())
    names = [n for n, _ in model.named_parameters()]

    pr = float(os.environ.get("ICON_PR", 0.4))
    sr = float(os.environ.get("ICON_SR", 0.2))
    pso = PolyStepOptimizer(model, polytope_type="orthoplex",
                            epsilon=CosineEpsilon(0.5, 0.05), step_radius=sr, probe_radius=pr,
                            num_probe=1, seed=seed, use_momentum=True,
                            momentum_init=0.5, momentum_final=0.9)

    def realized_cost(net, X, y):
        with torch.no_grad():
            yh = net(X).squeeze(-1).cpu().numpy()
        sols = batch_solve(solver, yh).numpy()          # (D,48) per-slot power for predicted prices
        return float((sols * y.cpu().numpy()).sum(1).mean())

    def closure(bp):
        K = bp[names[0]].shape[0]
        out = torch.zeros(K, device=DEV)
        base = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k in range(K):
            sd = dict(base)
            for n in names:
                sd[n] = bp[n][k]
            model.load_state_dict(sd)
            with torch.no_grad():
                yh = model(Xtr).squeeze(-1).cpu().numpy()
            sols = batch_solve(solver, yh).numpy()
            out[k] = float((sols * ytr.cpu().numpy()).sum(1).mean())
        model.load_state_dict(base)
        return out

    best = (float("inf"), {k: v.detach().clone() for k, v in model.state_dict().items()})
    for s in range(PS_STEPS):
        pso.step(closure)
        cur = realized_cost(model, Xval, yval)
        if cur < best[0]:
            best = (cur, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return model


# ------------------------------------------------------------------ driver
def run():
    results = {}
    for inst in INSTANCES:
        ntasks = {1: 10, 2: 15, 3: 20}[inst]
        param = data_reading(f"SchedulingInstances/load{inst}/day01.txt")
        print(f"\n=== instance {inst}: {param['nbMachines']} machines x {ntasks} tasks, "
              f"T={1440 // param['q']} slots ===", flush=True)
        per = {m: [] for m in METHODS}
        secs = {m: [] for m in METHODS}
        for seed in SEEDS:
            t0 = time.time()
            dm = build_data(param, seed)
            nfeat = dm.train_df.X.shape[-1]
            solver = SolveICON(relax=False, **param)
            solver.make_model()
            warm = None
            for m in METHODS:
                tm = time.time()
                if m == "two-stage":
                    net = train_lightning(baseline_mse, param, dm, seed, LR["two-stage"][inst])
                    warm = net
                    r = test_regret(net.net if hasattr(net, "net") else net, dm, solver)
                elif m == "SPO":
                    net = train_lightning(SPO, param, dm, seed, LR["SPO"][inst])
                    r = test_regret(net.net, dm, solver)
                elif m == "DBB":
                    net = train_lightning(DBB, param, dm, seed, LR["DBB"][inst],
                                          lambda_val=DBB_LAMBDA[inst])
                    r = test_regret(net.net, dm, solver)
                elif m == "PolyStep":
                    if warm is None:
                        warm = train_lightning(baseline_mse, param, dm, seed, LR["two-stage"][inst])
                    ps = train_polystep(param, dm, warm.net, seed, nfeat)
                    r = test_regret(ps, dm, solver)
                else:
                    continue
                dt = time.time() - tm
                per[m].append(r); secs[m].append(dt)
                print(f"  seed {seed} {m:10s} regret={r:.4f}  ({dt:.0f}s)", flush=True)
            print(f"  [seed {seed} done in {time.time() - t0:.0f}s]", flush=True)
        agg = {m: {"mean": float(np.mean(v)) if v else float("nan"),
                   "std": float(np.std(v)) if v else float("nan"),
                   "n": len(v), "raw": v,
                   "sec_mean": float(np.mean(secs[m])) if secs[m] else float("nan")}
               for m, v in per.items()}
        results[str(inst)] = {"ntasks": ntasks, "agg": agg}
        _write(results)  # incremental: survive a timeout with partial (per-instance) results
        print(f"  [instance {inst} written]", flush=True)
    print("\nwrote exp_results/icon.{json,md}\nDONE", flush=True)


def _write(results):
    payload = {"setting": {"instances": INSTANCES, "seeds": SEEDS, "epochs": EPOCHS,
                           "batch": BATCH, "methods": METHODS, "ps_days": PS_DAYS,
                           "ps_steps": PS_STEPS, "solver": "Gurobi exact ILP (SolveICON, relax=False)",
                           "metric": "normalized regret (authors' regret_fn), lower=better",
                           "data": "real SEMO 2013 prices (prices2013.dat)"},
               "results": results}
    with open(os.path.join(OUT, "icon.json"), "w") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(OUT, "icon.md"), "w") as f:
        f.write(to_md(payload))


def to_md(payload):
    s = payload["setting"]
    L = ["# ICON energy-cost-aware scheduling (Mandi et al.) -- two-stage + DFL baselines + PolyStep",
         "",
         f"Authors' code (baselines/predopt-benchmarks/Energy): data, ICON ILP solver, regret metric, "
         f"and baselines (two-stage MSE, SPO+, DBB). PolyStep is the only added method.", "",
         f"- instances: {s['instances']}  (3 machines x {{10,15,20}} tasks, T=48 half-hour slots)",
         f"- seeds: {s['seeds']}   epochs: {s['epochs']}   batch: {s['batch']}",
         f"- solver: {s['solver']}",
         f"- metric: {s['metric']}",
         f"- data: {s['data']}", ""]
    for inst, r in payload["results"].items():
        L.append(f"## instance {inst}  ({r['ntasks']} tasks)")
        L.append("| method | regret (mean +/- std) | n |")
        L.append("|---|---|---|")
        for m in payload["setting"]["methods"]:
            a = r["agg"].get(m)
            if a and a["n"]:
                L.append(f"| {m} | {a['mean']:.4f} +/- {a['std']:.4f} | {a['n']} |")
            else:
                L.append(f"| {m} | n/a | 0 |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    run()
