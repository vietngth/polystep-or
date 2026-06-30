"""ODECE head-to-head: PolyStep/SFGE vs the SOTA constraint-DFL benchmark (capacity prediction).

Reuses ODECE's EXACT instance generator (OptProblems.knapsack.kpdata.genCapacity) for the
multidimensional 0/1 knapsack with PREDICTED CAPACITY in the constraints (the regime where SPO+/
cvxpylayers/PFYL/IMLE are structurally N/A). We run two-stage, SFGE, and PolyStep on the same
generator and seeds; ODECE itself is run separately in its own env (it pins torch 2.6 + Lightning).

Why CAPACITY prediction (MDKP_CapaExp), not weight prediction: with a batched GREEDY solver the realized
objective is provably FLAT in the predicted WEIGHTS (verified: 0.0 change under sigma=3 perturbations,
because known costs dominate the greedy selection), so gradient-free methods cannot move there -- weight
prediction needs an EXACT solver, which ODECE has via Gurobi. The predicted CAPACITY gates the greedy
packing and HAS leverage (verified). This contrast is reported as an honest solver-boundary finding.

Realized objective = value of the deployed decision AFTER repair to true feasibility (drop overflowing
items): you cannot profit from infeasibility (a soft penalty can be gamed; repair cannot). The decision
is the greedy pack under PREDICTED capacity; repair is vs TRUE capacity; the true optimum (Gurobi) is the
regret denominator. Metric: normalized realized regret + pre-repair infeasibility rate. Problem matches
MDKP_CapaExp.sh: num_items=50, num_feat=10, dim=3, deg=6, noise=0.25, 1000 train / 500 test, seeds {11..15}.

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp_odece_mdkp.py [seeds]
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, "polystep/src")
sys.path.insert(0, "baselines/odece_neurips25")
import numpy as np
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
from OptProblems.knapsack.kpdata import genCapacity
from pto.solvers import mdkp_greedy, mdkp_repair
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from pto.seeding import seed_everything, device_generator
from pto.multiseed import summarize, wilcoxon_pair, md_table, write_json, write_md, fmt_mean_std

dev = "cuda" if torch.cuda.is_available() else "cpu"
NF, NIT, DIM, DEG, NOISE = 10, 50, 3, 6, 0.25
NTR, NTE = 1000, 500
# --- polytope x probe sweep (env-overridable; defaults preserve original behavior) ---
PS_POLYTOPE = os.environ.get("PS_POLYTOPE", "orthoplex")
PS_PROBES = int(os.environ.get("PS_PROBES", "1"))
OUT_TAG = os.environ.get("OUT_TAG", "")
_OUT_SFX = f"_{OUT_TAG}" if OUT_TAG else ""


def gen(seed):
    X, w, costs, cap = genCapacity(NTR + NTE, NF, NIT, dim=DIM, deg=DEG, noise_width=NOISE, seed=seed)
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
    X, w, costs, cap = t(X), t(w), t(costs), t(cap)
    return (X[:NTR], w[:NTR], costs[:NTR], cap[:NTR]), (X[NTR:], w[NTR:], costs[NTR:], cap[NTR:])


def gurobi_mdkp(costs, W, cap):
    n = len(costs); m = len(cap)
    md = gp.Model(); md.Params.OutputFlag = 0
    x = md.addVars(n, vtype=GRB.BINARY)
    md.setObjective(gp.quicksum(float(costs[i]) * x[i] for i in range(n)), GRB.MAXIMIZE)
    for j in range(m):
        md.addConstr(gp.quicksum(float(W[j, i]) * x[i] for i in range(n)) <= float(cap[j]))
    md.optimize()
    return np.array([x[i].X for i in range(n)])


def true_opt(costs, w, cap):
    cN, wN, capN = costs.cpu().numpy(), w.cpu().numpy(), cap.cpu().numpy()
    return torch.tensor([float(cN[i] @ gurobi_mdkp(cN[i], wN[i], capN[i])) for i in range(len(cN))],
                        dtype=torch.float32, device=dev)


def w_expand(w, K):
    return w.unsqueeze(0).expand(K, -1, -1, -1).reshape(K * w.shape[0], DIM, NIT)


def deploy_repaired(cap_pred_flat, v, A, cap_true_flat):
    """Greedy pack under predicted capacity, then repair (drop overflow) vs TRUE capacity. -> value (M,)."""
    sel = mdkp_greedy(v, A, cap_pred_flat.clamp(min=1.0))
    return mdkp_repair(sel, v, A, cap_true_flat)


def make_pred():
    return nn.Linear(NF, DIM, bias=True).to(dev)


def train_two_stage(Xtr, captr, epochs=60):
    m = make_pred(); opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(epochs):
        opt.zero_grad(); ((m(Xtr) - captr) ** 2).mean().backward(); opt.step()
    return m


def train_polystep(Xtr, w, costs, cap_true, warm, scale, steps=200, seed=0):
    m = make_pred()
    with torch.no_grad():
        m.weight.copy_(warm.weight); m.bias.copy_(warm.bias)
    pso = PolyStepOptimizer(m, polytope_type=PS_POLYTOPE, epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=0.4, probe_radius=0.8, num_probe=PS_PROBES, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    B = Xtr.shape[0]
    def closure(bp):
        pred = torch.einsum("kof,bf->kbo", bp["weight"], Xtr) + bp["bias"].unsqueeze(1)   # (K,B,DIM)
        K = pred.shape[0]
        v = costs.unsqueeze(0).expand(K, -1, -1).reshape(K * B, NIT)
        A = w_expand(w, K); ct = cap_true.unsqueeze(0).expand(K, -1, -1).reshape(K * B, DIM)
        realized = deploy_repaired(pred.reshape(K * B, DIM), v, A, ct).reshape(K, B)
        return -(realized / scale).mean(-1)
    for _ in range(steps):
        pso.step(closure)
    return m


def train_sfge(Xtr, w, costs, cap_true, warm, scale, epochs=200, n_samples=8, sigma=0.5, lr=1e-2, seed=0):
    m = make_pred()
    with torch.no_grad():
        m.weight.copy_(warm.weight); m.bias.copy_(warm.bias)
    opt = torch.optim.Adam(m.parameters(), lr); g = device_generator(seed, dev); B = Xtr.shape[0]
    for _ in range(epochs):
        pred = m(Xtr)
        with torch.no_grad():
            eps = torch.randn(n_samples, *pred.shape, device=dev, generator=g)
            chat = pred.unsqueeze(0) + sigma * eps; S = chat.shape[0]
            v = costs.unsqueeze(0).expand(S, -1, -1).reshape(S * B, NIT)
            A = w_expand(w, S); ct = cap_true.unsqueeze(0).expand(S, -1, -1).reshape(S * B, DIM)
            r = -(deploy_repaired(chat.reshape(S * B, DIM), v, A, ct).reshape(S, B) / scale)
            adv = r - r.mean(0, keepdim=True)
        logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
        surr = (adv * logp).mean(); opt.zero_grad(); surr.backward(); opt.step()
    return m


def evaluate(m, Xte, w, costs, cap_true, oracle):
    with torch.no_grad():
        cap_pred = m(Xte).clamp(min=1.0)
        sel = mdkp_greedy(costs, w, cap_pred)
        realized = mdkp_repair(sel, costs, w, cap_true)
        infeas = ((torch.einsum("mn,mjn->mj", sel.float(), w) > cap_true).any(-1)).float().mean().item()
        reg = ((oracle - realized) / oracle.clamp(min=1e-6)).mean().item()
    return reg, infeas


def main():
    seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [11, 12, 13, 14, 15]
    print(f"ODECE MDKP head-to-head (CAPACITY prediction) | seeds={seeds}", flush=True)
    print("  surrogate camp (SPO+/cvxpy/PFYL/IMLE): N/A. ODECE run separately in its own env.\n")
    acc = {m: {"regret": [], "infeas": []} for m in ("two-stage", "SFGE", "PolyStep")}
    for seed in seeds:
        seed_everything(seed)
        (Xtr, wtr, ctr, captr), (Xte, wte, cte, capte) = gen(seed)
        scale = float((ctr.sum(-1)).mean())
        oracle = true_opt(cte, wte, capte)
        ts = train_two_stage(Xtr, captr)
        ps = train_polystep(Xtr, wtr, ctr, captr, ts, scale, seed=seed)
        sf = train_sfge(Xtr, wtr, ctr, captr, ts, scale, seed=seed)
        for name, mdl in (("two-stage", ts), ("SFGE", sf), ("PolyStep", ps)):
            rg, inf = evaluate(mdl, Xte, wte, cte, capte, oracle)
            acc[name]["regret"].append(rg); acc[name]["infeas"].append(inf)
        print(f"  seed {seed}: " + "  ".join(
            f"{n} reg={acc[n]['regret'][-1]:.4f}/inf={acc[n]['infeas'][-1]:.2f}" for n in acc), flush=True)
    summ = {n: {"regret": summarize(acc[n]["regret"]), "infeas": summarize(acc[n]["infeas"])} for n in acc}
    best = min(("two-stage", "SFGE", "PolyStep"), key=lambda n: summ[n]["regret"]["mean"])
    p_ps = wilcoxon_pair(acc["PolyStep"]["regret"], acc["two-stage"]["regret"])
    p_sf = wilcoxon_pair(acc["SFGE"]["regret"], acc["two-stage"]["regret"])
    write_json(f"exp_results/odece_mdkp{_OUT_SFX}.json", {"seeds": seeds, "summary": summ, "raw": acc,
               "best": best, "p_polystep_lt_ts": p_ps, "p_sfge_lt_ts": p_sf})
    headers = ["method", "norm regret", "infeasibility (pre-repair)", "vs two-stage (p)"]
    rows = []
    for n in ("two-stage", "SFGE", "PolyStep"):
        p = p_ps if n == "PolyStep" else (p_sf if n == "SFGE" else None)
        c = fmt_mean_std(summ[n]["regret"])
        rows.append([n, f"**{c}**" if n == best else c, fmt_mean_std(summ[n]["infeas"]),
                     f"{p:.3f}" if p is not None else "-"])
    md = ("# ODECE head-to-head: multidimensional knapsack, predicted CAPACITY in constraints\n\n"
          f"seeds={seeds}; num_items={NIT}, dim={DIM}, deg={DEG}, noise={NOISE}. Surrogate/differentiable "
          "camp structurally N/A. Realized = repaired (feasible) value; greedy deploy on predicted capacity; "
          "Gurobi true-optimum oracle. ODECE's own numbers (its env, IPL/OPL losses) tabulated separately.\n\n"
          "Note: weight prediction is FLAT under the batched greedy solver (gradient-free cannot move; "
          "needs an exact solver as ODECE uses) -- so capacity prediction is the fair batched-solver "
          "head-to-head.\n\n" + md_table(headers, rows))
    write_md(f"exp_results/odece_mdkp{_OUT_SFX}.md", md)
    print(f"\nwrote exp_results/odece_mdkp{_OUT_SFX}.{{json,md}}\nDONE", flush=True)


if __name__ == "__main__":
    main()
