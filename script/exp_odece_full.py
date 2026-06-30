"""ODECE FULL head-to-head: gradient-free DFL (PolyStep, SFGE) vs the ODECE NeurIPS'25
constraint-DFL benchmark + CombOptNet + two-stage, on problems where the PREDICTED parameters
sit in the CONSTRAINTS (multidimensional 0/1 knapsack).

Two settings, both from ODECE's OWN generator (OptProblems.knapsack.kpdata), matched on
deg / noise / num_items / dim / seeds so our methods train+eval on the SAME instance
distribution as the ODECE repo:

  * capa   -- MDKP, predicted CAPACITY in the constraints (genCapacity, dim=3, deg=6, noise=0.25,
              num_items=50). Matches MDKP_CapaExp.sh / Results/KnapsackCapacity/NoFixedCosts/.
  * weight -- MDKP, predicted per-item WEIGHTS in the constraints (genWeights, capacity_ratio
              [0.18,0.2,0.22] => BINDING knapsack, dim=3, deg=6, noise=0.25, num_items=50).
              Matches KnapsackWeightExp.py / MDKP_WeightExp.sh. (Verified: with a binding capacity
              the realized objective HAS leverage in predicted weights under the batched greedy
              solver; with the non-binding genCapacity defaults it is flat -- that is why the
              capacity_ratio override is essential.)

Methods
  LIVE in our .venv (gradient-free + two-stage; same generator + seeds; our metric):
    - two-stage  : MSE predictor + greedy deploy + repair        (== ODECE-repo "mse" baseline)
    - SFGE       : score-function gradient estimator on realized cost
    - PolyStep   : gradient-free Sinkhorn-Step on realized cost   (OURS)
  CITED from baselines/odece_neurips25/Results/ (ODECE repo's published numbers, THEIR metric):
    - ODECE      : feasibility-aware DFL (alpha=0.5)              capa only (bundled)
    - MSE        : their two-stage baseline                       capa only (bundled)
  N/A:
    - CombOptNet : bundled in the repo but NOT in Results/, and NOT runnable in our .venv
                   (needs pytorch_lightning + einops + torch 2.6 pin). Documented, not run.
    - SPO+/PFYL/IMLE/cvxpylayers : structurally undefined -- the prediction parametrizes the
                   feasible region (no fixed feasible set, no objective cost vector to differentiate).

Metric.  LIVE methods: normalized realized regret = (oracle - realized)/oracle on the test set,
where oracle = Gurobi MDKP optimum on TRUE params and realized = value of the greedy decision under
PREDICTED params AFTER repair to TRUE feasibility (drop overflow). Plus pre-repair infeasibility rate.
Our post-repair regret <-> their test_posthoc_regret; our pre-repair infeasibility <-> their
test_infeasibility (recourse mechanisms differ, so treat as comparable-not-identical; the bridge is
two-stage(live) == mse(cited), same method).

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp_odece_full.py [smoke|full] [settings] [seeds]
      e.g.  ... exp_odece_full.py smoke capa,weight 11,12,13
            ... exp_odece_full.py full  capa,weight 11,12,13,14,15
"""
from __future__ import annotations
import sys, os, glob, csv
sys.path.insert(0, "polystep/src")
sys.path.insert(0, "baselines/odece_neurips25")
import numpy as np
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
from OptProblems.knapsack.kpdata import genCapacity, genWeights
from pto.solvers import mdkp_greedy, mdkp_repair
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from pto.seeding import seed_everything, device_generator
from pto.multiseed import summarize, wilcoxon_pair, md_table, write_json, write_md, fmt_mean_std

dev = "cuda" if torch.cuda.is_available() else "cpu"
NF, NIT, DIM, DEG, NOISE = 10, 50, 3, 6, 0.25
WEIGHT_CAP_RATIO = np.array([0.18, 0.2, 0.22])      # KnapsackWeightExp.py: binding knapsack
RESULTS_DIR = "baselines/odece_neurips25/Results/KnapsackCapacity/NoFixedCosts"


# --------------------------------------------------------------------------- data
def gen_capa(seed, ntr, nte):
    X, w, costs, cap = genCapacity(ntr + nte, NF, NIT, dim=DIM, deg=DEG, noise_width=NOISE, seed=seed)
    return _to_t(X, w, costs, cap, ntr)


def gen_weight(seed, ntr, nte):
    X, w, costs, cap = genWeights(ntr + nte, NF, NIT, WEIGHT_CAP_RATIO, DIM, DEG, NOISE, seed=seed)
    return _to_t(X, w, costs, cap, ntr)


def _to_t(X, w, costs, cap, ntr):
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
    X, w, costs, cap = t(X), t(w), t(costs), t(cap)
    tr = (X[:ntr], w[:ntr], costs[:ntr], cap[:ntr])
    te = (X[ntr:], w[ntr:], costs[ntr:], cap[ntr:])
    return tr, te


# --------------------------------------------------------------------------- oracle
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


# --------------------------------------------------------------------------- deploy / realize
# Targets are STANDARDIZED so the predictor outputs are O(1) (essential for the weight setting:
# physical weights are ~10-100, so an absolute sigma/step of 0.5 never crosses a greedy selection
# boundary -> flat -> no learning. MU/STD are set per-setting in run_setting; destd() maps a
# standardized prediction back to physical units before the solver.
MU = STD = None


def destd(pred):
    return pred * STD + MU


def deploy_realize_capa(cap_pred, v, A, cap_true):
    """greedy pack under predicted capacity, repair vs TRUE capacity -> realized value (M,)."""
    sel = mdkp_greedy(v, A, cap_pred.clamp(min=1.0))
    return mdkp_repair(sel, v, A, cap_true)


def deploy_realize_weight(w_pred, v, A_true, cap):
    """greedy pack under predicted weights, repair vs TRUE weights -> realized value (M,)."""
    sel = mdkp_greedy(v, w_pred.clamp(min=0.01), cap)
    return mdkp_repair(sel, v, A_true, cap)


# --------------------------------------------------------------------------- predictors / training
def make_pred(out):
    return nn.Linear(NF, out, bias=True).to(dev)


def train_two_stage(Xtr, Ytr, out, epochs):
    m = make_pred(out); opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(epochs):
        opt.zero_grad(); ((m(Xtr) - Ytr) ** 2).mean().backward(); opt.step()
    return m


def _warm(out, warm):
    m = make_pred(out)
    with torch.no_grad():
        m.weight.copy_(warm.weight); m.bias.copy_(warm.bias)
    return m


def train_polystep(cfg, Xtr, w, costs, cap, out, warm, scale, steps, ps_batch, seed):
    """closure minimizes -realized/scale; ps_batch subsamples train instances (bounds K*B).
    step_radius is per-setting: the high-dim weight predictor (1650 params) needs a larger move
    (0.8) to cross greedy selection boundaries; the 3-output capacity predictor converges at 0.4."""
    m = _warm(out, warm)
    sr = cfg["ps_sr"]
    pso = PolyStepOptimizer(m, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=sr, probe_radius=2 * sr, num_probe=1, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    g = device_generator(seed, dev); B = Xtr.shape[0]

    def closure(bp):
        if ps_batch is not None and ps_batch < B:
            idx = torch.randperm(B, device=dev, generator=g)[:ps_batch]
        else:
            idx = slice(None)
        Xb, wb, cb, capb = Xtr[idx], w[idx], costs[idx], cap[idx]
        Bb = Xb.shape[0]
        pred = torch.einsum("kof,bf->kbo", bp["weight"], Xb) + bp["bias"].unsqueeze(1)   # (K,Bb,out)
        K = pred.shape[0]
        realized = cfg["realize"](pred, wb, cb, capb, K, Bb)
        return -(realized / scale).mean(-1)
    for _ in range(steps):
        pso.step(closure)
    return m


def train_sfge(cfg, Xtr, w, costs, cap, out, warm, scale, epochs, n_samples, sigma, lr, seed):
    m = _warm(out, warm)
    opt = torch.optim.Adam(m.parameters(), lr); g = device_generator(seed, dev); B = Xtr.shape[0]
    for _ in range(epochs):
        pred = m(Xtr)
        with torch.no_grad():
            eps = torch.randn(n_samples, *pred.shape, device=dev, generator=g)
            chat = pred.unsqueeze(0) + sigma * eps; S = chat.shape[0]
            realized = cfg["realize"](chat, w, costs, cap, S, B)
            r = -(realized / scale)
            adv = r - r.mean(0, keepdim=True)
        logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
        surr = (adv * logp).mean(); opt.zero_grad(); surr.backward(); opt.step()
    return m


# --------------------------------------------------------------------------- setting configs
def realize_capa(pred, w, costs, cap_true, K, B):
    v = costs.unsqueeze(0).expand(K, -1, -1).reshape(K * B, NIT)
    A = w.unsqueeze(0).expand(K, -1, -1, -1).reshape(K * B, DIM, NIT)
    ct = cap_true.unsqueeze(0).expand(K, -1, -1).reshape(K * B, DIM)
    cp = destd(pred).reshape(K * B, DIM)
    return deploy_realize_capa(cp, v, A, ct).reshape(K, B)


def realize_weight(pred, w_true, costs, cap, K, B):
    v = costs.unsqueeze(0).expand(K, -1, -1).reshape(K * B, NIT)
    At = w_true.unsqueeze(0).expand(K, -1, -1, -1).reshape(K * B, DIM, NIT)
    cp = cap.unsqueeze(0).expand(K, -1, -1).reshape(K * B, DIM)
    wp = destd(pred).reshape(K * B, DIM, NIT)
    return deploy_realize_weight(wp, v, At, cp).reshape(K, B)


SETTINGS = {
    "capa": dict(out=DIM, target=lambda tr: tr[3], realize=realize_capa, ps_sr=0.4,
                 name="MDKP, predicted CAPACITY in constraints", cited=True),
    "weight": dict(out=DIM * NIT, target=lambda tr: tr[1].reshape(tr[1].shape[0], -1), ps_sr=0.8,
                   realize=realize_weight, name="MDKP, predicted WEIGHTS in constraints", cited=False),
}


def evaluate(cfg, m, te, oracle):
    Xte, wte, cte, capte = te
    with torch.no_grad():
        pred = destd(m(Xte))
        if cfg["out"] == DIM:                 # capacity
            cap_pred = pred.clamp(min=1.0)
            sel = mdkp_greedy(cte, wte, cap_pred)
            realized = mdkp_repair(sel, cte, wte, capte)
            infeas = ((torch.einsum("mn,mjn->mj", sel.float(), wte) > capte).any(-1)).float().mean().item()
        else:                                 # weight
            w_pred = pred.reshape(pred.shape[0], DIM, NIT).clamp(min=0.01)
            sel = mdkp_greedy(cte, w_pred, capte)
            realized = mdkp_repair(sel, cte, wte, capte)
            infeas = ((torch.einsum("mn,mjn->mj", sel.float(), wte) > capte).any(-1)).float().mean().item()
        reg = ((oracle - realized) / oracle.clamp(min=1e-6)).mean().item()
    return reg, infeas


# --------------------------------------------------------------------------- cited (read Results/)
def _parse_hparams(path):
    d = {}
    for ln in open(path):
        if ":" in ln:
            k, _, v = ln.partition(":")
            d[k.strip()] = v.strip()
    return d


def _final_test_row(path):
    rows = [r for r in csv.DictReader(open(path)) if r.get("test_posthoc_regret", "") not in ("", None)]
    return rows[-1] if rows else None


def read_cited(model):
    """Aggregate ODECE-repo published test metrics from Results/ for the capacity setting."""
    acc = {"test_regret": [], "test_posthoc_regret": [], "test_infeasibility": []}
    seeds = []
    for vdir in sorted(glob.glob(f"{RESULTS_DIR}/{model}_deg6_noise0.25_numitems50/version_*")):
        hp = os.path.join(vdir, "hparams.yaml"); mc = os.path.join(vdir, "metrics.csv")
        if not (os.path.exists(hp) and os.path.exists(mc)):
            continue
        h = _parse_hparams(hp)
        if h.get("max_epochs") != "20":            # skip partial (max_epochs=2) runs
            continue
        if model == "odece" and h.get("infeasibility_aversion_coeff") not in ("0.5", "0.5 "):
            continue
        row = _final_test_row(mc)
        if row is None:
            continue
        for k in acc:
            try:
                acc[k].append(float(row[k]))
            except (KeyError, ValueError):
                pass
        seeds.append(int(float(h.get("seed", -1))))
    return {k: summarize(v) for k, v in acc.items()}, sorted(seeds)


# --------------------------------------------------------------------------- run
def run_setting(key, seeds, hp):
    global MU, STD
    cfg = SETTINGS[key]
    print(f"\n[{key}] {cfg['name']}", flush=True)
    acc = {m: {"regret": [], "infeas": []} for m in ("two-stage", "SFGE", "PolyStep")}
    for seed in seeds:
        seed_everything(seed)
        tr = gen_capa(seed, hp["ntr"], hp["nte"]) if key == "capa" else gen_weight(seed, hp["ntr"], hp["nte"])
        (Xtr, wtr, ctr, captr), te = tr
        target = cfg["target"](tr[0])                       # physical units (M, out)
        MU = target.mean(0, keepdim=True)                   # standardize -> O(1) predictions
        STD = target.std(0, keepdim=True).clamp(min=1e-6)
        target_std = (target - MU) / STD
        scale = float(true_scale(tr[0]))
        oracle = true_opt(te[2], te[1], te[3])
        ts = train_two_stage(Xtr, target_std, cfg["out"], hp["ts_epochs"])
        ps = train_polystep(cfg, Xtr, wtr, ctr, captr, cfg["out"], ts, scale,
                            hp["ps_steps"], hp["ps_batch"], seed)
        sf = train_sfge(cfg, Xtr, wtr, ctr, captr, cfg["out"], ts, scale,
                        hp["sfge_epochs"], hp["sfge_samples"], hp["sigma"], hp["sfge_lr"], seed)
        for name, mdl in (("two-stage", ts), ("SFGE", sf), ("PolyStep", ps)):
            rg, inf = evaluate(cfg, mdl, te, oracle)
            acc[name]["regret"].append(rg); acc[name]["infeas"].append(inf)
        print(f"  seed {seed}: " + "  ".join(
            f"{n} reg={acc[n]['regret'][-1]:.4f}/inf={acc[n]['infeas'][-1]:.2f}" for n in acc), flush=True)
    summ = {n: {"regret": summarize(acc[n]["regret"]), "infeas": summarize(acc[n]["infeas"])} for n in acc}
    # opt-oracle alternative for Wilcoxon = best of {two-stage, SFGE} (PolyStep vs the best non-PolyStep)
    alt = min(("two-stage", "SFGE"), key=lambda n: summ[n]["regret"]["mean"])
    p_ps = wilcoxon_pair(acc["PolyStep"]["regret"], acc[alt]["regret"])
    p_ts = wilcoxon_pair(acc["PolyStep"]["regret"], acc["two-stage"]["regret"])
    out = {"summary": summ, "raw": acc, "wilcoxon_polystep_vs": alt,
           "p_polystep_lt_alt": p_ps, "p_polystep_lt_twostage": p_ts}
    if cfg["cited"]:
        out["cited"] = {}
        for model in ("odece", "mse"):
            s, sd = read_cited(model)
            out["cited"][model] = {"summary": s, "seeds": sd}
    return out


def true_scale(tr):
    """scale ~ mean total available 'budget' for normalization stability."""
    X, w, costs, cap = tr
    return (cap.sum(-1)).mean()


HP_FULL = dict(ntr=1000, nte=500, ts_epochs=60, ps_steps=300, ps_batch=256,
               sfge_epochs=200, sfge_samples=8, sigma=0.5, sfge_lr=1e-2)
HP_SMOKE = dict(ntr=200, nte=120, ts_epochs=40, ps_steps=80, ps_batch=96,
                sfge_epochs=60, sfge_samples=8, sigma=0.5, sfge_lr=1e-2)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    settings = sys.argv[2].split(",") if len(sys.argv) > 2 else ["capa", "weight"]
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else (
        [11, 12, 13] if mode == "smoke" else [11, 12, 13, 14, 15])
    hp = HP_SMOKE if mode == "smoke" else HP_FULL
    print(f"ODECE FULL head-to-head | mode={mode} settings={settings} seeds={seeds}", flush=True)
    print("  surrogate camp (SPO+/PFYL/IMLE/cvxpylayers): N/A -- prediction parametrizes feasible region",
          flush=True)
    results = {}
    for k in settings:
        results[k] = run_setting(k, seeds, hp)
    payload = {"mode": mode, "settings": settings, "seeds": seeds, "hp": hp,
               "problem": dict(num_feat=NF, num_items=NIT, dim=DIM, deg=DEG, noise=NOISE,
                               weight_capacity_ratio=[float(x) for x in WEIGHT_CAP_RATIO]),
               "results": results}
    sfx = "_smoke" if mode == "smoke" else ""
    write_json(f"exp_results/odece_full{sfx}.json", payload)
    write_md(f"exp_results/odece_full{sfx}.md", to_md(payload))
    print(f"\nwrote exp_results/odece_full{sfx}.{{json,md}}\nDONE", flush=True)


def to_md(p):
    L = ["# ODECE FULL head-to-head: predict-in-constraint MDKP (PolyStep / SFGE vs ODECE / CombOptNet / two-stage)",
         "",
         f"mode=**{p['mode']}**, seeds={p['seeds']}, "
         f"num_items={NIT}, dim={DIM}, deg={DEG}, noise={NOISE}; weight setting capacity_ratio="
         f"{[float(x) for x in WEIGHT_CAP_RATIO]} (binding).",
         "",
         "**Applicability.** The predicted parameters sit in the CONSTRAINTS, so SPO+ / PFYL / IMLE / "
         "cvxpylayers are structurally undefined (no fixed feasible set, no objective cost vector). Only "
         "methods that evaluate the realized outcome of a deployed decision apply.",
         "",
         "**Live vs cited.** *two-stage, SFGE, PolyStep* are run LIVE in our .venv on ODECE's own "
         "generator/seeds. Our metric = normalized realized regret (oracle - realized)/oracle on the test "
         "set, with a batched GREEDY deploy under the predicted parameters + drop-overflow REPAIR to true "
         "feasibility; oracle = Gurobi MDKP optimum on true params; infeasibility = pre-repair rate. "
         "*ODECE* and *MSE* are CITED from `baselines/odece_neurips25/Results/` (their PyTorch-Lightning "
         "test metrics; capacity setting only -- the repo bundles no weight/Alloy/CombOptNet result files). "
         "*CombOptNet* is bundled in the repo but has no Results/ entry and is not runnable in our .venv "
         "(needs pytorch_lightning + einops + torch 2.6); reported N/A. **Caveat on the live<->cited "
         "boundary:** our live pipeline deploys with a batched greedy solver + repair, whereas the cited "
         "ODECE/MSE deploy with an EXACT Gurobi solve + their own recourse, and their regret normalization "
         "differs from ours. So cross-boundary numbers are reference points, NOT directly equatable; the "
         "comparable axis is the qualitative feasibility/regret trade-off. The live **two-stage and the "
         "cited MSE are the same learning method** (MSE predictor), which calibrates the offset.",
         ""]
    for k in p["settings"]:
        r = p["results"][k]; cfg = SETTINGS[k]
        L.append(f"## {cfg['name']}  (`{k}`)")
        headers = ["method", "source", "norm regret (lower=better)", "infeasibility", "note"]
        rows = []
        s = r["summary"]
        best_live = min(("two-stage", "SFGE", "PolyStep"), key=lambda n: s[n]["regret"]["mean"])
        for n in ("two-stage", "SFGE", "PolyStep"):
            reg = fmt_mean_std(s[n]["regret"]); inf = fmt_mean_std(s[n]["infeas"])
            note = "OURS (gradient-free)" if n == "PolyStep" else (
                "MSE predictor (= cited MSE method)" if n == "two-stage" else "gradient-free")
            rows.append([n, "live", f"**{reg}**" if n == best_live else reg, inf, note])
        if cfg["cited"]:
            for model, lbl in (("mse", "MSE (their two-stage)"), ("odece", "ODECE (alpha=0.5)")):
                c = r["cited"].get(model, {}).get("summary", {})
                nan = {"mean": float("nan"), "std": float("nan")}
                reg = fmt_mean_std(c.get("test_posthoc_regret", nan))
                tr = fmt_mean_std(c.get("test_regret", nan))
                inf = fmt_mean_std(c.get("test_infeasibility", nan))
                rows.append([lbl, "cited", reg, inf,
                             f"their metric: posthoc_regret (test_regret={tr}); seeds {r['cited'][model]['seeds']}"])
            rows.append(["CombOptNet", "N/A", "-", "-", "bundled, no Results/, needs their env"])
        else:
            rows.append(["ODECE / CombOptNet", "TODO", "-", "-",
                         "weight Results not bundled; live run needs ODECE env (lightning+einops, torch2.6)"])
        L.append(md_table(headers, rows))
        pa, pt = r['p_polystep_lt_alt'], r['p_polystep_lt_twostage']
        fa = f"{pa:.3f}" if pa is not None else "n/a (ties/few seeds)"
        ft = f"{pt:.3f}" if pt is not None else "n/a (ties/few seeds)"
        L.append(f"\nWilcoxon (one-sided, PolyStep < alt): vs two-stage p={ft}; "
                 f"vs best non-PolyStep live ({r['wilcoxon_polystep_vs']}) p={fa}\n")
    L += ["## Alloy (Brass Alloy production)  -- TODO",
          "Not run: the Alloy solver (`OptProblems/alloyproduction/alloysolver.py`) + ODECE training "
          "harness require the ODECE env (pytorch_lightning, einops, torch 2.6) and a Gurobi LP per "
          "instance. Deferred; the two MDKP settings above are the matched head-to-head.",
          "",
          "## Takeaway",
          "On predict-in-constraint MDKP, the gradient-free evaluation-oracle methods (PolyStep, SFGE) "
          "train directly on realized post-repair cost with NO differentiable solver and NO imputation, "
          "and the surrogate camp (SPO+/PFYL/IMLE/cvxpylayers) cannot even be formulated. ODECE achieves "
          "low infeasibility via its feasibility-aware loss; our methods reach feasibility by deploy-time "
          "repair and are tuned on the realized objective. two-stage (=their MSE) is the shared anchor."]
    return "\n".join(L)


if __name__ == "__main__":
    main()
