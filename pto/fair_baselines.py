"""Review TODO #5 (colleague An): make the BASELINE camp fair + complete.

Two fairness fixes to the head-to-head:

(1) TUNE SFGE as carefully as PolyStep. SFGE -- not the surrogate camp -- is PolyStep's actual
    gradient-free peer, yet the harness ships an UNTUNED SFGE (epochs=120, n_samples=8,
    sigma=0.5, lr=1e-2). We sweep its four key hyperparameters per problem with a staged grid
    (16 + 6 configs) and SELECT the best config by VALIDATION regret on an 80/20 split carved
    from the TRAINING set only (test is never touched during selection).

(2) COMPLETE the surrogate camp beyond SPO+/IMLE by adding two more PyEPO surrogate losses:
      - NCE : noise-contrastive estimation  (Mulamba et al., NeurIPS 2021)
      - LTR : pairwise learning-to-rank      (Mandi et al., ICML 2022)
    (the listwise & pointwise LTR variants are also evaluated as supplementary, so the
     learning-to-rank camp is fully represented).

Head-to-head (normalized PyEPO test regret): two-stage, SPO+, IMLE, NCE, LTR, SFGE(default),
SFGE(tuned), PolyStep. PROTOCOL: every trainable method is warm-started from the SAME
two-stage MSE model (the project's fair-init convention; see sweep_lr.py / challenge_established.py)
so differences are attributable to the trainer alone. The default train_sfge in capability.py is
left untouched; the tuned variant is just train_sfge called with the selected hyperparameters.

Run (full):  .venv/bin/python -m pto.fair_baselines sp,knap,tsp,port 4 0,1,2
Run (smoke): .venv/bin/python -m pto.fair_baselines sp,knap 4 0 --smoke
"""
from __future__ import annotations
import sys, json, time, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from pyepo import metric
from pto.capability import (SETUPS, DFL, train_two_stage, train_polystep, train_sfge,
                            _adam, dev)

CAT = {"sp": "LP", "knap": "ILP", "tsp": "ILP", "port": "SOCP"}
DEFAULT_SFGE = dict(n_samples=8, sigma=0.5, lr=1e-2, epochs=120)   # the shipped UNTUNED config

# ---- SFGE tuning grid (staged: lr x sigma at fixed n/epochs, then n x epochs at the winner) ----
GRID_FULL = dict(lr=[3e-3, 1e-2, 3e-2, 1e-1], sigma=[0.1, 0.25, 0.5, 1.0],
                 n_samples=[8, 16, 32], epochs=[120, 240])
GRID_SMOKE = dict(lr=[1e-2, 3e-2], sigma=[0.25, 0.5], n_samples=[8], epochs=[120])

# surrogate-camp epochs (warm-started). IMLE/SPO+ use the per-instance solver -> kept modest.
SURR = ["SPO+", "IMLE", "NCE", "LTR"]
SUPP_LTR = ["lsLTR", "ptwLTR"]          # supplementary LTR variants (listwise / pointwise)


# ------------------------------------------------------------------ warm-started trainers
def train_dfl_warm(cfg, name, warm, epochs=30):
    """Mirror of capability.train_dfl, but warm-started from `warm` (the shared two-stage)."""
    build, kind, fwd = DFL[name]; om = cfg["om"]; sense = om.modelSense
    m = cfg["make"]()
    with torch.no_grad():
        if warm is not None: m.weight.copy_(warm.weight)
    opt = _adam(m); loss_mod = build(om, cfg["ds_tr"])
    for _ in range(epochs):
        for xb, cb, wb, zb in cfg["ld_tr"]:
            xb, cb, wb, zb = [t.float().to(dev) for t in (xb, cb, wb, zb)]
            pred = m(xb)
            if kind == "opt":
                loss = sense * (loss_mod(pred) * cb).sum(-1).mean()
            else:
                pick = {"pred": pred, "c": cb, "w": wb, "z": zb}
                out = loss_mod(*[pick[a] for a in fwd]); loss = out.mean() if out.dim() > 0 else out
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def train_sfge_warm(cfg, warm, hp):
    cfg = dict(cfg); cfg["warm"] = warm
    return train_sfge(cfg, epochs=hp["epochs"], n_samples=hp["n_samples"],
                      sigma=hp["sigma"], lr=hp["lr"])


# ------------------------------------------------------------------ validation split (no test leak)
def make_fit_val(cfg, frac=0.8):
    """Carve an 80/20 fit/validation split from the TRAINING set only and return
    (fit_cfg, val_loader, fit_warm). fit_cfg is a shallow copy of cfg with Xtr/Cs/ld_tr/ds_tr
    restricted to the fit slice; val_loader is a plain (x,c,w,z) loader over the val slice."""
    ds = cfg["ds_tr"]; n = ds.feats.shape[0]; nf = int(frac * n)
    fit = dict(cfg)
    fit["Xtr"] = cfg["Xtr"][:nf]
    fit["Cs"] = cfg["Cs"][:nf]
    fit["ld_tr"] = DataLoader(TensorDataset(ds.feats[:nf], ds.costs[:nf], ds.sols[:nf], ds.objs[:nf]),
                              batch_size=128, shuffle=True)
    # lightweight fit dataset view (only needed if a pool loss is tuned; SFGE ignores it)
    fit["ds_tr"] = ds
    val_loader = DataLoader(TensorDataset(ds.feats[nf:], ds.costs[nf:], ds.sols[nf:], ds.objs[nf:]),
                            batch_size=256)
    fit_warm = train_two_stage(fit)                       # two-stage on the FIT subset (no leak)
    return fit, val_loader, fit_warm


# ------------------------------------------------------------------ SFGE tuning by validation
def tune_sfge(pname, deg, seeds, grid, verbose=True):
    """Staged grid search; selection metric = mean VALIDATION regret over seeds.
    Returns (best_hp, table) where table maps a config-string -> mean/std val regret."""
    # build fit/val per seed once
    per = {}
    for s in seeds:
        cfg, _ = SETUPS[pname](s, deg)
        per[s] = (cfg,) + make_fit_val(cfg)
    table = {}

    def eval_hp(hp):
        key = f"n{hp['n_samples']}_s{hp['sigma']}_lr{hp['lr']:g}_e{hp['epochs']}"
        if key in table: return table[key][0]
        rs = []
        for s in seeds:
            cfg, fit, val_loader, fit_warm = per[s]
            m = train_sfge_warm(fit, fit_warm, hp)
            rs.append(metric.regret(m, cfg["om"], val_loader))
        table[key] = (float(np.mean(rs)), float(np.std(rs)), dict(hp))
        return table[key][0]

    # default config's validation regret (for the "did tuning help" check on val)
    eval_hp(DEFAULT_SFGE)

    # stage A: lr x sigma at n_samples=8, epochs=120
    best_a, best_av = None, np.inf
    for lr in grid["lr"]:
        for sg in grid["sigma"]:
            hp = dict(n_samples=8, epochs=120, lr=lr, sigma=sg)
            v = eval_hp(hp)
            if v < best_av: best_av, best_a = v, hp
    # stage B: n_samples x epochs at the stage-A winner's lr/sigma
    best_hp, best_v = dict(best_a), best_av
    for ns in grid["n_samples"]:
        for ep in grid["epochs"]:
            hp = dict(n_samples=ns, epochs=ep, lr=best_a["lr"], sigma=best_a["sigma"])
            v = eval_hp(hp)
            if v < best_v: best_v, best_hp = v, hp
    if verbose:
        print(f"  [tune SFGE] best={best_hp}  val_regret={best_v:.4f}  "
              f"(default val={table[_k(DEFAULT_SFGE)][0]:.4f}, {len(table)} configs)", flush=True)
    return best_hp, best_v, table


def _k(hp):
    return f"n{hp['n_samples']}_s{hp['sigma']}_lr{hp['lr']:g}_e{hp['epochs']}"


# ------------------------------------------------------------------ head-to-head on TEST
def head_to_head(pname, deg, seeds, best_hp, smoke):
    cols = ["two-stage", "SPO+", "IMLE", "NCE", "LTR",
            "SFGE-def", "SFGE-tuned", "PolyStep"] + SUPP_LTR
    acc = {c: [] for c in cols}; wall = {c: [] for c in cols}
    surr_ep = 5 if smoke else 30
    for s in seeds:
        cfg, _ = SETUPS[pname](s, deg)
        if smoke: cfg["ps_steps"] = min(cfg.get("ps_steps", 40), 40)
        ts = train_two_stage(cfg)
        def rec(name, fn):
            a = time.time(); m = fn(); dt = time.time() - a
            try: r = metric.regret(m, cfg["om"], cfg["ld_te"])
            except Exception: r = float("nan")
            acc[name].append(r); wall[name].append(dt)
        acc["two-stage"].append(metric.regret(ts, cfg["om"], cfg["ld_te"])); wall["two-stage"].append(0.0)
        for nm in SURR:
            rec(nm, lambda nm=nm: train_dfl_warm(cfg, nm, ts, epochs=surr_ep))
        for nm in SUPP_LTR:
            rec(nm, lambda nm=nm: train_dfl_warm(cfg, nm, ts, epochs=surr_ep))
        cfg["warm"] = ts
        rec("SFGE-def", lambda: train_sfge_warm(cfg, ts, DEFAULT_SFGE))
        rec("SFGE-tuned", lambda: train_sfge_warm(cfg, ts, best_hp))
        rec("PolyStep", lambda: train_polystep(cfg))
        print(f"  seed {s} done", flush=True)
    summ = {c: (float(np.nanmean(acc[c])), float(np.nanstd(acc[c])), float(np.mean(wall[c])))
            for c in cols}
    return summ, acc


# ------------------------------------------------------------------ driver
def run(problems, deg, seeds, smoke):
    grid = GRID_SMOKE if smoke else GRID_FULL
    results = []
    for p in problems:
        print(f"\n=== {p} ({CAT[p]}) deg={deg} seeds={seeds} {'[SMOKE]' if smoke else ''} ===", flush=True)
        best_hp, best_val, table = tune_sfge(p, deg, seeds, grid)
        default_val = table[_k(DEFAULT_SFGE)][:2]
        summ, _ = head_to_head(p, deg, seeds, best_hp, smoke)
        res = {"problem": p, "category": CAT[p], "deg": deg, "seeds": seeds,
               "sfge_default_hp": DEFAULT_SFGE, "sfge_best_hp": best_hp,
               "sfge_val_default": {"mean": default_val[0], "std": default_val[1]},
               "sfge_val_tuned": {"mean": best_val},
               "sfge_tuning_table": table,
               "test": {c: {"mean": summ[c][0], "std": summ[c][1], "wall_s": summ[c][2]} for c in summ}}
        results.append(res)
        t = res["test"]
        print(f"  TEST regret: " + "  ".join(
            f"{c} {t[c]['mean']:.4f}" for c in ["two-stage","SPO+","IMLE","NCE","LTR",
                                                "SFGE-def","SFGE-tuned","PolyStep"]), flush=True)
    return results


def write_outputs(results, deg, seeds, smoke):
    tag = "_smoke" if smoke else ""
    jpath = f"exp_results/fair_baselines{tag}.json"
    mpath = f"exp_results/fair_baselines{tag}.md"
    with open(jpath, "w") as f:
        json.dump({"deg": deg, "seeds": seeds, "smoke": smoke,
                   "grid": (GRID_SMOKE if smoke else GRID_FULL),
                   "default_sfge": DEFAULT_SFGE, "results": results}, f, indent=2)
    L = []
    L.append("# Review TODO #5 -- fair + complete baseline camp\n")
    L.append(f"mode=**{'smoke' if smoke else 'full'}**, deg={deg}, seeds={seeds}. "
             "Normalized PyEPO **test** regret (lower better); mean±std over seeds. "
             "All trainable methods warm-started from the SAME two-stage MSE init (project fair-init "
             "convention). SFGE tuned per problem by **validation** regret (80/20 split of the "
             "training set; test untouched).\n")
    L.append("**Surrogate camp completed:** SPO+ & IMLE (shipped) + **NCE** (noise-contrastive, "
             "Mulamba 2021) + **LTR** (pairwise learning-to-rank, Mandi 2022). Listwise/pointwise "
             "LTR reported as supplementary.\n")
    # main table
    head = ["problem","cat","two-stage","SPO+","IMLE","NCE","LTR","SFGE-def","SFGE-tuned","PolyStep"]
    L.append("## Head-to-head (test regret)\n")
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "|".join(["---"] * len(head)) + "|")
    for r in results:
        t = r["test"]
        cells = [r["problem"], r["category"]]
        for c in head[2:]:
            cells.append(f"{t[c]['mean']:.4f}±{t[c]['std']:.4f}")
        L.append("| " + " | ".join(cells) + " |")
    # SFGE tuning detail
    L.append("\n## SFGE: tuned vs default\n")
    L.append("| problem | default hp | default val-reg | tuned hp | tuned val-reg | "
             "test: SFGE-def | SFGE-tuned | Δtest |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        t = r["test"]; bh = r["sfge_best_hp"]
        dv = r["sfge_val_default"]["mean"]; tv = r["sfge_val_tuned"]["mean"]
        sd = t["SFGE-def"]["mean"]; st = t["SFGE-tuned"]["mean"]
        bhs = f"n{bh['n_samples']} σ{bh['sigma']} lr{bh['lr']:g} e{bh['epochs']}"
        L.append(f"| {r['problem']} | n8 σ0.5 lr0.01 e120 | {dv:.4f} | {bhs} | {tv:.4f} | "
                 f"{sd:.4f} | {st:.4f} | {st-sd:+.4f} |")
    # supplementary LTR variants
    L.append("\n## Supplementary: LTR variants (test regret)\n")
    L.append("| problem | LTR (pairwise) | listwise | pointwise |")
    L.append("|---|---|---|---|")
    for r in results:
        t = r["test"]
        L.append(f"| {r['problem']} | {t['LTR']['mean']:.4f} | {t['lsLTR']['mean']:.4f} | "
                 f"{t['ptwLTR']['mean']:.4f} |")
    # verdict
    L.append("\n## Does tuning SFGE change the SFGE-vs-PolyStep picture?\n")
    for r in results:
        t = r["test"]; st = t["SFGE-tuned"]["mean"]; ps = t["PolyStep"]["mean"]; sd = t["SFGE-def"]["mean"]
        who = "SFGE-tuned" if st < ps else ("PolyStep" if ps < st else "tie")
        L.append(f"- **{r['problem']}**: SFGE-def {sd:.4f} -> SFGE-tuned {st:.4f}; PolyStep {ps:.4f} "
                 f"-> winner **{who}** (|Δ|={abs(st-ps):.4f}).")
    with open(mpath, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nWROTE {jpath}\nWROTE {mpath}", flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    smoke = "--smoke" in sys.argv
    problems = args[0].split(",") if len(args) > 0 else ["sp", "knap", "tsp", "port"]
    deg = int(args[1]) if len(args) > 1 else 4
    seeds = [int(s) for s in args[2].split(",")] if len(args) > 2 else [0, 1, 2]
    print(f"FAIR BASELINES | problems={problems} deg={deg} seeds={seeds} smoke={smoke}", flush=True)
    t0 = time.time()
    results = run(problems, deg, seeds, smoke)
    write_outputs(results, deg, seeds, smoke)
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
