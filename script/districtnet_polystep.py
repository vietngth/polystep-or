"""DistrictNet x PolyStep: a real-world predict-then-optimize districting benchmark.

Reuses the DistrictNet (NeurIPS 2024) data and its precomputed realized-cost cache (no Julia/C++ build):
each city has ~450 candidate districts (connected blocks, size 2-4) with a cached expected routing cost
(deps LKH/Evaluator, baked into data/tspCosts). A districting decision selects districts that PARTITION
the blocks (cover each block once) into exactly r = num_blocks/target_size districts, minimizing total
realized routing cost.

PtO: predict each candidate district's cost from geometric features, solve the set-partition ILP on the
predicted costs (Gurobi), and incur the TRUE realized cost (cache sum) of the chosen partition. We train
the predictor with two-stage (MSE on cached costs), SFGE, and PolyStep (direct realized-cost
minimization). The predictor runs on GPU; the set-partition ILP is CPU (combinatorial, like DistrictNet's
own solver) -- this is the exact-solver regime, reported honestly. Generalizes across cities (district
features are city-agnostic): train on one set of cities, test on held-out cities.

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python districtnet_polystep.py [n_train] [n_test] [seed]
"""
from __future__ import annotations
import sys, json, glob, time
sys.path.insert(0, "polystep/src")
import numpy as np
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from pto.seeding import seed_everything, device_generator
from pto.multiseed import summarize, wilcoxon_pair, md_table, write_json, write_md, fmt_mean_std

dev = "cuda" if torch.cuda.is_available() else "cpu"
TARGET = 3
FEAT = 6


def mst_len(pts):
    """Euclidean MST length over a small point set (Prim). pts: (k,2)."""
    k = len(pts)
    if k <= 1:
        return 0.0
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    intree = [0]; total = 0.0
    while len(intree) < k:
        best, bj = 1e18, -1
        for i in intree:
            for j in range(k):
                if j not in intree and D[i, j] < best:
                    best, bj = D[i, j], j
        total += best; intree.append(bj)
    return total


def load_city(cf):
    d = json.load(open(cf)); fts = d["features"]
    n = len(fts)
    cent = np.array([[f["properties"]["CENTROID_X"], f["properties"]["CENTROID_Y"]] for f in fts])
    cid = cf.split("/")[-1].split(".")[0]
    cc = f"DistrictNet/data/tspCosts/{cid}_C_{n}_{TARGET}_tsp.train_and_test.json"
    cands = json.load(open(cc))["districts"]
    C = len(cands)
    mask = np.zeros((C, n), dtype=np.float32)
    cost = np.zeros(C, dtype=np.float32)
    feat = np.zeros((C, FEAT), dtype=np.float32)
    for i, dct in enumerate(cands):
        bl = dct["list-blocks"]; mask[i, bl] = 1.0; cost[i] = dct["average-cost"]
        pts = cent[bl]
        pw = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        feat[i] = [len(bl), mst_len(pts), pw.max(), pw.sum() / max(len(bl) * (len(bl) - 1), 1),
                   pts[:, 0].std(), pts[:, 1].std()]
    r = n // TARGET
    return {"cid": cid, "n": n, "r": r, "mask": mask, "cost": cost, "feat": feat}


def build_partition_model(city):
    """Prebuild a Gurobi set-partition model whose objective coeffs we reset per solve (fast)."""
    n, r, mask = city["n"], city["r"], city["mask"]
    C = mask.shape[0]
    md = gp.Model(); md.Params.OutputFlag = 0
    z = md.addVars(C, vtype=GRB.BINARY, name="z")
    for b in range(n):
        idx = np.nonzero(mask[:, b])[0]
        md.addConstr(gp.quicksum(z[int(c)] for c in idx) == 1)
    md.addConstr(gp.quicksum(z[c] for c in range(C)) == r)
    md.update()
    city["_md"], city["_z"], city["_C"] = md, z, C
    return city


def solve_partition(city, pred_cost):
    """Select districts minimizing sum pred_cost; return selection mask (C,) 0/1 (np)."""
    md, z, C = city["_md"], city["_z"], city["_C"]
    for c in range(C):
        z[c].Obj = float(pred_cost[c])
    md.ModelSense = GRB.MINIMIZE; md.update(); md.optimize()
    return np.array([z[c].X for c in range(C)])


def oracle_realized(city):
    """Best achievable realized cost: set-partition on TRUE cached costs."""
    sel = solve_partition(city, city["cost"])
    return float((sel * city["cost"]).sum())


def make_pred():
    # linear predictor: keeps the PolyStep candidate count (hence #ILP solves/step) small and tractable
    return nn.Linear(FEAT, 1, bias=True).to(dev)


def predict_costs(model, feat_t):
    return model(feat_t).squeeze(-1)                       # (C,)


def realized_of(model, cities):
    """Mean realized districting cost over cities for a single model (eval)."""
    tot = 0.0
    with torch.no_grad():
        for cy in cities:
            pc = predict_costs(model, cy["feat_t"]).cpu().numpy()
            sel = solve_partition(cy, pc)
            tot += float((sel * cy["cost"]).sum())
    return tot / len(cities)


def regret_of(model, cities):
    reg = []
    with torch.no_grad():
        for cy in cities:
            pc = predict_costs(model, cy["feat_t"]).cpu().numpy()
            sel = solve_partition(cy, pc)
            realized = float((sel * cy["cost"]).sum())
            reg.append((realized - cy["oracle"]) / cy["oracle"])
    return float(np.mean(reg))


def train_two_stage(cities, epochs=300, lr=1e-2):
    m = make_pred(); opt = torch.optim.Adam(m.parameters(), lr)
    feat = torch.cat([cy["feat_t"] for cy in cities]); tgt = torch.cat([cy["cost_t"] for cy in cities])
    fmean, fstd = feat.mean(0, keepdim=True), feat.std(0, keepdim=True) + 1e-6
    for cy in cities:
        cy["_norm"] = (fmean, fstd)
    for _ in range(epochs):
        opt.zero_grad(); ((predict_costs(m, (feat - fmean) / fstd) - tgt) ** 2).mean().backward(); opt.step()
    return m, (fmean, fstd)


def train_polystep(cities, warm, norm, scale, steps=60, seed=0):
    m = make_pred(); m.load_state_dict(warm.state_dict())
    fmean, fstd = norm
    pso = PolyStepOptimizer(m, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=0.4, probe_radius=0.8, num_probe=1, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    names = [n for n, _ in m.named_parameters()]
    from torch.func import functional_call
    def closure(bp):
        K = bp[names[0]].shape[0]; out = torch.zeros(K, device=dev)
        for k in range(K):
            params = {nm: bp[nm][k] for nm in names}
            realized = 0.0
            for cy in cities:
                pc = functional_call(m, params, ((cy["feat_t"] - fmean) / fstd,)).squeeze(-1)
                sel = solve_partition(cy, pc.detach().cpu().numpy())
                realized += float((sel * cy["cost"]).sum())
            out[k] = realized / len(cities) / scale
        return out
    for _ in range(steps):
        pso.step(closure)
    return m


def train_sfge(cities, warm, norm, scale, epochs=60, n_samples=6, sigma=0.5, lr=1e-2, seed=0):
    m = make_pred(); m.load_state_dict(warm.state_dict())
    fmean, fstd = norm
    opt = torch.optim.Adam(m.parameters(), lr); g = device_generator(seed, dev)
    feats = [(cy["feat_t"] - fmean) / fstd for cy in cities]
    for _ in range(epochs):
        preds = [predict_costs(m, f) for f in feats]                       # per-city (C,)
        loss_terms = []
        for cy, pred in zip(cities, preds):
            with torch.no_grad():
                eps = torch.randn(n_samples, pred.shape[0], device=dev, generator=g)
                chat = pred.unsqueeze(0) + sigma * eps                     # (S,C)
                realized = torch.tensor(
                    [(solve_partition(cy, chat[s].cpu().numpy()) * cy["cost"]).sum() for s in range(n_samples)],
                    device=dev) / scale
                adv = realized - realized.mean()
            logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
            loss_terms.append((adv * logp).mean())
        opt.zero_grad(); torch.stack(loss_terms).mean().backward(); opt.step()
    return m


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    seed_everything(seed)
    files = sorted(glob.glob("DistrictNet/data/geojson/city*.geojson"),
                   key=lambda f: int(f.split("city")[-1].split(".")[0]))
    print(f"DISTRICTNET x PolyStep | n_train={n_train} n_test={n_test} seed={seed} | loading cities...", flush=True)
    train = [build_partition_model(load_city(f)) for f in files[:n_train]]
    test = [build_partition_model(load_city(f)) for f in files[100:100 + n_test]]
    for cy in train + test:
        cy["feat_t"] = torch.tensor(cy["feat"], device=dev)
        cy["cost_t"] = torch.tensor(cy["cost"], device=dev)
        cy["oracle"] = oracle_realized(cy)
    scale = float(np.mean([cy["oracle"] for cy in train]))
    t0 = time.time()
    ts, norm = train_two_stage(train)
    for cy in train + test:
        cy.pop("_norm", None)
    ps = train_polystep(train, ts, norm, scale, seed=seed)
    sf = train_sfge(train, ts, norm, scale, seed=seed)
    res = {m: regret_of(mdl, test) for m, mdl in (("two-stage", ts), ("SFGE", sf), ("PolyStep", ps))}
    rcost = {m: realized_of(mdl, test) for m, mdl in (("two-stage", ts), ("SFGE", sf), ("PolyStep", ps))}
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    for m in ("two-stage", "SFGE", "PolyStep"):
        print(f"  {m:>10}: regret {res[m]:+.4f}  realized {rcost[m]:.2f}", flush=True)
    write_json("exp_results/districtnet.json", {"n_train": n_train, "n_test": n_test, "seed": seed,
               "regret": res, "realized": rcost, "oracle_mean_test": float(np.mean([cy["oracle"] for cy in test]))})
    md = ("# DistrictNet x PolyStep (real-world districting predict-then-optimize)\n\n"
          f"n_train={n_train} cities, n_test={n_test} held-out cities, seed={seed}. Predict candidate-district "
          "costs from geometric features; set-partition ILP (Gurobi) selects the districting; realized cost from "
          "the DistrictNet routing-cost cache. Normalized regret vs the cache-optimal partition (lower better).\n\n"
          + md_table(["method", "normalized regret", "mean realized cost"],
                     [[m, f"{res[m]:+.4f}", f"{rcost[m]:.2f}"] for m in ("two-stage", "SFGE", "PolyStep")]))
    write_md("exp_results/districtnet.md", md)
    print("\nwrote exp_results/districtnet.{json,md}\nDONE", flush=True)


if __name__ == "__main__":
    main()
