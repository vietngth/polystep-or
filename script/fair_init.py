"""INTEGRITY CHECK: was the SPO+ comparison unfair? Our runs warm-started SFGE/PolyStep from the
two-stage MSE solution but trained SPO+ from scratch. Re-run all three under the SAME init:
  - COLD: all from scratch
  - WARM: all warm-started from two-stage
Knapsack (ILP), deg 4, 3 seeds. If SFGE/PolyStep only beat SPO+ when warm-started, the earlier
'beats SPO+' claims are a warm-start artifact (consistent with Mandi: SFGE !> SPO+).
"""
import sys, numpy as np, torch
sys.path.insert(0, "polystep/src")
import pyepo.func as F
from pyepo import metric
from pto.capability import setup_knap, train_two_stage, train_sfge, train_polystep, dev

def train_spo(cfg, warm, epochs=30):
    m = cfg["make"]()
    if warm is not None:
        with torch.no_grad(): m.weight.copy_(warm.weight)
    opt = torch.optim.Adam(m.parameters(), 1e-2); spop = F.SPOPlus(cfg["om"])
    for _ in range(epochs):
        for xb, cb, wb, zb in cfg["ld_tr"]:
            xb, cb, wb, zb = [t.float().to(dev) for t in (xb, cb, wb, zb)]
            opt.zero_grad(); spop(m(xb), cb, wb, zb).mean().backward(); opt.step()
    return m

SEEDS = [0, 1, 2]
res = {c: {m: [] for m in ("SPO+", "SFGE", "PolyStep")} for c in ("COLD", "WARM")}
for seed in SEEDS:
    cfg, _ = setup_knap(seed, 4)
    ts = train_two_stage(cfg)
    # COLD — all from scratch
    cfg["warm"] = None
    res["COLD"]["SPO+"].append(metric.regret(train_spo(cfg, None), cfg["om"], cfg["ld_te"]))
    res["COLD"]["SFGE"].append(metric.regret(train_sfge(cfg), cfg["om"], cfg["ld_te"]))
    res["COLD"]["PolyStep"].append(metric.regret(train_polystep(cfg), cfg["om"], cfg["ld_te"]))
    # WARM — all from two-stage
    cfg["warm"] = ts
    res["WARM"]["SPO+"].append(metric.regret(train_spo(cfg, ts), cfg["om"], cfg["ld_te"]))
    res["WARM"]["SFGE"].append(metric.regret(train_sfge(cfg), cfg["om"], cfg["ld_te"]))
    res["WARM"]["PolyStep"].append(metric.regret(train_polystep(cfg), cfg["om"], cfg["ld_te"]))

print("FAIR-INIT CHECK | knapsack (ILP) deg=4, 3 seeds | normalized regret")
print(f"{'init':>6} | {'two-stage(ref)':>14} {'SPO+':>8} {'SFGE':>8} {'PolyStep':>9}")
ts_ref = np.mean([metric.regret(train_two_stage(setup_knap(s, 4)[0]), setup_knap(s, 4)[0]["om"],
                                setup_knap(s, 4)[0]["ld_te"]) for s in SEEDS])
for c in ("COLD", "WARM"):
    m = {k: np.mean(v) for k, v in res[c].items()}
    print(f"{c:>6} | {ts_ref:>14.4f} {m['SPO+']:>8.4f} {m['SFGE']:>8.4f} {m['PolyStep']:>9.4f}", flush=True)
