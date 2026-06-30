"""PolyStep on the Amazon Last-Mile Routing Research Challenge (ALMRRC 2021).

METHODOLOGY (strict): we reuse the AWS reference solver and the official scorer VERBATIM:
  * decision solver  : aro.model.zone_utils.zone_based_tsp  (its PPM-rollout zone ordering +
                       per-zone OR-tools TSP, aro.model.ortools_helper.run_ortools).
  * scorer           : baselines/rc-cli/scoring/score.py  (evaluate / score = seq_dev x erp_per_edit).
  * protocol         : train on almrrc2021-data-training, score on almrrc2021-data-evaluation
                       with the official metric (submission_score = mean route score).

The ONLY new component is the PolyStep-learned zone preference model that REPLACES the hand-crafted
PPM (aro.model.ppm.PPM).  We expose a `LearnedZoneModel` with the SAME `.query(preceding, following)`
scalar interface the solver expects, so `zone_based_tsp` runs UNCHANGED.  The scalar it returns is a
zone-transition affinity:   reward(a->b) = -PRIOR_W * dist(a,b)/scale  +  g_theta(features(a,b)),
i.e. a learned correction (driver preference) on top of a travel-time / distance prior.  g_theta is a
small MLP whose parameters are PolyStep's optimisation variables.  PolyStep minimises the deployed
route's REALISED official rc-cli score directly (experience / cost-min regime -- no per-stop labels).

Features per zone: centroid (km, vs station), #stops, mean planned service time, summed parcel volume,
mean time-window start.  Pair features add (dx, dy, dist, same-major-cluster, same-sub-cluster).

Data prep is done faithfully from the raw challenge JSON (same node ordering -- station first, then
stops in dict-insertion order -- and the same symmetric-mean travel-time matrix the reference builds).
We avoid the reference parquet/MDS feature pipeline (it is brittle under pandas 3.x and its MDS coords
are not needed by this predictor); everything that touches the *solver* and *scorer* is verbatim.

Run (cluster compute node):
  .venv/bin/python exp_amazon_polystep.py --n_train 800 --n_eval full --steps 40
"""
from __future__ import annotations
import argparse, json, os, sys, time, pickle, math
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "baselines/amazon-routing-challenge-sol"))
sys.path.insert(0, os.path.join(ROOT, "baselines/rc-cli/scoring"))
sys.path.insert(0, os.path.join(ROOT, "polystep/src"))

import torch
import torch.nn as nn
import torch.func as tfunc

import score as rc_score                                   # official rc-cli scorer (verbatim)
from aro.model.zone_utils import zone_based_tsp            # reference solver (verbatim)
from aro.model.ppm import build_ppm_model                  # reference PPM trainer (verbatim)
from polystep import PolyStepOptimizer                     # the new component's optimiser

DATA = os.environ.get("AMAZON_DATA", os.path.join(ROOT, "data/amazon_lmrrc"))
TRAIN_DIR = os.path.join(DATA, "almrrc2021-data-training")
EVAL_DIR = os.path.join(DATA, "almrrc2021-data-evaluation")
CACHE = os.environ.get("AMAZON_CACHE", os.path.join(ROOT, "exp_cache/amazon"))
os.makedirs(CACHE, exist_ok=True)
DEV = "cpu"   # solver/scorer-bound; tiny MLP -> CPU is fine and avoids host<->dev churn
PRIOR_W = 3.0

# ----------------------------------------------------------------------------- raw json subset
def _defloat(o):
    """ijson yields Decimal for numbers; rc-cli good_format requires float/int. Recurse."""
    import decimal
    if isinstance(o, decimal.Decimal):
        return float(o)
    if isinstance(o, dict):
        return {k: _defloat(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_defloat(v) for v in o]
    return o


def _stream_subset(path, wanted):
    """Stream a giant top-level-dict json, keep only keys in `wanted`."""
    import ijson
    out = {}
    wset = set(wanted)
    with open(path, "rb") as f:
        for k, v in ijson.kvitems(f, ""):
            if k in wset:
                out[k] = _defloat(v)
                if len(out) == len(wset):
                    break
    return out


def _load_filter(path, rids):
    """stdlib json.load (tolerates bare NaN/Infinity, unlike ijson) then keep only `rids`."""
    rset = set(rids)
    with open(path) as f:
        d = json.load(f)
    return {k: v for k, v in d.items() if k in rset}


def _stream_pick(path, valid, n):
    """Stream travel_times from the front; collect the first n routes that are in `valid`.
    Returns (rids, tt_subset). Avoids a full scan of the multi-GB file for subsets."""
    import ijson
    rids, tt = [], {}
    with open(path, "rb") as f:
        for k, v in ijson.kvitems(f, ""):
            if k in valid:
                rids.append(k)
                tt[k] = _defloat(v)
                if len(rids) >= n:
                    break
    return rids, tt


def load_split(split, n, seed=0):
    """Return subset dicts for a split, cached. split in {train,eval}."""
    tag = f"{split}_{n}"
    cpath = os.path.join(CACHE, f"raw_{tag}.pkl")
    if os.path.exists(cpath):
        with open(cpath, "rb") as f:
            return pickle.load(f)
    if split == "train":
        d = TRAIN_DIR
        f_act = f"{d}/model_build_inputs/actual_sequences.json"
        f_route = f"{d}/model_build_inputs/route_data.json"
        f_pkg = f"{d}/model_build_inputs/package_data.json"
        f_tt = f"{d}/model_build_inputs/travel_times.json"
        f_inv = f"{d}/model_build_inputs/invalid_sequence_scores.json"
    else:
        d = EVAL_DIR
        f_act = f"{d}/model_score_inputs/eval_actual_sequences.json"
        f_route = f"{d}/model_apply_inputs/eval_route_data.json"
        f_pkg = f"{d}/model_apply_inputs/eval_package_data.json"
        f_tt = f"{d}/model_apply_inputs/eval_travel_times.json"
        f_inv = f"{d}/model_score_inputs/eval_invalid_sequence_scores.json"

    print(f"[load_split {split}] reading actual sequences ...", flush=True)
    with open(f_act) as f:
        actual = json.load(f)
    with open(f_inv) as f:
        invalid = json.load(f)
    valid = set(actual.keys())
    if n is None or n >= len(valid):
        rids = set(valid)
        print(f"[load_split {split}] FULL = {len(rids)}; streaming travel times (full scan) ...", flush=True)
        tt = _stream_subset(f_tt, rids)
    else:
        print(f"[load_split {split}] picking first {n} routes from travel-time stream ...", flush=True)
        picked, tt = _stream_pick(f_tt, valid, n)
        rids = set(picked)
    print(f"[load_split {split}] subset routes = {len(rids)}; loading route/pkg ...", flush=True)
    route = _load_filter(f_route, rids)
    pkg = _load_filter(f_pkg, rids)
    # keep only routes that have everything
    keep = [r for r in rids if r in route and r in pkg and r in tt and r in actual]
    out = {
        "rids": sorted(keep),
        "actual": {r: actual[r] for r in keep},
        "invalid": {r: invalid.get(r, 1.0) for r in keep},
        "route": route, "pkg": pkg, "tt": tt,
    }
    with open(cpath, "wb") as f:
        pickle.dump(out, f)
    print(f"[load_split {split}] cached {len(keep)} routes -> {cpath}", flush=True)
    return out


# ----------------------------------------------------------------------------- per-route geometry
def parse_zone(z):
    """hierarchy pieces a la aro.model.ppm: (major, sub, ssc)."""
    if z == "stz" or z is None:
        return ("_", "_", "_")
    major = z[0]
    sub = z.split(".")[0].split("-")[-1]
    ssc = z.split(".")[-1]
    return (major, sub, ssc)


def build_route(rid, route, pkg, tt):
    """Faithful reconstruction of the reference node ordering + distance matrix + zone list.

    Mirrors preprocessing.gen_zone_list / gen_distance_matrix: iterate stops dict insertion order,
    station(s) first, others appended; zone 'stz' for station; None zones filled by nearest haversine.
    Distance matrix = symmetric mean of raw travel times, same node order. Returns dict or None.
    """
    stops_dict = route[rid]["stops"]
    tt_r = tt[rid]
    pkg_r = pkg.get(rid, {})
    nodes, coords, zones = [], [], []
    # station(s) first
    order = list(stops_dict.items())
    sta = [(k, v) for k, v in order if v.get("type") == "Station"]
    oth = [(k, v) for k, v in order if v.get("type") != "Station"]
    for k, v in sta + oth:
        nodes.append(k)
        coords.append([v["lat"], v["lng"]])
        if v.get("type") == "Station":
            zones.append("stz")
        else:
            z = v.get("zone_id")
            zones.append(None if (z is None or (isinstance(z, float) and math.isnan(z))) else z)
    if not sta:
        return None
    coords = np.array(coords, dtype=float)
    # fill None zones by nearest non-None, non-stz (haversine), as in gen_zone_list
    none_idx = [i for i, z in enumerate(zones) if z is None]
    if none_idx:
        rad = np.radians(coords)
        for i in none_idx:
            d = np.sum((rad - rad[i]) ** 2, axis=1)  # cheap surrogate of haversine ordering
            for j in np.argsort(d):
                if zones[j] is not None and zones[j] != "stz":
                    zones[i] = zones[j]
                    break
            if zones[i] is None:
                zones[i] = "stz"
    # distance matrix: symmetric mean of travel times, node order above
    nN = len(nodes)
    M = np.zeros((nN, nN), dtype=float)
    for i, a in enumerate(nodes):
        ta = tt_r.get(a, {})
        for j, b in enumerate(nodes):
            ab = ta.get(b)
            ba = tt_r.get(b, {}).get(a)
            if ab is None and ba is None:
                v = 0.0
            elif ab is None:
                v = float(ba)
            elif ba is None:
                v = float(ab)
            else:
                v = 0.5 * (float(ab) + float(ba))
            M[i, j] = v
    # per-stop package aggregates -> per-zone features
    svc = np.zeros(nN); vol = np.zeros(nN); tws = np.zeros(nN)
    for i, k in enumerate(nodes):
        info = pkg_r.get(k, {})
        if not isinstance(info, dict):
            continue
        s = 0.0; vv = 0.0; tstart = []
        for _, pi in info.items():
            if not isinstance(pi, dict):
                continue
            st = pi.get("planned_service_time_seconds")
            if st is not None:
                s += float(st)
            dim = pi.get("dimensions") or {}
            try:
                vv += float(dim.get("depth_cm", 0)) * float(dim.get("height_cm", 0)) * float(dim.get("width_cm", 0))
            except Exception:
                pass
            tw = pi.get("time_window") or {}
        svc[i] = s; vol[i] = vv
    return {"rid": rid, "nodes": nodes, "coords": coords, "zones": zones,
            "matrix": M, "svc": svc, "vol": vol}


def utm_xy(coords):
    import utm
    xy = np.zeros((len(coords), 2))
    for i, (la, lo) in enumerate(coords):
        x, y, _, _ = utm.from_latlon(la, lo)
        xy[i] = [x, y]
    return xy / 1000.0  # km


def zone_feature_pack(rt):
    """Per-route: unique zones (incl 'stz'), per-zone feature matrix, centroid-distance matrix."""
    xy = utm_xy(rt["coords"])
    zones = rt["zones"]
    uniq = list(dict.fromkeys(zones))           # preserves order; 'stz' first (station first)
    zi = {z: i for i, z in enumerate(uniq)}
    Z = len(uniq)
    cent = np.zeros((Z, 2)); nst = np.zeros(Z); svc = np.zeros(Z); vol = np.zeros(Z)
    cnt = np.zeros(Z)
    for i, z in enumerate(zones):
        j = zi[z]
        cent[j] += xy[i]; nst[j] += 1; svc[j] += rt["svc"][i]; vol[j] += rt["vol"][i]; cnt[j] += 1
    cent /= np.maximum(cnt[:, None], 1)
    sta = cent[zi["stz"]] if "stz" in zi else cent[0]
    # per-zone raw feature: rel-x, rel-y, n_stops, mean_svc, sum_vol
    feat = np.stack([cent[:, 0] - sta[0], cent[:, 1] - sta[1], nst,
                     svc / np.maximum(cnt, 1), vol], axis=1)         # (Z,5)
    # centroid distance matrix
    D = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))  # (Z,Z)
    par = [parse_zone(z) for z in uniq]
    same_major = np.array([[1.0 if par[a][0] == par[b][0] else 0.0 for b in range(Z)] for a in range(Z)])
    same_sub = np.array([[1.0 if par[a][1] == par[b][1] else 0.0 for b in range(Z)] for a in range(Z)])
    return {"uniq": uniq, "zi": zi, "feat": feat.astype(np.float32), "D": D.astype(np.float32),
            "same_major": same_major.astype(np.float32), "same_sub": same_sub.astype(np.float32)}


# ----------------------------------------------------------------------------- predictor
ZF = 5                  # per-zone raw features
PF = 2 * ZF + 6         # pair feat: feat_a, feat_b, dx, dy, dist, same_major, same_sub, ones (even!)


class ZoneNet(nn.Module):
    """Learned zone-transition affinity. hid=0 -> a single linear (interpretable, 16 params, tiny N);
    hid>0 -> one hidden layer. Final layer bias disabled so every param has an even element count
    (PolyStep reshapes each param to (num_particles, particle_dim=2))."""
    def __init__(self, hid=0):
        super().__init__()
        if hid <= 0:
            self.net = nn.Sequential(nn.Linear(PF, 1, bias=False))
        else:
            self.net = nn.Sequential(nn.Linear(PF, hid), nn.Tanh(), nn.Linear(hid, 1, bias=False))
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.05)
            else:
                nn.init.zeros_(p)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pair_features(zp, fmean, fstd):
    """Build (Z*Z, PF) pair-feature tensor for a route from its zone pack."""
    feat = (zp["feat"] - fmean) / fstd                      # (Z,ZF) standardised
    Z = feat.shape[0]
    fa = np.repeat(feat[:, None, :], Z, axis=1)             # (Z,Z,ZF) from
    fb = np.repeat(feat[None, :, :], Z, axis=0)             # (Z,Z,ZF) to
    dxy = (zp["feat"][None, :, :2] - zp["feat"][:, None, :2])   # raw rel positions diff (km)
    ones = np.ones((Z, Z, 1), dtype=np.float32)                 # intercept term
    geo = np.concatenate([dxy,
                          zp["D"][:, :, None],
                          zp["same_major"][:, :, None],
                          zp["same_sub"][:, :, None],
                          ones], axis=2)                          # (Z,Z,6)
    P = np.concatenate([fa, fb, geo], axis=2).reshape(Z * Z, PF)
    return torch.from_numpy(P.astype(np.float32))


class LearnedZoneModel:
    """Drop-in replacement for aro.model.ppm.PPM exposing .query(preceding, following).

    reward(a->b) = -PRIOR_W * D(a,b)/scale + g_theta(features).  Higher = preferred (the reference
    rollout maximises reward), matching PPM's log-prob convention (higher = preferred)."""
    def __init__(self, zp, S):
        self.zi = zp["zi"]
        self.S = S

    def query(self, preceding_zone_list, following_zone, no_context_panelty=1.0,
              consider_hierarchy=True, cluster_weights=None):
        a = preceding_zone_list[-1] if preceding_zone_list else "stz"
        ia = self.zi.get(a, self.zi.get("stz", 0))
        ib = self.zi.get(following_zone, ia)
        return float(self.S[ia, ib])


def build_S(zp, scores):
    """scores: (Z,Z) learned correction. Add geometric prior. Return reward matrix (Z,Z)."""
    D = zp["D"]
    scale = float(np.median(D[D > 0])) if np.any(D > 0) else 1.0
    return (-PRIOR_W * D / max(scale, 1e-6)) + scores


# ----------------------------------------------------------------------------- parallel workers
# zone_based_tsp costs ~2 s/route (per-zone OR-tools), so route-solves are parallelised across CPUs.
# A fork-based Pool lets workers inherit these module globals copy-on-write (no pickling of routes).
G_ROUTES = {}      # rid -> route dict
G_PACKS = {}       # rid -> zone pack
G_PPM = None       # reference PPM model (for baseline parallel deploy)


def _w_learned(task):
    """task = (rid, S float32 array). Returns (rid, score, proposed_dict)."""
    rid, S = task
    rt = G_ROUTES[rid]; zp = G_PACKS[rid]
    try:
        prop = deploy_route(rt, zp, S)
        sv = route_score(rt["actual"], prop, rt["cost"])
        return rid, (1.0 if sv is None else sv), prop
    except Exception:
        return rid, 1.0, None


def _w_ppm(rid):
    rt = G_ROUTES[rid]
    try:
        prop = deploy_route_ppm(rt, G_PPM)
        return rid, prop
    except Exception:
        return rid, None


# ----------------------------------------------------------------------------- deploy + score
def deploy_route(rt, zp, S):
    """Run the reference solver with the given zone reward matrix; return proposed seq dict."""
    model = LearnedZoneModel(zp, S)
    tour = zone_based_tsp(rt["matrix"], list(rt["zones"]), model, rt["rid"],
                          cluster_weights=[0.25, 0.25, 0.25, 0.25], zone_sort_algo="ppm")
    tour = [int(x) for x in tour]
    rank = [-1] * len(rt["nodes"])
    for i, node in enumerate(tour):
        rank[node] = i
    seq = {rt["nodes"][i]: rank[i] for i in range(len(rt["nodes"]))}
    return {"proposed": seq}


def deploy_route_ppm(rt, ppm):
    tour = zone_based_tsp(rt["matrix"], list(rt["zones"]), ppm, rt["rid"],
                          cluster_weights=[0.25, 0.25, 0.25, 0.25], zone_sort_algo="ppm")
    tour = [int(x) for x in tour]
    rank = [-1] * len(rt["nodes"])
    for i, node in enumerate(tour):
        rank[node] = i
    seq = {rt["nodes"][i]: rank[i] for i in range(len(rt["nodes"]))}
    return {"proposed": seq}


def route_score(actual_dict, proposed_dict, cost_mat):
    """Official rc-cli single-route score, verbatim functions.

    rc_score.score -> normalize_matrix MUTATES the cost dict in place, so we hand it a fresh
    nested copy each call (the official file scorer re-reads from JSON, so it never reuses)."""
    actual = rc_score.route2list(actual_dict)
    sub = rc_score.route2list(proposed_dict)
    if rc_score.isinvalid(actual, sub):
        return None
    cm = {o: {d: v for d, v in row.items()} for o, row in cost_mat.items()}
    return float(rc_score.score(actual, sub, cm))


# ----------------------------------------------------------------------------- official file scorer
def official_evaluate(routes, props, data, tag):
    """Write subset actual/submission/cost/invalid JSON and call rc-cli evaluate() verbatim."""
    sub_path = os.path.join(CACHE, f"submission_{tag}.json")
    act_path = os.path.join(CACHE, f"actual_{tag}.json")
    cost_path = os.path.join(CACHE, f"costs_{tag}.json")
    inv_path = os.path.join(CACHE, f"invalid_{tag}.json")
    submission = {r: props[r] for r in routes if r in props}
    actual = {r: data["actual"][r] for r in routes}
    costs = {r: data["tt"][r] for r in routes}
    invalid = {r: data["invalid"][r] for r in routes}
    for p, o in [(sub_path, submission), (act_path, actual), (cost_path, costs), (inv_path, invalid)]:
        with open(p, "w") as f:
            json.dump(o, f)
    res = rc_score.evaluate(act_path, sub_path, cost_path, inv_path)
    return res


# ----------------------------------------------------------------------------- PolyStep training
def _S_for(net, params, PFs, packs, rid):
    """Compute the (Z,Z) reward matrix for one route under a parameter set (main process, fast)."""
    with torch.no_grad():
        sc = tfunc.functional_call(net, params, (PFs[rid],))
    Z = len(packs[rid]["uniq"])
    return build_S(packs[rid], sc.reshape(Z, Z).numpy()).astype(np.float32)


def train_polystep(net, PFs, packs, rids, pool, steps, n_probe, seed, hid, batch=None, log=print):
    """PolyStep over the small ZoneNet; closure parallelises route-solves across `pool`.
    The reference solver + rc-cli scorer run inside the workers, unchanged."""
    names = [n for n, _ in net.named_parameters()]
    rng = np.random.default_rng(seed)

    _seen = {"n": False}

    def scored(param_list, route_list):
        """param_list: list of param-dicts; route_list: rids. Returns array [len(param_list)] of mean score."""
        tasks = []
        for params in param_list:
            for r in route_list:
                tasks.append((r, _S_for(net, params, PFs, packs, r)))
        res = pool.map(_w_learned, tasks, chunksize=max(1, len(tasks) // (pool._processes * 4)))
        sc = np.array([x[1] for x in res]).reshape(len(param_list), len(route_list))
        return sc.mean(1)

    def closure(bp):
        N = bp[names[0]].shape[0]
        if not _seen["n"]:
            log(f"[polystep] closure batch N={N}; route-solves/step={N * (batch or len(rids))}", flush=True)
            _seen["n"] = True
        chosen = rids if (batch is None or batch >= len(rids)) else \
            [rids[i] for i in rng.choice(len(rids), batch, replace=False)]
        param_list = [{n: bp[n][k] for n in names} for k in range(N)]
        out = scored(param_list, chosen)
        return torch.tensor(out, dtype=torch.float32)

    pso = PolyStepOptimizer(net, polytope_type="orthoplex",
                            epsilon=0.3, step_radius=0.4, probe_radius=0.8,
                            num_probe=n_probe, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    vrng = np.random.default_rng(seed + 1)
    val_ids = rids if len(rids) <= 120 else [rids[i] for i in vrng.choice(len(rids), 120, replace=False)]

    def full_obj():
        params = {n: p.detach() for n, p in net.named_parameters()}
        return float(scored([params], val_ids)[0])

    best_val = full_obj()
    best_state = {n: p.detach().clone() for n, p in net.named_parameters()}
    log(f"[polystep] init val score = {best_val:.4f}  (params={sum(p.numel() for p in net.parameters())}, "
        f"hid={hid}, train_routes={len(rids)}, val={len(val_ids)})", flush=True)
    for s in range(steps):
        t0 = time.time()
        pso.step(closure)
        cur = full_obj()
        if cur < best_val:
            best_val = cur
            best_state = {n: p.detach().clone() for n, p in net.named_parameters()}
        log(f"[polystep] step {s+1}/{steps} val={cur:.4f} best={best_val:.4f} ({time.time()-t0:.1f}s)", flush=True)
    with torch.no_grad():
        for n, p in net.named_parameters():
            p.copy_(best_state[n])
    return net, best_val


def net_score_matrix(net, zp, fmean, fstd):
    P = pair_features(zp, fmean, fstd)
    with torch.no_grad():
        sc = net(P)
    Z = len(zp["uniq"])
    return build_S(zp, sc.reshape(Z, Z).numpy())


# ----------------------------------------------------------------------------- main
def prepare(split, n, seed):
    raw = load_split(split, n, seed)
    routes, packs = {}, {}
    for r in raw["rids"]:
        rt = build_route(r, raw["route"], raw["pkg"], raw["tt"])
        if rt is None:
            continue
        rt["actual"] = raw["actual"][r]
        rt["cost"] = raw["tt"][r]
        routes[r] = rt
        packs[r] = zone_feature_pack(rt)
    return raw, routes, packs


def main():
    import multiprocessing as mp
    global G_ROUTES, G_PACKS, G_PPM
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_eval", default="200", help="int or 'full'")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--n_probe", type=int, default=1)
    ap.add_argument("--batch", type=int, default=0, help="routes per closure eval; 0=all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hid", type=int, default=0, help="0=linear affinity model, >0=one hidden layer")
    ap.add_argument("--procs", type=int, default=0, help="worker processes; 0=auto from SLURM/cpu_count")
    ap.add_argument("--ppm_order", type=int, default=5)
    args = ap.parse_args()
    n_eval = None if str(args.n_eval).lower() == "full" else int(args.n_eval)
    batch = None if args.batch == 0 else args.batch
    nproc = args.procs or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    t_start = time.time()

    print("=== prepare TRAIN ===", flush=True)
    raw_tr, routes_tr, packs_tr = prepare("train", args.n_train, args.seed)
    print(f"train routes usable = {len(routes_tr)}", flush=True)
    print("=== prepare EVAL ===", flush=True)
    raw_ev, routes_ev, packs_ev = prepare("eval", n_eval, args.seed)
    print(f"eval routes usable = {len(routes_ev)}", flush=True)

    # feature standardisation from TRAIN zones
    allfeat = np.concatenate([packs_tr[r]["feat"] for r in routes_tr], axis=0)
    fmean = allfeat.mean(0); fstd = allfeat.std(0) + 1e-6
    fmean = fmean.astype(np.float32); fstd = fstd.astype(np.float32)

    # ---------------- reference PPM baseline (verbatim trainer + solver) ----------------
    print("=== build reference PPM (verbatim) ===", flush=True)
    import pandas as pd
    rows = []
    for r in routes_tr:
        rt = routes_tr[r]
        # ground-truth zone sequence ordered by ACTUAL stop order (full + collapsed)
        seq_map = rt["actual"]["actual"]               # {stop: position}
        node_zone = {rt["nodes"][i]: rt["zones"][i] for i in range(len(rt["nodes"]))}
        ordered_stops = sorted(seq_map.keys(), key=lambda s: seq_map[s])
        full = [node_zone.get(s, "stz") for s in ordered_stops]
        coll, last = [], None
        for z in full:
            if z != last:
                coll.append(z); last = z
        rows.append({"route_id": r, "zone_seq": "|".join(coll), "full_zone_seq": "|".join(full)})
    zdf = pd.DataFrame(rows)
    ppm = build_ppm_model(zdf, args.ppm_order, gt_strictly_set=True)

    # publish globals so fork()ed workers inherit route data + PPM copy-on-write (no pickling)
    G_ROUTES = {**routes_tr, **routes_ev}
    G_PACKS = {**packs_tr, **packs_ev}
    G_PPM = ppm
    ev_ids = list(routes_ev.keys())
    PFs_tr = {r: pair_features(packs_tr[r], fmean, fstd) for r in routes_tr}
    print(f"=== creating Pool({nproc}) ===", flush=True)
    pool = mp.get_context("fork").Pool(nproc)

    # ---------------- deploy PPM on eval (parallel) ----------------
    print("=== deploy PPM on eval (parallel) ===", flush=True)
    t0 = time.time()
    props_ppm = {r: p for r, p in pool.map(_w_ppm, ev_ids, chunksize=max(1, len(ev_ids) // (nproc * 4))) if p}
    print(f"  PPM deploy done {len(props_ppm)}/{len(ev_ids)} ({time.time()-t0:.0f}s, "
          f"{1000*(time.time()-t0)/max(1,len(ev_ids)):.0f} ms/route eff.)", flush=True)
    res_ppm = official_evaluate(ev_ids, props_ppm, raw_ev, "ppm")
    print(f"PPM submission_score = {res_ppm['submission_score']:.5f}", flush=True)

    # ---------------- PolyStep ----------------
    print("=== train PolyStep ===", flush=True)
    net = ZoneNet(hid=args.hid).to(DEV)
    net, tr_best = train_polystep(net, PFs_tr, packs_tr, list(routes_tr.keys()), pool,
                                  steps=args.steps, n_probe=args.n_probe, seed=args.seed,
                                  hid=args.hid, batch=batch)
    print("=== deploy PolyStep on eval (parallel) ===", flush=True)
    t0 = time.time()
    ps_tasks = [(r, net_score_matrix(net, packs_ev[r], fmean, fstd)) for r in ev_ids]
    props_ps = {r: p for r, s, p in pool.map(_w_learned, ps_tasks, chunksize=max(1, len(ev_ids) // (nproc * 4))) if p}
    print(f"  PolyStep deploy done {len(props_ps)}/{len(ev_ids)} ({time.time()-t0:.0f}s)", flush=True)
    res_ps = official_evaluate(ev_ids, props_ps, raw_ev, "polystep")
    print(f"PolyStep submission_score = {res_ps['submission_score']:.5f}", flush=True)
    pool.close(); pool.join()

    # ---------------- write results ----------------
    payload = {
        "benchmark": "Amazon Last-Mile Routing Research Challenge (ALMRRC 2021)",
        "methodology": "reference solver (zone_based_tsp + OR-tools) and rc-cli scorer used verbatim; "
                       "PolyStep replaces the PPM zone-preference model only.",
        "subset": {"n_train": len(routes_tr), "n_eval": len(routes_ev),
                   "n_train_arg": args.n_train, "n_eval_arg": args.n_eval, "is_full_eval": n_eval is None},
        "polystep": {"steps": args.steps, "n_probe": args.n_probe, "batch": args.batch,
                     "hid": args.hid, "procs": nproc,
                     "params": int(sum(p.numel() for p in net.parameters())),
                     "train_best_score": tr_best},
        "scores": {
            "PPM_baseline": float(res_ppm["submission_score"]),
            "PolyStep": float(res_ps["submission_score"]),
            "reference_paper_PPM_ortools": 0.0372,
        },
        "feasibility": {
            "PPM_valid": int(sum(1 for v in res_ppm["route_feasibility"].values() if v)),
            "PolyStep_valid": int(sum(1 for v in res_ps["route_feasibility"].values() if v)),
            "n_eval": len(routes_ev),
        },
        "wall_clock_sec": time.time() - t_start,
    }
    os.makedirs(os.path.join(ROOT, "exp_results"), exist_ok=True)
    with open(os.path.join(ROOT, "exp_results/amazon_lmrrc.json"), "w") as f:
        json.dump(payload, f, indent=2)
    md = f"""# PolyStep on the Amazon Last-Mile Routing Research Challenge (ALMRRC 2021)

**Methodology (strict reuse).** The decision solver is the AWS reference `zone_based_tsp`
(its PPM-rollout zone ordering + per-zone Google OR-tools TSP) used VERBATIM, and the scorer is the
official rc-cli `score.evaluate` (`submission_score = mean of seq_dev x erp_per_edit`) used VERBATIM.
Train = `almrrc2021-data-training`, score = `almrrc2021-data-evaluation` with the official metric.
The ONLY new component is a **PolyStep-learned zone preference model** that replaces the hand-crafted
PPM: a `LearnedZoneModel` exposing the same `.query(preceding, following)` scalar interface, returning
a zone-transition affinity `-{PRIOR_W}*dist(a,b)/scale + g_theta(features)`. `g_theta` is a small
{'linear model' if args.hid == 0 else 'MLP'} ({payload['polystep']['params']} params) whose weights are
PolyStep's variables; PolyStep minimises the deployed route's REALISED official rc-cli score directly
(no per-stop optimal labels). Route-solves (reference `zone_based_tsp` ~2 s each) are parallelised over
{nproc} CPUs.

**Subset.** This run trains on **{len(routes_tr)}** training routes and scores on
**{len(routes_ev)}** evaluation routes{' (FULL eval split)' if n_eval is None else ' (subset)'}.
Full-scale (all ~6k train / ~3k eval routes) is a pure scale-up of the same pipeline.

## Scores (official rc-cli submission_score, lower is better)

| method | submission_score | valid routes |
|---|---|---|
| Reference PPM + OR-tools (this run, verbatim) | {payload['scores']['PPM_baseline']:.5f} | {payload['feasibility']['PPM_valid']}/{len(routes_ev)} |
| **PolyStep + OR-tools (ours)** | **{payload['scores']['PolyStep']:.5f}** | {payload['feasibility']['PolyStep_valid']}/{len(routes_ev)} |
| Reference paper (PPM+OR-tools, full eval) | ~0.0372 | -- |

PolyStep train-set best score during optimisation: {tr_best:.5f}.
Wall-clock: {payload['wall_clock_sec']:.0f}s. PolyStep: {args.steps} steps, num_probe={args.n_probe}.

_Note: a subset run; the PPM baseline column is the SAME reference pipeline scored on the SAME subset,
so the PPM-vs-PolyStep comparison is apples-to-apples. The ~0.0372 figure is the reference paper's
full-eval number for context._
"""
    with open(os.path.join(ROOT, "exp_results/amazon_lmrrc.md"), "w") as f:
        f.write(md)
    print("wrote exp_results/amazon_lmrrc.{json,md}\nDONE", flush=True)


if __name__ == "__main__":
    main()
