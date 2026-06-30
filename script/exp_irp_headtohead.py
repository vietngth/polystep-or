"""DSIRP head-to-head: PolyStep (eval-oracle, label-free) vs Toni et al.'s InferOpt-FY (opt-oracle,
anticipative-expert labels) vs untrained -- ALL evaluated in the IDENTICAL simulator.

WHY THIS IS APPLES-TO-APPLES. Toni's trained policy is a GLM (PINN) whose weights serialize to JSON
as weights["1"]=w_inv(72), weights["2"]=w_pen(72) -- EXACTLY our PINN's parameters. We train with
THEIR faithful Julia pipeline (FenchelYoungLoss over a PerturbedAdditive CPCTSP maximizer, imitating
an ANTICIPATIVE-EXPERT IRP-MILP, DAgger state sampling; run on HiGHS), export those 72+72 weights,
load them into OUR PINN, and evaluate with OUR rollout simulator -- the same oracle/sim/instances
PolyStep uses. The ONLY thing that differs is the TRAINING SIGNAL:
  * Toni  : imitation of an OPTIMIZATION+anticipative oracle (needs labels).            [opt-oracle]
  * PolyStep: minimize realized rollout cost directly, gradient-free (no labels).        [eval-oracle]

FAIR PROTOCOL (out-of-sample). Both methods learn from each instance's demand HISTORY and are scored
on its HELD-OUT demand_eval scenarios:
  * Toni   : trained by their pipeline on the instance (history-derived anticipative scenarios).
  * PolyStep: trained here on BOOTSTRAP scenarios resampled from demand_hist (never sees demand_eval).
  * Both + untrained: evaluated on demand_eval[0..4] with the same horizon and simulator.

Run:  CACHE=/media/anindex/Data/project-cache/ot-or-project \
      .venv/bin/python exp_irp_headtohead.py full
"""
from __future__ import annotations
import os, sys, glob, json, time
import numpy as np
import torch

# reuse the verified scaffold (oracle, simulator, PINN, PolyStep trainer)
import exp_irp_polystep as B
import irp_gpu as G                                    # GPU-batched CPCTSP + rollout for PolyStep

CACHE = os.environ.get("CACHE", "/media/anindex/Data/project-cache/ot-or-project")
REPO = os.path.join(CACHE, "InferOpt_DSIRP")
B.INSTDIR = os.path.join(REPO, "instances")          # instances now live on the Data disk
# look_ahead MUST match the value Toni's models were trained with (it sets the feature/weight dim:
# NB_OBS = look_ahead * |quantiles|). Override consistently so the exported weights load into our PINN.
_LA = int(os.environ.get("IRP_LOOK_AHEAD", str(B.LOOK_AHEAD)))
if _LA != B.LOOK_AHEAD:
    B.LOOK_AHEAD = _LA
    B.NB_OBS = _LA * len(B.QUANTILES)
    print(f"[look_ahead override] LOOK_AHEAD={B.LOOK_AHEAD} NB_OBS={B.NB_OBS}")
PATTERN = os.environ.get("IRP_PATTERN", "normal")    # demand pattern: normal | uniform | bimodal
# ---- polytope-robustness sweep knobs (mirror the cheap experiments; defaults keep the working config) ----
PS_POLYTOPE = os.environ.get("PS_POLYTOPE", "orthoplex")   # orthoplex | simplex | cube
PS_PROBES   = int(os.environ.get("PS_PROBES", "1"))
OUT_TAG     = os.environ.get("OUT_TAG", "")
_OUT_SFX    = f"_{OUT_TAG}" if OUT_TAG else ""
SOLDIR = os.path.join(REPO, f"training/solutions/dagger/{PATTERN}/penalty_200")
EVAL_H = int(os.environ.get("IRP_EVAL_H", "15"))     # held-out demand_eval scenarios are length 15
P = B.P


def toni_weights(instance_name):
    """Load the latest VALID InferOpt-FY trained GLM weights -> (w_inv(72,), w_pen(72,), src, best_it).
    Skips partial/aborted files (best_iteration=None, no epoch key) -- training writes incrementally."""
    cands = sorted(glob.glob(os.path.join(SOLDIR, instance_name, "*_solutions.json")))
    for f in reversed(cands):                                  # newest valid first
        sol = json.load(open(f))["pctsp"]
        bi = sol.get("best_iteration")
        epochs = [k for k in sol if k.isdigit()]
        key = None
        if bi is not None and str(bi) in sol and "weights" in sol[str(bi)]:
            key = str(bi)
        elif epochs and "weights" in sol[max(epochs, key=int)]:
            key = max(epochs, key=int)
        if key is None:
            continue
        w = sol[key]["weights"]
        w_inv = np.asarray(w["1"], dtype=np.float64).flatten()
        w_pen = np.asarray(w["2"], dtype=np.float64).flatten()
        if w_inv.size == B.NB_OBS and w_pen.size == B.NB_OBS:
            return w_inv, w_pen, os.path.basename(f), int(key)
    return None


def predict_from_weights(w_inv, w_pen):
    wi = torch.tensor(w_inv, dtype=torch.float32, device=B.DEV)
    wp = torch.tensor(w_pen, dtype=torch.float32, device=B.DEV)
    return lambda h, p: B.theta_from_params(wi, wp, h, p)


TEST_H = int(os.environ.get("IRP_TEST_H", "30"))      # stable long test-trajectory rollout


def eval_on_heldout(predict_theta, inst, horizon=TEST_H):
    """STABLE held-out metric: one long rollout on the realized demand_test trajectory (H steps).
    demand_test is never used in training (PolyStep trains on bootstrapped history; Toni on
    anticipative history scenarios), so it is genuinely held out for both. Also returns the 5
    demand_eval-scenario costs as a secondary signal."""
    dtest = [inst["demand_test"][c] for c in range(inst["C"])]
    h = min(horizon, min(len(x) for x in dtest))   # clamp to available test-trajectory length (bimodal ships len-15, normal len-90)
    c_test, _ = B.rollout(predict_theta, inst, dtest, horizon=h)
    evals = []
    for s in range(5):
        try:
            evals.append(B.rollout(predict_theta, inst, B.demand_seq(inst, "eval", s), horizon=EVAL_H)[0])
        except Exception:
            break
    return float(c_test), evals


def bootstrap_scenarios(inst, n_scen, horizon, seed):
    """Resample per-customer demand trajectories (with replacement) from demand_hist -> train demand.
    Returned as a list of n_scen demand sequences, each [C][horizon]. Out-of-sample vs demand_eval."""
    rng = np.random.default_rng(seed)
    C = inst["C"]
    scens = []
    for _ in range(n_scen):
        dseq = [rng.choice(inst["demand_hist"][c], size=horizon, replace=True) for c in range(C)]
        scens.append(dseq)
    return scens


def train_polystep_instance(inst, train_scens, steps, seed, horizon):
    """PolyStep on ONE instance, minimizing mean realized cost over its bootstrap TRAIN scenarios.

    GPU-BATCHED (author's guidance): the closure rolls out ALL K orthoplex probes in lock-step through
    the horizon via irp_gpu.batched_rollout -- a batched GPU CPCTSP (exact, == gurobipy) instead of K
    serial gurobipy solves. This is PolyStep's whole point (many cheap batched forward solves) and is
    ~100x faster per probe. Toni's FY training stays on CPU (HiGHS) for reproducibility."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"   # batched rollout runs on GPU (PINN/eval stay on B.DEV/CPU)
    dt = torch.float32
    tour, memb = G.precompute_subsets(inst["dist"], inst["C"], device=dev, dtype=dt)
    inst_t = G.inst_to_tensors(inst, dev, dtype=dt)
    dseqs_t = [torch.tensor(np.stack([s[c][:horizon] for c in range(inst["C"])]), dtype=dt, device=dev)
               for s in train_scens]

    def batched_mean(Wi, Wp):                                   # Wi,Wp: (K, NB_OBS) -> (K,) mean cost
        tot = torch.zeros(Wi.shape[0], device=dev, dtype=dt)
        for d in dseqs_t:
            tot += G.batched_rollout(Wi, Wp, inst_t, d, horizon, B.LOOK_AHEAD, B.QUANTILES, tour, memb)
        return tot / len(dseqs_t)

    from polystep import PolyStepOptimizer
    from polystep.epsilon import CosineEpsilon
    pr = float(os.environ.get("IRP_PR", "0.5"))
    sr = float(os.environ.get("IRP_SR", "0.6"))
    model = B.PINN().to(B.DEV)
    pso = PolyStepOptimizer(model, polytope_type=PS_POLYTOPE, epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=sr, probe_radius=pr, num_probe=PS_PROBES, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)

    def closure(bp):
        Wi = bp["w_inv"].to(device=dev, dtype=dt); Wp = bp["w_pen"].to(device=dev, dtype=dt)
        return batched_mean(Wi, Wp).to(B.DEV)                  # PolyStep consumes the (K,) cost vector

    best = (float("inf"), None)
    for s in range(steps):
        pso.step(closure)
        cur = float(batched_mean(model.w_inv.detach().to(device=dev, dtype=dt)[None],
                                 model.w_pen.detach().to(device=dev, dtype=dt)[None])[0])
        if cur < best[0]:
            best = (cur, {k: v.detach().clone() for k, v in model.state_dict().items()})
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model


# --------------------------------------------------------------------------------------------------
# SFGE TRAINER  (score-function / REINFORCE, output-space, episodic over the rollout horizon)
# --------------------------------------------------------------------------------------------------
# The PINN output theta (CPCTSP prize vector) is treated as a Gaussian policy MEAN: each period we sample
# theta_t ~ N(theta_pred_t, sigma), roll the closed loop, accrue the realized horizon cost, and take a
# REINFORCE step  g = E[(cost - baseline) * sum_t (theta_t - theta_pred_t)/sigma^2 * d theta_pred_t/dw].
# This is the SAME score-function estimator as the constraint experiments (exp4_constraints.train_sfge /
# exp_polystep_vs_sfge.sfge_train_traj), just made EPISODIC over the multi-period rollout. The rollout /
# CPCTSP solver run under no_grad (the GPU-batched EXACT oracle G.cpctsp_batched, identical to PolyStep's
# -- the M batch dim now indexes the n_samples Monte-Carlo episodes); theta_pred_t is grad-tracked so the
# score gradient flows back through the PINN. A per-scenario mean-cost baseline reduces variance; Adam @ lr.
# Defaults from a normal-pattern sigma/lr sweep (steps=40, n_samples=32): sigma=150 is the ONLY robust
# value -- it ties/edges PolyStep (3-inst mean 36.9k vs 39.5k), while sigma in {50,100,250} catastrophically
# diverges on >=1 instance (cost 150k-240k). SFGE works here, but only inside a NARROW sigma band (the
# score-function-variance / sigma-sensitivity story); PolyStep needs no such knife-edge tuning.
SFGE_SIGMA   = float(os.environ.get("IRP_SFGE_SIGMA", "150.0"))   # theta is O(380) here -> sigma on that scale
SFGE_LR      = float(os.environ.get("IRP_SFGE_LR", "0.05"))
SFGE_NSAMPLE = int(os.environ.get("IRP_SFGE_NSAMPLE", "32"))      # Monte-Carlo episodes per scenario/step


def _sfge_batched_rollout(Wi, Wp, inst_t, dseq_t, horizon, look_ahead, quantiles, tour, memb,
                          sigma, n_samples, gen):
    """One episodic SFGE rollout batch: M=n_samples MC episodes advanced in lock-step on a shared
    realized demand `dseq_t`, each with INDEPENDENT per-period Gaussian theta noise. Wi,Wp: (NB_OBS,)
    grad-tracked predictor weights (on dev). Returns (tot:(M,) DETACHED realized cost,
    logp_sum:(M,) GRAD-tracked sum_t log N(theta_t | theta_pred_t, sigma))."""
    dev, dt = Wi.device, Wi.dtype
    M = n_samples
    C = inst_t["holding"].shape[0]
    start_inv = inst_t["start_inv0"].unsqueeze(0).expand(M, C).clone()
    max_inv, holding, penalty = inst_t["max_inv"], inst_t["holding"], inst_t["penalty"]
    v_cap = inst_t["v_cap"]
    hist = [h.clone() for h in inst_t["demand_hist"]]
    Q = torch.tensor(quantiles, device=dev, dtype=dt)
    tot = torch.zeros(M, device=dev, dtype=dt)
    logp_sum = torch.zeros(M, device=dev, dtype=dt)
    WiM = Wi.unsqueeze(0).expand(M, -1)                         # (M,NB) grad-tracked (shared model weights)
    WpM = Wp.unsqueeze(0).expand(M, -1)
    for t in range(horizon):
        if t > 0:
            for c in range(C):
                hist[c] = torch.cat([hist[c], dseq_t[c, t - 1:t]])
        qv = torch.stack([torch.quantile(hist[c], Q) for c in range(C)])            # (C,|Q|) shared
        hold_t, pen_t = G.batched_period_terms(start_inv.detach(), qv, holding, penalty, look_ahead)
        theta_pred = torch.einsum("mcn,mn->mc", hold_t, WiM) + torch.einsum("mcn,mn->mc", pen_t, WpM)
        eps = torch.randn(M, C, device=dev, dtype=dt, generator=gen)
        theta_s = (theta_pred.detach() + sigma * eps)          # sampled prizes = the ACTION (detached)
        logp_sum = logp_sum - ((theta_s - theta_pred) ** 2).sum(1) / (2.0 * sigma ** 2)
        with torch.no_grad():
            deliver = max_inv.unsqueeze(0) - start_inv
            visited, routing = G.cpctsp_batched(theta_s, deliver, float(v_cap), tour, memb)
            q = deliver * visited
            nxt = start_inv + q - dseq_t[:, t].unsqueeze(0)
            short = torch.clamp(-nxt, min=0.0); carry = torch.clamp(nxt, min=0.0)
            tot = tot + (carry * holding[None]).sum(1) + (short * penalty[None]).sum(1) + routing
            start_inv = carry
    return tot, logp_sum


def train_sfge_instance(inst, train_scens, steps, seed, horizon,
                        sigma=None, lr=None, n_samples=None):
    """SFGE on ONE instance, minimizing mean realized rollout cost over its bootstrap TRAIN scenarios --
    mirror of train_polystep_instance, score-function update instead of the OT step. Same instance, same
    bootstrap scenarios, same horizon, same GPU-batched EXACT oracle, same held-out eval => apples-to-apples."""
    sigma = SFGE_SIGMA if sigma is None else sigma
    lr = SFGE_LR if lr is None else lr
    n_samples = SFGE_NSAMPLE if n_samples is None else n_samples
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32
    tour, memb = G.precompute_subsets(inst["dist"], inst["C"], device=dev, dtype=dt)
    inst_t = G.inst_to_tensors(inst, dev, dtype=dt)
    dseqs_t = [torch.tensor(np.stack([s[c][:horizon] for c in range(inst["C"])]), dtype=dt, device=dev)
               for s in train_scens]
    model = B.PINN().to(B.DEV)
    opt = torch.optim.Adam(model.parameters(), lr)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)

    def det_mean(Wi, Wp):                                       # deterministic mean cost (best-tracking)
        tot = torch.zeros(Wi.shape[0], device=dev, dtype=dt)
        for d in dseqs_t:
            tot += G.batched_rollout(Wi, Wp, inst_t, d, horizon, B.LOOK_AHEAD, B.QUANTILES, tour, memb)
        return tot / len(dseqs_t)

    best = (float("inf"), None)
    for s in range(steps):
        opt.zero_grad()
        Wi = model.w_inv.to(device=dev, dtype=dt)               # grad-tracked move cpu-leaf -> dev
        Wp = model.w_pen.to(device=dev, dtype=dt)
        surrogate = torch.zeros((), device=dev, dtype=dt)
        for d in dseqs_t:
            tot, logp = _sfge_batched_rollout(Wi, Wp, inst_t, d, horizon, B.LOOK_AHEAD, B.QUANTILES,
                                              tour, memb, sigma, n_samples, gen)
            adv = (tot - tot.mean()).detach()                  # per-scenario mean-cost baseline
            surrogate = surrogate + (adv * logp).mean()
        (surrogate / len(dseqs_t)).backward()
        opt.step()
        cur = float(det_mean(model.w_inv.detach().to(device=dev, dtype=dt)[None],
                             model.w_pen.detach().to(device=dev, dtype=dt)[None])[0])
        if cur < best[0]:
            best = (cur, {k: v.detach().clone() for k, v in model.state_dict().items()})
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    smoke = mode == "smoke"
    n_inst = 1 if smoke else int(os.environ.get("IRP_N_INST", "5"))
    ps_steps = 10 if smoke else 40
    ps_train_scen = 4 if smoke else 8
    seed = int(os.environ.get("IRP_SEED", "0"))
    P(f"=== DSIRP HEAD-TO-HEAD | {'SMOKE' if smoke else 'FULL'} | n_inst={n_inst} eval_H={EVAL_H} "
      f"ps_steps={ps_steps} train_scen={ps_train_scen} PR={os.environ.get('IRP_PR','3.0')} "
      f"| polytope={PS_POLYTOPE} num_probe={PS_PROBES} seed={seed} tag={OUT_TAG or '(none)'} ===")
    files = sorted(glob.glob(os.path.join(B.INSTDIR, f"{PATTERN}-10_*.json")))[:n_inst]
    insts = [B.load_instance(f, PATTERN) for f in files]
    inames = [os.path.splitext(os.path.basename(f))[0] for f in files]   # filename stem == instance_id
    t0 = time.time()

    rows = []
    for inst, iname in zip(insts, inames):
        # untrained
        unt = B.PINN().to(B.DEV)
        c_unt, _ = eval_on_heldout(lambda h, p: unt.theta(h, p), inst)
        # toni
        tw = toni_weights(iname) if iname else None
        if tw is not None:
            c_toni, _ = eval_on_heldout(predict_from_weights(tw[0], tw[1]), inst)
            toni_src = tw[2]
        else:
            c_toni, toni_src = float("nan"), "MISSING"
        # polystep (trained on bootstrap, evaluated held-out)
        train_scens = bootstrap_scenarios(inst, ps_train_scen, EVAL_H, seed)
        ps_model = train_polystep_instance(inst, train_scens, ps_steps, seed, EVAL_H)
        c_ps, _ = eval_on_heldout(lambda h, p: ps_model.theta(h, p), inst)
        # SFGE (score-function, output-space episodic) -- SAME instance/bootstrap scens/horizon/eval as PolyStep
        sfge_model = train_sfge_instance(inst, train_scens, ps_steps, seed, EVAL_H)
        c_sfge, _ = eval_on_heldout(lambda h, p: sfge_model.theta(h, p), inst)
        rows.append(dict(instance=iname, untrained=c_unt, toni_fy=c_toni, polystep=c_ps, sfge=c_sfge,
                         toni_src=toni_src))
        P(f"  {iname[:40] if iname else inst['name'][:40]}: untrained={c_unt:,.1f}  "
          f"ToniFY={c_toni:,.1f}  PolyStep={c_ps:,.1f}  SFGE={c_sfge:,.1f}")

    # aggregate
    def col(k):
        v = np.array([r[k] for r in rows], float)
        return v[~np.isnan(v)]
    mu = {k: float(np.mean(col(k))) for k in ("untrained", "toni_fy", "polystep", "sfge")}
    P("\n--- HEAD-TO-HEAD (mean held-out rollout cost over instances) ---")
    P(f"  untrained : {mu['untrained']:,.1f}")
    P(f"  Toni-FY   : {mu['toni_fy']:,.1f}   ({100*(mu['untrained']-mu['toni_fy'])/mu['untrained']:+.1f}% vs untrained)")
    P(f"  PolyStep  : {mu['polystep']:,.1f}   ({100*(mu['untrained']-mu['polystep'])/mu['untrained']:+.1f}% vs untrained)")
    P(f"  SFGE      : {mu['sfge']:,.1f}   ({100*(mu['untrained']-mu['sfge'])/mu['untrained']:+.1f}% vs untrained)")
    if not np.isnan(mu["toni_fy"]):
        P(f"  PolyStep vs Toni-FY: {100*(mu['toni_fy']-mu['polystep'])/mu['toni_fy']:+.1f}% "
          f"({'PolyStep cheaper' if mu['polystep']<mu['toni_fy'] else 'Toni cheaper'})")
    P(f"  PolyStep vs SFGE: {100*(mu['sfge']-mu['polystep'])/mu['sfge']:+.1f}% "
      f"({'PolyStep cheaper' if mu['polystep']<mu['sfge'] else 'SFGE cheaper'})")

    out = dict(mode=mode, n_inst=n_inst, eval_horizon=EVAL_H, ps_steps=ps_steps,
               ps_train_scen=ps_train_scen, probe_radius=float(os.environ.get("IRP_PR", "3.0")),
               polytope=PS_POLYTOPE, num_probe=PS_PROBES, seed=seed, pattern=PATTERN,
               sfge_sigma=SFGE_SIGMA, sfge_lr=SFGE_LR, sfge_nsample=SFGE_NSAMPLE,
               rows=rows, means=mu, wall_s=time.time() - t0)
    os.makedirs("exp_results", exist_ok=True)
    json.dump(out, open(f"exp_results/irp_headtohead_{mode}_{PATTERN}{_OUT_SFX}.json", "w"), indent=1)
    P(f"[total {time.time()-t0:.0f}s] -> exp_results/irp_headtohead_{mode}_{PATTERN}{_OUT_SFX}.json")


if __name__ == "__main__":
    main()
