"""
Axis 3 (black-box-only): a predict-then-optimize problem the ENTIRE gradient/surrogate camp cannot
even be formulated on, but gradient-free direct-regret methods train normally.

Setup — KNAPSACK WITH UNCERTAIN RESOURCE CONSUMPTION (predicted parameters live in the CONSTRAINT):
  - item VALUES v are known and fixed; item CONSUMPTIONS (weights) are UNKNOWN and predicted from
    features x; capacity C is known.
  - deploy: greedy-by-predicted-density pack until predicted capacity -> selection z.
  - realize: true consumption w is revealed; realized objective = vᵀz − λ·max(0, wᵀz − C)
    (collect value, pay λ per unit of capacity overrun). This is the true decision objective.

WHY THE GRADIENT/SURROGATE CAMP IS INAPPLICABLE (structural, not "hard"):
  SPO+ / PFYL / IMLE / cvxpylayers all require a PyEPO optModel with a FIXED feasible region and a
  PREDICTED OBJECTIVE COST VECTOR — regret and every surrogate are defined as cᵀw over that region.
  Here the prediction parametrizes the CONSTRAINT, so (a) you cannot even build knapsackModel(weights=…)
  without the unknown weights, and (b) the realized objective is non-linear & non-differentiable in the
  prediction (relu of a constraint violation through an argmax selection). There is no cost vector for
  the surrogate to consume. -> N/A.

WHO CAN RUN: two-stage (MSE on consumption), PolyStep, SFGE — they only evaluate the realized outcome
of a deployed decision (a black box). PolyStep/SFGE optimize that realized objective directly.

Run: .venv/bin/python blackbox_constraint.py [deg] [seeds]
"""
import os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "polystep/src")
from pyepo.data import knapsack
from pto.solvers import knap1_dp
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon

dev = "cuda" if torch.cuda.is_available() else "cpu"
PF = 5; NIT = 16
# --- polytope x probe sweep (env-overridable; defaults preserve original behavior) ---
PS_POLYTOPE = os.environ.get("PS_POLYTOPE", "orthoplex")
PS_PROBES = int(os.environ.get("PS_PROBES", "1"))
OUT_TAG = os.environ.get("OUT_TAG", "")   # no file output here; results print to stdout (captured per-job)


def dp_knap(wp, v, C):
    """Batched EXACT 0/1 knapsack: max value s.t. (rounded) consumption <= capacity C.
    wp (K,N) consumptions (rounded to int>=1 inside), v (N,) values -> z (K,N) binary selection."""
    wi = wp.round().clamp(min=1.0)
    return knap1_dp(v.unsqueeze(0).expand(wp.shape[0], -1), wi, int(C))[1].to(wp.dtype)


def realized(z, v, w_true, C, lam):
    """Realized decision objective (maximize): value collected minus λ·capacity overrun."""
    value = (z * v).sum(-1)
    overflow = (z * w_true).sum(-1) - C
    return value - lam * overflow.clamp(min=0)


def gen(seed, deg):
    """values v = fixed known item weights; per-instance consumption c = uncertain (predicted)."""
    Wfix, _, _ = knapsack.genData(2, PF, NIT, dim=1, deg=1, seed=1)
    v = torch.tensor(Wfix[0], dtype=torch.float32, device=dev)           # known values, fixed
    _, x, c = knapsack.genData(1400, PF, NIT, dim=1, deg=deg, noise_width=0, seed=seed)
    X = torch.tensor(x, dtype=torch.float32, device=dev)
    W = torch.tensor(c, dtype=torch.float32, device=dev)                 # true consumption per instance
    ntr = 400
    C = float(int(0.5 * W[:ntr].sum(-1).mean()))                         # integer capacity ~ half the load
    return (X[:ntr], W[:ntr], X[ntr:], W[ntr:], v, C)


def make():
    return nn.Linear(PF, NIT, bias=True).to(dev)                         # bias lets it predict conservatively


def train_two_stage(Xtr, Wtr, epochs=60):
    m = make(); opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(epochs):
        opt.zero_grad(); ((m(Xtr) - Wtr) ** 2).mean().backward(); opt.step()
    return m


def train_polystep(Xtr, Wtr, v, C, lam, warm, steps=200, seed=0):
    m = make()
    with torch.no_grad():
        m.weight.copy_(warm.weight); m.bias.copy_(warm.bias)
    pso = PolyStepOptimizer(m, polytope_type=PS_POLYTOPE, epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=0.4, probe_radius=0.8, num_probe=PS_PROBES, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    scale = float(v.sum())                                              # normalize objective -> O(1) for OT
    def closure(bp):
        pred = torch.einsum("mnf,bf->mbn", bp["weight"], Xtr) + bp["bias"].unsqueeze(1)   # (M,B,N)
        M, B, N = pred.shape
        z = dp_knap(pred.reshape(M * B, N), v, C).reshape(M, B, N)
        r = realized(z, v, Wtr.unsqueeze(0), C, lam)                     # (M,B) maximize
        return -(r / scale).mean(-1)                                     # minimize negative (normalized)
    for _ in range(steps):
        pso.step(closure)
    return m


def train_sfge(Xtr, Wtr, v, C, lam, warm, epochs=200, n_samples=8, sigma=0.5, lr=1e-2, seed=0):
    m = make()
    with torch.no_grad():
        m.weight.copy_(warm.weight); m.bias.copy_(warm.bias)
    opt = torch.optim.Adam(m.parameters(), lr); g = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(epochs):
        pred = m(Xtr)                                                    # (B,N)
        with torch.no_grad():
            eps = torch.randn(n_samples, *pred.shape, device=dev, generator=g)
            chat = pred.unsqueeze(0) + sigma * eps
            S, B, N = chat.shape
            z = dp_knap(chat.reshape(S * B, N), v, C).reshape(S, B, N)
            r = realized(z, v, Wtr.unsqueeze(0), C, lam)                 # (S,B)
            adv = r - r.mean(0, keepdim=True)
        logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
        surrogate = -(adv * logp).mean()                                # maximize realized -> minimize -adv*logp
        opt.zero_grad(); surrogate.backward(); opt.step()
    return m


def evaluate(m, Xte, Wte, v, C, lam):
    with torch.no_grad():
        z = dp_knap(m(Xte), v, C)
        ach = realized(z, v, Wte, C, lam)
        z_or = dp_knap(Wte, v, C)                                        # oracle: EXACT on TRUE consumption
        opt = realized(z_or, v, Wte, C, lam)                            # feasible -> true optimum, no penalty
        reg = ((opt - ach) / opt.clamp(min=1e-6)).mean().item()
        overrun = (z * Wte).sum(-1).sub(C).clamp(min=0).gt(0).float().mean().item()
    return reg, overrun


def main():
    deg = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    lams = [1.0, 5.0, 20.0]
    print(f"PolyStep cfg: polytope={PS_POLYTOPE} num_probe={PS_PROBES} OUT_TAG={OUT_TAG or '(none)'}")
    print(f"BLACK-BOX-ONLY | knapsack with UNCERTAIN CONSUMPTION (predicted param in the CONSTRAINT)")
    print(f"deg={deg} | {len(seeds)} seeds | normalized realized-regret (lower better)\n")
    print("  SPO+ / PFYL / IMLE / cvxpylayers : N/A  (no optModel without the predicted weights;")
    print("                                          realized objective non-linear & non-diff)\n")
    hdr = f"{'lambda':>7} | {'two-stage':>20} {'SFGE':>20} {'PolyStep':>20} | best"
    print(hdr); print("-" * len(hdr))
    for lam in lams:
        acc = {k: [] for k in ("two-stage", "SFGE", "PolyStep")}
        for seed in seeds:
            Xtr, Wtr, Xte, Wte, v, C = gen(seed, deg)
            ts = train_two_stage(Xtr, Wtr)
            sf = train_sfge(Xtr, Wtr, v, C, lam, ts, seed=seed)
            ps = train_polystep(Xtr, Wtr, v, C, lam, ts, seed=seed)
            acc["two-stage"].append(evaluate(ts, Xte, Wte, v, C, lam)[0])
            acc["SFGE"].append(evaluate(sf, Xte, Wte, v, C, lam)[0])
            acc["PolyStep"].append(evaluate(ps, Xte, Wte, v, C, lam)[0])
        m = {k: (np.mean(v_), np.std(v_)) for k, v_ in acc.items()}
        best = min(m, key=lambda k: m[k][0])
        gain = (m["two-stage"][0] - m[best][0]) / max(m["two-stage"][0], 1e-9) * 100
        row = "  ".join(f"{m[k][0]:>7.4f}+/-{m[k][1]:<6.4f}" for k in ("two-stage", "SFGE", "PolyStep"))
        print(f"{lam:>7} | {row} | {best} ({gain:+.0f}% vs two-stage)", flush=True)


if __name__ == "__main__":
    main()
