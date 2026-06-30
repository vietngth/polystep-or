"""Experiment 2b -- Solve-budget Pareto on more COMMON OR problems.

Extends the Pareto frontier (regret vs cost) beyond shortest-path/knapsack/portfolio to staple
predict-then-optimize problems, each with a fast BATCHED GPU forward solver (for SFGE/PolyStep) and an
EXACT per-instance solver (for SPO+'s subgradient, the regret oracle, and the Gurobi/exact-call cost axis):

  assignment      : n x n linear assignment (scheduling, worker-task). exact = Hungarian (scipy);
                    batched = entropic-OT Sinkhorn + greedy round.
  transportation  : min-cost flow with fixed supplies/demands (logistics). exact = Gurobi LP;
                    batched = Sinkhorn (its native solver).
  mdkp            : multi-dimensional 0/1 knapsack, predicted OBJECTIVE values (resource allocation).
                    exact = Gurobi ILP; batched = greedy (pto.solvers.mdkp_greedy).

Methods: two-stage (MSE), SPO+ (manual subgradient via the exact solver), SFGE, PolyStep. Three honest
cost axes: exact-solver (Hungarian/Gurobi) calls, wall-clock, batched-GPU forward solves. The message:
gradient-free pays ZERO exact-solver calls on every one of these common problems.

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp2b_pareto_common.py [check|run] [problems] [seeds]
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, "polystep/src")
import numpy as np
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linear_sum_assignment
from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from pto.solvers import mdkp_greedy
from pto.budget import SolveCounter, Timer
from pto.seeding import seed_everything, device_generator
from pto.multiseed import summarize, md_table, write_json, write_md

dev = "cuda" if torch.cuda.is_available() else "cpu"
PF = 5


def genpoly(n, dim, deg, seed):
    """Elmachtoub-Grigas style features -> positive cost vector (n,dim) with degree-deg misspecification."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, PF).astype(np.float32)
    B = rng.rand(dim, PF).astype(np.float32)
    base = (X @ B.T) / np.sqrt(PF) + 3.0
    C = (np.sign(base) * np.abs(base) ** deg) / (3.5 ** deg) + 1.0
    C *= rng.uniform(0.5, 1.5, size=C.shape).astype(np.float32)
    return X, C.astype(np.float32)


# ---------------- batched GPU forward solvers ----------------
def sinkhorn(cost, a, b, eps=0.05, iters=80):
    """Batched entropic-OT plan. cost (M,r,c) (lower=better to transport), a (r,), b (c,). -> T (M,r,c)."""
    K = (-cost / eps)
    f = torch.zeros(cost.shape[0], cost.shape[1], device=cost.device)
    la, lb = torch.log(a + 1e-30), torch.log(b + 1e-30)
    g = torch.zeros(cost.shape[0], cost.shape[2], device=cost.device)
    for _ in range(iters):
        f = la - torch.logsumexp(K + g.unsqueeze(1), dim=2)
        g = lb - torch.logsumexp(K + f.unsqueeze(2), dim=1)
    return torch.exp(f.unsqueeze(2) + K + g.unsqueeze(1))


def assign_forward(pred, n):
    """pred (M,n*n) predicted cost matrix -> hard permutation (M,n*n) via Sinkhorn + greedy round."""
    M = pred.shape[0]
    C = pred.reshape(M, n, n)
    a = torch.full((n,), 1.0 / n, device=dev); b = a
    T = sinkhorn(C, a, b)                                       # soft doubly-stochastic-ish
    perm = torch.zeros(M, n, n, device=dev)
    taken = torch.zeros(M, n, dtype=torch.bool, device=dev)
    score = T.clone()
    for i in range(n):
        score[:, i, :] = torch.where(taken, torch.full_like(score[:, i, :], -1e9), score[:, i, :])
        j = score[:, i, :].argmax(-1)
        perm[torch.arange(M), i, j] = 1.0
        taken[torch.arange(M), j] = True
    return perm.reshape(M, n * n)


def transport_forward(pred, r, c, a, b):
    M = pred.shape[0]
    T = sinkhorn(pred.reshape(M, r, c), a, b)
    return T.reshape(M, r * c)


# ---------------- exact (per-instance) solvers ----------------
def assign_exact(c_np, n):
    ri, ci = linear_sum_assignment(c_np.reshape(n, n))
    w = np.zeros((n, n)); w[ri, ci] = 1.0
    return w.reshape(-1)


def transport_exact(c_np, r, c, a_np, b_np):
    md = gp.Model(); md.Params.OutputFlag = 0
    x = md.addVars(r, c, lb=0.0)
    md.setObjective(gp.quicksum(float(c_np.reshape(r, c)[i, j]) * x[i, j] for i in range(r) for j in range(c)), GRB.MINIMIZE)
    for i in range(r): md.addConstr(gp.quicksum(x[i, j] for j in range(c)) == float(a_np[i]))
    for j in range(c): md.addConstr(gp.quicksum(x[i, j] for i in range(r)) == float(b_np[j]))
    md.optimize()
    return np.array([[x[i, j].X for j in range(c)] for i in range(r)]).reshape(-1)


def mdkp_exact(v_np, W, cap):
    n = len(v_np); m = len(cap)
    md = gp.Model(); md.Params.OutputFlag = 0
    z = md.addVars(n, vtype=GRB.BINARY)
    md.setObjective(gp.quicksum(float(v_np[i]) * z[i] for i in range(n)), GRB.MAXIMIZE)
    for j in range(m): md.addConstr(gp.quicksum(float(W[j, i]) * z[i] for i in range(n)) <= float(cap[j]))
    md.optimize()
    return np.array([z[i].X for i in range(n)])


# ---------------- problem registry ----------------
def make_problem(name, seed, deg=4, n_train=200, n_test=200):
    if name == "assignment":
        n = 6; dim = n * n
        X, C = genpoly(n_train + n_test, dim, deg, seed)
        fwd = lambda pred: assign_forward(pred, n)
        ex = lambda c: assign_exact(c, n)
        sense = "min"
    elif name == "transportation":
        r = c = 5; dim = r * c
        rng = np.random.RandomState(seed + 7)
        a = np.ones(r, dtype=np.float32); b = np.ones(c, dtype=np.float32)             # uniform marginals
        at, bt = torch.tensor(a, device=dev), torch.tensor(b, device=dev)
        X, C = genpoly(n_train + n_test, dim, deg, seed)
        fwd = lambda pred: transport_forward(pred, r, c, at, bt)
        ex = lambda cc: transport_exact(cc, r, c, a, b)
        sense = "min"
    elif name == "mdkp":
        n = 16; m = 3; dim = n
        rng = np.random.RandomState(seed + 3)
        W = rng.randint(1, 6, size=(m, n)).astype(np.float32)
        cap = (W.sum(1) * 0.5).astype(np.float32)
        Wt = torch.tensor(W, device=dev)
        X, V = genpoly(n_train + n_test, dim, deg, seed)                                 # predict item VALUES
        C = V
        capb = torch.tensor(cap, device=dev)
        fwd = lambda pred: mdkp_greedy(pred.clamp(min=1e-3), Wt.unsqueeze(0).expand(pred.shape[0], -1, -1),
                                       capb.unsqueeze(0).expand(pred.shape[0], -1)).float()
        ex = lambda v: mdkp_exact(v, W, cap)
        sense = "max"
    else:
        raise ValueError(name)
    Xtr, Ctr = X[:n_train], C[:n_train]; Xte, Cte = X[n_train:], C[n_train:]
    return dict(name=name, dim=dim, sense=sense, fwd=fwd, exact=ex,
                Xtr=torch.tensor(Xtr, device=dev), Ctr=torch.tensor(Ctr, device=dev),
                Xte=torch.tensor(Xte, device=dev), Cte=torch.tensor(Cte, device=dev),
                Ctr_np=Ctr, Cte_np=Cte)


def check(name):
    """Verify the batched forward solver tracks the exact solver (objective gap on random costs)."""
    p = make_problem(name, 0)
    C = p["Cte"][:50]; sgn = 1.0 if p["sense"] == "min" else -1.0
    wf = p["fwd"](C)
    obj_fwd = (wf * C).sum(-1)
    obj_ex = torch.tensor([float((torch.tensor(p["exact"](C[i].cpu().numpy()), device=dev) * C[i]).sum())
                           for i in range(C.shape[0])], device=dev)
    gap = (sgn * (obj_fwd - obj_ex) / obj_ex.abs().clamp(min=1e-6)).mean().item()
    print(f"  [{name}] batched-vs-exact mean objective gap = {gap:+.3%} (>=0 means batched is suboptimal)")


# ---------------- trainers ----------------
def make_model(dim):
    return nn.Linear(PF, dim, bias=False).to(dev)


def regret_eval(model, p):
    sgn = 1.0 if p["sense"] == "min" else -1.0
    with torch.no_grad():
        w = p["fwd"](model(p["Xte"]))
        realized = sgn * (w * p["Cte"]).sum(-1)
        opt = torch.tensor([sgn * float((torch.tensor(p["exact"](p["Cte_np"][i]), device=dev) * p["Cte"][i]).sum())
                            for i in range(p["Cte"].shape[0])], device=dev)
        return ((realized - opt) / opt.abs().clamp(min=1e-6)).mean().item()


def train_two_stage(p, epochs=40):
    m = make_model(p["dim"]); opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(epochs):
        opt.zero_grad(); ((m(p["Xtr"]) - p["Ctr"]) ** 2).mean().backward(); opt.step()
    return m


def train_spoplus(p, warm, epochs, counter, lr=1e-2):
    sgn = 1.0 if p["sense"] == "min" else -1.0
    m = make_model(p["dim"])
    with torch.no_grad(): m.weight.copy_(warm.weight)
    opt = torch.optim.Adam(m.parameters(), lr)
    Cnp = p["Ctr_np"]; n = Cnp.shape[0]
    wstar = np.stack([p["exact"](Cnp[i] if sgn > 0 else Cnp[i]) for i in range(n)])      # w*(c)
    wstar = torch.tensor(wstar, dtype=torch.float32, device=dev)
    counter.calls += n; counter.instances += n
    C = p["Ctr"]
    for _ in range(epochs):
        chat = m(p["Xtr"])
        spo_arg = (2 * chat - C).detach().cpu().numpy()
        wspo = np.stack([p["exact"]((spo_arg[i] if sgn > 0 else -spo_arg[i]))
                         for i in range(n)])                                              # w*(2chat-c)
        counter.calls += n; counter.instances += n
        wspo = torch.tensor(wspo, dtype=torch.float32, device=dev)
        grad_dir = 2.0 * (wstar - wspo)                                                  # SPO+ subgradient wrt chat
        opt.zero_grad(); (chat * grad_dir.detach()).sum(-1).mean().backward(); opt.step()
    return m


def train_sfge(p, warm, epochs, counter, n_samples=8, sigma=0.5, lr=1e-2, seed=0):
    sgn = 1.0 if p["sense"] == "min" else -1.0
    m = make_model(p["dim"])
    with torch.no_grad(): m.weight.copy_(warm.weight)
    opt = torch.optim.Adam(m.parameters(), lr); g = device_generator(seed, dev); C = p["Ctr"]
    for _ in range(epochs):
        pred = m(p["Xtr"])
        with torch.no_grad():
            eps = torch.randn(n_samples, *pred.shape, device=dev, generator=g)
            chat = pred.unsqueeze(0) + sigma * eps; S, B, D = chat.shape
            w = p["fwd"](chat.reshape(S * B, D)); counter.calls += 1; counter.instances += S * B
            r = sgn * (w.reshape(S, B, D) * C.unsqueeze(0)).sum(-1)
            adv = r - r.mean(0, keepdim=True)
        logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
        (adv * logp).mean().backward(); opt.step(); opt.zero_grad()
    return m


def train_polystep(p, warm, steps, counter, seed=0):
    sgn = 1.0 if p["sense"] == "min" else -1.0
    m = make_model(p["dim"])
    with torch.no_grad(): m.weight.copy_(warm.weight)
    pso = PolyStepOptimizer(m, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                            step_radius=0.4, probe_radius=0.8, num_probe=1, seed=seed,
                            use_momentum=True, momentum_init=0.5, momentum_final=0.9)
    X, C = p["Xtr"], p["Ctr"]
    def closure(bp):
        pred = torch.einsum("nef,bf->nbe", bp["weight"], X); N, B, D = pred.shape
        w = p["fwd"](pred.reshape(N * B, D)); counter.calls += 1; counter.instances += N * B
        return sgn * (w.reshape(N, B, D) * C.unsqueeze(0)).sum(-1).mean(-1)
    for _ in range(steps): pso.step(closure)
    return m


PS_STEPS = [10, 25, 50, 100]
SFGE_EPOCHS = [15, 30, 60, 120]
SPO_EPOCHS = [5, 15, 30, 60]


def pareto_point(p, warm, method, budget, seed):
    sc = SolveCounter(lambda x: x); sc.calls = 0; sc.instances = 0
    with Timer() as t:
        if method == "SPO+":
            m = train_spoplus(p, warm, budget, sc)
            gcalls = sc.instances; fwd = 0
        elif method == "SFGE":
            m = train_sfge(p, warm, budget, sc, seed=seed); gcalls = 0; fwd = sc.instances
        else:
            m = train_polystep(p, warm, budget, sc, seed=seed); gcalls = 0; fwd = sc.instances
    return {"budget": budget, "regret": regret_eval(m, p), "wall_clock_s": t.seconds,
            "exact_solver_calls": gcalls, "batched_fwd_solves": fwd}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    problems = sys.argv[2].split(",") if len(sys.argv) > 2 else ["assignment", "transportation", "mdkp"]
    if mode == "check":
        print("Solver fidelity check (batched forward vs exact):")
        for nm in problems: check(nm)
        return
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2]
    print(f"PARETO (common problems) | problems={problems} seeds={seeds}", flush=True)
    results = {}
    for nm in problems:
        print(f"[pareto] {nm}", flush=True)
        per = {"SPO+": [], "SFGE": [], "PolyStep": []}
        for s in seeds:
            seed_everything(s)
            p = make_problem(nm, s)
            warm = train_two_stage(p)
            for b in SPO_EPOCHS: per["SPO+"].append((s, pareto_point(p, warm, "SPO+", b, s)))
            for b in SFGE_EPOCHS: per["SFGE"].append((s, pareto_point(p, warm, "SFGE", b, s)))
            for b in PS_STEPS: per["PolyStep"].append((s, pareto_point(p, warm, "PolyStep", b, s)))
            print(f"    seed {s} done", flush=True)
        agg = {}
        for mth, pts in per.items():
            byb = {}
            for _, pt in pts: byb.setdefault(pt["budget"], []).append(pt)
            agg[mth] = [{"budget": b,
                         **{k: summarize([q[k] for q in qs]) for k in
                            ("regret", "wall_clock_s", "exact_solver_calls", "batched_fwd_solves")}}
                        for b, qs in sorted(byb.items())]
        results[nm] = agg
    write_json("exp_results/pareto_common.json", {"problems": problems, "seeds": seeds, "results": results})
    L = ["# Experiment 2b -- Solve-budget Pareto on common OR problems", "",
         f"seeds={seeds}. Three honest cost axes; gradient-free makes ZERO exact-solver (Hungarian/Gurobi) "
         "calls on every problem.", ""]
    for nm in problems:
        L.append(f"## {nm}")
        rows = []
        for mth in ("SPO+", "SFGE", "PolyStep"):
            for rec in results[nm][mth]:
                rows.append([mth, rec["budget"], f"{rec['regret']['mean']:.4f}",
                             f"{rec['wall_clock_s']['mean']:.2f}", f"{rec['exact_solver_calls']['mean']:.0f}",
                             f"{rec['batched_fwd_solves']['mean']:.0f}"])
        L.append(md_table(["method", "budget", "regret", "wall_clock_s", "exact_calls", "batched_fwd"], rows))
        L.append("")
    write_md("exp_results/pareto_common.md", "\n".join(L))
    print("\nwrote exp_results/pareto_common.{json,md}\nDONE", flush=True)


if __name__ == "__main__":
    main()
