"""The five classical benchmarks of Mandi et al. (2024) as polystep_or problems.

Data, splits, predictors and solvers are the authors' (third_party/predopt-benchmarks); the
energy problem also offers an exact HiGHS formulation of the authors' Gurobi model.
"""

import contextlib
import multiprocessing
import os
import sys
import time
import types

import numpy as np
import torch

from experiments.common import enter, third_party
from polystep_or import Problem

SPLITS = (("train", "Train"), ("val", "Validation"), ("test", "Test"))
ENERGY_SOLVER = None


def authors_dir(folder):
    """Enter one problem folder of the authors' repository and return it."""
    path = third_party("predopt-benchmarks") / folder
    enter(path)
    return path


def read_csv_split(path, prefix, suffix):
    """Read the authors' feature and target CSV files of the three splits."""
    import pandas as pd

    data = {}
    for split, stem in SPLITS:
        x = pd.read_csv(path / f"{prefix}/{stem}dataX_{suffix}.csv", header=None)
        y = pd.read_csv(path / f"{prefix}/{stem}datay_{suffix}.csv", header=None)
        data[split] = (torch.from_numpy(x.T.values.astype(np.float32)),
                       torch.from_numpy(y.T.values.astype(np.float32)))
    return data


def stub_removed_ortools_module():
    """Provide ortools.graph.pywrapgraph, removed from OR-Tools and imported but unused here."""
    import ortools.graph as graph

    if not hasattr(graph, "pywrapgraph"):
        graph.pywrapgraph = types.SimpleNamespace(LinearSumAssignment=None)
        sys.modules["ortools.graph.pywrapgraph"] = graph.pywrapgraph


def shortest_path_rows(rows):
    """Solve shortest-path rows with the authors' OR-Tools solver."""
    from Trainer.optimizer_module import spsolver

    return np.stack([spsolver.shortest_pathsolution(r) for r in rows])


def matching_rows(job):
    """Solve matching rows with the authors' diverse bipartite matching solver."""
    rows, masks, p, q = job
    stub_removed_ortools_module()
    from Trainer.bipartite import bmatching_diverse

    solver = bmatching_diverse(p=p, q=q)
    return np.stack([solver.solve(rows[i], masks[i]) for i in range(len(rows))])


def energy_worker_init(param, attempt=0):
    """Build one HiGHS energy solver per worker process."""
    global ENERGY_SOLVER
    os.environ["OMP_NUM_THREADS"] = "1"
    ENERGY_SOLVER = HighsIcon(param, attempt=attempt)


def energy_rows(rows):
    """Solve energy rows with the worker's HiGHS solver."""
    return [ENERGY_SOLVER.solve(r) for r in rows]


class HighsIcon:
    """Exact HiGHS formulation of the authors' ICON scheduling model (SolveICON).

    The constraint matrix, bounds and integrality are read from the authors' Gurobi model,
    which is built but never optimized, and the objective is their price expression.
    """

    def __init__(self, param, attempt=0):
        from Trainer.comb_solver import SolveICON

        self.ref = SolveICON(relax=False, **param)
        self.ref.make_model()
        model = self.ref.model
        model.update()
        cons, variables = model.getConstrs(), model.getVars()
        self.integrality = np.array([0 if t == "C" else 1
                                     for t in model.getAttr("VType", variables)])
        self.matrix = model.getA().tocsr()
        rhs = np.array(model.getAttr("RHS", cons), dtype=np.float64)
        sense = np.array(model.getAttr("Sense", cons))
        self.low = np.where(sense == "<", -np.inf, rhs)
        self.high = np.where(sense == ">", np.inf, rhs)
        self.lb = np.array(model.getAttr("LB", variables), dtype=np.float64)
        self.ub = np.array(model.getAttr("UB", variables), dtype=np.float64)
        self.index = [tuple(map(int, v.VarName[2:-1].split(","))) for v in variables]
        self.periods = 1440 // self.ref.q
        self.columns = np.arange(len(self.lb), dtype=np.int32)
        self.highs = self._build(int(attempt))

    def _build(self, attempt):
        import highspy

        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.setOptionValue("threads", 1)
        if attempt:
            highs.setOptionValue("random_seed", attempt)
        if attempt >= 3:
            highs.setOptionValue("mip_heuristic_effort", 0.0)
        n, inf = len(self.lb), highspy.kHighsInf
        highs.addVars(n, np.where(np.isinf(self.lb), -inf, self.lb),
                      np.where(np.isinf(self.ub), inf, self.ub))
        kinds = [highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous
                 for i in self.integrality]
        highs.changeColsIntegrality(n, np.arange(n, dtype=np.int32), kinds)
        a = self.matrix
        highs.addRows(a.shape[0], np.where(np.isinf(self.low), -inf, self.low),
                      np.where(np.isinf(self.high), inf, self.high), a.nnz,
                      a.indptr[:-1].astype(np.int32), a.indices.astype(np.int32),
                      a.data.astype(np.float64))
        return highs

    def solve(self, price):
        """Return the per-period consumption of the optimal schedule under price."""
        import highspy

        durations, power, q = self.ref.D, self.ref.P, self.ref.q
        price = np.asarray(price, dtype=np.float64).reshape(-1)
        cost = np.zeros(len(self.index))
        for i, (task, _, start) in enumerate(self.index):
            if start <= self.periods - durations[task]:
                cost[i] = price[start:start + durations[task]].sum() * power[task] * q / 60
        out = np.zeros(self.periods)
        self.highs.changeColsCost(len(cost), self.columns, cost)
        self.highs.run()
        if self.highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return out
        x = np.asarray(self.highs.getSolution().col_value)
        x = np.round(x) if self.integrality.all() else x
        on = np.zeros((self.ref.nbTasks, self.ref.nbMachines, self.periods))
        for i, (task, machine, start) in enumerate(self.index):
            on[task, machine, start] = x[i]
        for t in range(self.periods):
            out[t] = sum(np.sum(on[task, :, max(0, t - durations[task] + 1):t + 1]) * power[task]
                         for task in range(self.ref.nbTasks))
        return out * q / 60


class ClassicalProblem(Problem):
    """A benchmark with a linear objective; cost is the signed objective value."""

    sign = 1.0

    def __init__(self, workers=1):
        self.workers = int(workers)
        self.pool = None
        self.data = {}

    def load_data(self, split):
        """Return the authors' features and true parameters of a split."""
        return self.data[split]

    def cost(self, decisions, parameters):
        """Return sign times the objective value of each decision."""
        return self.sign * (decisions * parameters).sum(-1)

    def optimal(self, parameters, context=None):
        """Return the optimal decisions under the true parameters."""
        return self.solve(parameters, context)

    def regret(self, decisions, parameters, context=None):
        """Return the authors' relative regret of each decision."""
        optimal = self.optimal(parameters, context)
        gap = self.sign * ((decisions - optimal) * parameters).sum(-1)
        return gap / (optimal * parameters).sum(-1)

    def fork_pool(self, **kwargs):
        """Return a pool of forked solver processes, created once."""
        if self.pool is None:
            self.pool = multiprocessing.get_context("fork").Pool(self.workers, **kwargs)
        return self.pool


class ShortestPath(ClassicalProblem):
    """Shortest path on a 5x5 grid: fixed CSV splits, OR-Tools solver, minimized."""

    def __init__(self, degree, seed, workers=8):
        super().__init__(workers)
        self.name = f"shortestpath_{degree}"
        path = authors_dir("ShortestPath")
        self.data = read_csv_split(path, "SyntheticData", f"N_1000_noise_0.5_deg_{degree}")

    def predictor(self):
        """The authors' linear predictor."""
        return torch.nn.Linear(5, 40)

    def solve(self, parameters, context=None):
        """Solve one shortest-path problem per row."""
        rows = parameters.detach().numpy()
        chunks = 4 * self.workers
        if self.workers > 1 and len(rows) >= chunks:
            parts = [c for c in np.array_split(rows, chunks) if len(c)]
            out = np.concatenate(self.fork_pool().map(shortest_path_rows, parts))
        else:
            out = shortest_path_rows(rows)
        return torch.from_numpy(out).float()


class Knapsack(ClassicalProblem):
    """Energy-price knapsack: the authors' per-seed splits and solver, maximized."""

    sign = -1.0

    def __init__(self, capacity, seed, workers=1):
        super().__init__(workers)
        self.name = f"knapsack_{capacity}"
        authors_dir("Knapsack")
        from Trainer.comb_solver import knapsack_solver
        from Trainer.data_utils import KnapsackDataModule

        module = KnapsackDataModule(capacity=int(capacity), seed=int(seed), num_workers=0)
        self.solver = knapsack_solver(module.weights, capacity=int(capacity),
                                      n_items=module.n_items)
        for split, frame in (("train", module.train_df), ("val", module.valid_df),
                             ("test", module.test_df)):
            self.data[split] = (torch.from_numpy(np.asarray(frame.X, dtype=np.float32)),
                                torch.from_numpy(np.asarray(frame.y, dtype=np.float32)))

    def predictor(self):
        """The authors' item-wise linear predictor."""
        return torch.nn.Linear(8, 1)

    def solve(self, parameters, context=None):
        """Solve one knapsack per row."""
        rows = [row.detach().numpy().astype(np.float64) for row in parameters]
        return torch.from_numpy(np.stack([self.solver.solve(r) for r in rows])).float()


class Energy(ClassicalProblem):
    """ICON energy-cost-aware scheduling: the authors' data pipeline and model, minimized.

    The split construction follows the authors' EnergyDataModule, including the optimal
    schedules of every split, computed when the data are built.
    """

    def __init__(self, load, seed, workers=1, solver="highs", hang_s=300.0, attempts=6):
        super().__init__(workers)
        self.name = f"energy_{load}"
        self.hang_s, self.attempts = float(hang_s), int(attempts)
        path = authors_dir("Energy")
        from Trainer.comb_solver import SolveICON, data_reading

        self.param = data_reading(str(path / f"SchedulingInstances/load{load}/day01.txt"))
        self.solver = None
        if solver == "gurobi":
            self.solver = SolveICON(relax=False, **self.param)
            self.solver.make_model()
        elif self.workers <= 1:
            self.solver = HighsIcon(self.param)
        self.optimal_by_split = {}
        self._build(int(seed))

    def _build(self, seed):
        import sklearn.utils
        from sklearn.preprocessing import StandardScaler
        from Trainer.get_energy import get_energy

        x_train, y_train, x_test, y_test = get_energy(fname="Trainer/prices2013.dat")
        x_train, x_test = x_train[:, 1:], x_test[:, 1:]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).reshape(-1, 48, x_train.shape[1])
        x_test = scaler.transform(x_test).reshape(-1, 48, x_test.shape[1])
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train.reshape(-1, 48), y_test.reshape(-1, 48)), axis=0)
        x, y = sklearn.utils.shuffle(x, y, random_state=seed)
        parts = {"train": slice(0, 550), "val": slice(550, 650), "test": slice(650, None)}
        for split, part in parts.items():
            features = torch.from_numpy(np.asarray(x[part], dtype=np.float32))
            parameters = torch.from_numpy(np.asarray(y[part], dtype=np.float32))
            self.data[split] = (features, parameters)
            self.optimal_by_split[id(parameters)] = self.solve(parameters)

    def predictor(self):
        """The authors' period-wise linear predictor."""
        return torch.nn.Linear(8, 1)

    def optimal(self, parameters, context=None):
        """Return the stored optimal schedules of a split, or solve for new parameters."""
        stored = self.optimal_by_split.get(id(parameters))
        return self.solve(parameters) if stored is None else stored

    def solve(self, parameters, context=None):
        """Solve one scheduling problem per row of prices."""
        rows = [row.detach().numpy().astype(np.float64) for row in parameters]
        if self.solver is not None:
            out = [self.solver.solve(r) for r in rows]
        else:
            out = self._pool_solve(rows)
        return torch.from_numpy(np.stack(out).astype(np.float32))

    def _pool_solve(self, rows):
        size = max(1, len(rows) // (2 * self.workers))
        pending = {i: rows[i:i + size] for i in range(0, len(rows), size)}
        done, attempt = {}, 0
        while pending:
            pool = self.fork_pool(initializer=energy_worker_init, initargs=(self.param, attempt))
            jobs = {i: pool.apply_async(energy_rows, (chunk,)) for i, chunk in pending.items()}
            deadline = time.time() + self.hang_s * (attempt + 1)
            for i, job in jobs.items():
                with contextlib.suppress(multiprocessing.TimeoutError):
                    done[i] = job.get(timeout=max(0.0, deadline - time.time()))
            pending = {i: chunk for i, chunk in pending.items() if i not in done}
            if pending or attempt:
                self.pool.terminate()
                self.pool = None
            if pending:
                attempt += 1
                if attempt > self.attempts:
                    raise RuntimeError(f"energy solves still running after {attempt} attempts")
                pending = {i + t: [r] for i, chunk in pending.items() for t, r in enumerate(chunk)}
        return [r for i in sorted(done) for r in done[i]]


class Portfolio(ClassicalProblem):
    """Markowitz portfolio with a risk constraint: the authors' Gurobi QP, maximized.

    The paper reports the absolute regret for this problem.
    """

    sign = -1.0

    def __init__(self, degree, seed, workers=1):
        super().__init__(workers)
        self.name = f"portfolio_{degree}"
        path = authors_dir("Portfolio")
        from Trainer.optimizer_module import gurobi_portfolio_solver

        suffix = f"N_1000_noise_1_deg_{degree}"
        risk = np.load(path / f"SyntheticPortfolioData/GammaSigma_{suffix}.npz")
        self.solver = gurobi_portfolio_solver(cov=risk["sigma"], gamma=risk["gamma"])
        self.data = read_csv_split(path, "SyntheticPortfolioData", suffix)

    def predictor(self):
        """The authors' linear predictor."""
        return torch.nn.Linear(5, 50)

    def regret(self, decisions, parameters, context=None):
        """Return the absolute regret of each decision."""
        optimal = self.optimal(parameters, context)
        return (self.sign * (decisions - optimal) * parameters).sum(-1)

    def solve(self, parameters, context=None):
        """Solve one portfolio problem per row."""
        rows = [row.detach().numpy().astype(np.float64) for row in parameters]
        return torch.from_numpy(np.stack([self.solver.solve(r) for r in rows])).float()


class Matching(ClassicalProblem):
    """Diverse bipartite matching on CORA: 17/5/5 instances, the authors' solver, maximized.

    The same-field matrix of each instance is solver context.
    """

    sign = -1.0
    DIVERSITY = {1: 0.1, 2: 0.25, 3: 0.5}

    def __init__(self, instance, seed, workers=8):
        super().__init__(workers)
        self.name = f"matching_{instance}"
        self.p = self.q = self.DIVERSITY[int(instance)]
        authors_dir("Matching")
        stub_removed_ortools_module()
        from Trainer.bipartite import get_cora
        from Trainer.NNModels import cora_net

        self.network = cora_net
        x, y, m = (torch.from_numpy(np.asarray(a, dtype=np.float32)) for a in get_cora())
        self.masks = {}
        for split, part in (("train", slice(0, 17)), ("val", slice(17, 22)),
                            ("test", slice(22, 27))):
            self.data[split] = (x[part], y[part])
            self.masks[split] = m[part]

    def predictor(self):
        """The authors' default network."""
        return self.network(n_layers=2)

    def context(self, split):
        """Return the same-field matrices of a split."""
        return self.masks[split]

    def solve(self, parameters, context=None):
        """Solve one matching per row, with its same-field matrix."""
        if context is None:
            raise ValueError("matching needs the same-field matrix of every instance")
        rows = parameters.detach().numpy().astype(np.float64)
        masks = context.detach().numpy()
        if self.workers > 1 and len(rows) >= 2 * self.workers:
            bounds = np.array_split(np.arange(len(rows)), 2 * self.workers)
            jobs = [(rows[b], masks[b], self.p, self.q) for b in bounds if len(b)]
            out = np.concatenate(self.fork_pool().map(matching_rows, jobs))
        else:
            out = matching_rows((rows, masks, self.p, self.q))
        return torch.from_numpy(out).float()


PROBLEMS = {
    "shortestpath": ShortestPath,
    "knapsack": Knapsack,
    "energy": Energy,
    "matching": Matching,
    "portfolio": Portfolio,
}
