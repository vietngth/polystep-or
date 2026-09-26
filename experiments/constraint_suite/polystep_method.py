"""PolystepOR as a method of the constraint suite of Silvestri et al. (sfge-dfl).

The module is linked into the authors' repository as methods/polystep.py. It trains the
authors' point predictor on the post-hoc cost of its decisions, computed with the authors'
solve_from_torch and get_objective_values, and leaves their data, splits, early stopping,
checkpointing and metrics unchanged.
"""

import multiprocessing

import numpy as np
import torch
from dfl_module import DFLModule
from optimization_problems import TOTAL_COST
from torch.func import functional_call

from polystep_or import Learner

POOL_PROBLEM = None


def pool_init(problem):
    """Store the optimization problem in a solver worker."""
    global POOL_PROBLEM
    POOL_PROBLEM = problem


def pool_task(task):
    """Return the signed post-hoc cost of one decision in a solver worker."""
    return signed_cost(POOL_PROBLEM, task)


def signed_cost(problem, task):
    """Solve for predicted parameters and return the post-hoc cost, lower is better."""
    y_hat, y, params, solve_params = task
    solution = problem.solve_from_torch(y_torch=y_hat, opt_prob_params=params, **solve_params)
    sign = 1 if problem.is_minimization_problem else -1
    cost, _ = problem.get_objective_values(y=torch.squeeze(y), sols=torch.squeeze(solution),
                                           opt_prob_params=params)
    return sign * float(cost[TOTAL_COST])


class PolyStepModule(DFLModule):
    """Trains the point predictor with PolyStep from post-hoc costs, without gradients."""

    def __init__(self, net, optimization_problem, annealer, lr=0.1, monitor=None, min_delta=0,
                 config="flatSNN", overrides=None, horizon=None, seed=0, workers=0):
        super().__init__(net=net, optimization_problem=optimization_problem, annealer=annealer,
                         lr=lr, monitor=monitor, min_delta=min_delta)
        self.automatic_optimization = False
        self.learner = Learner(config, **(overrides or {}))
        self.horizon, self.seed, self.workers = horizon, int(seed), int(workers)
        self._hyperparams.update({"config": config, "overrides": overrides, "horizon": horizon,
                                  "seed": self.seed})
        self.optimizer_ = None
        self.pool = None

    @property
    def raw_net(self):
        """The predictor without the authors' fixed output scaling layer."""
        return self.net[0] if isinstance(self.net, torch.nn.Sequential) else self.net

    def scale(self, y):
        """Apply the authors' output scaling layer."""
        return self.net[1](y) if isinstance(self.net, torch.nn.Sequential) else y

    def configure_optimizers(self):
        """Return a zero-step optimizer that only advances Lightning's step count."""
        return torch.optim.SGD(self.net.parameters(), lr=0.0)

    def signed_costs(self, tasks):
        """Return the signed post-hoc cost of every task, in parallel when workers > 1."""
        if self.workers <= 1:
            return [signed_cost(self._optimization_problem, t) for t in tasks]
        if self.pool is None:
            self.pool = multiprocessing.get_context("spawn").Pool(
                self.workers, initializer=pool_init, initargs=(self._optimization_problem,))
        return self.pool.map(pool_task, tasks, chunksize=max(1, len(tasks) // (2 * self.workers)))

    @staticmethod
    def tasks(y_hat, y, solve_params, params, cache=True):
        """Return one solve task per instance of a batch."""
        extra = {} if cache else {"allow_cache_sampling": False, "allow_cache_adding": False}
        return [(y_hat[i], y[i], params[i],
                 {**{k: v[i] for k, v in solve_params.items()}, **extra}) for i in range(len(y))]

    def validation_step(self, batch, batch_idx, log=True, testing=False):
        """Log the validation post-hoc cost; at test time run the authors' step."""
        if testing:
            return super().validation_step(batch, batch_idx, log=log, testing=True)
        x, y, _, _, solve_params, params = batch
        params = [{k: params[k][j] for k in params} for j in range(len(y))]
        sign = 1 if self._optimization_problem.is_minimization_problem else -1
        with torch.no_grad():
            y_hat = self(x).squeeze().view(y.shape)
            mse = torch.nn.functional.mse_loss(y_hat, y)
        signed = float(np.mean(self.signed_costs(self.tasks(y_hat, y, solve_params, params,
                                                             cache=False))))
        if log:
            for name, value in (("val_mse", mse), ("val_cost", sign * signed),
                                ("val_signed_cost", signed)):
                self.log(name, value, prog_bar=True, on_step=False, on_epoch=True,
                         batch_size=len(y))
        return {"val_mse": mse, "val_signed_cost": signed}

    def training_step(self, batch, batch_idx):
        """Take one PolyStep step on the post-hoc costs of a training batch."""
        x, y, _, _, solve_params, params = super().training_step(batch, batch_idx)
        if self.optimizer_ is None:
            steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 1000)
            self.optimizer_ = self.learner.optimizer(self.raw_net, self.seed,
                                                     horizon=self.horizon or steps,
                                                     max_iterations=steps)

        def closure(stacked):
            n_cand = next(iter(stacked.values())).shape[0]
            with torch.no_grad():
                candidates = [{n: stacked[n][k] for n in stacked} for k in range(n_cand)]
                preds = [self.scale(functional_call(self.raw_net, p, (x,))).squeeze().view(y.shape)
                         for p in candidates]
            tasks = [t for y_hat in preds for t in self.tasks(y_hat, y, solve_params, params)]
            values = self.signed_costs(tasks)
            costs = [sum(values[k * len(y):(k + 1) * len(y)]) / len(y) for k in range(n_cand)]
            return torch.tensor(costs, dtype=torch.float32)

        cost = self.optimizer_.step(closure)
        placeholder = self.optimizers()
        placeholder.step()
        placeholder.zero_grad()
        self.log("train_regret", float(cost), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=len(y))
        return None
