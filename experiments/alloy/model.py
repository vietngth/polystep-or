"""PolystepOR on the brass alloy benchmark of the ODECE repository (Mandi et al., 2025).

Imported after entering the authors' repository. The authors' dataset, predictor, solver,
Lightning module and metrics are used unchanged; PolyStep replaces the training step and
trains on the post-hoc cost of the decisions.
"""

import json
import multiprocessing
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
from ML.TorchML import AlloyNet
from OptProblems import opt
from OptProblems.alloyproduction.alloydataset import alloy_dataset
from OptProblems.alloyproduction.alloysolver import alloy_solver
from pytorch_lightning.loggers import CSVLogger
from src.pfl import PFL
from torch.func import functional_call
from torch.utils.data import DataLoader

from polystep_or import Learner

N_SUPPLIERS = 10
UNSOLVABLE_PURCHASE = 100.0
POOL_SOLVER = None


def posthoc_cost(solver, predicted, true, costs, penalty):
    """Return the post-hoc cost of the decision for predicted parameters, in objective units.

    A decision infeasible under the true parameters is corrected by the authors' recourse and
    charged their penalty coefficient times the objective gap. An unsolvable prediction is
    charged the empty decision when maximizing and a large purchase when minimizing.
    """
    coefficient = penalty.mean()
    sign = -1.0 if solver.modelSense == opt.MAXIMIZE else 1.0
    try:
        decision = solver.solve(predicted, costs)
    except Exception:
        return 0.0 if sign < 0 else coefficient * UNSOLVABLE_PURCHASE * float(np.abs(costs).sum())
    realized = solver.evaluate_solution(costs, true, decision)
    if solver.check_feasibility(true, decision):
        return sign * realized
    corrected = solver.evaluate_solution(costs, true, solver.correct_feasibility(true, decision))
    return sign * corrected + coefficient * abs(realized - corrected)


def pool_init():
    """Build one alloy solver per worker process."""
    global POOL_SOLVER
    os.environ["OMP_NUM_THREADS"] = "1"
    POOL_SOLVER = alloy_solver(N_SUPPLIERS)


def pool_task(task):
    """Return the post-hoc cost of one decision in a worker process."""
    return posthoc_cost(POOL_SOLVER, *task)


class PolyStepPFL(PFL):
    """The authors' PFL module, trained with PolyStep on post-hoc costs."""

    def __init__(self, predictor, solver, config, polystep_seed, seed, max_epochs, lr=0.005,
                 solver_workers=1, val_log=None):
        super().__init__([predictor], solver, 1, None, False, False, lr=lr,
                         max_epochs=max_epochs, seed=seed)
        self.learner = Learner(config)
        self.polystep_seed = int(polystep_seed)
        self.solver_workers = int(solver_workers)
        self.val_log = val_log
        self.index = next(iter(self.constr_predictors))
        self.optimizer_ = None
        self.pool = None
        self.solve_seconds = 0.0

    @property
    def net(self):
        """The authors' predictor searched by PolyStep."""
        return self.constr_predictors[self.index]

    def configure_optimizers(self):
        """Return a zero-step optimizer that only advances Lightning's step count."""
        return [torch.optim.SGD(self.net.parameters(), lr=0.0)]

    def tasks(self, predicted, true, costs, penalty):
        """Return one post-hoc cost task per instance of a batch."""
        predicted_tuple, _ = self._create_params_tuple([predicted], true, costs)
        predicted_np = [t.detach().cpu().numpy() for t in predicted_tuple]
        true_np = [t.detach().cpu().numpy() for t in true]
        costs_np, penalty_np = costs.numpy(), penalty.numpy()
        return [(tuple(p[i] for p in predicted_np), tuple(t[i] for t in true_np), costs_np[i],
                 penalty_np[i]) for i in range(len(costs_np))]

    def costs_of(self, tasks):
        """Return the post-hoc cost of every task, in parallel when solver_workers > 1."""
        start = time.time()
        if self.solver_workers > 1:
            if self.pool is None:
                self.pool = multiprocessing.get_context("spawn").Pool(self.solver_workers,
                                                                      initializer=pool_init)
            chunk = max(1, len(tasks) // (4 * self.solver_workers))
            values = self.pool.map(pool_task, tasks, chunksize=chunk)
        else:
            values = [posthoc_cost(self.optsolver, *t) for t in tasks]
        self.solve_seconds += time.time() - start
        return values

    def training_step(self, batch, batch_idx):
        """Take one PolyStep step on the post-hoc costs of a training batch."""
        features, true, costs, _, _, penalty = batch
        if self.optimizer_ is None:
            steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 1000)
            self.optimizer_ = self.learner.optimizer(self.net, self.polystep_seed, horizon=steps,
                                                     max_iterations=steps)

        def closure(stacked):
            n_cand = next(iter(stacked.values())).shape[0]
            with torch.no_grad():
                preds = [functional_call(self.net, {n: stacked[n][k] for n in stacked},
                                         (features,)) for k in range(n_cand)]
            tasks = [t for pred in preds for t in self.tasks(pred, true, costs, penalty)]
            values = np.asarray(self.costs_of(tasks), dtype=np.float64).reshape(n_cand, -1)
            return torch.tensor(values.mean(axis=1), dtype=torch.float32)

        cost = self.optimizer_.step(closure)
        placeholder = self.optimizers()
        placeholder = placeholder[0] if isinstance(placeholder, list) else placeholder
        placeholder.step()
        placeholder.zero_grad()
        self.log("train_loss", float(cost), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=len(costs))

    def validation_step(self, batch, batch_idx, prefix="val"):
        """Log the authors' validation metrics and the validation post-hoc cost."""
        out = super().validation_step(batch, batch_idx, prefix)
        features, true, costs, _, _, penalty = batch
        with torch.no_grad():
            predicted = self.forward(features)[0]
        value = float(np.mean([posthoc_cost(self.optsolver, *t)
                               for t in self.tasks(predicted, true, costs, penalty)]))
        self.log(f"{prefix}_posthoc_cost", value, on_epoch=True, batch_size=len(costs))
        return out

    def on_validation_epoch_end(self):
        """Append the validation metrics of the epoch to the validation log."""
        super().on_validation_epoch_end()
        if self.val_log:
            record = {"epoch": int(self.current_epoch), "step": int(self.global_step)}
            record.update({k: float(v) for k, v in self.trainer.callback_metrics.items()})
            with open(self.val_log, "a") as f:
                f.write(json.dumps(record) + "\n")


def train_and_test(config, seed, out, settings, epochs=None):
    """Train PolystepOR on the brass alloy data at one seed and return the test metrics."""
    start = time.time()
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    sets = [alloy_dataset(mode=m, penaltyTerm=settings["penalty_term"])
            for m in ("train", "val", "test")]
    predictor = AlloyNet(num_layers=1, num_features=4096, num_targets=1)
    solver = alloy_solver(N_SUPPLIERS)
    workers = settings["loader_workers"]
    kwargs = {"num_workers": workers, "persistent_workers": workers > 0}
    evaluation = {"batch_size": settings["eval_batch_size"], "shuffle": False, **kwargs}
    loaders = [DataLoader(sets[0], batch_size=settings["batch_size"], shuffle=True, **kwargs),
               DataLoader(sets[1], **evaluation), DataLoader(sets[2], **evaluation)]
    val_log = os.path.join(out, "val_log.jsonl")
    open(val_log, "w").close()
    epochs = epochs or settings["epochs"]
    model = PolyStepPFL(predictor, solver, config, settings["polystep_seed"], seed, epochs,
                        solver_workers=settings["solver_workers"], val_log=val_log)
    trainer = pl.Trainer(max_epochs=epochs, check_val_every_n_epoch=1,
                         logger=CSVLogger(out, name="", version=""), enable_checkpointing=False,
                         accelerator="cpu", num_sanity_val_steps=2)
    trainer.validate(model, dataloaders=loaders[1])
    trainer.fit(model, train_dataloaders=loaders[0], val_dataloaders=loaders[1])
    test = trainer.test(model, dataloaders=loaders[2])
    if model.pool is not None:
        model.pool.close()
    return {"test": test[0] if test else {}, "solve_seconds": model.solve_seconds,
            "elapsed_s": time.time() - start}
