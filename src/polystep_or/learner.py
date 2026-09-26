"""Training a torch predictor with PolyStep on the realized cost of its decisions."""

import time

import torch
from torch.func import functional_call, vmap

from polystep_or.configs import get_config, make_optimizer

BATCH_SEED_OFFSET = 202608


def predict(model, features, parameters):
    """Return model predictions shaped like the parameters."""
    out = model(features)
    return out.squeeze(-1) if out.dim() > parameters.dim() else out


def repeat_rows(rows, times):
    """Stack times copies of rows along the first dimension."""
    return rows.unsqueeze(0).expand(times, *rows.shape).reshape(times * rows.shape[0],
                                                                 *rows.shape[1:])


def evaluate(problem, model, split="test"):
    """Return the realized cost of the model's decision on every instance of a split."""
    features, parameters = problem.load_data(split)
    with torch.no_grad():
        decisions = problem.solve(predict(model, features, parameters), problem.context(split))
        return problem.cost(decisions, parameters)


def load_validation(problem):
    """Return the validation split of a problem, or None when it has none."""
    try:
        return problem.load_data("val"), problem.context("val")
    except (KeyError, NotImplementedError):
        return None


class Learner:
    """Trains any torch predictor on a Problem with PolyStep, from realized costs only.

    Every step draws a random batch of training instances, and PolyStep scores each candidate
    parameter vector by the mean realized cost of its decisions. Every eval_every steps the
    realized cost on the validation split is measured; the best validation weights are kept
    and training stops after patience checks without improvement.
    """

    def __init__(self, config="cosB", steps=100, batch_size=128, patience=10, eval_every=None,
                 **overrides):
        self.config = get_config(config, **overrides)
        self.steps = int(steps)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.eval_every = int(eval_every) if eval_every else max(1, self.steps // 20)

    def optimizer(self, model, seed, horizon=None, max_iterations=None):
        """Return the PolyStep optimizer of this configuration for an external training loop."""
        return make_optimizer(model, self.config, horizon or self.steps, seed, max_iterations)

    def fit(self, problem, model, seed=0):
        """Train model in place on problem and return the training history."""
        features, parameters = problem.load_data("train")
        context = problem.context("train")
        validation = load_validation(problem)
        generator = torch.Generator().manual_seed(int(seed) + BATCH_SEED_OFFSET)
        batch = min(self.batch_size, features.shape[0])
        optimizer = self.optimizer(model, seed)
        best = {"cost": float("inf"), "state": None, "step": self.steps}
        stale, stopped_at, costs = 0, self.steps, []
        start = time.time()
        for step in range(self.steps):
            index = torch.randperm(features.shape[0], generator=generator)[:batch]
            closure = self._closure(problem, model, features[index], parameters[index],
                                    None if context is None else context[index])
            costs.append(float(optimizer.step(closure)))
            last = step == self.steps - 1
            if validation is None or not ((step + 1) % self.eval_every == 0 or last):
                continue
            cost = self._validation_cost(problem, model, validation)
            if cost < best["cost"] - 1e-9:
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best, stale = {"cost": cost, "state": state, "step": step + 1}, 0
                continue
            stale += 1
            if stale >= self.patience:
                stopped_at = step + 1
                break
        if best["state"] is not None:
            model.load_state_dict(best["state"])
        return {
            "steps": self.steps,
            "stopped_at": stopped_at,
            "val_best": None if validation is None else best["cost"],
            "val_best_step": None if validation is None else best["step"],
            "batch_size": batch,
            "n_params": sum(p.numel() for p in model.parameters()),
            "wall_s": time.time() - start,
            "train_costs": costs,
        }

    @staticmethod
    def _closure(problem, model, features, parameters, context):
        def forward(params):
            out = functional_call(model, params, (features,))
            return out.squeeze(-1) if out.dim() > parameters.dim() else out

        def closure(stacked):
            preds = vmap(forward)(stacked)
            n_cand, n_inst = preds.shape[:2]
            flat = preds.reshape(n_cand * n_inst, *preds.shape[2:])
            ctx = None if context is None else repeat_rows(context, n_cand)
            decisions = problem.solve(flat, ctx)
            realized = problem.cost(decisions, repeat_rows(parameters, n_cand))
            return realized.reshape(n_cand, n_inst).mean(-1)

        return closure

    @staticmethod
    def _validation_cost(problem, model, validation):
        (features, parameters), context = validation
        with torch.no_grad():
            decisions = problem.solve(predict(model, features, parameters), context)
            return float(problem.cost(decisions, parameters).mean())


def run(problem, predictor, config="cosB", seed=0, **kwargs):
    """Train predictor on problem with a named or custom configuration and test it once."""
    learner = Learner(config, **kwargs)
    history = learner.fit(problem, predictor, seed)
    test = evaluate(problem, predictor, "test")
    return {"test_cost": float(test.mean()), **history}
