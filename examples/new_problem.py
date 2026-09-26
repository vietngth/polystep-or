"""
Example of how you can add a new problem of interest. Here you can find an example of adding a 
knapsack whose item weights are predicted from features.

The problem subclasses polystep_or.Problem with its data, its solver (scipy.optimize.milp)
and its realized cost, which includes a recourse action (in-constraint preds). Any torch predictor is then trained
from realized costs alone with polystep_or.run. Run from the repository root:

    python examples/new_problem.py

Hint: you can speed up significantly if your solvers are GPU-friendly!
"""

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp

from polystep_or import Problem, evaluate, run

N_ITEMS = 6
N_FEATURES = 3
CAPACITY = 2.0
PENALTY = 2.0
MIN_WEIGHT = 1e-3


class PredictedWeightKnapsack(Problem):
    """Select items of known value under a capacity, with weights predicted from features.

    Once the true weights are revealed, selected items are removed in order of increasing
    value per weight until the selection fits, and each removed item costs PENALTY times its
    value. The realized cost is the negated kept value plus that penalty.
    """

    name = "predicted_weight_knapsack"

    def __init__(self, sizes=(200, 50, 50), seed=0):
        rng = np.random.default_rng(seed)
        self.values = rng.uniform(1.0, 2.0, N_ITEMS)
        mixing = rng.normal(size=(N_FEATURES, N_ITEMS))
        self.data = {}
        for split, size in zip(("train", "val", "test"), sizes):
            features = rng.normal(size=(size, N_FEATURES))
            noise = rng.uniform(0.9, 1.1, size=(size, N_ITEMS))
            weights = 0.1 + 0.5 * np.exp(0.5 * features @ mixing) * noise
            self.data[split] = (torch.tensor(features, dtype=torch.float32),
                                torch.tensor(weights, dtype=torch.float32))

    def load_data(self, split):
        """Return features and true weights of a split."""
        return self.data[split]

    def solve(self, parameters, context=None):
        """Solve one 0-1 knapsack per row of predicted weights."""
        weights = np.clip(parameters.detach().numpy().astype(float), MIN_WEIGHT, None)
        decisions = [self._solve_one(w) for w in weights]
        return torch.tensor(np.array(decisions), dtype=torch.float32)

    def _solve_one(self, weights):
        result = milp(-self.values, integrality=np.ones(N_ITEMS), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(weights[None, :], -np.inf, CAPACITY))
        return np.round(result.x)

    def recourse(self, decisions, parameters):
        """Remove items until each selection fits the true weights."""
        kept = decisions.clone()
        values = torch.tensor(self.values, dtype=parameters.dtype)
        for row, weights in zip(kept, parameters):
            order = torch.argsort(values / weights)
            for item in order:
                if float((row * weights).sum()) <= CAPACITY:
                    break
                row[item] = 0.0
        return kept

    def cost(self, decisions, parameters):
        """Return the negated kept value plus the penalty on removed items."""
        kept = self.recourse(decisions, parameters)
        values = torch.tensor(self.values, dtype=parameters.dtype)
        removed = ((decisions - kept) * values).sum(-1)
        return -(kept * values).sum(-1) + PENALTY * removed


def main():
    """Train a linear weight predictor from realized costs and compare test costs."""
    problem = PredictedWeightKnapsack()
    _, true_weights = problem.load_data("test")
    optimal = float(problem.cost(problem.solve(true_weights), true_weights).mean())
    torch.manual_seed(0)
    predictor = torch.nn.Linear(N_FEATURES, N_ITEMS)
    before = float(evaluate(problem, predictor).mean())
    result = run(problem, predictor, config="cosB", steps=30, batch_size=12, eval_every=5)
    print(f"test cost: untrained {before:.3f}, trained {result['test_cost']:.3f}, "
          f"true weights {optimal:.3f} (lower is better)")
    print(f"validation best {result['val_best']:.3f} at step {result['val_best_step']}, "
          f"{result['wall_s']:.1f} s")


if __name__ == "__main__":
    main()
