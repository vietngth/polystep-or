"""Validation-only hyperparameter search with stored, parallel-friendly trials."""

import itertools
import json
import warnings
from pathlib import Path

import numpy as np

STRATEGIES = ("sobol", "optuna", "grid")


def is_range(spec):
    """Return True for a range axis written as ("log" | "lin", low, high)."""
    return (isinstance(spec, (list, tuple)) and len(spec) == 3 and spec[0] in ("log", "lin")
            and all(isinstance(v, (int, float)) for v in spec[1:]))


def from_unit(spec, u, digits=None):
    """Map a number in [0, 1) to a value of one axis."""
    if is_range(spec):
        kind, low, high = spec
        if kind == "log":
            value = 10 ** (np.log10(low) + u * (np.log10(high) - np.log10(low)))
        else:
            value = low + u * (high - low)
        return float(np.round(value, digits)) if digits is not None else float(value)
    choices = list(spec)
    return choices[min(int(u * len(choices)), len(choices) - 1)]


class Tuner:
    """Selects a configuration by its mean validation cost over the tuning seeds.

    space maps each hyperparameter to a range ("log" | "lin", low, high) or to a list of
    choices. strategy "sobol" draws a scrambled Sobol design with one coordinate per axis in
    the order of space, "grid" takes every combination of the choices, and "optuna" runs
    TPE. The objective is called as objective(config, seed) and returns a validation cost;
    the test split is never seen. With a store directory, each (trial, seed) result is one
    JSON file, so an interrupted search continues where it stopped and trials can run as
    separate jobs.
    """

    def __init__(self, space, strategy="sobol", n_trials=32, seeds=(0,), seed=0, digits=None,
                 store=None):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")
        self.space = dict(space)
        self.strategy = strategy
        self.n_trials = int(n_trials)
        self.seeds = [int(s) for s in seeds]
        self.seed = int(seed)
        self.digits = digits
        self.store = Path(store) if store else None

    def configs(self):
        """Return the trial configurations of a Sobol or grid design."""
        if self.strategy == "grid":
            if any(is_range(v) for v in self.space.values()):
                raise ValueError("a grid needs lists of choices, not ranges")
            keys = list(self.space)
            return [dict(zip(keys, combo)) for combo in itertools.product(*self.space.values())]
        if self.strategy != "sobol":
            raise ValueError("the optuna strategy proposes its configurations while it runs")
        from scipy.stats import qmc

        units = qmc.Sobol(d=len(self.space), scramble=True, seed=self.seed).random(self.n_trials)
        return [{k: from_unit(spec, row[i], self.digits) for i, (k, spec) in
                 enumerate(self.space.items())} for row in units]

    def evaluate(self, objective, trial, config, seed):
        """Return the stored value of one (trial, seed) unit, computing it when missing."""
        path = None if self.store is None else self.store / f"trial{trial:03d}_seed{seed}.json"
        if path is not None and path.exists():
            with open(path) as f:
                record = json.load(f)
            if record["config"] != json.loads(json.dumps(config)):
                raise ValueError(f"{path} holds a different configuration")
            return record["value"]
        if objective is None:
            return None
        value = float(objective(config, seed))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump({"trial": trial, "seed": seed, "config": config, "value": value}, f)
            tmp.replace(path)
        return value

    def run(self, objective, trials=None):
        """Evaluate the design and return the best configuration and the trial table.

        trials restricts a Sobol or grid run to some trial indices, for parallel jobs.
        With objective None only stored results are read and missing units stay empty.
        """
        if self.strategy == "optuna":
            return self._run_optuna(objective)
        table = []
        for i, config in enumerate(self.configs()):
            if trials is not None and i not in trials:
                continue
            values = {s: self.evaluate(objective, i, config, s) for s in self.seeds}
            table.append(self._row(i, config, values))
        return self._best(table), table

    def _run_optuna(self, objective):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(seed=self.seed, multivariate=True,
                                             n_startup_trials=max(4, min(self.n_trials // 3, 10)))
        study = optuna.create_study(direction="minimize", sampler=sampler)
        table = []
        for i in range(self.n_trials):
            trial = study.ask()
            config = {k: self._suggest(trial, k, spec) for k, spec in self.space.items()}
            values = {s: self.evaluate(objective, i, config, s) for s in self.seeds}
            row = self._row(i, config, values)
            table.append(row)
            if row["mean"] is None:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                study.tell(trial, row["mean"])
        return self._best(table), table

    @staticmethod
    def _suggest(trial, name, spec):
        if is_range(spec):
            kind, low, high = spec
            return trial.suggest_float(name, float(low), float(high), log=kind == "log")
        return trial.suggest_categorical(name, list(spec))

    @staticmethod
    def _row(trial, config, values):
        complete = all(v is not None for v in values.values())
        mean = float(np.mean(list(values.values()))) if complete else None
        return {"trial": trial, "config": config, "values": values, "mean": mean}

    @staticmethod
    def _best(table):
        done = [row for row in table if row["mean"] is not None]
        if not done:
            return None
        return min(done, key=lambda row: row["mean"])["config"]
