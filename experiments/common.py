"""Paths, configuration files and result records shared by the experiment scripts."""

import json
import os
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def third_party(name):
    """Return the directory of a third-party clone made by scripts/setup_third_party.sh."""
    base = Path(os.environ.get("POLYSTEP_OR_THIRD_PARTY", ROOT / "third_party"))
    path = base / name
    if not path.is_dir():
        raise SystemExit(f"{path} is missing; run scripts/setup_third_party.sh first")
    return path


def results_dir(experiment):
    """Return the results directory of an experiment."""
    return Path(os.environ.get("POLYSTEP_OR_RESULTS", ROOT / "results")) / experiment


def load_config(experiment):
    """Return the configuration file of an experiment as a dictionary."""
    with open(ROOT / "experiments" / experiment / "config.yaml") as f:
        return yaml.safe_load(f)


def enter(directory):
    """Run from a third-party directory and import its modules, as its own scripts do."""
    os.chdir(directory)
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def write_json(path, record):
    """Write a JSON record atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1)
    tmp.replace(path)


def read_json(path):
    """Read a JSON record."""
    with open(path) as f:
        return json.load(f)


def mean_std(values):
    """Return the mean and the population standard deviation of a list."""
    return statistics.fmean(values), statistics.pstdev(values)


def paired_test(values, reference):
    """Return the two-sided Wilcoxon signed-rank p-value of paired values."""
    from scipy.stats import wilcoxon

    diff = [a - b for a, b in zip(values, reference)]
    return float(wilcoxon(diff).pvalue) if any(d != 0 for d in diff) else 1.0


def sign_flip_test(differences, draws=20000, seed=0):
    """Return the two-sided sign-flip permutation p-value of paired differences."""
    import numpy as np

    d = np.asarray(differences, dtype=float)
    flips = np.random.default_rng(seed).choice([-1, 1], size=(draws, len(d)))
    extreme = np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-15
    return float((extreme.sum() + 1) / (draws + 1))
