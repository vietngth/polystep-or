"""Constraint suite of Silvestri et al. (2026) with PolystepOR at the flatSNN configuration.

Each run is one (data set, split) pair of a cell, trained and tested by the authors' handler.

    python -m experiments.constraint_suite.run train --cell sckp_50_penalty_5 --seed 0 --split 0
    python -m experiments.constraint_suite.run commands > commands.txt
    python -m experiments.constraint_suite.run score
"""

import argparse
import copy
import csv
import math
import os
import subprocess
import sys

from experiments.common import (
    load_config,
    mean_std,
    paired_test,
    results_dir,
    third_party,
    write_json,
)

CONFIG = load_config("constraint_suite")
METRICS = ("test_relative_regret", "test_num_infeasible_solutions", "test_mse")


def cell_config(cell):
    """Return the authors' data and training settings of a cell."""
    spec = dict(CONFIG["cells"][cell])
    family = CONFIG["families"][spec.pop("family")]
    return {**copy.deepcopy(CONFIG["common"]), **family, **spec}


def horizon(settings):
    """Return the step count of the schedule horizon, schedule_epochs epochs of batches."""
    n_train = int(settings["n"] * settings["prop_training"])
    n_train -= int(math.ceil(n_train * settings["prop_validation"]))
    epochs = CONFIG["polystep"]["schedule_epochs"]
    return int(epochs * math.ceil(n_train / settings["batch_size"]))


def method_config(method, settings, seed, split, workers=None):
    """Return the method entry of the authors' configuration file."""
    if method != "PolyStep":
        return {"name": method, **CONFIG["authors_methods"][method]}
    ps = CONFIG["polystep"]
    return {"name": "PolyStep", "lr": ps["lr"], "monitor": ps["monitor"],
            "min_delta": ps["min_delta"], "config": ps["config"], "overrides": ps["overrides"],
            "horizon": horizon(settings), "seed": 1000 * seed + split,
            "workers": workers or ps["workers"][settings["optimization_problem"]]}


def run_dir(cell, seed, split, method):
    """Return the authors' output directory of one run."""
    pair = f"seed-{seed}/rnd-split-seed-{split}"
    return results_dir("constraint_suite") / cell / pair / method


def last_value(metrics_file, column):
    """Return the last logged value of a column of an authors' metrics.csv file."""
    with open(metrics_file) as f:
        values = [row[column] for row in csv.DictReader(f) if row.get(column) not in (None, "")]
    return float(values[-1]) if values else None


def test_metrics(cell, seed, split, method):
    """Return the test metrics of one finished run, or None."""
    files = sorted(run_dir(cell, seed, split, method).glob("run_*/metrics.csv"))
    if not files:
        return None
    values = {m: last_value(files[-1], m) for m in METRICS}
    return None if values["test_relative_regret"] is None else values


def train(args):
    """Run one (data set, split) pair of a cell with the authors' handler."""
    settings = cell_config(args.cell)
    settings.update({"numpy_seed": [args.seed], "rnd_split_seed": [args.split],
                     "method_configurations": [method_config(args.method, settings, args.seed,
                                                             args.split, args.workers)]})
    if args.epochs:
        settings["epochs"] = settings["patience"] = args.epochs
    out = results_dir("constraint_suite") / args.cell
    tag = f"{args.method}_s{args.seed}_p{args.split}"
    write_json(out / "configs" / f"{tag}.json", settings)
    repo = third_party("sfge-dfl")
    env = dict(os.environ, PYTHONPATH=str(repo), CUDA_VISIBLE_DEVICES="")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    command = [sys.executable, "experiments/experiment_handler.py",
               str(out / "configs" / f"{tag}.json"), str(out), "--mode", "train"]
    (out / "logs").mkdir(parents=True, exist_ok=True)
    with open(out / "logs" / f"{tag}.log", "w") as log:
        subprocess.run(command, cwd=repo, env=env, check=True, stdout=log,
                       stderr=subprocess.STDOUT)
    result = test_metrics(args.cell, args.seed, args.split, args.method)
    print(f"{args.cell} seed {args.seed} split {args.split} {args.method}: {result}")


def commands(args):
    """Print one command per cell, data set and split that has no result yet."""
    for cell in CONFIG["cells"]:
        for seed in CONFIG["seeds"]:
            for split in CONFIG["splits"]:
                if test_metrics(cell, seed, split, args.method) is None:
                    print(f"python -m experiments.constraint_suite.run train --cell {cell} "
                          f"--seed {seed} --split {split} --method {args.method}")


def score(args):
    """Print test relative regret per cell over the 15 runs, with paired tests."""
    pairs = [(s, p) for s in CONFIG["seeds"] for p in CONFIG["splits"]]
    summary = {}
    print(f"{'cell':<24}{'method':<10}{'runs':>5}  regret (mean +- std)  infeasible  paired p")
    for cell in CONFIG["cells"]:
        runs = {m: {pair: test_metrics(cell, *pair, m) for pair in pairs}
                for m in ("PolyStep", *CONFIG["authors_methods"])}
        runs = {m: {k: v for k, v in r.items() if v is not None} for m, r in runs.items()}
        polystep = runs["PolyStep"]
        for method, rows in runs.items():
            if not rows:
                continue
            regret = [r["test_relative_regret"] for r in rows.values()]
            infeasible = [r["test_num_infeasible_solutions"] or 0.0 for r in rows.values()]
            mean, std = mean_std(regret)
            shared = sorted(set(rows) & set(polystep))
            p = None
            if method != "PolyStep" and len(shared) >= 5:
                p = paired_test([polystep[k]["test_relative_regret"] for k in shared],
                                [rows[k]["test_relative_regret"] for k in shared])
            summary.setdefault(cell, {})[method] = {"n": len(rows), "mean": mean, "std": std,
                                                    "infeasible": mean_std(infeasible)[0],
                                                    "p_vs_polystep": p}
            p_text = "" if p is None else f"{p:.3f}"
            print(f"{cell:<24}{method:<10}{len(rows):>5}  {mean:.4f} +- {std:.4f}"
                  f"{mean_std(infeasible)[0]:>12.2f}  {p_text}")
    write_json(results_dir("constraint_suite") / "summary.json", summary)


def main():
    """Parse the command line and run a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    methods = ["PolyStep", *CONFIG["authors_methods"]]
    run = sub.add_parser("train")
    run.add_argument("--cell", required=True, choices=list(CONFIG["cells"]))
    run.add_argument("--seed", required=True, type=int)
    run.add_argument("--split", required=True, type=int)
    run.add_argument("--method", default="PolyStep", choices=methods)
    run.add_argument("--epochs", type=int)
    run.add_argument("--workers", type=int,
                     help="parallel solver processes of PolyStep (default: the configuration file)")
    listing = sub.add_parser("commands")
    listing.add_argument("--method", default="PolyStep", choices=methods)
    sub.add_parser("score")
    args = parser.parse_args()
    {"train": train, "commands": commands, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
