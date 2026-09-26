"""Classical benchmarks of Mandi et al. (2024) with PolystepOR at fixed configurations.

    python -m experiments.classical.run train --problem shortestpath --value 1 --seed 0
    python -m experiments.classical.run commands > commands.txt
    python -m experiments.classical.run score
"""

import argparse
import time
from pathlib import Path

import torch

from experiments.classical.problems import PROBLEMS
from experiments.common import load_config, mean_std, read_json, results_dir, write_json
from polystep_or import Learner, predict

CONFIG = load_config("classical")


def make_problem(name, value, seed, workers=None):
    """Build one benchmark instance with the options of the configuration file."""
    spec = CONFIG["problems"][name]
    options = {"workers": workers or spec.get("workers", 1)}
    if name == "energy":
        options["solver"] = spec["solver"][int(value)]
    return PROBLEMS[name](int(value), int(seed), **options)


def result_path(name, value, seed):
    """Return the record path of one run."""
    config = CONFIG["problems"][name]["config"]
    return results_dir("classical") / f"{name}_{value}" / f"{config}_seed{seed}.json"


def train(args):
    """Train and test PolystepOR on one benchmark cell and seed."""
    spec = CONFIG["problems"][args.problem]
    path = result_path(args.problem, args.value, args.seed)
    if args.out:
        path = Path(args.out).resolve()
    problem = make_problem(args.problem, args.value, args.seed, args.workers)
    torch.manual_seed(args.seed)
    model = problem.predictor()
    learner = Learner(spec["config"], steps=args.steps or spec["steps"],
                      batch_size=spec["batch_size"], patience=CONFIG["patience"],
                      **spec.get("overrides", {}))
    start = time.time()
    history = learner.fit(problem, model, args.seed)
    features, parameters = problem.load_data("test")
    context = problem.context("test")
    with torch.no_grad():
        decisions = problem.solve(predict(model, features, parameters), context)
        regret = problem.regret(decisions, parameters, context)
    record = {"problem": args.problem, "value": args.value, "config": spec["config"],
              "seed": args.seed, "test_regret": float(regret.mean()),
              "test_regret_per_instance": [float(r) for r in regret],
              "wall_s": time.time() - start, **history}
    write_json(path, record)
    print(f"{args.problem} {args.value} seed {args.seed}: test regret "
          f"{record['test_regret']:.6f}, validation best {history['val_best']} at step "
          f"{history['val_best_step']}")


def commands(args):
    """Print one training command per benchmark cell and seed."""
    for name, spec in CONFIG["problems"].items():
        for value in spec["values"]:
            for seed in CONFIG["seeds"]:
                if not result_path(name, value, seed).exists():
                    print(f"python -m experiments.classical.run train --problem {name} "
                          f"--value {value} --seed {seed}")


def score(args):
    """Print mean and standard deviation of the test regret per cell."""
    summary = []
    print(f"{'problem':<14}{'value':>6}{'config':>8}{'seeds':>7}  test regret (mean +- std)")
    for name, spec in CONFIG["problems"].items():
        scale = spec.get("scale", 1.0)
        for value in spec["values"]:
            paths = [result_path(name, value, s) for s in CONFIG["seeds"]]
            values = [read_json(p)["test_regret"] * scale for p in paths if p.exists()]
            if not values:
                continue
            mean, std = mean_std(values)
            summary.append({"problem": name, "value": value, "config": spec["config"],
                            "n": len(values), "mean": mean, "std": std, "scale": scale,
                            "metric": spec.get("metric", "relative regret")})
            print(f"{name:<14}{value:>6}{spec['config']:>8}{len(values):>7}  "
                  f"{mean:.4f} +- {std:.4f}")
    write_json(results_dir("classical") / "summary.json", summary)


def main():
    """Parse the command line and run a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("train")
    run.add_argument("--problem", required=True, choices=sorted(PROBLEMS))
    run.add_argument("--value", required=True, type=int)
    run.add_argument("--seed", required=True, type=int)
    run.add_argument("--steps", type=int)
    run.add_argument("--out")
    run.add_argument("--workers", type=int,
                     help="parallel solver processes (default: the configuration file)")
    sub.add_parser("commands")
    sub.add_parser("score")
    args = parser.parse_args()
    {"train": train, "commands": commands, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
