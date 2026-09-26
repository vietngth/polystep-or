"""Brass alloy benchmark of Mandi et al. (2025) with PolystepOR tuned by a Sobol design.

Per radius mode, 32 Sobol points are trained on tuning seeds 7 and 8, the point with the
lowest mean last-epoch validation post-hoc cost is selected, and it is retrained and tested
on seeds 11 to 15.

    python -m experiments.alloy.run tune --mode constant --trial 0 --seed 7
    python -m experiments.alloy.run select --mode constant
    python -m experiments.alloy.run final --mode constant --seed 11 [--reported]
    python -m experiments.alloy.run commands --stage tune > commands.txt
    python -m experiments.alloy.run score
"""

import argparse
import json
import math

from experiments.common import (
    enter,
    load_config,
    mean_std,
    read_json,
    results_dir,
    third_party,
    write_json,
)
from polystep_or import Tuner

CONFIG = load_config("alloy")
WORKERS_HELP = "parallel solver processes (default: the configuration file)"
METRICS = ("test_regret", "test_infeasibility", "test_posthoc_regret")


def tuner(mode):
    """Return the Sobol tuner of a radius mode, storing results under results/alloy."""
    spec = CONFIG["tuning"]
    space = dict(spec["space"])
    if mode == "moving":
        space.update(spec["moving_space"])
    return Tuner(space, strategy=spec["strategy"], n_trials=spec["n_trials"], seeds=spec["seeds"],
                 seed=spec["seed"], digits=spec["digits"],
                 store=results_dir("alloy") / f"tuning_{mode}")


def polystep_config(trial, mode):
    """Return the PolyStep configuration of one design point."""
    eps0, step, probe = trial["eps0"], trial["step_radius"], trial["probe_radius"]
    config = dict(CONFIG["polystep"], rank=trial["subspace_rank"], num_probe=trial["num_probe"],
                  momentum=trial["use_momentum"], probe_radius=(probe, probe))
    if mode == "moving":
        config.update(epsilon=(eps0, eps0), radius_schedule="linear",
                      step_radius=(step, step * trial["step_decay_ratio"]))
    else:
        config.update(epsilon=(eps0, CONFIG["eps1"]), step_radius=(step, step))
    return config


def train_unit(trial, mode, seed, out, epochs=None, workers=None):
    """Train and test one configuration at one seed; return the last validation cost."""
    enter(third_party("odece_neurips25"))
    from experiments.alloy.model import train_and_test

    settings = dict(CONFIG, solver_workers=workers or CONFIG["solver_workers"])
    record = train_and_test(polystep_config(trial, mode), seed, str(out), settings, epochs)
    with open(out / "val_log.jsonl") as f:
        values = [json.loads(line)["val_posthoc_cost"] for line in f if "val_posthoc_cost" in line]
    record.update({"trial": trial, "mode": mode, "seed": seed, "val_posthoc_cost": values[-1]})
    write_json(out / "done.json", record)
    print(f"alloy {mode} seed {seed}: validation post-hoc cost {values[-1]:.4f}, "
          f"test {record['test']}")
    return values[-1]


def tune(args):
    """Train one design point at one tuning seed and store its validation cost."""
    search = tuner(args.mode)
    config = search.configs()[args.trial]
    out = results_dir("alloy") / f"tune_{args.mode}" / f"trial{args.trial:03d}_seed{args.seed}"
    search.evaluate(lambda c, s: train_unit(c, args.mode, s, out, args.epochs, args.workers), args.trial, config,
                    args.seed)


def select(args):
    """Select the design point with the lowest mean validation cost over the tuning seeds."""
    best, table = tuner(args.mode).run(None)
    missing = sum(row["mean"] is None for row in table)
    if missing:
        raise SystemExit(f"{missing} of {len(table)} design points are not finished")
    write_json(results_dir("alloy") / f"best_{args.mode}.json", {"best": best, "table": table})
    print(f"{args.mode}: {best}")


def final(args):
    """Train and test the selected configuration at one evaluation seed."""
    if args.reported:
        best = CONFIG["reported_best"][args.mode]
    else:
        best = read_json(results_dir("alloy") / f"best_{args.mode}.json")["best"]
    out = results_dir("alloy") / f"final_{args.mode}" / f"seed{args.seed}"
    train_unit(best, args.mode, args.seed, out, args.epochs, args.workers)


def commands(args):
    """Print the unfinished commands of a stage."""
    root = results_dir("alloy")
    for mode in CONFIG["modes"]:
        if args.stage == "tune":
            for trial in range(CONFIG["tuning"]["n_trials"]):
                for seed in CONFIG["tuning"]["seeds"]:
                    stored = root / f"tuning_{mode}" / f"trial{trial:03d}_seed{seed}.json"
                    if not stored.exists():
                        print(f"python -m experiments.alloy.run tune --mode {mode} "
                              f"--trial {trial} --seed {seed}")
        else:
            for seed in CONFIG["final_seeds"]:
                if not (root / f"final_{mode}" / f"seed{seed}" / "done.json").exists():
                    print(f"python -m experiments.alloy.run final --mode {mode} --seed {seed}")


def score(args):
    """Print the test metrics of the final seeds per radius mode."""
    summary = {}
    for mode in CONFIG["modes"]:
        paths = [results_dir("alloy") / f"final_{mode}" / f"seed{s}" / "done.json"
                 for s in CONFIG["final_seeds"]]
        records = [read_json(p)["test"] for p in paths if p.exists()]
        if not records:
            continue
        values = {m: [r[m] for r in records if r.get(m) is not None and not math.isnan(r[m])]
                  for m in METRICS}
        summary[mode] = {m: mean_std(v) for m, v in values.items() if v}
        text = ", ".join(f"{m} {v[0]:.4f} +- {v[1]:.4f}" for m, v in summary[mode].items())
        summary[mode]["n"] = len(records)
        print(f"{mode} ({len(records)} seeds): {text}")
    write_json(results_dir("alloy") / "summary.json", summary)


def main():
    """Parse the command line and run a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    unit = sub.add_parser("tune")
    unit.add_argument("--mode", required=True, choices=CONFIG["modes"])
    unit.add_argument("--trial", required=True, type=int)
    unit.add_argument("--seed", required=True, type=int)
    unit.add_argument("--epochs", type=int)
    unit.add_argument("--workers", type=int, help=WORKERS_HELP)
    choose = sub.add_parser("select")
    choose.add_argument("--mode", required=True, choices=CONFIG["modes"])
    last = sub.add_parser("final")
    last.add_argument("--mode", required=True, choices=CONFIG["modes"])
    last.add_argument("--seed", required=True, type=int)
    last.add_argument("--epochs", type=int)
    last.add_argument("--workers", type=int, help=WORKERS_HELP)
    last.add_argument("--reported", action="store_true",
                      help="use the configuration selected in the paper instead of select")
    listing = sub.add_parser("commands")
    listing.add_argument("--stage", required=True, choices=["tune", "final"])
    sub.add_parser("score")
    args = parser.parse_args()
    actions = {"tune": tune, "select": select, "final": final, "commands": commands,
               "score": score}
    actions[args.command](args)


if __name__ == "__main__":
    main()
