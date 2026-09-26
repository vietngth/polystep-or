"""DistrictNet benchmark of Ahmed et al. (2024) with PolystepOR (Julia) at the cosB configuration.

The authors' fixed-seed baselines (BD, FIG, AvgTSP) are trained and solved once in a base
working copy. For every seed, PolystepOR, DistrictNet and PredGNN are trained at that seed in
the seed's working copy, all instances are solved, and every method is evaluated on the same
demand scenarios, so each comparison is paired within a seed.

    python -m experiments.districtnet.run commands --stage train > commands.txt
    python -m experiments.districtnet.run train --method polyStep --seed 17
    python -m experiments.districtnet.run solve --method polyStep --seed 17 \
        --instance Bristol:120:3
    python -m experiments.districtnet.run scenario --seed 17
    python -m experiments.districtnet.run evaluate --seed 17 --instance Bristol:120:3
    python -m experiments.districtnet.run score

Stages run in the order train, solve, scenario, evaluate.
"""

import argparse
import os
import shlex
import shutil
import subprocess

import numpy as np

from experiments.common import (
    ROOT,
    load_config,
    paired_test,
    read_json,
    results_dir,
    sign_flip_test,
    third_party,
    write_json,
)
from polystep_or import get_config

CONFIG = load_config("districtnet")
EXPERIMENT = "Experiment_General_multisize_cities"
PRIVATE = ("deps/Scenario/output", "models", "output")
LEFTOVERS = (".par", ".tsp", ".tour", ".sol")


def instances(include_large=True):
    """Return every evaluated instance as (city, size, target)."""
    grid = CONFIG["problems"]
    out = [(c, grid["size"], t) for c in grid["cities"] for t in grid["targets"]]
    varying = CONFIG["size_varying"]
    out += [(c, n, varying["target"]) for c, sizes in varying["sizes"].items() for n in sizes]
    large = CONFIG["large"]
    return out + ([(large["city"], large["size"], large["target"])] if include_large else [])


def parse_instance(text):
    """Parse CITY:SIZE:TARGET."""
    city, size, target = text.split(":")
    return city, int(size), int(target)


def is_large(instance):
    """Return True for the Ile-de-France instance."""
    return instance[0] == CONFIG["large"]["city"]


def work_root():
    """Return the directory holding the working copies."""
    return results_dir("districtnet") / "work"


def copy_tree(src, dst):
    """Copy a file or tree with hard links, or plainly across file systems."""
    if subprocess.run(["cp", "-al", str(src), str(dst)], stderr=subprocess.DEVNULL).returncode:
        shutil.rmtree(dst, ignore_errors=True)
        subprocess.run(["cp", "-a", str(src), str(dst)], check=True)


def link_copy(src, dst):
    """Copy a working copy for one task and make its mutable parts private."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    copy_tree(src, dst)
    for path in dst.iterdir():
        if path.is_file() and path.suffix in LEFTOVERS:
            path.unlink()
    for sub in PRIVATE:
        linked = dst / sub
        if linked.exists():
            shutil.rmtree(linked)
        if (src / sub).exists():
            shutil.copytree(src / sub, linked, symlinks=True)
        else:
            linked.mkdir(parents=True)
    (dst / "output/solution" / EXPERIMENT).mkdir(parents=True, exist_ok=True)
    (dst / "output/solution/Comparaison" / EXPERIMENT).mkdir(parents=True, exist_ok=True)


def workdir(seed):
    """Return the working copy of a seed, or of the fixed-seed baselines when seed is None."""
    path = work_root() / ("base" if seed is None else f"seed_{seed}")
    if not path.exists():
        source = third_party("DistrictNet")
        path.mkdir(parents=True)
        for entry in source.iterdir():
            if entry.name not in (".git", *PRIVATE):
                copy_tree(entry, path / entry.name)
        for sub in PRIVATE:
            (path / sub).mkdir(parents=True, exist_ok=True)
        for sub in (EXPERIMENT, f"Comparaison/{EXPERIMENT}"):
            (path / "output/solution" / sub).mkdir(parents=True, exist_ok=True)
    if seed is not None:
        write_json(path / "polystep_config.json", polystep_config(seed))
    return path


def task_dir(seed, name):
    """Return a private copy of a working copy for one parallel task."""
    task = work_root() / "tasks" / f"{'base' if seed is None else seed}_{name}"
    if task.exists():
        shutil.rmtree(task)
    link_copy(workdir(seed), task)
    return task


def polystep_config(seed):
    """Return the PolyStep settings read by the Julia estimator."""
    spec = CONFIG["polystep"]
    config = get_config(spec["config"], **spec.get("overrides", {}))
    if (config.subspace, config.scale_cost, config.solver) != ("hybrid", "mean", "softmax"):
        raise SystemExit("the Julia estimator implements the hybrid subspace, mean scaling and "
                         "softmax weights only")
    return {"tag": spec["config"], "seed": int(seed), "steps": int(spec["steps"]),
            "epsilon": list(config.epsilon), "step_radius": list(config.step_radius),
            "probe_radius": list(config.probe_radius), "rank": config.rank,
            "max_subspace_dim": config.max_subspace_dim, "num_probe": config.num_probe,
            "momentum": config.momentum_range[1] if config.momentum else 0.0,
            "polytope": config.polytope}


def environment(method, seed, directory, instance=None):
    """Return the environment of a Julia run."""
    env = dict(os.environ, NB_SCENARIO=str(CONFIG["scenarios"]), JULIA_NUM_THREADS="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", GKSwstype="100",
               JULIA_PROJECT=str(directory))
    if method == "polyStep":
        env["POLYSTEP_CONFIG"] = str(directory / "polystep_config.json")
    if method in ("districtNet", "predictGnn"):
        env.update(DN_SEEDED="1", DN_SEED=str(seed))
    if method == "predictGnn":
        env["PREDGNN_EPOCHS"] = str(CONFIG["predictgnn_epochs"])
    if instance is not None:
        limits = CONFIG["time_limit"]
        large = is_large(instance) or instance[1] >= limits["large_from"]
        env["DN_MAX_TIME"] = str(limits["large"] if large else limits["default"])
    return env


def julia(arguments, directory, env, log, workers=0):
    """Run Julia in a DistrictNet working copy."""
    executable = shlex.split(os.environ.get("JULIA", CONFIG["julia"]))
    command = [*executable, *(["-p", str(workers)] if workers else []), "--project=.", *arguments]
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        subprocess.run(command, cwd=directory, env=env, check=True, stdout=f,
                       stderr=subprocess.STDOUT)


def seed_of(method, seed):
    """Return the seed of a method's working copy: None for the fixed-seed baselines."""
    if method in CONFIG["fixed_methods"]:
        return None
    if seed is None:
        raise SystemExit(f"{method} needs --seed")
    return seed


def logs(seed):
    """Return the log directory of a seed."""
    return results_dir("districtnet") / ("base" if seed is None else f"seed_{seed}") / "logs"


def train(args):
    """Train one method on the authors' 100 training cities."""
    seed = seed_of(args.method, args.seed)
    directory = workdir(seed)
    workers = (args.workers or CONFIG["polystep"]["workers"]) if args.method == "polyStep" else 0
    arguments = ["experiments.jl", "2", "city1", args.method, "train", "20", "120", "C",
                 str(CONFIG["training_cities"])]
    julia(arguments, directory, environment(args.method, seed, directory),
          logs(seed) / f"train_{args.method}.log", workers)


def solve(args):
    """Solve one instance with one trained method and keep its districting."""
    seed, instance = seed_of(args.method, args.seed), parse_instance(args.instance)
    city, size, target = instance
    task = task_dir(seed, f"solve_{args.method}_{city}_{size}_{target}")
    arguments = ["experiments.jl", "2", city, args.method, "solve", str(target), str(size), "C",
                 str(CONFIG["training_cities"])]
    julia(arguments, task, environment(args.method, seed, task, instance),
          logs(seed) / f"solve_{args.method}_{city}_{size}_{target}.log")
    name = f"{city}_C_{size}_{target}.{args.method}.txt"
    shutil.copy2(task / "output/solution" / EXPERIMENT / name,
                 workdir(seed) / "output/solution" / EXPERIMENT / name)
    shutil.rmtree(task)


def scenario(args):
    """Draw the demand scenarios of the large instance once for a seed."""
    large = CONFIG["large"]
    directory = workdir(args.seed)
    arguments = [str(ROOT / "experiments/districtnet/evaluate.jl"), "scenario", large["city"],
                 str(large["size"]), str(large["target"])]
    julia(arguments, directory, environment(None, args.seed, directory),
          logs(args.seed) / "scenario_large.log")


def evaluate(args):
    """Score every method's districting of an instance on the seed's demand scenarios."""
    instance = parse_instance(args.instance)
    city, size, target = instance
    label = args.method or "all"
    task = task_dir(args.seed, f"evaluate_{label}_{city}_{size}_{target}")
    base = workdir(None) / "output/solution" / EXPERIMENT
    for method in CONFIG["fixed_methods"]:
        solution = base / f"{city}_C_{size}_{target}.{method}.txt"
        if solution.exists():
            shutil.copy2(solution, task / "output/solution" / EXPERIMENT / solution.name)
    out = results_dir("districtnet") / f"seed_{args.seed}"
    record = out / (f"{city}_{size}_t{target}.json" if args.method is None
                    else f"{city}_{size}_t{target}_{args.method}.json")
    mode = "methods" if is_large(instance) else "instance"
    arguments = [str(ROOT / "experiments/districtnet/evaluate.jl"), mode, city, str(size),
                 str(target), str(record)]
    if args.method:
        arguments.append(args.method)
    julia(arguments, task, environment(None, args.seed, task),
          logs(args.seed) / f"evaluate_{label}_{city}_{size}_{target}.log")
    shutil.rmtree(task)


def commands(args):
    """Print the commands of one stage for the fixed-seed baselines and every seed."""
    run = "python -m experiments.districtnet.run"
    seeds = args.seeds or CONFIG["seeds"]
    fixed, per_seed = CONFIG["fixed_methods"], CONFIG["seed_methods"]
    for seed in [None, *seeds]:
        methods = fixed if seed is None else per_seed
        flag = "" if seed is None else f" --seed {seed}"
        if args.stage == "train":
            print("\n".join(f"{run} train --method {m}{flag}" for m in methods))
        elif args.stage == "solve":
            print("\n".join(f"{run} solve --method {m}{flag} --instance {c}:{n}:{t}"
                            for m in methods for c, n, t in instances()))
        elif seed is not None and args.stage == "scenario":
            print(f"{run} scenario{flag}")
        elif seed is not None and args.stage == "evaluate":
            large = CONFIG["large"]
            print("\n".join(f"{run} evaluate{flag} --instance {c}:{n}:{t}"
                            for c, n, t in instances(include_large=False)))
            print("\n".join(f"{run} evaluate{flag} --instance "
                            f"{large['city']}:{large['size']}:{large['target']} --method {m}"
                            for m in [*fixed, *per_seed]))


def seed_costs(seed):
    """Return the evaluated costs of a seed: {(city, size, target): {method: cost}}."""
    out = {}
    for path in (results_dir("districtnet") / f"seed_{seed}").glob("*_t*.json"):
        record = read_json(path)
        key = (record["city"], int(record["size"]), int(record["target"]))
        out.setdefault(key, {}).update(record["costs"])
    return out


def relative(values, reference):
    """Return the mean relative cost of values against a reference in percent."""
    values, reference = np.asarray(values), np.asarray(reference)
    return float(100.0 * np.mean((values - reference) / reference))


def score(args):
    """Write the seed-matched tables, the seed summary and the all-instance table.

    Per seed, PolystepOR is compared with DistrictNet retrained at the same seed and with the
    other methods as printed in the paper (Table 7, Table 2); the seed summary also compares
    with PredGNN retrained at the same seed, and the all-instance table pairs PolystepOR and
    DistrictNet on the scenarios of each seed.
    """
    grid, reference = CONFIG["problems"], CONFIG["reference"]
    problems = [(c, grid["size"], t) for c in grid["cities"] for t in grid["targets"]]
    large = (CONFIG["large"]["city"], CONFIG["large"]["size"], CONFIG["large"]["target"])
    per_seed, matched, ratios = {}, {"districtNet": [], "predictGnn": []}, {}
    for seed in CONFIG["seeds"]:
        costs = seed_costs(seed)
        for key, row in costs.items():
            if key != large and "polyStep" in row and "districtNet" in row:
                ratios.setdefault(key, []).append(row["polyStep"] / row["districtNet"])
        if any("polyStep" not in costs.get(p, {}) for p in problems):
            continue
        ps = [costs[p]["polyStep"] for p in problems]
        rows = {}
        for index, method in enumerate(reference["methods"]):
            same_seed = method == "districtNet"
            values = ([costs[p].get(method, np.nan) for p in problems] if same_seed else
                      [reference["table7"][p[0]][p[2]][index] for p in problems])
            if any(np.isnan(values)):
                continue
            rows[method] = {"relative_percent": relative(values, ps),
                            "p": paired_test(values, ps),
                            "polystep_cheaper": int(sum(v > c for v, c in zip(values, ps))),
                            "source": "same seed" if same_seed else "paper Table 7"}
        for method in matched:
            if all(method in costs[p] for p in problems):
                matched[method].append(relative([costs[p][method] for p in problems], ps))
        table = {"problems": rows, "polystep_mean_cost": float(np.mean(ps))}
        evaluated = costs.get(large, {})
        if "polyStep" in evaluated:
            base = evaluated["polyStep"]
            printed = {m: reference["table2"][m] for m in reference["methods"]
                       if m != "districtNet"}
            printed.update({m: evaluated[m] for m in ("districtNet",) if m in evaluated})
            table["large"] = {"evaluated_percent": {m: 100.0 * (c - base) / base
                                                    for m, c in evaluated.items()},
                              "reference_percent": {m: 100.0 * (c - base) / base
                                                    for m, c in printed.items()},
                              "polystep_cost": base}
        per_seed[seed] = table
    summary = {m: {"seeds": len(v), "mean_relative_percent": float(np.mean(v)),
                   "sd_relative_percent": float(np.std(v, ddof=1))}
               for m, v in matched.items() if len(v) >= 2}
    instance_ratios = {f"{c}:{n}:{t}": float(np.mean(r))
                       for (c, n, t), r in sorted(ratios.items())}
    values = list(instance_ratios.values())
    if len(values) >= 5:
        summary["all_instances"] = {"n": len(values),
                                    "polystep_vs_districtnet_percent": 100 * (np.mean(values) - 1),
                                    "wilcoxon_p": paired_test(values, [1.0] * len(values)),
                                    "sign_flip_p": sign_flip_test([v - 1 for v in values]),
                                    "polystep_cheaper": int(sum(v < 1 for v in values))}
    write_json(results_dir("districtnet") / "summary.json",
               {"per_seed": per_seed, "summary": summary, "instance_ratios": instance_ratios})
    for seed, table in per_seed.items():
        text = ", ".join(f"{m} {r['relative_percent']:+.2f}% (p {r['p']:.3f})"
                         for m, r in table["problems"].items())
        print(f"seed {seed}: {text}")
    for name, entry in summary.items():
        print(f"{name}: {entry}")


def main():
    """Parse the command line and run a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    methods = [*CONFIG["seed_methods"], *CONFIG["fixed_methods"]]
    fit = sub.add_parser("train")
    fit.add_argument("--method", required=True, choices=methods)
    fit.add_argument("--seed", type=int)
    fit.add_argument("--workers", type=int,
                     help="Julia worker processes of PolyStep (default: the configuration file)")
    run = sub.add_parser("solve")
    run.add_argument("--method", required=True, choices=methods)
    run.add_argument("--seed", type=int)
    run.add_argument("--instance", required=True)
    draw = sub.add_parser("scenario")
    draw.add_argument("--seed", required=True, type=int)
    rate = sub.add_parser("evaluate")
    rate.add_argument("--seed", required=True, type=int)
    rate.add_argument("--instance", required=True)
    rate.add_argument("--method", choices=methods)
    listing = sub.add_parser("commands")
    listing.add_argument("--stage", required=True,
                         choices=["train", "solve", "scenario", "evaluate"])
    listing.add_argument("--seeds", type=int, nargs="+")
    sub.add_parser("score")
    args = parser.parse_args()
    actions = {"train": train, "solve": solve, "scenario": scenario, "evaluate": evaluate,
               "commands": commands, "score": score}
    actions[args.command](args)


if __name__ == "__main__":
    main()
