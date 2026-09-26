# PolystepOR

Implementation for PolyStepOR: Learning to Decide Without Optimal Solutions

## Quick start

```bash
uv sync                                                          # install uv and run this to set up the environment
source .venv/bin/activate
bash scripts/setup_third_party.sh clone                          # benchmark code at pinned commits + patches
bash scripts/setup_third_party.sh data                           # benchmark data
python -m experiments.classical.run commands > classical.txt     # list every run of an experiment
bash scripts/run_local.sh classical.txt                          # run one experiment locally
```

`uv sync` installs every Python library of the experiments from `uv.lock`, including the PolyStep optimizer (`polystep`; Le, 2026) on which this repository's `polystep_or` builds.

Portfolio, energy load 1, the in-constraint suite and brass alloy production use Gurobi and need a Gurobi licence; the other problems use HiGHS, OR-Tools or GLPK.

Every experiment is a 1-command line: `python -m experiments.<name>.run <command> [args]`, where `<name>` is a folder under `experiments/`. Results are written under `results/`.

### DistrictNet (Julia)
Run the following commands to install additional packages for DistrictNet:
```bash
curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.10
julia --project=third_party/DistrictNet -e 'using Pkg; Pkg.add(url="https://github.com/anindex/PolyStep.jl", rev="v0.2.0"); Pkg.instantiate()'
PYTHON=$PWD/.venv/bin/python julia --project=third_party/DistrictNet -e 'using Pkg; Pkg.build("PyCall")'
julia --project=third_party/DistrictNet third_party/DistrictNet/buildCpp.jl
```

1. Julia 1.10 through juliaup, the version of DistrictNet's `Manifest.toml`.
2. Installs DistrictNet's Julia packages and PolyStep.jl version 0.2.0.
3. Points PyCall to the uv environment, which holds the Python libraries DistrictNet imports (geopandas, shapely).
4. Compiles LKH, the cost evaluator and the scenario generator with `cmake` and a C++ compiler.

## Structure

```
src/polystep_or/   the package: Problem, Learner, run, Tuner, named configurations
experiments/       one folder per experiment (classical, constraint_suite, alloy, districtnet): run.py and config.yaml
patches/           changes to the third-party code, applied by scripts/setup_third_party.sh
scripts/           setup_third_party.sh, run_local.sh, slurm_array.sh
examples/          new_problem.py: adding a new problem
tests/             unit tests of the package and the example
LICENSE            MIT license
pyproject.toml     package metadata and pinned dependencies
uv.lock            locked versions of every Python dependency
third_party/       benchmark code and data (gitignored, regenerated or downloaded)
results/           experiment outputs (gitignored)
```

## Running example

```bash
python -m experiments.classical.run train --problem shortestpath --value 1 --seed 0
python -m experiments.constraint_suite.run train --cell sckp_50_penalty_5 --seed 0 --split 0
python -m experiments.alloy.run final --mode constant --seed 11 --reported
python -m experiments.districtnet.run train --method polyStep --seed 17
```

## Notes
Every PolyStepOR step runs the predictor forward on each probe candidate and then solves one optimization problem per instance, so the solver calls dominate the cost and run in parallel. You can set `--workers N` to increase the number of parallel solver processes (`train` of classical and constraint_suite, `tune` and `final` of alloy, Julia worker processes `julia -p N` of districtnet). The default settings are in each `config.yaml`.

We build on the PolyStep optimizer (Le, 2026), imported as a package in its Python and Julia versions. DistrictNet and the other benchmark repositories are upstream clones with a small, documented adapter layer. You can find the differences recorded in the `patches/` folder. 
