# PolyStep for operations research

Gradient-free, label-free decision-focused learning with PolyStep, and its comparison against the optimization-oracle camp (SPO+, PFYL, IMLE, DBB) and the evaluation-oracle peer SFGE, across predict-then-optimize, prediction-in-constraint, districting, and inventory-routing problems.

## Quick start

```bash
./run.sh setup                       # install PolyStep (editable) + harness deps
./run.sh list                        # list every experiment
./run.sh challenge_established        # run one experiment locally
./run.sh exp_irp_headtohead full     # a compound experiment (canonical env preset)
./run.sh submit irp_uniform          # submit script/slurm/<name>.sbatch on a cluster
```

Every experiment is a single line: `./run.sh <name> [args]`, where `<name>` is a script stem under `script/`. Results are written under `results/`.

The Julia components build separately:

```bash
cd DistrictNet     && julia --project=. -e 'using Pkg; Pkg.instantiate()'
cd InferOpt_DSIRP  && julia --project=. setup_environment.jl
```

## Structure

```
polystep/          PolyStep library, pristine clone of github.com/anindex/polystep (the optimizer)
pto/               experiment harness: problems, solvers, baselines, multi-seed, budget, seeding
script/            every experiment as a runnable script; script/slurm/ holds the SLURM specs
DistrictNet/       clone of cheikh025/DistrictNet + minimal adapter (see DistrictNet/POLYSTEP_CHANGES.md)
InferOpt_DSIRP/    clone of tonigreif/InferOpt_DSIRP + minimal adapter (see InferOpt_DSIRP/POLYSTEP_CHANGES.md)
results/           experiment outputs (gitignored, synced from the server)
data/              datasets and instances (gitignored, regenerated or downloaded)
run.sh             one-line runner and SLURM submitter
```

## Provenance

PolyStep is used unmodified from its own repository. DistrictNet and the Toni DSIRP code are upstream clones with a small, documented adapter layer; each carries a `POLYSTEP_CHANGES.md` and a `POLYSTEP_CHANGES.patch` recording the exact differences. Benchmarks reuse the authors' code at the paper-default configuration; problems with no released code are implemented from the cited paper and marked as such in `docs/EXPERIMENTS.pdf`.
