#!/bin/bash
#SBATCH --job-name=repro-trainatscale_combined
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -uo pipefail
# CPU-only: at-scale PolyStep trained on BIG (target-20) + SMALL (target-3) origin instances combined.
# No GPU (pure-CPU Flux + C++ SAA routing evaluator). Result -> results/trainatscale_combined.md
ROOT="${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"; cd "$ROOT"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="$ROOT/julia-depot:$HOME/.julia"
export JULIA_NUM_GC_THREADS=1
export TAS_NB_SCENARIO="${TAS_NB_SCENARIO:-100}"
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
cd "$ROOT/DistrictNet"
echo "=== [trainatscale_combined] host=$(hostname) @ $(date) ==="
julia +1.10 -t 1 --project=. trainatscale_combined.jl 2>&1
echo "=== [trainatscale_combined] rc=$? @ $(date) ==="
