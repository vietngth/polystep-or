#!/bin/bash
#SBATCH --job-name=repro-timing_smallscale
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -uo pipefail
# CPU-only: from-scratch small-scale (target-3) training wall time for DistrictNet original (FY) and
# PolyStep (Julia). No GPU (the DistrictNet Julia stack is pure-CPU Flux). Result -> results/timing_smallscale.md
ROOT="${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"; cd "$ROOT"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="$ROOT/julia-depot:$HOME/.julia"
export JULIA_NUM_GC_THREADS=1
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
cd "$ROOT/DistrictNet"
echo "=== [timing_smallscale] host=$(hostname) @ $(date) ==="
julia +1.10 -t 1 --project=. timing_smallscale.jl 2>&1
echo "=== [timing_smallscale] rc=$? @ $(date) ==="
