#!/bin/bash
#SBATCH --job-name=repro-dn_trainatscale
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -uo pipefail
# DistrictNet "train-at-scale": label-free PolyStep trained directly on large real-city instances
# (target-20, 120-200 BU) where the exact districting label is intractable, vs DistrictNet/PolyStep
# trained small (30BU/t3) and deployed large. Self-contained; no GPU (CPU: Kruskal+ILS+GLPK+C++ SAA).
# GLPK is NOT thread-safe -> single Julia thread + single GC thread.
ROOT="${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"; cd "$ROOT/DistrictNet"
export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="$ROOT/julia-depot:$HOME/.julia"
export JULIA_NUM_GC_THREADS=1
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"   # set per convention (this run uses GLPK, not Gurobi)

# train-at-scale hyperparameters (defaults baked into the driver; override here for the heavy run)
export PS_SCALE_RETRAIN=1
export PS_SCALE_STEPS="${PS_SCALE_STEPS:-25}"
export PS_SCALE_PROBES="${PS_SCALE_PROBES:-10}"
export PS_SIZE_PENALTY="${PS_SIZE_PENALTY:-0.0}"      # set >0 (e.g. 200) for the feasibility-aware variant
export TAS_REGEN_SCEN=1                               # regenerate scenarios at NB_SCENARIO=100 (SAA consistency)
export PS_SCALE_ID="${PS_SCALE_ID:-trainatscale}"
# TAS_TRAIN / TAS_TEST default in the driver to:
#   train: London@200, Leeds@190, Bristol@160, Manchester@120, Lyon@120, Paris@120  (target-20)
#   test : Marseille@120, Birmingham@120 (held-out cities), London@300, Leeds@240 (held-out sizes)

echo "=== [dn_trainatscale] host=$(hostname) @ $(date) ==="
echo "steps=$PS_SCALE_STEPS probes=$PS_SCALE_PROBES size_penalty=$PS_SIZE_PENALTY id=$PS_SCALE_ID"
julia +1.10 -t 1 --project=. trainatscale_driver.jl 2>&1
rc=$?
echo "=== [dn_trainatscale] rc=$rc @ $(date) ==="
echo "--- results ---"; cat exp_results/districtnet_trainatscale.md 2>/dev/null || echo "(no md)"
exit $rc
