#!/bin/bash
#SBATCH --job-name=repro-dn_scale_morecities
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -uo pipefail
# Experiment 1: firm the label-free-vs-DistrictNet comparison with MORE cities and MORE scenarios.
# Train PolyStep at scale (label-free) on a broad city set at moderate size, NB_SCENARIO=100 (vs 50),
# evaluate on held-out cities/sizes. More instances + lower-variance SAA -> statistical power on the tie.
ROOT="${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"; cd "$ROOT/DistrictNet"
export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="$ROOT/julia-depot:$HOME/.julia"
export JULIA_NUM_GC_THREADS=1
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
export PS_SCALE_RETRAIN=1
export PS_SCALE_STEPS="${PS_SCALE_STEPS:-25}"
export PS_SCALE_PROBES="${PS_SCALE_PROBES:-10}"
export PS_SIZE_PENALTY="${PS_SIZE_PENALTY:-0.0}"
export TAS_REGEN_SCEN=1
export TAS_NB_SCENARIO=100
export PS_SCALE_ID=morecities
export TAS_TRAIN="London:120,Leeds:120,Bristol:120,Manchester:120,Lyon:120,Paris:120,Birmingham:120"
export TAS_TEST="Marseille:120,London:200,Leeds:150"
echo "=== [dn_scale_morecities] host=$(hostname) @ $(date) ==="
echo "train=$TAS_TRAIN  test=$TAS_TEST  nb_scen=$TAS_NB_SCENARIO  id=$PS_SCALE_ID"
julia +1.10 -t 1 --project=. trainatscale_driver.jl 2>&1
rc=$?
echo "=== [dn_scale_morecities] rc=$rc @ $(date) ==="
echo "--- results ---"; cat "exp_results/districtnet_trainatscale_${PS_SCALE_ID}.md" 2>/dev/null || echo "(no md)"
exit $rc
