#!/bin/bash
#SBATCH --job-name=repro-dn_scale_bigger
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -uo pipefail
# Experiment 2: combine with BIGGER instances where the optimal-decision label is intractable, so only
# the label-free (realized-cost) route applies. Train PolyStep directly at large sizes, NB_SCENARIO=100,
# and test generalization to held-out LARGER sizes and a held-out city. DistrictNet here can only
# train-small-deploy-large (its label is not solvable at these sizes).
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
export PS_SCALE_ID=bigger
export TAS_TRAIN="London:400,Leeds:290,Bristol:260,Manchester:120,Lyon:120,Paris:120"
export TAS_TEST="London:600,Leeds:240,Bristol:210,Marseille:120"
echo "=== [dn_scale_bigger] host=$(hostname) @ $(date) ==="
echo "train=$TAS_TRAIN  test=$TAS_TEST  nb_scen=$TAS_NB_SCENARIO  id=$PS_SCALE_ID"
julia +1.10 -t 1 --project=. trainatscale_driver.jl 2>&1
rc=$?
echo "=== [dn_scale_bigger] rc=$rc @ $(date) ==="
echo "--- results ---"; cat "exp_results/districtnet_trainatscale_${PS_SCALE_ID}.md" 2>/dev/null || echo "(no md)"
exit $rc
