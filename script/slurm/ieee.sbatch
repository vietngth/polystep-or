#!/bin/bash
#SBATCH --job-name=repro-ieee
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
# CPU-only (Gurobi MILP); cluster auto-routes CPU-only jobs to the cpu partition.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$PATH"
export TQDM_DISABLE=1
# Gurobi licence (WLS). Adjust path if your licence lives elsewhere.
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
# Engine working dir off any shared/system tmp (storage rule). Bundled data lives in the MA&RE repo.
export IEEE_CACHE="${IEEE_CACHE:-$HOME/ot-or-project-cache/ieee_cis/IEEE-CIS Predict+Optimize}"
echo "=== [ieee] on $(hostname) @ $(date) ==="
echo "(cpu node — Gurobi MILP)"
MODE="${1:-full}"
echo ">>> exp_ieee_polystep.py $MODE"
.venv/bin/python exp_ieee_polystep.py "$MODE"
rc=$?
echo "=== done [ieee] rc=$rc @ $(date) ==="
exit $rc
