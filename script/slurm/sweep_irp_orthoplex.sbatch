#!/bin/bash
#SBATCH --job-name=sweep_irp_orthoplex
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
POLY=orthoplex
echo "=== sweep_irp_$POLY on $(hostname) @ $(date) | 3 seeds x polytope=$POLY (pattern=normal LA=3) ==="
for S in 0 1 2; do
  echo "---- irp headtohead full: polytope=$POLY seed=$S ----"
  CACHE="$PWD" IRP_PATTERN=normal IRP_LOOK_AHEAD=3 IRP_PR=0.5 IRP_SR=0.6 \
    PS_POLYTOPE=$POLY PS_PROBES=1 IRP_SEED=$S OUT_TAG=${POLY}_p1_s${S} \
    .venv/bin/python exp_irp_headtohead.py full
  echo "---- wrote exp_results/irp_headtohead_full_normal_${POLY}_p1_s${S}.json (rc=$?) ----"
done
echo "=== sweep_irp_$POLY done @ $(date) ==="
