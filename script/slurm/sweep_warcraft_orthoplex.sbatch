#!/bin/bash
#SBATCH --job-name=sweep_warcraft_orthoplex
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
POLY=orthoplex; PD=8
echo "=== sweep_warcraft_$POLY on $(hostname) @ $(date) | 3 seeds x polytope=$POLY (q=512 steps=400 pdim=$PD) ==="
for S in 0 1 2; do
  echo "---- warcraft full: polytope=$POLY pdim=$PD seed=$S ----"
  t0=$(date +%s)
  PS_POLYTOPE=$POLY WC_PDIM=$PD PS_PROBES=1 WC_SEED=$S OUT_TAG=${POLY}_p1_s${S} \
    .venv/bin/python exp_warcraft.py full
  echo "---- $POLY seed=$S rc=$? wall=$(( $(date +%s)-t0 ))s -> exp_results/warcraft_full_${POLY}_p1_s${S}.json ----"
done
echo "=== sweep_warcraft_$POLY done @ $(date) ==="
