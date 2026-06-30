#!/bin/bash
#SBATCH --job-name=smoke_wc_timing
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== smoke_wc_timing on $(hostname) @ $(date) ==="
for spec in "orthoplex 8" "cube 4" "simplex 8"; do
  set -- $spec; POLY=$1; PD=$2
  echo "########## warcraft full timing: polytope=$POLY pdim=$PD WC_MAIN_STEPS=12 ##########"
  t0=$(date +%s)
  PS_POLYTOPE=$POLY WC_PDIM=$PD PS_PROBES=1 WC_SEED=0 WC_MAIN_STEPS=12 OUT_TAG=tmg_${POLY} \
    .venv/bin/python exp_warcraft.py full
  echo "########## $POLY wall=$(( $(date +%s)-t0 ))s ; result: ##########"
  ls -la exp_results/warcraft_full_tmg_${POLY}.json 2>&1
done
echo "=== smoke_wc_timing done @ $(date) ==="
