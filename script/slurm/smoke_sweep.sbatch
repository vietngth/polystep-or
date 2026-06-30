#!/bin/bash
#SBATCH --job-name=smoke_sweep
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
echo "=== smoke_sweep on $(hostname) @ $(date) ==="

echo "###### WARCRAFT timing smoke (full data, WC_MAIN_STEPS=12) ######"
for spec in "orthoplex 8" "cube 4"; do
  set -- $spec; POLY=$1; PD=$2
  echo "---- warcraft full timing: polytope=$POLY pdim=$PD ----"
  PS_POLYTOPE=$POLY WC_PDIM=$PD PS_PROBES=1 WC_SEED=0 WC_MAIN_STEPS=12 OUT_TAG=smoke_${POLY} \
    .venv/bin/python exp_warcraft.py full
  echo "---- result file ----"; ls -la exp_results/warcraft_full_smoke_${POLY}.json 2>&1
done

echo "###### IRP head-to-head smoke (all 3 polytopes) ######"
for POLY in orthoplex simplex cube; do
  echo "---- irp headtohead smoke: polytope=$POLY ----"
  CACHE="$PWD" IRP_PATTERN=normal IRP_LOOK_AHEAD=3 IRP_PR=0.5 IRP_SR=0.6 \
    PS_POLYTOPE=$POLY PS_PROBES=1 IRP_SEED=0 OUT_TAG=smoke_${POLY} \
    .venv/bin/python exp_irp_headtohead.py smoke
  echo "---- result file ----"; ls -la exp_results/irp_headtohead_smoke_normal_smoke_${POLY}.json 2>&1
done
echo "=== smoke_sweep done @ $(date) ==="
