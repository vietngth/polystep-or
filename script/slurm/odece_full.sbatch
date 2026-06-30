#!/bin/bash
#SBATCH --job-name=repro-odece_full
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
echo "=== odece_full (predict-in-constraint MDKP capa+weight; PolyStep/SFGE live vs cited ODECE/MSE) @ $(date) ==="
.venv/bin/python exp_odece_full.py full capa,weight 11,12,13,14,15
echo "=== odece_full done @ $(date) ==="
