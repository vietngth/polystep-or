#!/bin/bash
#SBATCH --job-name=repro-sfge_suite
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
echo "=== sfge_suite (WSMC + KP-unknown-weights, predict-in-constraint) @ $(date) ==="
.venv/bin/python exp_sfge_suite.py 0,1,2,3,4
echo "=== sfge_suite done @ $(date) ==="
