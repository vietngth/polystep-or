#!/bin/bash
#SBATCH --job-name=repro-spo_convergence
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
echo "=== spo_convergence (SPO+/SFGE epoch sweep vs PolyStep ref) @ $(date) ==="
.venv/bin/python exp_spo_convergence.py sp,knap,tsp,port 5,10,20,40,80,160 0,1,2,3,4
echo "=== spo_convergence done @ $(date) ==="
