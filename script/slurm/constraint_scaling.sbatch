#!/bin/bash
#SBATCH --job-name=repro-constraint_scaling
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
echo "=== [constraint_scaling] on $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU)"
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python exp_constraint_scaling.py 1,2,4,8,16,32,64 0,1,2,3,4
rc=$?
echo "=== done [constraint_scaling] rc=$rc @ $(date) ==="
exit $rc
