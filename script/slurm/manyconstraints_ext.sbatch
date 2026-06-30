#!/bin/bash
#SBATCH --job-name=repro-manyconstraints_ext
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
echo "=== [manyconstraints_ext] on $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU)"
MC_SEEDS=0,1,2,3,4 MC_MRES=1,2,4,8,16,32,64 .venv/bin/python exp_manyconstraints.py
rc=$?
echo "=== done [manyconstraints_ext] rc=$rc @ $(date) ==="
exit $rc
