#!/bin/bash
#SBATCH --job-name=repro-polystep_vs_sfge
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
echo "=== polystep_vs_sfge (probe SFGE's 3 weaknesses: var-vs-dim, HP-sensitivity, sample-eff) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(cpu node)"
KNAP_CHUNK="${KNAP_CHUNK:-4096}" .venv/bin/python exp_polystep_vs_sfge.py
rc=$?
echo "=== polystep_vs_sfge done rc=$rc @ $(date) ==="
exit $rc
