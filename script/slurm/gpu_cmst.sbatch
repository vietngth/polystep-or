#!/bin/bash
#SBATCH --job-name=repro-gpu_cmst
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:1
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# PYTHONPATH covers repo root (exp script), script/ (gpu_cmst module) and the pristine PolyStep library.
export PYTHONPATH="$(pwd):$(pwd)/script:$(pwd)/polystep/src:${PYTHONPATH:-}"
STEPS="${STEPS:-40}"; SEED="${SEED:-0}"
echo "=== [gpu_cmst] batched GPU CMST vs CPU CMST on $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU)"
.venv/bin/python exp_gpu_cmst.py "$STEPS" "$SEED"
rc=$?
echo "=== done [gpu_cmst] rc=$rc @ $(date) ==="
exit $rc
