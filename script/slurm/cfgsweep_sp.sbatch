#!/bin/bash
#SBATCH --job-name=repro-cfgsweep_sp
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== [cfgsweep_sp] on $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU)"
.venv/bin/python exp_config_sweep.py sp orthoplex,simplex,cube 1,3,5,8 0,1,2,3,4
rc=$?
echo "=== done [cfgsweep_sp] rc=$rc @ $(date) ==="
exit $rc
