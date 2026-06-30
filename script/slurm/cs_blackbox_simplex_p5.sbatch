#!/bin/bash
#SBATCH --job-name=repro-cs_blackbox_simplex_p5
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:1
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
export PS_POLYTOPE=simplex PS_PROBES=5 OUT_TAG=simplex_p5
echo "=== [cs_blackbox_simplex_p5] on $(hostname) @ $(date)  PS_POLYTOPE=$PS_POLYTOPE PS_PROBES=$PS_PROBES OUT_TAG=$OUT_TAG ==="
.venv/bin/python blackbox_constraint.py 4 0,1,2,3,4 && .venv/bin/python blackbox_constraint.py 6 0,1,2,3,4
rc=$?
echo "=== done [cs_blackbox_simplex_p5] rc=$rc @ $(date) ==="
exit $rc
