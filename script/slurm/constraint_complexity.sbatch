#!/bin/bash
#SBATCH --job-name=repro-constraint_complexity
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# Constraint-TYPE complexity ladder (L0 box -> L1 capacity -> L2 cardinality -> L3 conic -> L4 coupled).
# Self-contained: full grids (9 sigma x 6 probe_radius), 5 seeds. Gurobi used only as the per-level
# exactness reference for the batched greedy (needs GRB_LICENSE_FILE).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
echo "=== [constraint_complexity] on $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(cpu node)"
.venv/bin/python exp_constraint_complexity.py
rc=$?
echo "--- results ---"; cat exp_results/constraint_complexity.md 2>/dev/null | head -60 || echo "(no md)"
echo "=== done [constraint_complexity] rc=$rc @ $(date) ==="
exit $rc
