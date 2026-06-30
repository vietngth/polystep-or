#!/bin/bash
#SBATCH --job-name=repro-icon
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:1
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
echo "=== ICON energy-cost-aware scheduling (Mandi et al.) + PolyStep @ $(date) ==="
# pytorch_lightning is required by the authors' Energy code (pre-installed on this cluster via uv);
# install via uv if somehow missing (cluster .venv is uv-managed and has no pip).
.venv/bin/python -c "import pytorch_lightning" 2>/dev/null || uv pip install --python .venv/bin/python pytorch_lightning
# Paper-default scheduling setting: 3 machines x {10,15,20} tasks (instances 1/2/3), T=48 half-hour
# slots, real SEMO prices, Gurobi exact ILP, 20 epochs. SEEDS: the paper uses 10; SPO/DBB call the
# ILP once per training sample per epoch (very Gurobi-heavy), so we run 3 seeds here to finish within
# wall-clock (results are written incrementally per instance). To reproduce the full 10 seeds, set
# ICON_SEEDS=0,1,2,3,4,5,6,7,8,9 and raise --time.
ICON_INSTANCES="${ICON_INSTANCES:-1,2,3}" ICON_SEEDS="${ICON_SEEDS:-0,1,2}" ICON_EPOCHS="${ICON_EPOCHS:-20}" \
  ICON_PS_DAYS="${ICON_PS_DAYS:-40}" ICON_PS_STEPS="${ICON_PS_STEPS:-40}" \
  .venv/bin/python exp_icon_polystep.py
echo "=== ICON done @ $(date) ==="
