#!/bin/bash
#SBATCH --job-name=repro-spotree
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPOTREE_DATA="$PWD/baselines/SPOTree/data/ShortestPathData"
echo "=== [spotree] on $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU)"
# Paper-default 4x4-grid shortest path: n_train=200, deg in {2,10}, eps=0,
# max_depth=3, min_weights_per_node=20, 10 dataset replications.
.venv/bin/python exp_spotree_polystep.py \
  --n_train 200 --degs 2-10 --eps 0.0 --depth 3 --min_obs 20 \
  --reps_st 0 --reps_end 10 --steps 60 --particles 20 --seed 0
rc=$?
echo "=== done [spotree] rc=$rc @ $(date) ==="
exit $rc
