#!/bin/bash
#SBATCH --job-name=irp_sfge_normal
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
echo "=== irp_sfge_normal on $(hostname) @ $(date) | 4-way head-to-head (untrained/Toni-FY/PolyStep/SFGE) ==="
CACHE="$PWD" IRP_PATTERN=normal IRP_LOOK_AHEAD=3 IRP_PR=0.5 IRP_SR=0.6 \
  IRP_SFGE_SIGMA=150 IRP_SFGE_LR=0.05 IRP_SFGE_NSAMPLE=32 \
  PS_POLYTOPE=orthoplex PS_PROBES=1 IRP_SEED=0 OUT_TAG=sfge4way \
  .venv/bin/python exp_irp_headtohead.py full
echo "=== irp_sfge_normal done (rc=$?) @ $(date) -> exp_results/irp_headtohead_full_normal_sfge4way.json ==="
