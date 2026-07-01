#!/bin/bash
#SBATCH --job-name=repro-irp_uniform_eval
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
set -uo pipefail
# GPU phase of the uniform DSIRP experiment: the 4-way head-to-head (untrained, Toni-FY, PolyStep,
# SFGE). Only this phase uses the GPU, to run the batched CPCTSP rollout (irp_gpu.py). Submit with
# --dependency=afterok:<irp_uniform_train job> so it takes a GPU only once the CPU training finishes.
ROOT="${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"; cd "$ROOT"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="$ROOT/julia-depot:$HOME/.julia"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"

echo "=== [irp_uniform_eval] host=$(hostname) @ $(date) | IRP_PATTERN=uniform 4-way head-to-head on GPU ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "(no gpu visible)"
CACHE="$PWD" IRP_N_INST=3 IRP_PATTERN=uniform IRP_LOOK_AHEAD=3 IRP_PR=0.5 IRP_SR=0.6 \
  IRP_SFGE_SIGMA=150 IRP_SFGE_LR=0.05 IRP_SFGE_NSAMPLE=32 \
  PS_POLYTOPE=orthoplex PS_PROBES=1 IRP_SEED=0 OUT_TAG=sfge4way \
  .venv/bin/python exp_irp_headtohead.py full
rc=$?
echo "=== [irp_uniform_eval] head-to-head rc=$rc @ $(date) ==="
cat exp_results/irp_headtohead_full_uniform*.md 2>/dev/null || true
exit $rc
