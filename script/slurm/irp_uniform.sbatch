#!/bin/bash
#SBATCH --job-name=repro-irp_uniform
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -uo pipefail
# DSIRP head-to-head, UNIFORM demand pattern (third pattern after normal/bimodal).
# PolyStep/SFGE are label-free (no retrain), but the "Toni" (Greif et al.) column needs DAgger-trained
# GLM weights on disk. Those exist for normal/bimodal but NOT uniform, so we (1) train Toni's faithful
# Julia pipeline on the SAME 5 uniform-10 instances the head-to-head evaluates (sorted glob, first 5),
# look_ahead=3 -> 36-dim weights matching IRP_LOOK_AHEAD=3, then (2) run the Python 4-way head-to-head.
# Weights land in InferOpt_DSIRP/training/solutions/dagger/uniform/penalty_200/<instance>/.
ROOT="${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"; cd "$ROOT"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="$ROOT/julia-depot:$HOME/.julia"   # InferOpt_DSIRP deps (ArgParse, ...) live here
export JULIA_NUM_GC_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"   # set per convention (training uses HiGHS)

# The instances exp_irp_headtohead.py evaluates: sorted(glob("uniform-10_*.json"))[:IRP_N_INST], stem == instance_id.
# Reduced from 5 to the first 3: per-instance DAgger training is ~4.5h single-node, so 5 would not clear the
# 24h wall. IRP_N_INST=3 below keeps the head-to-head aligned to exactly these 3 trained instances.
INSTANCES=(
  uniform-10_202212-1314-0054-8985d3c2-4e23-4d82-81b9-5da4bc03369e
  uniform-10_202212-1314-0054-b3c1d321-3484-4e23-97ff-e7dc017f0c8a
  uniform-10_202212-1314-0054-e4963fca-0a80-4e36-a2d9-ca32560849ba
)

echo "=== [irp_uniform] host=$(hostname) @ $(date) | train Toni(dagger, look_ahead=3, penalty=200, HiGHS) on 3 uniform instances, then 4-way head-to-head ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(cpu node)"

cd "$ROOT/InferOpt_DSIRP"
i=0
for ID in "${INSTANCES[@]}"; do
  i=$((i+1))
  echo "--- [train Toni $i/3] $ID @ $(date) ---"
  julia +1.10 -t 1 --project=. train_pipeline.jl \
    --instance_id "$ID" \
    --paradigm dagger \
    --shortage_penalty 200 \
    --look_ahead 3 \
    --milp_solver gurobi 2>&1
  echo "--- [train Toni $i/3] $ID rc=$? @ $(date) ---"
done

cd "$ROOT"
echo "=== [head-to-head] IRP_PATTERN=uniform @ $(date) ==="
CACHE="$PWD" IRP_N_INST=3 IRP_PATTERN=uniform IRP_LOOK_AHEAD=3 IRP_PR=0.5 IRP_SR=0.6 \
  IRP_SFGE_SIGMA=150 IRP_SFGE_LR=0.05 IRP_SFGE_NSAMPLE=32 \
  PS_POLYTOPE=orthoplex PS_PROBES=1 IRP_SEED=0 OUT_TAG=sfge4way \
  .venv/bin/python exp_irp_headtohead.py full
rc=$?
echo "=== [irp_uniform] head-to-head rc=$rc @ $(date) ==="
echo "--- results tail ---"
cat exp_results/irp_headtohead_full_uniform*.md 2>/dev/null || true
tail -c 4000 exp_results/irp_headtohead_full_uniform*.json 2>/dev/null || echo "(no json produced)"
exit $rc
