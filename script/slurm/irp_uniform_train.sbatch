#!/bin/bash
#SBATCH --job-name=repro-irp_uniform_train
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -uo pipefail
# CPU-ONLY phase of the uniform DSIRP experiment: Toni (Greif et al.) DAgger training in Julia
# with a Gurobi anticipative MILP expert. This uses no GPU at all, so it runs on spare cores
# alongside the GPU campaign. The GPU head-to-head is the separate irp_uniform_eval.sbatch,
# gated on this job. Trained GLM weights land in
# InferOpt_DSIRP/training/solutions/dagger/uniform/penalty_200/<instance>/.
ROOT="${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"; cd "$ROOT"
export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="$ROOT/julia-depot:$HOME/.julia"
export JULIA_NUM_GC_THREADS=1
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"

# The instances the head-to-head evaluates: sorted(glob("uniform-10_*.json"))[:IRP_N_INST=3].
INSTANCES=(
  uniform-10_202212-1314-0054-8985d3c2-4e23-4d82-81b9-5da4bc03369e
  uniform-10_202212-1314-0054-b3c1d321-3484-4e23-97ff-e7dc017f0c8a
  uniform-10_202212-1314-0054-e4963fca-0a80-4e36-a2d9-ca32560849ba
)
echo "=== [irp_uniform_train] host=$(hostname) @ $(date) | CPU-only Toni DAgger (look_ahead=3, penalty=200, gurobi) on 3 uniform instances ==="
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
echo "=== [irp_uniform_train] done @ $(date) ==="
