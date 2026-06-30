#!/bin/bash
#SBATCH --job-name=repro-amazon
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/ot-or-project}"
export PATH="$HOME/.local/bin:$PATH"
export TQDM_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
echo "=== Amazon Last-Mile Routing Challenge (ALMRRC 2021) + PolyStep @ $(date) ==="
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"
# CPU/solver-bound: Google OR-tools per-zone TSP (~2 s/route) + rc-cli ERP scorer; no GPU.
# Reference solver (aro.model.zone_utils.zone_based_tsp) + PPM trainer + rc-cli score.py reused VERBATIM.
# PolyStep replaces only the PPM zone-preference model with a learned zone-transition affinity, minimising
# the deployed route's REALISED official rc-cli submission_score. Route-solves parallelised over the CPUs.
# Train on a SUBSET (800 routes) for tractability; score on the FULL evaluation split (official metric).
.venv/bin/python -u exp_amazon_polystep.py \
  --n_train 800 --n_eval full --steps 35 --hid 0 --batch 120 --n_probe 1 \
  --procs "${SLURM_CPUS_PER_TASK:-16}"
echo "=== Amazon done @ $(date) ==="
