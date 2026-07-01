#!/usr/bin/env bash
# Submit every experiment SLURM spec under a ROLLING-WINDOW dependency so idle GPUs are
# used without flooding the shared node. Each job depends on the job WINDOW positions
# earlier in the submission order, so at most WINDOW jobs are unblocked ahead of
# completions: as one finishes, the next is released. Effective concurrency is
# min(WINDOW, free GPUs). This replaces the old rigid wave-barrier scheme, which stalled
# every pending job behind the slowest job of the previous wave.
#
# Run from the repo root on the cluster:  bash script/slurm/submit_waves.sh [WINDOW]
# Specs whose job name is already RUNNING or PENDING are skipped, so it is safe to rerun
# after cancelling only the pending jobs (the in-flight ones are left untouched).
# Job ids and the submission map are logged to results/wave_submission.log.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# sbatch propagates this environment to every job (default --export=ALL), so the clean
# package layout (pto at the root, polystep under polystep/src) resolves inside each job.
export PYTHONPATH="$REPO:$REPO/script:$REPO/polystep/src:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
WINDOW="${1:-6}"
LOG="$REPO/results/wave_submission.log"
mkdir -p "$REPO/results"; : > "$LOG"

# job names already in the queue (running or pending); skip these to avoid duplicates
active="$(squeue -u "$USER" -h -o '%j' 2>/dev/null)"

mapfile -t SPECS < <(ls script/slurm/*.sbatch | grep -vE "/(smoke_|submit_waves)" | sort)
echo "$(date) | ${#SPECS[@]} specs, rolling window $WINDOW" | tee -a "$LOG"

ids=()          # submitted job ids, in order
for spec in "${SPECS[@]}"; do
  name="$(basename "$spec" .sbatch)"
  jobname="$(grep -m1 -oE '#SBATCH --job-name=\S+' "$spec" | cut -d= -f2)"
  jobname="${jobname:-$name}"
  if grep -qxF "$jobname" <<<"$active"; then
    echo "  skip (already queued): $name" | tee -a "$LOG"; continue
  fi
  dep=""
  n=${#ids[@]}
  if (( n >= WINDOW )); then dep="--dependency=afterany:${ids[n-WINDOW]}"; fi
  jid=$(sbatch --parsable $dep "$spec" 2>>"$LOG")
  if [ -n "$jid" ]; then
    ids+=("$jid")
    echo "  $jid  $name ${dep:+<- afterany:${ids[n-WINDOW]}}" | tee -a "$LOG"
  else
    echo "  FAILED to submit $name" | tee -a "$LOG"
  fi
done
echo "$(date) | submitted ${#ids[@]} jobs, rolling window $WINDOW" | tee -a "$LOG"
