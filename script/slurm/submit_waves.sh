#!/usr/bin/env bash
# Submit every experiment SLURM spec in throttled, dependency-chained waves so the
# shared node is not flooded. Each wave starts only after the previous wave finishes.
# Run from the repo root on the cluster:  bash script/slurm/submit_waves.sh [WAVE_SIZE]
# Job ids and the wave map are logged to results/wave_submission.log.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
WAVE_SIZE="${1:-8}"
LOG="$REPO/results/wave_submission.log"
mkdir -p "$REPO/results"; : > "$LOG"

# every spec except the smoke jobs and the orchestrator itself; skip irp_uniform if a copy is already running
mapfile -t SPECS < <(ls script/slurm/*.sbatch | grep -vE "/(smoke_|submit_waves)" | sort)
echo "$(date) | ${#SPECS[@]} specs, wave size $WAVE_SIZE" | tee -a "$LOG"

prev_ids=""
wave=0
i=0
while [ $i -lt ${#SPECS[@]} ]; do
  wave=$((wave+1))
  dep=""; [ -n "$prev_ids" ] && dep="--dependency=afterany:${prev_ids}"
  this_ids=""
  echo "--- wave $wave ${dep:+(after $prev_ids)} ---" | tee -a "$LOG"
  for ((k=0; k<WAVE_SIZE && i<${#SPECS[@]}; k++, i++)); do
    spec="${SPECS[$i]}"; name="$(basename "$spec" .sbatch)"
    jid=$(sbatch --parsable $dep "$spec" 2>>"$LOG")
    if [ -n "$jid" ]; then
      this_ids="${this_ids:+$this_ids:}$jid"
      echo "  $jid  $name" | tee -a "$LOG"
    else
      echo "  FAILED to submit $name" | tee -a "$LOG"
    fi
  done
  prev_ids="$this_ids"
done
echo "$(date) | submitted $i jobs across $wave waves" | tee -a "$LOG"
