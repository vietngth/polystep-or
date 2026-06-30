#!/bin/bash
# Capture RAW (pre-repair) districtings to expose the orphan, for the before/after figures.
# Saves <sol>.raw without clobbering the existing feasible (after) solution.
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT/DistrictNet"
export PATH="$HOME/.juliaup/bin:$PATH"; export JULIA_DEPOT_PATH="$ROOT/julia-depot:$HOME/.julia"
export JULIA_NUM_GC_THREADS=1 DN_NO_REPAIR=1
D=output/solution/Experiment_General_multisize_cities
for spec in "Bristol 120 polyStep" "ilsdefrance 2000 districtNet"; do
  set -- $spec; CITY=$1; BU=$2; METHOD=$3
  SOL="$D/${CITY}_C_${BU}_20.${METHOD}.txt"
  [ -f "$SOL" ] && cp "$SOL" "${SOL}.afterfix.keep"
  echo "=== RAW (no repair) $CITY ${BU}BU $METHOD ==="
  julia +1.10 -t 1 --project=. experiments.jl 2 "$CITY" "$METHOD" solve 20 "$BU" C 10 2>&1 || echo "FAILED $CITY $METHOD"
  [ -f "$SOL" ] && mv "$SOL" "${SOL}.raw"
  [ -f "${SOL}.afterfix.keep" ] && mv "${SOL}.afterfix.keep" "$SOL"
  echo "=== saved ${SOL}.raw ==="
done
echo "=== raw done ==="
