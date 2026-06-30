#!/bin/bash
# ABLATION: UNTRAINED (random GNN) -> CMST inference for BOTH methods on the paper's instances.
# Saves <sol>.untrained.<method>. districtNet-untrained should == polyStep-untrained (same arch+seed).
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT/DistrictNet"
export PATH="$HOME/.juliaup/bin:$PATH"; export JULIA_DEPOT_PATH="$ROOT/julia-depot:$HOME/.julia"
export JULIA_NUM_GC_THREADS=1 DN_UNTRAINED=1 DN_SEED=1234
D=output/solution/Experiment_General_multisize_cities
for spec in "Manchester 120" "London 120" "Bristol 120" "Leeds 120" "ilsdefrance 2000"; do
  set -- $spec; CITY=$1; BU=$2
  for METHOD in districtNet polyStep; do
    SOL="$D/${CITY}_C_${BU}_20.${METHOD}.txt"
    [ -f "$SOL" ] && cp "$SOL" "${SOL}.trainedkeep"
    echo "=== UNTRAINED $CITY ${BU}BU $METHOD ==="
    julia +1.10 -t 1 --project=. experiments.jl 2 "$CITY" "$METHOD" solve 20 "$BU" C 10 2>&1 || echo "FAILED $CITY $METHOD"
    [ -f "$SOL" ] && mv "$SOL" "${SOL}.untrained"
    [ -f "${SOL}.trainedkeep" ] && mv "${SOL}.trainedkeep" "$SOL"
  done
done
echo "=== untrained ablation done ==="
