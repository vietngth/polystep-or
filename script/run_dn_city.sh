#!/bin/bash
# Re-run a single test city's PolyStep deployment with the CURRENT cascade-repair (PS_SIZE_PENALTY=0),
# loading a copied baseline model (no retrain) under a separate id to avoid clobbering nb=200.
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT/DistrictNet"
export PATH="$HOME/.juliaup/bin:$PATH"; export JULIA_DEPOT_PATH="$ROOT/julia-depot:$HOME/.julia"; export JULIA_NUM_GC_THREADS=1
CITY="${CITY:-Bristol}"; BU="${BU:-120}"; NB="${NB:-999}"
M="models/GeneralPolyStep_${NB}.jld2"
if [ ! -f "$M" ]; then
  for src in models/GeneralPolyStep_200.jld2.nopenalty models/GeneralPolyStep_200.jld2 models/GeneralPolyStep_200.jld2.penalty200; do
    [ -f "$src" ] && cp "$src" "$M" && echo "loaded model from $src" && break
  done
fi
echo "=== [CITY-FIX] $CITY ${BU}BU t20, cascade-repair, model id=$NB ==="
julia +1.10 -t 1 --project=. experiments.jl 2 "$CITY" polyStep solve 20 "$BU" C "$NB" 2>&1
echo "=== done $CITY $BU ==="
