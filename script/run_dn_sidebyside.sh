#!/bin/bash
# Run feasibility_sidebyside on a city -> output/feasibility_sidebyside_<city>.json (per-case districts,
# sizes, cost, violated indices) for the before/after map figures.
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT/DistrictNet"
export PATH="$HOME/.juliaup/bin:$PATH"; export JULIA_DEPOT_PATH="$ROOT/julia-depot:$HOME/.julia"; export JULIA_NUM_GC_THREADS=1
export DN_MODEL_DNET=models/GeneralDistrictNet_10.jld2
export DN_MODEL_PSCOST=models/GeneralPolyStep_10.jld2
export DN_MODEL_PSPEN=models/GeneralPolyStep_10.jld2.penalty200
for spec in "ilsdefrance 2000 20" "Bristol 120 20"; do
  set -- $spec
  export DN_CITY="$1" DN_NBBU="$2" DN_TARGET="$3"
  echo "=== SIDEBYSIDE $DN_CITY ${DN_NBBU}BU t$DN_TARGET ==="
  julia +1.10 -t 1 --project=. feasibility_sidebyside.jl 2>&1 || echo "FAILED $DN_CITY"
  [ -f output/feasibility_sidebyside.json ] && mv output/feasibility_sidebyside.json "output/feasibility_sidebyside_${DN_CITY}.json"
done
echo "=== sidebyside done ==="
