#!/bin/bash
# Train PolyStep WITH the size-penalty constraint (PS_SIZE_PENALTY) on the small cities, then infer the
# big instance (ilsdefrance) and report feasibility + cost. Single-threaded (GLPK is not thread-safe).
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT/DistrictNet"
export PATH="$HOME/.juliaup/bin:$PATH"; export JULIA_DEPOT_PATH="$ROOT/julia-depot:$HOME/.julia"
export JULIA_NUM_GC_THREADS=1
export PS_SIZE_PENALTY="${PS_SIZE_PENALTY:-200}"
NB="${NB:-200}"
M="models/GeneralPolyStep_${NB}.jld2"
[ -f "$M" ] && mv "$M" "${M}.nopenalty"        # force retrain WITH penalty (don't clobber baseline)
echo "=== [PENALTY] train+infer: PS_SIZE_PENALTY=$PS_SIZE_PENALTY nb=$NB -> ilsdefrance (2000 BU, target 20) ==="
julia +1.10 -t 1 --project=. experiments.jl 2 ilsdefrance polyStep solve 20 2000 C "$NB" 2>&1
[ -f "$M" ] && mv "$M" "${M}.penalty${PS_SIZE_PENALTY}"   # keep penalty model
[ -f "${M}.nopenalty" ] && mv "${M}.nopenalty" "$M"       # restore baseline
echo "=== [PENALTY] done; penalty model saved as ${M}.penalty${PS_SIZE_PENALTY} ==="
