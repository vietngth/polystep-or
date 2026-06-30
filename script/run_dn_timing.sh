#!/bin/bash
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT/DistrictNet"
export PATH="$HOME/.juliaup/bin:$PATH"; export JULIA_DEPOT_PATH="$ROOT/julia-depot:$HOME/.julia"
NTH=${NTH:-10}; export DN_NBDATA=${DN_NBDATA:-30}; export DN_EPOCHS=${DN_EPOCHS:-5}
echo "=== DistrictNet timing run: nb_data=$DN_NBDATA epochs=$DN_EPOCHS threads=$NTH ==="
julia +1.10 -t "$NTH" --project=. time_districtnet.jl
echo "=== dn_timing done ==="
