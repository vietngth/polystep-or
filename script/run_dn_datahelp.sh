#!/bin/bash
# "Does data help?" controlled experiment: for each 120-BU city, solve BOTH untrained (random GNN)
# and trained models, then SAA-evaluate each -> untrained-vs-trained TRUE routing cost delta.
# Untrained districtNet == untrained polyStep (sanity); trained should be markedly cheaper if data helps.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}/DistrictNet"
export PATH="$HOME/.juliaup/bin:$PATH"
JL="julia +1.10 --project=."
CITIES="${DH_CITIES:-Manchester London Bristol Leeds}"
T=20; BU=120; ND=10
for city in $CITIES; do
  for mode in untrained trained; do
    if [ "$mode" = untrained ]; then export DN_UNTRAINED=1; else export DN_UNTRAINED=0; fi
    for method in districtNet polyStep; do
      echo "=== SOLVE $city $mode $method (DN_UNTRAINED=$DN_UNTRAINED) ==="
      NTH=1 JULIA_NUM_GC_THREADS=1 $JL experiments.jl 2 "$city" "$method" solve $T $BU C $ND || echo "SOLVE FAILED $city $mode $method"
    done
    echo "=== DATAHELP-EVAL $city $mode ==="
    NTH=1 JULIA_NUM_GC_THREADS=1 $JL datahelp_eval.jl "$city" $T $BU $ND || echo "EVAL FAILED $city $mode"
  done
done
echo "=== run_dn_datahelp DONE ==="
