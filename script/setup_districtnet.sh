#!/bin/bash
# login-node setup (install only): instantiate Julia env + build C++ libs (CxxWrap Evaluator/Scenario, LKH)
set -uo pipefail
ROOT="${1:-$HOME/ot-or-project}"; cd "$ROOT/DistrictNet"
export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="$ROOT/julia-depot:$HOME/.julia"
echo "=== tools ==="; which cmake gcc g++ make 2>/dev/null || echo "MISSING BUILD TOOLS"
echo "=== instantiate DistrictNet project (Flux/GNN/InferOpt/CxxWrap) ==="
julia +1.10 --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); println("DN_INSTANTIATE_OK")' || exit 3
echo "=== build C++ deps ==="
julia +1.10 --project=. buildCpp.jl && echo "DN_BUILDCPP_OK" || echo "DN_BUILDCPP_FAILED"
echo "=== built libs ==="; ls -la deps/Evaluator/build/*.so deps/Scenario/build/*.so 2>/dev/null
