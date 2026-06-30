#!/usr/bin/env bash
# One-line runner for every experiment in this repo.
#
#   ./run.sh list                 show every available experiment
#   ./run.sh setup                install the PolyStep library and harness deps
#   ./run.sh <name> [args...]     run experiment <name> locally
#   ./run.sh submit <name>        submit script/slurm/<name>.sbatch on a SLURM cluster
#
# <name> is the script stem under script/ (for example exp_warcraft, challenge_established,
# exp_fair_batched_spo). A few compound experiments set their canonical environment below.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"   # scripts use paths relative to the repo root (polystep/src, data/, results/)
export PYTHONPATH="$REPO:$REPO/script:$REPO/polystep/src:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8        # deterministic seeding (pto/seeding.py)
PY="${PYTHON:-python3}"
[ -x "$REPO/.venv/bin/python" ] && PY="$REPO/.venv/bin/python"

cmd="${1:-}"; shift || true

case "$cmd" in
  ""|-h|--help|help)
    sed -n '2,14p' "$0"; exit 0 ;;

  list)
    echo "Experiments (script/<name>.py), run with: ./run.sh <name>"
    ls "$REPO/script"/*.py | xargs -n1 basename | sed 's/\.py$//' | sort | column -c 100 2>/dev/null || \
    ls "$REPO/script"/*.py | xargs -n1 basename | sed 's/\.py$//' | sort
    echo; echo "SLURM specs (./run.sh submit <name>):"
    ls "$REPO/script/slurm"/*.sbatch 2>/dev/null | xargs -n1 basename | sed 's/\.sbatch$//'
    exit 0 ;;

  setup)
    echo "Installing PolyStep (editable) and harness dependencies into the active environment ..."
    $PY -m pip install -e "$REPO/polystep" && \
    $PY -m pip install -r "$REPO/requirements.txt"
    echo "Julia components (DistrictNet, InferOpt_DSIRP) build separately:"
    echo "  cd DistrictNet      && julia --project=. -e 'using Pkg; Pkg.instantiate()'"
    echo "  cd InferOpt_DSIRP   && julia --project=. setup_environment.jl"
    exit 0 ;;

  submit)
    name="${1:?usage: ./run.sh submit <name>}"
    spec="$REPO/script/slurm/${name}.sbatch"
    [ -f "$spec" ] || { echo "no SLURM spec: $spec"; exit 1; }
    exec sbatch "$spec" ;;
esac

# Compound experiments with a canonical environment. Results land under results/.
RESULTS="$REPO/results"; mkdir -p "$RESULTS"
case "$cmd" in
  exp_irp_headtohead)
    export IRP_N_INST="${IRP_N_INST:-5}" IRP_PATTERN="${IRP_PATTERN:-normal}" IRP_LOOK_AHEAD=3 \
           IRP_PR=0.5 IRP_SR=0.6 IRP_SFGE_SIGMA=150 IRP_SFGE_LR=0.05 IRP_SFGE_NSAMPLE=32 \
           PS_POLYTOPE=orthoplex PS_PROBES=1 IRP_SEED=0
    exec $PY "$REPO/script/exp_irp_headtohead.py" "${@:-full}" ;;
  *)
    script="$REPO/script/${cmd}.py"
    [ -f "$script" ] || { echo "unknown experiment '$cmd' (try ./run.sh list)"; exit 1; }
    exec $PY "$script" "$@" ;;
esac
