#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${POLYSTEP_OR_THIRD_PARTY:-$ROOT/third_party}"
PATCHES="$ROOT/patches"
PYTHON="${PYTHON:-python}"
read -r -a JULIA <<< "${JULIA:-julia}"
STAGE="${1:-all}"

PREDOPT_URL="${PREDOPT_BENCHMARKS_SOURCE:-https://github.com/PredOpt/predopt-benchmarks}"
PREDOPT_COMMIT=1a8e048c5aa640f73a05e29878b8e1e8f6f73610
SFGE_URL="${SFGE_DFL_SOURCE:-https://github.com/matsilv/sfge-dfl}"
SFGE_COMMIT=c62241ad8c186496751402606f68982abedafacc
ODECE_URL="${ODECE_SOURCE:-https://github.com/JayMan91/OdeceDFLforConstraintsNeurips25}"
ODECE_COMMIT=1b432e03ce9ded1e504df60fd4c44cf936334d4f
DISTRICTNET_URL="${DISTRICTNET_SOURCE:-https://github.com/cheikh025/DistrictNet}"
DISTRICTNET_COMMIT=7a624a33a6d04eac379b364c8b7895eaff206274
POLYSTEP_JL_COMMIT="v0.2.0"

RDR=https://rdr.kuleuven.be/api/access/datafile
BRASS=https://huggingface.co/datasets/JayMan91/OdeceBrassAlloy/resolve/main/brass.zip

fetch() {
    local name="$1" url="$2" commit="$3" dir="$THIRD_PARTY/$1"
    if [ ! -d "$dir/.git" ]; then
        git clone --quiet "$url" "$dir"
    fi
    git -C "$dir" checkout --quiet "$commit"
    local patch="$PATCHES/$name.patch"
    if [ -f "$patch" ] && ! git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
        git -C "$dir" apply "$patch"
    fi
    echo "$name at $commit"
}

source_dir() {
    case "$1" in
        predopt-benchmarks) echo "${PREDOPT_BENCHMARKS_SOURCE:-}" ;;
        sfge-dfl) echo "${SFGE_DFL_SOURCE:-}" ;;
        odece_neurips25) echo "${ODECE_SOURCE:-}" ;;
    esac
}

provide() {
    local name="$1" path="$2"
    shift 2
    local target="$THIRD_PARTY/$name/$path" src
    src="$(source_dir "$name")"
    if [ -e "$target" ]; then
        return
    fi
    mkdir -p "$(dirname "$target")"
    if [ -n "$src" ] && [ -e "$src/$path" ]; then
        ln -s "$(readlink -f "$src/$path")" "$target"
        echo "linked $name/$path"
        return
    fi
    "$@"
}

download_rdr() {
    local dir="$1" id="$2" tmp
    tmp="$(mktemp -d)"
    curl -L --fail -o "$tmp/data.tar.gz" "$RDR/$id"
    tar -xzf "$tmp/data.tar.gz" -C "$dir"
    rm -rf "$tmp"
}

download_brass() {
    local dir="$THIRD_PARTY/odece_neurips25/data/Alloy production" tmp
    tmp="$(mktemp -d)"
    curl -L --fail -o "$tmp/brass.zip" "$BRASS"
    unzip -q "$tmp/brass.zip" -d "$dir"
    rm -rf "$tmp"
}

generate_sfge_data() {
    local dir="$THIRD_PARTY/sfge-dfl" seeds="0 1 2 3 4" p
    (
        cd "$dir"
        export PYTHONPATH="$dir"
        "$PYTHON" data/generation_scripts/generate_stochastic_capacity_kp_data.py --input-dim 5 \
            --output-dim 50 --degree 5 --num-instances 1000 --multiplicative-noise 0.1 \
            --penalty 5 --correlate-values-weights 1 --rho 0 --seeds $seeds
        "$PYTHON" data/generation_scripts/generate_stochastic_weights_kp_data.py --input-dim 5 \
            --output-dim 50 --penalty 5 --relative-capacity 0.2 --degree 5 --num-instances 1000 \
            --multiplicative-noise 0.1 --additive-noise 0.03 --correlate-values-weights 1 --rho 0 \
            --seeds $seeds
        for p in 1 5 10; do
            "$PYTHON" data/generation_scripts/generate_wsmc_data.py --input-dim 5 --output-dim 5 \
                --num-sets 25 --num-products 5 --penalty "$p" --degree 5 --num-instances 1000 \
                --multiplicative-noise 0.5 --additive-noise 0.03 --seeds $seeds
            "$PYTHON" data/generation_scripts/generate_wsmc_data.py --input-dim 5 --output-dim 10 \
                --num-sets 50 --num-products 10 --penalty "$p" --degree 5 --num-instances 1000 \
                --multiplicative-noise 0.5 --additive-noise 0.03 --seeds $seeds
        done
    )
}

stage_clone() {
    mkdir -p "$THIRD_PARTY"
    fetch predopt-benchmarks "$PREDOPT_URL" "$PREDOPT_COMMIT"
    fetch sfge-dfl "$SFGE_URL" "$SFGE_COMMIT"
    fetch odece_neurips25 "$ODECE_URL" "$ODECE_COMMIT"
    fetch DistrictNet "$DISTRICTNET_URL" "$DISTRICTNET_COMMIT"
    ln -sfn "$ROOT/experiments/constraint_suite/polystep_method.py" \
        "$THIRD_PARTY/sfge-dfl/methods/polystep.py"
    ln -sfn "$ROOT/experiments/districtnet/polyStep.jl" \
        "$THIRD_PARTY/DistrictNet/src/Estimators/polyStep.jl"
}

stage_data() {
    local pb="$THIRD_PARTY/predopt-benchmarks"
    provide predopt-benchmarks ShortestPath/SyntheticData download_rdr "$pb/ShortestPath" 169872
    provide predopt-benchmarks Portfolio/SyntheticPortfolioData download_rdr "$pb/Portfolio" 169874
    provide predopt-benchmarks Matching/data download_rdr "$pb/Matching" 169871
    provide sfge-dfl data/data generate_sfge_data
    provide odece_neurips25 "data/Alloy production/brass" download_brass
}

stage_julia() {
    export POLYSTEP_JL_URL="${POLYSTEP_JL_URL:-https://github.com/anindex/PolyStep.jl}"
    (
        cd "$THIRD_PARTY/DistrictNet"
        "${JULIA[@]}" --project=. -e "using Pkg; Pkg.add(url=ENV[\"POLYSTEP_JL_URL\"], \
            rev=\"$POLYSTEP_JL_COMMIT\"); Pkg.instantiate(); Pkg.precompile()"
        bindir="$("${JULIA[@]}" -e 'print(Sys.BINDIR)')"
        PATH="$bindir:$PATH" "$bindir/julia" --project=. buildCpp.jl
    )
}

case "$STAGE" in
    clone) stage_clone ;;
    data) stage_data ;;
    julia) stage_julia ;;
    all) stage_clone; stage_data; stage_julia ;;
    *) echo "usage: $0 [clone|data|julia|all]" >&2; exit 2 ;;
esac
