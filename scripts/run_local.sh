#!/usr/bin/env bash
set -euo pipefail

LIST="${1:?usage: scripts/run_local.sh COMMANDS_FILE [PARALLEL_JOBS]}"
JOBS="${2:-1}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
grep -v '^[[:space:]]*$' "$LIST" | xargs -d '\n' -P "$JOBS" -I CMD bash -c CMD
