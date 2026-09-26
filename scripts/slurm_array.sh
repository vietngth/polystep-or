#!/usr/bin/env bash
set -euo pipefail

LIST="${1:?usage: sbatch --array=1-N [options] scripts/slurm_array.sh COMMANDS_FILE}"
ROOT="${POLYSTEP_OR_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
INDEX="${SLURM_ARRAY_TASK_ID:?submit as a job array, one task per line of $LIST}"

cd "$ROOT"
COMMAND="$(grep -v '^[[:space:]]*$' "$LIST" | sed -n "${INDEX}p")"
if [ -z "$COMMAND" ]; then
    echo "no command at line $INDEX of $LIST" >&2
    exit 2
fi
echo "task $INDEX: $COMMAND"
bash -c "$COMMAND"
