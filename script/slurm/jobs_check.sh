#!/bin/bash
# Cluster job monitor — run locally; ssh's to the 4090 box. Summarizes queue + GPU utilization +
# node health, flags FAILED jobs (with error tails) and stuck/severe pending states, and fetches
# exp_results/ back. Drives the 15-min check.
#   bash remote/jobs_check.sh [minutes_window]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIN="${1:-20}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 gpu4090)

echo "===== QUEUE (counts by state) @ $(date '+%F %T') ====="
timeout 30 "${SSH[@]}" 'squeue -u $USER -h -o "%T" | sort | uniq -c; echo "total in queue: $(squeue -u $USER -h | wc -l)"'

echo ""
echo "===== NODE HEALTH + GPU allocation (watch for DOWN/DRAIN) ====="
timeout 25 "${SSH[@]}" 'scontrol show node vm-compute 2>/dev/null | grep -oiE "State=[^ ]+|CPULoad=[^ ]+|AllocTRES=[^ ]+" | tr "\n" "  "; echo'

echo ""
echo "===== SEVERE/STUCK pending reasons (ReqNodeNotAvail / DOWN / DRAINED / Reserved) ====="
timeout 25 "${SSH[@]}" 'squeue -u $USER -h -o "%.10i %.22j %R" | grep -iE "Nodes required|Down|Drain|Reserved|BadConstraints|launch failed" | head -10 || echo "(none — only normal Priority/Resources waits)"'

echo ""
echo "===== GPU UTILIZATION (via srun --overlap on running GPU jobs) ====="
timeout 60 "${SSH[@]}" 'for jid in $(squeue -u $USER -h -t RUNNING -o "%i %b" | grep "gpu:" | awk "{print \$1}" | head -4); do
  u=$(srun --overlap --jobid=$jid nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
  echo "job $jid -> $u"
done; echo "(if a GPU job sits at 0% util for many checks, it may be hung)"'

echo ""
echo "===== sacct last ${WIN} min: COMPLETED / FAILED / other (non-running) ====="
timeout 30 "${SSH[@]}" "sacct -u \$USER --starttime now-${WIN}minutes -o JobID,JobName%26,State,Elapsed -n 2>/dev/null | grep -vE '\.ba|\.ex|RUNNING|PENDING'"

echo ""
echo "===== ERROR TAILS for FAILED jobs in window ====="
timeout 70 "${SSH[@]}" "cd ot-or-project
faildids=\$(sacct -u \$USER --starttime now-${WIN}minutes -o JobID,State -n 2>/dev/null | grep -iE 'FAILED|TIMEOUT|OUT_OF_ME|CANCELLED|NODE_FAIL' | grep -vE '\.ba|\.ex' | awk '{print \$1}')
for j in \$faildids; do
  f=\$(ls -t *-\$j.err 2>/dev/null | head -1); [ -z \"\$f\" ] && f=\$(ls -t *-\$j.out 2>/dev/null | head -1)
  echo \"--- \$j (\$f) ---\"; tail -4 \"\$f\" 2>/dev/null | cut -c1-160
done
[ -z \"\$faildids\" ] && echo '(no failures in window)'"

echo ""
echo "===== FETCH exp_results/ back ====="
mkdir -p "$HERE/../exp_results"
timeout 150 rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=15" gpu4090:ot-or-project/exp_results/ "$HERE/../exp_results/" 2>&1 | tail -2
echo "=== jobs_check done @ $(date '+%F %T') ==="
