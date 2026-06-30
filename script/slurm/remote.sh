#!/usr/bin/env bash
# Local->remote bridge for the 4090 SLURM box. Driven from the LOCAL Claude session.
#   ./remote/remote.sh probe          # inventory the cluster (scheduler, uv/julia, internet, storage)
#   ./remote/remote.sh sync           # rsync code up (no data/.venv/.git)
#   ./remote/remote.sh data           # one-time: rsync the big data/ dir up
#   ./remote/remote.sh submit <exp>   # sbatch a job (exp = combresnet_sup|combresnet|warcraft|districtnet|<path.py>)
#   ./remote/remote.sh status         # squeue for my jobs
#   ./remote/remote.sh logs [jobid]   # tail the newest %x-%j.out
#   ./remote/remote.sh watch <jobid>  # block until that job leaves the queue (one completion signal)
#   ./remote/remote.sh fetch          # rsync exp_results/ back down
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/remote.env"
SSH=(ssh -o StrictHostKeyChecking=accept-new "$REMOTE_HOST")
EXCL=(--exclude '.git' --exclude '.venv' --exclude 'env_dfl' --exclude 'env_*' --exclude '*env/'
      --exclude 'polystep-visualization' --exclude 'data' --exclude 'InferOpt_DSIRP' --exclude 'julia-depot' --exclude 'project-cache' --exclude '__pycache__'
      --exclude '*.tar.gz' --exclude '*.jld2.bak' --exclude '*.out' --exclude '*.err'
      --exclude '*.pdf' --exclude '*.log' --exclude 'exp_results/districtnet_avg/logs')

case "${1:-}" in
  probe)
    "${SSH[@]}" 'echo "host=$(hostname)"; echo "--scheduler--"; command -v sbatch squeue sacct seff 2>/dev/null;
      echo "--toolchain--"; for t in uv python3 julia juliaup gcc g++ make curl rsync git; do printf "%-9s " $t; (command -v $t||echo MISSING); done;
      echo "--internet(login)--"; (curl -sI -m6 https://pypi.org >/dev/null && echo pypi=yes)||echo pypi=no;
      echo "--storage--"; echo "HOME=$HOME"; df -h "$HOME" 2>/dev/null|tail -1; [ -d /scratch ]&&df -h /scratch|tail -1;
      echo "--gpu partitions/gres--"; sinfo -o "%P %G %D %t %m" 2>/dev/null|head -15;
      echo "--project dir?--"; ls -d ~/ot-or-project 2>/dev/null||echo "no project dir yet"' ;;
  sync)
    rsync -azP --delete "${EXCL[@]}" "$HERE/../" "$REMOTE_HOST:$REMOTE_DIR/"
    echo "synced code -> $REMOTE_HOST:$REMOTE_DIR" ;;
  data)
    "${SSH[@]}" "mkdir -p $REMOTE_DIR/data"
    rsync -azP "$HERE/../data/" "$REMOTE_HOST:$REMOTE_DIR/data/"
    echo "synced data/ (big, one-time)" ;;
  submit)
    shift; EXP="${1:-combresnet_sup}"
    "${SSH[@]}" "cd $REMOTE_DIR && sbatch --gres=$GRES --time=$TIME --cpus-per-task=$CPUS --mem=$MEM remote/run.sbatch '$EXP'" ;;
  status)
    "${SSH[@]}" 'squeue -u $USER -o "%.18i %.12P %.24j %.8T %.10M %.6D %R"' ;;
  logs)
    JOB="${2:-}"; PAT="${JOB:+*-$JOB.out}"; PAT="${PAT:-*.out}"
    "${SSH[@]}" "f=\$(ls -t $REMOTE_DIR/$PAT 2>/dev/null | head -1); echo \"== \$f ==\"; tail -n 80 \"\$f\"" ;;
  watch)
    JOB="${2:?usage: watch <jobid>}"
    "${SSH[@]}" "while squeue -h -j $JOB 2>/dev/null | grep -q $JOB; do sleep 30; done; echo \"job $JOB left the queue\"; sacct -nj $JOB --format=JobID,JobName,State,Elapsed,MaxRSS 2>/dev/null | head" ;;
  fetch)
    mkdir -p "$HERE/../exp_results"
    rsync -azP "$REMOTE_HOST:$REMOTE_DIR/exp_results/" "$HERE/../exp_results/"
    echo "fetched exp_results/ <- $REMOTE_HOST" ;;
  *)
    grep -E '^#( |$)' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -12; exit 1 ;;
esac
