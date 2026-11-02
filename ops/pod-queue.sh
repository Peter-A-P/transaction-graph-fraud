#!/usr/bin/env bash
# Arm N's runs on the pod, one after another, each line of ops/gnn-runs.txt a train-gnn
# argument list, logged ok or FAIL with its time; a failed run does not stop the rest. At the end
# the outcomes are packed for the laptop to copy back; the watchdog removes the idle pod.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
while read -r args; do
  [ -z "$args" ] && continue
  case "$args" in \#*) continue ;; esac
  s=$(date +%s)
  if uv run consort train-gnn $args > /root/last-run.log 2>&1; then
    echo "ok $(( $(date +%s) - s ))s: $args"
  else
    echo "FAIL $(( $(date +%s) - s ))s: $args"; tail -5 /root/last-run.log
  fi
  tar -cf /root/results.tar data/runs  # after every run, so a cut-short queue keeps what it did
done < "${RUNS_FILE:-ops/gnn-runs.txt}"
echo QUEUEDONE
