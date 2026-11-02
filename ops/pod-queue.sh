#!/usr/bin/env bash
# The pod's runs, one after another, each line of ops/gnn-runs.txt a consort command line
# (train-gnn, train-tabfm), logged ok or FAIL with its time; a failed run does not stop the rest. At the end
# the outcomes are packed for the laptop to copy back; the watchdog removes the idle pod.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
# Lines may name the machine and its price ($MACHINE, $PRICE, $PRICE_DATE), which the laptop
# wrote into .env; only those are read into the shell, never the keys.
eval "$(grep -E '^(MACHINE|PRICE|PRICE_DATE)=' .env | sed 's/^/export /')"
while read -r args; do
  [ -z "$args" ] && continue
  case "$args" in \#*) continue ;; esac
  s=$(date +%s)
  if eval "uv run consort $args" > /root/last-run.log 2>&1; then
    echo "ok $(( $(date +%s) - s ))s: $args"
  else
    echo "FAIL $(( $(date +%s) - s ))s: $args"; tail -5 /root/last-run.log
  fi
  # after every run, so a cut-short queue keeps what it did
  tar -cf /root/results.tar data/runs $(ls -d data/serve-runs results/serving.json 2>/dev/null)
done < "${RUNS_FILE:-ops/gnn-runs.txt}"
echo QUEUEDONE
