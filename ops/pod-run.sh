#!/usr/bin/env bash
# From the laptop: rent one pod, run ops/pod-setup.sh then ops/pod-queue.sh on it, copy the
# outcomes back into data/runs, and delete the pod. The pod's own watchdog deletes it if this
# script or the laptop goes away; this script deletes it on every exit path it controls.
#
#   NAME=consort-gnn-1 bash ops/pod-run.sh
#   NAME=consort-serve-1 CLOUD=SECURE SERVE=1 RUNS_FILE=ops/serve-runs.txt bash ops/pod-run.sh
#
# GPUS lists the cards to try, cheapest first (docs/gpu.md); the first one in supply is rented.
#
# The code goes over ssh as a git archive of HEAD (it was written while the repository was private;
# no GitHub token goes on a rented machine), and the pod's .env carries only the RunPod key,
# the pod's own id and the Kaggle token. Nothing here prints them.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${NAME:-consort-gnn-1}"
CLOUD="${CLOUD:-COMMUNITY}"
GPUS="${GPUS:-NVIDIA RTX A5000,NVIDIA GeForce RTX 3090,NVIDIA RTX 5000 Ada Generation,NVIDIA RTX A6000,NVIDIA A40}"
KEY="${KEY:-$HOME/.ssh/runpod_smallprint}"
mkdir -p data/pod-logs

# On every exit, terminate every pod with this run's name: a launch that failed half way
# (2026-11-01: two pods outlived a failed launch) leaves nothing running.
sweep() { uv run --no-sync consort pod-down --name "$NAME" || echo "SWEEP FAILED: check the console"; }
trap 'echo "exit: sweeping pods named $NAME"; sweep' EXIT
info=""
IFS=, read -ra cards <<< "$GPUS"
# Community stock comes and goes by the hour: try the list up to TRIES times, WAIT seconds apart.
for attempt in $(seq 1 "${TRIES:-1}"); do
  for GPU in "${cards[@]}"; do
    if info=$(uv run --no-sync consort pod-up --name "$NAME" --gpu "$GPU" --cloud "$CLOUD" 2>data/pod-logs/pod-up.err); then break 2; fi
    info=""
    if ! grep -q "SUPPLY_CONSTRAINT\|no longer any instances\|does not have the resources" data/pod-logs/pod-up.err; then
      echo "pod-up failed on $GPU:"; tail -5 data/pod-logs/pod-up.err; exit 1
    fi
  done
  echo "attempt $attempt, $(date -u +%FT%TZ): no card in supply"
  [ "$attempt" -lt "${TRIES:-1}" ] && sleep "${WAIT:-1200}"
done
[ -n "$info" ] || { echo "no card in supply after ${TRIES:-1} attempts"; exit 1; }
read -r id host port cost < <(python -c "import json,sys; d=json.loads(sys.argv[1]); print(d['id'], d['host'], d['port'], d['cost_per_hr'])" "$info")
echo "pod $id ($GPU, US\$$cost/h) at $host:$port, $(date -u +%FT%TZ)"

SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=30 -p "$port" "root@$host")
# A fresh pod's sshd sometimes resets the first connections (2026-11-02): each step is retried.
retry() { for i in 1 2 3 4 5 6; do "$@" && return 0; echo "retry $i: $1"; sleep 20; done; return 1; }
for i in $(seq 1 20); do "${SSH[@]}" true 2>/dev/null && break; sleep 15; done
push_code() { git archive --format=tar HEAD | "${SSH[@]}" "mkdir -p /root/consort && tar -x -C /root/consort"; }
push_env() {
  {
    grep -E '^(RUNPOD_API_KEY|KAGGLE_API_TOKEN)=' .env
    echo "RUNPOD_POD_ID=$id"
    echo "MACHINE='RunPod $CLOUD $GPU'"
    echo "PRICE=$cost"
    echo "PRICE_DATE=$(date -u +%F)"
  } | "${SSH[@]}" "cat > /root/consort/.env && chmod 600 /root/consort/.env"
}
retry push_code
retry push_env
SETUP="DATASETS='${DATASETS:-}' bash ops/pod-setup.sh"
[ -n "${SERVE:-}" ] && SETUP="$SETUP && bash ops/serve-setup.sh"
# The queue runs in its own session with no terminal, so it outlives the ssh call that
# started it; and the start counts only once the log has a line in it. On 2026-11-02 a pod sat
# idle for an hour and a half with an empty log, its queue never started, its watchdog with
# it: `test -f` had taken the empty log for a running queue.
start() {
  "${SSH[@]}" "cd /root/consort && test -s /root/run.log || { setsid nohup bash -c '$SETUP && RUNS_FILE=${RUNS_FILE:-ops/gnn-runs.txt} bash ops/pod-queue.sh; echo ALLDONE' > /root/run.log 2>&1 < /dev/null & disown; }" &&
    for _ in 1 2 3 4 5 6; do sleep 10; "${SSH[@]}" "test -s /root/run.log" && return 0; done
  echo "the queue left no log line within a minute"; return 1
}
retry start

# Copy the log and the packed outcomes back on every poll, so whatever finished is on the
# laptop even if the pod is lost (its watchdog ends it after 10 hours).
pull() {
  scp -q -i "$KEY" -P "$port" "root@$host:/root/run.log" "data/pod-logs/$NAME.log" 2>/dev/null
  if "${SSH[@]}" "test -f /root/results.tar" 2>/dev/null; then
    scp -q -i "$KEY" -P "$port" "root@$host:/root/results.tar" "data/pod-logs/$NAME.tar" 2>/dev/null &&
      tar -xf "data/pod-logs/$NAME.tar"  # data/runs, and for SERVE data/serve-runs and results
  fi
}
misses=0
until "${SSH[@]}" "grep -q ALLDONE /root/run.log" 2>/dev/null; do
  if "${SSH[@]}" true 2>/dev/null; then misses=0; pull || true; else misses=$((misses + 1)); fi
  if [ "$misses" -ge 10 ]; then echo "pod unreachable for 10 polls; giving up"; exit 1; fi
  sleep 120
done
retry pull
grep -E "^(ok|FAIL)" "data/pod-logs/$NAME.log" || true
echo "done $(date -u +%FT%TZ)"
