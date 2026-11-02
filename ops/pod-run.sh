#!/usr/bin/env bash
# From the laptop: rent one pod, run ops/pod-setup.sh then ops/pod-queue.sh on it, copy the
# outcomes back into data/runs, and delete the pod. The pod's own watchdog deletes it if this
# script or the laptop goes away; this script deletes it on every exit path it controls.
#
#   NAME=consort-gnn-1 GPU="NVIDIA RTX A5000" bash ops/pod-run.sh
#
# The code goes over ssh as a git archive of HEAD (the repository is private until it ships;
# no GitHub token goes on a rented machine), and the pod's .env carries only the RunPod key,
# the pod's own id and the Kaggle token. Nothing here prints them.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${NAME:-consort-gnn-1}"
GPU="${GPU:-NVIDIA RTX A5000}"
KEY="${KEY:-$HOME/.ssh/runpod_smallprint}"
mkdir -p data/pod-logs

info=$(uv run --no-sync consort pod-up --name "$NAME" --gpu "$GPU")
read -r id host port cost < <(python -c "import json,sys; d=json.loads(sys.argv[1]); print(d['id'], d['host'], d['port'], d['cost_per_hr'])" "$info")
echo "pod $id ($GPU, US\$$cost/h) at $host:$port, $(date -u +%FT%TZ)"
down() { uv run --no-sync consort pod-down "$id" || true; }
trap 'echo "stopping: deleting $id"; down' EXIT

SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=30 -p "$port" "root@$host")
for i in $(seq 1 20); do "${SSH[@]}" true 2>/dev/null && break; sleep 15; done
git archive --format=tar HEAD | "${SSH[@]}" "mkdir -p /root/consort && tar -x -C /root/consort"
{
  grep -E '^(RUNPOD_API_KEY|KAGGLE_API_TOKEN)=' .env
  echo "RUNPOD_POD_ID=$id"
} | "${SSH[@]}" "cat > /root/consort/.env && chmod 600 /root/consort/.env"
"${SSH[@]}" "cd /root/consort && nohup bash -c 'bash ops/pod-setup.sh && bash ops/pod-queue.sh; echo ALLDONE' > /root/run.log 2>&1 &"

misses=0
until "${SSH[@]}" "grep -q ALLDONE /root/run.log" 2>/dev/null; do
  if "${SSH[@]}" true 2>/dev/null; then misses=0; else misses=$((misses + 1)); fi
  if [ "$misses" -ge 10 ]; then echo "pod unreachable for 10 polls; giving up"; exit 1; fi
  sleep 120
done
scp -i "$KEY" -P "$port" "root@$host:/root/run.log" "data/pod-logs/$NAME.log"
grep -E "^(ok|FAIL)" "data/pod-logs/$NAME.log" || true
if "${SSH[@]}" "test -f /root/results.tar"; then
  scp -i "$KEY" -P "$port" "root@$host:/root/results.tar" "data/pod-logs/$NAME.tar"
  tar -xf "data/pod-logs/$NAME.tar"
fi
echo "done $(date -u +%FT%TZ)"
