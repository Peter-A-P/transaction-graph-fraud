#!/usr/bin/env bash
# On a rented pod, from the unpacked repository with its .env (RUNPOD_API_KEY, RUNPOD_POD_ID,
# KAGGLE_API_TOKEN) copied beside it. Order matters:
#   1. the watchdog, before anything else, with the image's own Python (it needs only the
#      standard library), so a pod whose setup hangs or fails still deletes itself;
#   2. the card: driver, and a matrix multiply under the pinned torch, which a bad host fails
#      (06 rented two such hosts); the pod deletes itself if it does;
#   3. the environment and the data, each fetched and checked against the committed manifests.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" UV_LINK_MODE=copy PYTHONUNBUFFERED=1

nohup python3 -c "
from consort import env; env.load()
import os
from consort.ops import watchdog as wd
wd.watch(wd.RunPod(wd.key_from_env()), wd.Policy(idle_minutes=30, max_hours=${MAX_HOURS:-10}),
         self_id=os.environ['RUNPOD_POD_ID'])
" > /root/watchdog.log 2>&1 &
echo "watchdog pid $!"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen --extra gnn --extra export
uv run python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available under the pinned torch"
a = torch.randn(4096, 4096, device="cuda")
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), float((a @ a).sum()))
PY
for source in ${DATASETS:-elliptic ellipticpp ellipticpp-txs ibm-HI-Small ibm-LI-Small}; do
  uv run consort data fetch "$source"
done
echo SETUPDONE
