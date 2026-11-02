# Rented GPUs

Arm N's full runs, TabFM inference and the serving benchmark run on rented machines; the
tabular arms and Elliptic's development runs stay on the laptop, which never reports a timing.

## Prices

Read from RunPod's API (`gpuTypes`) on 2026-11-01, US$ an hour, on-demand. The API returned no
interruptible price for any of these cards that day.

| Card | Memory (GB) | Community cloud | Secure cloud |
|---|---:|---:|---:|
| RTX A5000 | 24 | 0.16 | 0.27 |
| RTX 3090 | 24 | 0.22 | 0.50 |
| RTX 4090 | 24 | 0.34 | 0.89 |
| L4 | 24 | 0.44 | 0.59 |
| RTX A6000 | 48 | 0.33 | 0.59 |
| A40 | 48 | 0.35 | 0.59 |
| L40S | 48 | 0.79 | 1.09 |

**Choice.** Arm N on Elliptic, HI-Small and LI-Small and TabFM inference on a 24 GB card in the
community cloud (the A5000 or 3090, whichever is free); HI-Medium, if run, on a 48 GB A40 or
A6000. The serving benchmark on a secure-cloud pod, so the machine named beside the latency
numbers is one that can be rented again. These are below the plan's estimate (US$0.27 to 0.50
for the 24 GB class), so the estimate stands as an upper bound.

**The cloud comparison** (06's rule: take GCP or Azure spot if within 20% of the
marketplace), read 2026-11-01 from third-party price trackers, which Azure's and Google's own
pages should confirm before any rental there:

| Option | Spot US$ an hour | What it gives |
|---|---:|---|
| Azure `Standard_NC4as_T4_v3` | about 0.15 ([Vantage](https://instances.vantage.sh/azure/vm/nc4ast4-v3?currency=USD), [Spare Cores](https://sparecores.com/server/azure/Standard_NC4as_T4_v3)) | 16 GB T4, 4 vCPUs, 28 GB memory |
| GCP L4 | about 0.33 ([computeprices.com](https://computeprices.com/providers/google/gpus/l4)) | 24 GB L4, before the machine it is attached to |
| RunPod RTX A5000, Community | 0.16 (its API) | 24 GB, at least 8 vCPUs and 48 GB memory asked for |

The Azure T4 is within 20% on price but not on the machine: arm N's sampling is CPU-bound and
the IBM graphs need more than 28 GB, so the size that fits costs several times as much. GCP's
L4 is about twice the price. RunPod stays.

## The watchdog

`consort watchdog` deletes this project's pods (names starting `consort-`) when idle for 30
minutes (GPU at or under 5% and CPU at or under 10% on every poll) or up for more than 8 hours.
It never touches a pod without the prefix: the RunPod account is shared with other projects.

- **On the pod, first:** `consort watchdog --self &` before any work. It watches only the pod
  it runs on (`RUNPOD_POD_ID`, set by RunPod), deletes it when idle or past its lifetime, and
  deletes it if the watchdog itself fails for any reason but a network error. A laptop that
  sleeps or drops its connection cannot leave the pod running; on 2026-11-01 another project's
  pod sat idle for ten hours at US$2.59 an hour for exactly that reason.
- **From the laptop, as a second line:** `uv run consort watchdog` over every `consort-` pod.
  It does not delete anything when it fails itself.
- **The spend cap** is the account balance: credit is topped up by hand for this project, with
  automatic reload off.

The key is `RUNPOD_API_KEY`, from the environment or `.env`, never printed or committed. The
idle and lifetime rules, the prefix and every exit path are tested in `tests/test_watchdog.py`.

## Running arm N on a pod

    NAME=consort-gnn-1 GPU="NVIDIA RTX A5000" bash ops/pod-run.sh

From the laptop, one command rents the pod, runs everything on it, copies the outcomes back
into `data/runs` and deletes it:

1. `consort pod-up` rents one Community pod with a public IP (06 learned that without one SSH
   goes through a proxy that copies no files) and waits for its public SSH port; a pod that
   never shows one is deleted.
2. The code goes over SSH as a `git archive` of HEAD, and a `.env` with only the RunPod key,
   the pod's own id and the Kaggle token. No GitHub token goes on a rented machine.
3. On the pod, `ops/pod-setup.sh` starts the watchdog first, with the image's Python, before
   anything is installed; then checks the card (driver, and a 4096 by 4096 matrix multiply
   under the pinned torch, which a bad host fails); then installs the locked environment and
   fetches every dataset against the committed manifests.
4. `ops/pod-queue.sh` runs each line of `ops/gnn-runs.txt` through `consort train-gnn`,
   logging ok or FAIL with the time, and packs the outcomes after every run.
5. The laptop polls every two minutes, copies the log to `data/pod-logs/` and the outcomes
   into `data/runs`, and deletes the pod on every exit path. If the laptop goes away, the
   pod's watchdog deletes it after 30 idle minutes or 10 hours.

Training time is not a reported number, so a Community pod is right here. Arm N's neighbour
sampling runs on the CPU, so the pod asks for at least 8 vCPUs and 48 GB of memory.

