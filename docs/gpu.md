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

**Terminating goes through GraphQL (`podTerminate`).** RunPod's REST delete answers 403 to this
account's key. That was found on 2026-11-01, the hard way: a launch failed after two pods were
created (a malformed query), its clean-up called the REST delete, and both pods kept running
until they were found by listing the account and terminated through GraphQL, minutes later.
The launcher now sweeps every pod with its run's name on every exit, and the whole path (rent,
wait for the public SSH port, sweep by name, confirm the account empty) was run live on a
US$0.12 card before the first real queue.

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

Training time is not a reported number, so a Community pod is right here. The pod asks for
at least 4 vCPUs and 24 GB of memory: the sampler is mostly one thread, and the largest graph
(IBM HI-Small, about 5 million transfers) fits well within it. On 2026-11-01 (a Sunday evening)
no A5000, 3090, A6000 or A40 was in stock on the Community cloud with a public IP, at 8 vCPUs
and 48 GB or at the lower minimums; the cheapest card in stock with a public IP and enough
memory for two-hop batches was the 32 GB RTX 5000 Ada at US$0.49 an hour, added to the list
after the A5000 and 3090.

**What was rented, 2026-11-02 (UTC).** Community stock stayed empty for every card with a
public IP from about 02:00 to 03:00 UTC, so the smoke run (one short arm N run on Elliptic, the
whole path from rent to sweep) and then the first queue went to a Secure Cloud L4 at US$0.59
an hour: at the queue's few hours, a few dollars more than waiting for the Community cloud and
a night sooner. The smoke run showed torch 2.15 on CUDA 13.2 under the L4's driver, the
datasets verified against the committed manifests on the pod, and arm N's two epochs on
Elliptic in 17 seconds against 96 on the laptop.


**The first full queue, 2026-11-02.** No L4 was in stock after an hour of retries, so the
queue ran on a Secure Cloud RTX 4090 at US$0.89 an hour from 04:27 UTC. Elliptic's arm N runs
took about 20 seconds each and Elliptic++'s about 11 minutes. GraphSAGE at one hop on IBM
HI-Small took 4 hours 5 minutes with the card at 0% utilisation: training shuffled batches
across days, so nearly every batch rebuilt its day's graph on the CPU. The two-hop run was
stopped after 40 minutes and the IBM runs taken off the pod; the rest of the list (Elliptic's
hop curve, GAT on the real sets, TabFM) went on. The fix (a day's batches run together, and
the sampler without Python sets) is timed on the laptop before IBM goes back on a pod.

**IBM's arm N, 2026-11-02.** With a day's batches run together, the vectorised sampler and a
fifth of the negatives drawn afresh each epoch (PLAN.md 2.5), the six IBM runs took 15 to 30
minutes each on a second Secure Cloud RTX 4090 at US$0.89 an hour, against four hours for one
run before. That pod sat idle for an hour and a half first: the launcher's start had left an
empty log, took it for a running queue and never started one, and with the queue went the
watchdog. The queue was started by hand. The launcher now starts it in its own session with
no terminal, counts the start only once the log has a line, and treats an empty log as no
queue.

**What did not finish.** The first pod became unreachable while TabFM ran on Elliptic++'s
graph features with a context of 30,000 rows, and the launcher swept it after ten missed
polls. That run, the context swap at step 43 and TabFM on Elliptic's graph features (which
had run out of card memory before the query batches were cut to 1,000 rows) are still to run.
