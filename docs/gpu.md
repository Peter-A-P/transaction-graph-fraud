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

**Not yet checked:** the plan's rule from 06 compares GCP and Azure spot prices for the same
class and takes them if within 20% of the marketplace. That comparison is made, and dated here,
before the first rental.

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
