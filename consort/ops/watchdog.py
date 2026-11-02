"""The pod watchdog: deletes this project's RunPod pods when they sit idle or run too long.

16 lost about US$15 to pods left running, and on 2026-11-01 another project's pod sat idle for
ten hours at US$2.59 an hour because the laptop watching it lost its connection. So the
watchdog's first place is on the pod itself (`--self`), where nothing between it and the pod
can drop: it watches the pod it runs on and deletes it when idle, past its lifetime, or when
the watchdog itself fails for any reason other than a network error (which it retries, since a
deletion needs the network anyway). From the laptop it is a second line over every pod whose
name starts with the project prefix; it never touches another project's pods on the shared
account.

Idle means GPU utilisation and container CPU both at or under a threshold on every poll for
the idle window. CPU counts because arm N's sampler and the tabular arms work the CPU while
the GPU waits. The key is RUNPOD_API_KEY, read from the environment or `.env` and never
printed.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

GRAPHQL = "https://api.runpod.io/graphql"
PREFIX = "consort-"
QUERY = (
    "query { myself { pods { id name desiredStatus costPerHr runtime { uptimeInSeconds "
    "gpus { gpuUtilPercent } container { cpuPercent } } } } }"
)


@dataclass(frozen=True)
class Sample:
    pod_id: str
    name: str
    at: float  # seconds, monotonic or wall, as long as one watcher uses one clock
    uptime_s: float
    gpu_pct: float  # the busiest GPU
    cpu_pct: float
    cost_per_hr: float


@dataclass(frozen=True)
class Policy:
    prefix: str = PREFIX
    idle_minutes: float = 30.0
    gpu_idle_pct: float = 5.0
    cpu_idle_pct: float = 10.0
    max_hours: float = 8.0


@dataclass
class Tracker:
    """Remembers since when each pod has been idle; says which to delete and why."""

    policy: Policy
    idle_since: dict[str, float] = field(default_factory=dict)

    def observe(self, samples: Iterable[Sample]) -> list[tuple[str, str]]:
        p = self.policy
        out: list[tuple[str, str]] = []
        seen = set()
        for s in samples:
            if not s.name.startswith(p.prefix):
                continue
            seen.add(s.pod_id)
            if s.uptime_s >= p.max_hours * 3600:
                out.append(
                    (s.pod_id, f"{s.name}: up {s.uptime_s / 3600:.1f} h, past {p.max_hours} h")
                )
                continue
            if s.gpu_pct <= p.gpu_idle_pct and s.cpu_pct <= p.cpu_idle_pct:
                since = self.idle_since.setdefault(s.pod_id, s.at)
                if s.at - since >= p.idle_minutes * 60:
                    out.append(
                        (
                            s.pod_id,
                            f"{s.name}: idle {(s.at - since) / 60:.0f} min at US${s.cost_per_hr}/h",
                        )
                    )
            else:
                self.idle_since.pop(s.pod_id, None)
        for gone in set(self.idle_since) - seen:
            del self.idle_since[gone]
        return out


def parse(payload: Any, at: float, include_starting: bool = False) -> list[Sample]:
    """Samples from the GraphQL answer. A pod with no runtime yet (starting) is skipped for
    the idle rule, or kept with zero uptime when a sweep needs every pod."""
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"RunPod answered without data: {payload.get('errors')}")
    out = []
    for pod in data["myself"]["pods"]:
        rt = pod.get("runtime")
        if not rt:
            if include_starting:
                out.append(Sample(pod["id"], pod.get("name") or "", at, 0.0, 0.0, 0.0, 0.0))
            continue
        gpus = [float(g.get("gpuUtilPercent") or 0) for g in (rt.get("gpus") or [])]
        out.append(
            Sample(
                pod_id=pod["id"],
                name=pod.get("name") or "",
                at=at,
                uptime_s=float(rt.get("uptimeInSeconds") or 0),
                gpu_pct=max(gpus, default=0.0),
                cpu_pct=float((rt.get("container") or {}).get("cpuPercent") or 0),
                cost_per_hr=float(pod.get("costPerHr") or 0),
            )
        )
    return out


class RunPod:
    def __init__(self, key: str, timeout: float = 30.0) -> None:
        self._key = key
        self._timeout = timeout

    def _call(self, url: str, method: str, body: bytes | None = None) -> bytes:
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Authorization", f"Bearer {self._key}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=self._timeout) as r:
            return bytes(r.read())

    def pods(self, at: float, include_starting: bool = False) -> list[Sample]:
        raw = self._call(GRAPHQL, "POST", json.dumps({"query": QUERY}).encode())
        return parse(json.loads(raw), at, include_starting)

    def delete(self, pod_id: str) -> None:
        """Terminate the pod through GraphQL. The REST endpoint answers 403 to this account's
        key (found 2026-11-01, when two pods outlived a failed launch), so it is not used."""
        raw = self._call(
            GRAPHQL, "POST", json.dumps({"query": terminate_mutation(pod_id)}).encode()
        )
        out = json.loads(raw)
        if out.get("errors"):
            raise RuntimeError(f"RunPod refused to terminate {pod_id}: {out['errors']}")


def terminate_mutation(pod_id: str) -> str:
    return f"mutation {{ podTerminate(input: {{podId: {json.dumps(pod_id)}}}) }}"


def watch(
    api: RunPod,
    policy: Policy,
    self_id: str | None = None,
    poll_s: float = 60.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
    max_polls: int | None = None,
) -> None:
    """Poll, delete what the tracker names. With `self_id`, watch only that pod, stop once it
    is deleted, and delete it if the watcher fails for any reason but the network."""
    tracker = Tracker(policy)
    polls = 0
    try:
        while max_polls is None or polls < max_polls:
            polls += 1
            try:
                samples = api.pods(clock())
            except (urllib.error.URLError, TimeoutError) as e:
                log(f"poll failed, retrying: {e}")
                sleep(poll_s)
                continue
            if self_id is not None:
                samples = [s for s in samples if s.pod_id == self_id]
            for pod_id, why in tracker.observe(samples):
                log(f"deleting {pod_id}: {why}")
                api.delete(pod_id)
                if pod_id == self_id:
                    return
            sleep(poll_s)
    except BaseException as e:
        if self_id is not None and not isinstance(e, urllib.error.URLError):
            log(f"watchdog failed ({type(e).__name__}); deleting {self_id}")
            api.delete(self_id)
        raise


def key_from_env() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError("RUNPOD_API_KEY is not set (environment or .env)")
    return key
