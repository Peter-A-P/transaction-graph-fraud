"""Renting a RunPod pod for arm N, TabFM and the serving benchmark, and giving it back.

Learned by 06 and kept here (docs/gpu.md): ask for a public IP, or a Community pod offers SSH
only through a proxy that carries no file copies; wait for the pod's runtime to show its
public SSH port rather than a fixed time; and check the card before installing anything (the
driver must suit the pinned torch's CUDA, and a matrix multiply must run), which
ops/pod-setup.sh does first. Every pod this project rents is named with the watchdog's prefix,
so the watchdog on the pod and the one on the laptop both cover it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from consort.ops.watchdog import GRAPHQL, PREFIX, RunPod

IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


@dataclass(frozen=True)
class PodSpec:
    name: str
    gpu_type: str  # a RunPod gpuTypes id, such as "NVIDIA RTX A5000"
    cloud: str = "COMMUNITY"
    container_disk_gb: int = 80
    min_vcpu: int = 8
    min_memory_gb: int = 48
    image: str = IMAGE

    def __post_init__(self) -> None:
        if not self.name.startswith(PREFIX):
            raise ValueError(f"pod name {self.name!r} must start with {PREFIX!r} (the watchdog's)")
        if self.cloud not in ("COMMUNITY", "SECURE"):
            raise ValueError(f"cloud must be COMMUNITY or SECURE, not {self.cloud}")

    def mutation(self) -> str:
        fields = {
            "cloudType": self.cloud,
            "gpuCount": 1,
            "gpuTypeId": json.dumps(self.gpu_type),
            "name": json.dumps(self.name),
            "imageName": json.dumps(self.image),
            "containerDiskInGb": self.container_disk_gb,
            "volumeInGb": 0,
            "minVcpuCount": self.min_vcpu,
            "minMemoryInGb": self.min_memory_gb,
            "ports": json.dumps("22/tcp"),
            "supportPublicIp": "true",
            "startSsh": "true",
        }
        body = ", ".join(f"{k}: {v}" for k, v in fields.items())
        return (
            f"mutation {{ podFindAndDeployOnDemand(input: {{{body}}}) "
            "{ id machineId costPerHr } }"
        )


@dataclass(frozen=True)
class Ssh:
    host: str
    port: int

    def command(self, key: str = "~/.ssh/runpod_smallprint") -> str:
        return f"ssh -i {key} -p {self.port} root@{self.host}"


POD_QUERY = (
    'query {{ pod(input: {{podId: "{id}"}}) {{ id desiredStatus costPerHr '
    "machine { gpuDisplayName } runtime { uptimeInSeconds ports { ip isIpPublic privatePort "
    "publicPort type } } } }"
)


class Pods(RunPod):
    def _graphql(self, query: str) -> dict[str, Any]:
        raw = self._call(GRAPHQL, "POST", json.dumps({"query": query}).encode())
        out: dict[str, Any] = json.loads(raw)
        if out.get("errors"):
            raise RuntimeError(f"RunPod: {out['errors']}")
        return out

    def deploy(self, spec: PodSpec) -> tuple[str, float]:
        pod = self._graphql(spec.mutation())["data"]["podFindAndDeployOnDemand"]
        if not pod:
            raise RuntimeError(f"no {spec.gpu_type} available in {spec.cloud} cloud")
        return str(pod["id"]), float(pod.get("costPerHr") or 0.0)

    def ssh(self, pod_id: str) -> Ssh | None:
        pod = self._graphql(POD_QUERY.format(id=pod_id))["data"]["pod"]
        return public_ssh(pod)

    def wait_ssh(
        self,
        pod_id: str,
        timeout_s: float = 900,
        poll_s: float = 15,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Ssh:
        waited = 0.0
        while waited < timeout_s:
            found = self.ssh(pod_id)
            if found is not None:
                return found
            sleep(poll_s)
            waited += poll_s
        raise TimeoutError(f"pod {pod_id} showed no public SSH port in {timeout_s:.0f} s")


def public_ssh(pod: dict[str, Any] | None) -> Ssh | None:
    """The pod's public SSH endpoint, once its runtime shows one."""
    if not pod or not pod.get("runtime"):
        return None
    for p in pod["runtime"].get("ports") or []:
        if p.get("privatePort") == 22 and p.get("isIpPublic") and p.get("type") == "tcp":
            return Ssh(str(p["ip"]), int(p["publicPort"]))
    return None
