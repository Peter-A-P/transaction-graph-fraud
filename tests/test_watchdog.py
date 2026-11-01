"""The pod watchdog: idle and lifetime rules, other projects' pods untouched, every exit path."""

from __future__ import annotations

import pytest

from consort.ops.watchdog import Policy, RunPod, Sample, Tracker, parse, watch

P = Policy(idle_minutes=30, max_hours=8)


def _s(
    at_min: float, gpu: float = 0, cpu: float = 0, name: str = "consort-n", up_h: float = 1
) -> Sample:
    return Sample("p1", name, at_min * 60, up_h * 3600, gpu, cpu, 0.4)


def test_idle_for_the_whole_window_is_deleted_and_not_before() -> None:
    t = Tracker(P)
    assert t.observe([_s(0)]) == []
    assert t.observe([_s(29)]) == []
    assert [pid for pid, _ in t.observe([_s(30)])] == ["p1"]


def test_any_busy_poll_resets_the_window() -> None:
    t = Tracker(P)
    t.observe([_s(0)])
    t.observe([_s(20, cpu=80)])  # the sampler working the CPU while the GPU waits
    assert t.observe([_s(45)]) == []
    assert t.observe([_s(75)]) != []


def test_lifetime_cap_deletes_a_busy_pod() -> None:
    assert Tracker(P).observe([_s(0, gpu=99, up_h=8.1)]) != []


def test_another_projects_pod_is_never_touched() -> None:
    t = Tracker(P)
    t.observe([_s(0, name="itx-gpu-phase2", up_h=10)])
    assert t.observe([_s(600, name="itx-gpu-phase2", up_h=20)]) == []


def test_parse_takes_the_busiest_gpu_and_skips_a_starting_pod() -> None:
    payload = {
        "data": {
            "myself": {
                "pods": [
                    {
                        "id": "a",
                        "name": "consort-x",
                        "costPerHr": 0.4,
                        "runtime": {
                            "uptimeInSeconds": 60,
                            "gpus": [{"gpuUtilPercent": 3}, {"gpuUtilPercent": 70}],
                            "container": {"cpuPercent": 5},
                        },
                    },
                    {"id": "b", "name": "consort-y", "runtime": None},
                ]
            }
        }
    }
    (s,) = parse(payload, 0.0)
    assert (s.pod_id, s.gpu_pct, s.cpu_pct) == ("a", 70.0, 5.0)


class FakeAPI(RunPod):
    def __init__(self, samples: list[list[Sample]], fail_at: int | None = None) -> None:
        super().__init__("unused")
        self.samples = samples
        self.fail_at = fail_at
        self.deleted: list[str] = []
        self.n = 0

    def pods(self, at: float) -> list[Sample]:
        self.n += 1
        if self.fail_at == self.n:
            raise ValueError("unexpected answer")
        return self.samples[min(self.n - 1, len(self.samples) - 1)]

    def delete(self, pod_id: str) -> None:
        self.deleted.append(pod_id)


def test_self_mode_deletes_itself_when_idle_and_stops() -> None:
    api = FakeAPI([[_s(0)], [_s(31)]])
    clock = iter(range(0, 10_000, 1860))
    watch(
        api,
        P,
        self_id="p1",
        clock=lambda: float(next(clock)),
        sleep=lambda _: None,
        log=lambda _: None,
    )
    assert api.deleted == ["p1"]


def test_self_mode_deletes_itself_when_the_watcher_fails() -> None:
    api = FakeAPI([[_s(0, gpu=90)]], fail_at=2)
    with pytest.raises(ValueError):
        watch(api, P, self_id="p1", clock=lambda: 0.0, sleep=lambda _: None, log=lambda _: None)
    assert api.deleted == ["p1"]


def test_laptop_mode_does_not_delete_on_its_own_failure() -> None:
    api = FakeAPI([[_s(0, gpu=90)]], fail_at=2)
    with pytest.raises(ValueError):
        watch(api, P, clock=lambda: 0.0, sleep=lambda _: None, log=lambda _: None)
    assert api.deleted == []
