"""Pod launching: the watchdog's prefix is enforced, and SSH is only the public endpoint."""

from __future__ import annotations

import pytest

from consort.ops.pod import PodSpec, public_ssh


def test_a_pod_without_the_watchdog_prefix_is_refused() -> None:
    with pytest.raises(ValueError, match="consort-"):
        PodSpec("gnn", "NVIDIA RTX A5000")


def test_the_deploy_request_asks_for_a_public_ip_and_ssh() -> None:
    m = PodSpec("consort-gnn", "NVIDIA RTX A5000").mutation()
    assert "supportPublicIp: true" in m and "startSsh: true" in m
    assert 'gpuTypeId: "NVIDIA RTX A5000"' in m and 'name: "consort-gnn"' in m


def test_only_a_public_ssh_port_counts() -> None:
    proxied = {
        "runtime": {
            "ports": [
                {
                    "ip": "10.0.0.2",
                    "isIpPublic": False,
                    "privatePort": 22,
                    "publicPort": 22,
                    "type": "tcp",
                }
            ]
        }
    }
    public = {
        "runtime": {
            "ports": [
                {
                    "ip": "1.2.3.4",
                    "isIpPublic": True,
                    "privatePort": 22,
                    "publicPort": 40123,
                    "type": "tcp",
                }
            ]
        }
    }
    assert public_ssh(None) is None
    assert public_ssh({"runtime": None}) is None
    assert public_ssh(proxied) is None
    s = public_ssh(public)
    assert s is not None and (s.host, s.port) == ("1.2.3.4", 40123)


def test_queries_are_well_formed_graphql_strings() -> None:
    from consort.ops.pod import pod_query
    from consort.ops.watchdog import terminate_mutation

    for q in (pod_query("abc123"), terminate_mutation("abc123")):
        assert q.count("{") == q.count("}")
        assert '"abc123"' in q
    assert terminate_mutation("x").startswith("mutation { podTerminate(")


def test_a_sweep_sees_pods_that_are_still_starting() -> None:
    from consort.ops.watchdog import parse

    payload = {"data": {"myself": {"pods": [{"id": "s", "name": "consort-x", "runtime": None}]}}}
    assert parse(payload, 0.0) == []
    assert [p.pod_id for p in parse(payload, 0.0, include_starting=True)] == ["s"]
