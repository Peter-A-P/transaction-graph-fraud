"""The demo site's ring: chosen without reading a score, accounts named by place only, every
arm's verdict on the same transfers, and a case file for each transfer arm G flagged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("onnxmltools")

from consort.data import ibm
from consort.data.tasks import ibm_task
from consort.models.run import run_tabular
from consort.report import site as st
from tests.toy import write_ibm


def test_the_ring_is_picked_blind_and_scored_by_every_arm(tmp_path: Path) -> None:
    task = ibm_task(ibm.load(write_ibm(tmp_path, n_background=1500, n_cycles=15), "T"))
    runs = {
        "R": run_tabular(task, "R", 0, 0, seed=0, out_dir=tmp_path / "r"),
        "G": run_tabular(task, "G", 1, 10, seed=0, out_dir=tmp_path / "r", onnx_dir=tmp_path / "m"),
    }
    (model,) = (tmp_path / "m").glob("*.onnx")
    # The pick takes the task alone, no outcomes: it cannot have been chosen by a score.
    attempt = st.pick_ring(task, "CYCLE")
    t = task.data.transfers if task.data is not None else None
    assert t is not None and (t["attempt"] == attempt).any()

    ring = st.ring(task, attempt, runs, model, hops=1, cap=10, budget=0.5)
    names = {a["name"] for a in ring["accounts"]}
    assert names == {f"A{i + 1}" for i in range(len(names))}
    text = json.dumps(ring)
    assert task.data is not None
    for acct in task.data.accounts["acct"].head(50):
        assert f'"{acct}"' not in text  # the generator's account numbers never leave
    scored = [t for t in ring["transfers"] if t["arms"]]
    assert scored and all(set(t["arms"]) == {"R", "G"} for t in scored)
    flagged_g = sum(t["arms"]["G"]["flagged"] for t in scored)
    assert len(ring["cases"]) == flagged_g
    for case in ring["cases"]:
        assert all(n["account"] for n in case["top"]) and 0 <= case["score"] <= 1


def test_a_transfers_rank_is_what_puts_it_in_the_queue() -> None:
    import numpy as np

    from consort.eval import outcomes as oc

    # Scores squeezed near 1, as the graph neural network's are; a tie broken by item id.
    o = oc.Outcomes(
        item_id=np.array(["e", "d", "c", "b", "a"]),
        group=np.zeros(5, dtype=np.int64),
        label=np.array([0, 1, 0, 1, 0], dtype=np.int8),
        score=np.array([0.9951, 0.9961, 0.9949, 0.9961, 0.9]),
    )
    r = st.ranks(o)
    assert r == {"b": 0.2, "d": 0.4, "e": 0.6, "c": 0.8, "a": 1.0}
    for budget in (0.2, 0.4, 0.6):
        assert st.queue(o, budget) == {i for i, share in r.items() if share <= budget + 1e-9}
