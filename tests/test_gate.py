"""Gate side files: one verdict per illicit item, the queue's cut exact and deterministic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from consort.eval import outcomes as oc
from consort.gate.export import caught, export


def test_caught_is_recall_at_the_budget_per_positive() -> None:
    # 200 items, 1% budget = 2 alerts; positives at ranks 1, 3 and 150.
    n = 200
    labels = np.zeros(n, dtype=int)
    labels[[0, 2, 149]] = 1
    o = oc.Outcomes.build(np.arange(n).astype(str), np.zeros(n), labels, -np.arange(n, dtype=float))
    c = caught(o, 0.01)
    assert c == {"0": True, "2": False, "149": False}


def test_ties_are_broken_by_item_id_not_by_order_of_arrival() -> None:
    labels = np.array([1, 1, 0, 0] + [0] * 96)
    ids = np.array(["b", "a"] + [f"z{i}" for i in range(98)])
    o = oc.Outcomes.build(ids, np.zeros(100), labels, np.r_[1.0, 1.0, np.zeros(98)])
    assert caught(o, 0.01) == {"a": True, "b": False}


def test_export_writes_a_spec_and_matching_side_files(tmp_path: Path) -> None:
    n = 100
    labels = np.r_[np.ones(10), np.zeros(90)].astype(int)
    paths = []
    for arm, score in (("R", np.random.default_rng(0).random(n)), ("G", labels + 0.0)):
        o = oc.Outcomes.build(np.arange(n).astype(str), np.arange(n) % 5, labels, score)
        meta = oc.RunMeta(
            run=f"t.temporal.{arm}.h0.cap0.s0",
            dataset="t",
            split="s",
            arm=arm,
            model="m",
            hops=0,
            seed=0,
        )
        paths.append(oc.write(o, meta, tmp_path / "runs"))
    files = export(paths[0], paths[1:], tmp_path / "gate", "t")
    sides = [json.loads(f.read_text()) for f in files]
    assert set(sides[0]["suites"]["t"]["outcomes"]) == set(sides[1]["suites"]["t"]["outcomes"])
    assert "kind: outcomes_file" in (tmp_path / "gate" / "spec.yaml").read_text()
