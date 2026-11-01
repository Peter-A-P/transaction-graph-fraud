"""Recall by pattern, the shift slices and the neighbour-label-rate experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from consort.data import ibm
from consort.data.splits import ELLIPTIC_TEMPORAL
from consort.data.tasks import ibm_task
from consort.eval import outcomes as oc
from consort.eval.bypattern import recall_by_pattern
from consort.eval.leak import neighbour_label_rate
from consort.eval.shift import by_slice
from tests.toy import write_ibm


def test_recall_by_pattern_shares_one_queue() -> None:
    # 200 items; the top 1% (2 items) are both CYCLE positives; FAN-IN positives rank low.
    n = 200
    ids = np.arange(n).astype(str)
    labels = np.zeros(n, dtype=int)
    labels[[0, 1, 2, 3]] = 1
    scores = -np.arange(n, dtype=float)
    pattern = pd.Series(["CYCLE", "CYCLE", "FAN-IN", "FAN-IN"] + ["NONE"] * (n - 4), index=ids)
    o = oc.Outcomes.build(ids, np.arange(n) % 10, labels, scores)
    est = recall_by_pattern(o, pattern, budget=0.01, n_boot=20)
    assert est["CYCLE"].point == pytest.approx(1.0)
    assert est["FAN-IN"].point == pytest.approx(0.0)
    assert "NONE" not in est


def test_shift_slices_split_the_test_steps() -> None:
    rng = np.random.default_rng(0)
    steps = np.repeat(np.arange(39, 50), 30)
    labels = (rng.random(steps.shape[0]) < 0.2).astype(int)
    # The model is good before step 43 and random from it.
    scores = np.where(
        steps < 43, labels + rng.normal(0, 0.1, steps.shape[0]), rng.random(steps.shape[0])
    )
    o = oc.Outcomes.build(np.arange(steps.shape[0]), steps, labels, scores)
    out = by_slice(o, ELLIPTIC_TEMPORAL, n_boot=50)
    assert out["before-43"]["auprc"].point > 0.95
    assert out["from-43"]["auprc"].point < 0.6
    assert out["before-43"]["auprc"].n_groups == 4 and out["from-43"]["auprc"].n_groups == 7


def test_neighbour_label_rate_reads_only_visible_labels(tmp_path: Path) -> None:
    data = ibm.load(write_ibm(tmp_path, n_background=1500, n_cycles=20), "T")
    task = ibm_task(data, boot_group="day")
    visible = (task.partition() == "train") & task.labelled()
    before = neighbour_label_rate(task, visible)
    flipped = task.items.copy()
    hidden = ~visible
    flipped.loc[hidden, "label"] = 1 - flipped.loc[hidden, "label"]
    task.items = flipped
    after = neighbour_label_rate(task, visible)
    pd.testing.assert_frame_equal(before, after)
