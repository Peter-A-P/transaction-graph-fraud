"""Arms R and G end to end on a toy set where only the graph carries the signal."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from consort.data import ibm
from consort.data.tasks import ibm_task
from consort.eval import outcomes as oc
from consort.eval.metrics import evaluate, paired_delta
from consort.models.run import run_tabular
from tests.toy import write_ibm


def test_arms_score_identical_items_and_the_graph_finds_the_cycles(tmp_path: Path) -> None:
    data = ibm.load(write_ibm(tmp_path), "T")
    task = ibm_task(data, boot_group="day")
    r = oc.read(run_tabular(task, "R", 0, 0, seed=0, out_dir=tmp_path / "runs"))
    g = oc.read(run_tabular(task, "G", 2, 20, seed=0, out_dir=tmp_path / "runs"))
    assert np.array_equal(np.sort(r.item_id), np.sort(g.item_id))
    # Every test item is from the test days.
    test_days = set(task.split.groups["test"])
    scored = task.items["item_id"].isin(r.item_id).to_numpy()
    assert set(task.items["split_group"].to_numpy()[scored].tolist()) <= test_days
    er = evaluate(r, n_boot=50)
    eg = evaluate(g, n_boot=50)
    assert eg["auprc"].point > 0.8 > er["auprc"].point
    delta = paired_delta(r, g, n_boot=50)["auprc"]
    assert delta.lo > 0


def test_random_split_run_is_labelled_as_such(tmp_path: Path) -> None:
    data = ibm.load(write_ibm(tmp_path, n_background=2000, n_cycles=20), "T")
    task = ibm_task(data, boot_group="day")
    path = run_tabular(task, "R", 0, 0, seed=0, out_dir=tmp_path, split_kind="random")
    meta = oc.read_meta(path)
    assert meta.split == "ibm-T-random"
    assert meta.val_auprc is not None and 0.0 <= meta.val_auprc <= 1.0
