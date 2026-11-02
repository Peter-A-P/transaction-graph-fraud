"""Arm N end to end on toy sets. Skipped where the gnn extra is not installed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch_geometric")

from consort.data import elliptic, ibm
from consort.data.tasks import elliptic_task, ibm_task
from consort.eval import outcomes as oc
from consort.eval.metrics import evaluate
from consort.models.gnn import GNNConfig, run
from consort.models.run import run_tabular
from tests.toy import write_elliptic, write_ibm


def test_node_gnn_scores_the_same_items_as_arm_r(tmp_path: Path) -> None:
    task = elliptic_task(elliptic.load(write_elliptic(tmp_path), check_counts=False))
    r = oc.read(run_tabular(task, "R", 0, 0, seed=0, out_dir=tmp_path / "runs"))
    n = oc.read(
        run(
            task,
            GNNConfig(hops=1, cap=10, epochs=15, patience=15, batch_size=512),
            tmp_path / "runs",
        )
    )
    assert np.array_equal(np.sort(r.item_id), np.sort(n.item_id))
    # The chain members' own features are noise; their hub's marker is one hop away.
    assert evaluate(n, n_boot=20)["auprc"].point > evaluate(r, n_boot=20)["auprc"].point


@pytest.mark.parametrize(
    "cfg",
    [
        GNNConfig(hops=0, epochs=2, batch_size=1024),
        GNNConfig(hops=2, cap=5, epochs=2, batch_size=1024, reverse_mp=True, ego_ids=True),
        GNNConfig(arch="gat", hops=1, cap=5, epochs=2, batch_size=1024),
    ],
)
def test_edge_gnn_variants_run_and_score_every_test_transfer(
    tmp_path: Path, cfg: GNNConfig
) -> None:
    task = ibm_task(
        ibm.load(write_ibm(tmp_path, n_background=2000, n_cycles=20), "T"), boot_group="day"
    )
    o = oc.read(run(task, cfg, tmp_path / "runs"))
    assert len(o) == int((task.partition() == "test").sum())
    assert np.isfinite(o.score).all()
    assert (
        oc.read_meta(
            tmp_path
            / "runs"
            / f"{task.name}.temporal.{cfg.tag}.h{cfg.hops}.cap{cfg.cap}.s0.outcomes.parquet"
        ).arm
        == cfg.tag
    )


def test_edge_views_draw_each_day_from_earlier_days_and_batch_within_a_day(tmp_path: Path) -> None:
    from consort.models.gnn import _EdgeViews

    task = ibm_task(
        ibm.load(write_ibm(tmp_path, n_background=1500, n_cycles=10), "T"), boot_group="day"
    )
    views = _EdgeViews(task, GNNConfig(hops=1, cap=5))
    day = task.items["split_group"].to_numpy()
    test = np.flatnonzero(task.partition() == "test").astype(np.int64)
    d = int(day[test].max())
    assert views("test", d).sampler.g.n_edges == int((day < d).sum())
    batches = views.batches(test, 64, True, np.random.default_rng(0))
    assert all((day[b] == bd).all() for bd, b in batches)
    assert sorted(np.concatenate([b for _, b in batches]).tolist()) == test.tolist()


def test_node_gnn_on_elliptic_pp_scores_every_new_test_address(tmp_path: Path) -> None:
    from consort.data import ellipticpp
    from consort.data.splits import temporal_split
    from consort.data.tasks import ellipticpp_task
    from tests.toy import write_ellipticpp

    split = temporal_split("toy", train=range(1, 8), val=range(8, 10), test=range(10, 14))
    task = ellipticpp_task(ellipticpp.load(*write_ellipticpp(tmp_path), check_counts=False), split)
    o = oc.read(run(task, GNNConfig(hops=1, cap=5, epochs=2, batch_size=64), tmp_path / "r"))
    test = (task.partition() == "test") & task.labelled()
    assert len(o) == int(test.sum())
    assert np.isfinite(o.score).all()
