"""The TabFM column and the context swap, with a stand-in scorer: outcomes shape, validation
recorded, and no context row from the step it scores or later."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from consort.data import elliptic
from consort.data.tasks import elliptic_task
from consort.eval import outcomes as oc
from consort.models.tabfm_run import (
    LeakedStep,
    check_strictly_before,
    run_context_swap,
    run_tabfm,
)
from tests.toy import write_elliptic


class Stand:
    """Scores by the first column's z-score; remembers the groups of every context."""

    seen: ClassVar[list[npt.NDArray[np.int64]]] = []

    def __init__(self, size: int, seed: int) -> None:
        self.size = size

    def fit(self, X: pd.DataFrame, y: npt.ArrayLike, groups: npt.ArrayLike) -> Stand:
        Stand.seen.append(np.asarray(groups))
        self.mu = float(X.iloc[:, 0].mean())
        return self

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray(X.iloc[:, 0].to_numpy() - self.mu, dtype=np.float64)


@pytest.fixture(scope="module")
def task(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    return elliptic_task(
        elliptic.load(write_elliptic(tmp_path_factory.mktemp("e")), check_counts=False)
    )


def test_tabfm_column_scores_every_labelled_test_item(task, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = run_tabfm(task, "R", 0, 0, 200, 0, tmp_path, Stand)
    o, meta = oc.read(path), oc.read_meta(path)
    test = (task.partition() == "test") & task.labelled()
    assert len(o) == int(test.sum()) and meta.arm == "TabFM-R"
    assert meta.val_auprc is not None
    assert int(Stand.seen[-1].max()) <= max(task.split.groups["train"])


def test_context_swap_never_reads_the_step_it_scores(task, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    Stand.seen = []
    a, b = run_context_swap(task, "R", 0, 0, 150, 4, 0, tmp_path, Stand)
    test_steps = sorted(task.split.groups["test"])
    assert all(len(g) <= 150 for g in Stand.seen)  # the swap samples before fitting
    # One context per test step, each from the four steps before it.
    assert len(Stand.seen) == len(test_steps)
    for s, groups in zip(test_steps, Stand.seen, strict=True):
        assert set(groups.tolist()) <= set(range(s - 4, s))
    oa, ob = oc.read(a), oc.read(b)
    assert np.array_equal(np.sort(oa.item_id), np.sort(ob.item_id))
    assert oc.read_meta(a).arm == "TabFM-R-swap" and oc.read_meta(b).arm == "R-retrain"


def test_strictly_before_refuses_the_scored_step() -> None:
    check_strictly_before([40, 41, 42], 43)
    with pytest.raises(LeakedStep):
        check_strictly_before([41, 43], 43)
