"""The metric arithmetic against closed-form answers, the bootstrap grouping, paired deltas."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from consort.eval import outcomes as oc
from consort.eval.metrics import (
    MismatchedItems,
    Ranked,
    average_precision,
    budget_ceiling,
    evaluate,
    group_weights,
    paired_delta,
    recall_at_budget,
    recall_at_fpr,
)


def _ranked(labels: Sequence[int], scores: Sequence[float]) -> Ranked:
    return Ranked.of(labels, scores, np.zeros(len(labels)))


def _w(n: int) -> npt.NDArray[np.float64]:
    return np.ones(n)


def test_average_precision_hand_example() -> None:
    # Positives at ranks 1 and 3: precision 1 then 2/3, each worth half the recall.
    r = _ranked([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6])
    assert average_precision(r, _w(4)) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


@pytest.mark.parametrize("m", [2, 5, 20])
def test_average_precision_closed_form_stream(m: int) -> None:
    """A score stream with a positive at every m-th rank: precision at each positive is 1/m,
    so AUPRC is exactly 1/m; the top b items hold floor(b/m) of the P positives."""
    n_pos = 50
    n = n_pos * m
    labels = np.zeros(n, dtype=int)
    labels[m - 1 :: m] = 1
    scores = -np.arange(n, dtype=float)
    r = Ranked.of(labels, scores, np.arange(n))
    assert average_precision(r, _w(n)) == pytest.approx(1 / m)
    # Budgets that take a whole number of m-rank strides: the top b*n items hold b*n/m of
    # the n/m positives, so recall at budget b is exactly b.
    for budget in (0.02, 0.1, 0.38):
        assert recall_at_budget(budget)(r, _w(n)) == pytest.approx(budget)


def test_all_tied_scores_give_prevalence_and_proportional_budget() -> None:
    labels = [1] * 10 + [0] * 90
    r = _ranked(labels, [0.5] * 100)
    assert average_precision(r, _w(100)) == pytest.approx(0.1)
    # Flagging 5% of a single tie block takes 5% of its positives in expectation.
    assert recall_at_budget(0.05)(r, _w(100)) == pytest.approx(0.05)


def test_perfect_ranking() -> None:
    r = _ranked([1, 1, 0, 0, 0], [5, 4, 3, 2, 1])
    assert average_precision(r, _w(5)) == 1.0
    assert recall_at_fpr(0.0)(r, _w(5)) == 1.0
    assert recall_at_budget(0.2)(r, _w(5)) == pytest.approx(0.5)


def test_budget_ceiling_is_alerts_over_positives_and_a_perfect_ranking_reaches_it() -> None:
    # 1,000 items, 60 positive: a 1% budget is 10 alerts, so no ranking passes 10/60.
    labels = [1] * 60 + [0] * 940
    r = _ranked(labels, list(range(1000, 0, -1)))
    assert budget_ceiling(0.01)(r, _w(1000)) == pytest.approx(10 / 60)
    assert recall_at_budget(0.01)(r, _w(1000)) == pytest.approx(10 / 60)
    assert budget_ceiling(0.1)(r, _w(1000)) == 1.0


def test_recall_at_fpr_stops_at_the_first_negative_over_the_rate() -> None:
    # 100 negatives: 1% FPR allows exactly one negative above the threshold.
    labels = [1, 0, 1, 0, 1] + [0] * 98
    scores = list(range(103, 0, -1))
    r = _ranked(labels, scores)
    assert recall_at_fpr(0.01)(r, _w(103)) == pytest.approx(2 / 3)


def test_bootstrap_never_splits_a_group() -> None:
    rng = np.random.default_rng(3)
    codes = np.repeat(np.arange(7), [5, 1, 9, 2, 2, 4, 30])
    for _ in range(50):
        w = group_weights(codes, 7, rng)
        for g in range(7):
            assert np.unique(w[codes == g]).shape[0] == 1


def test_interval_brackets_the_point_and_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    labels = (rng.random(n) < 0.05).astype(int)
    scores = labels * 1.0 + rng.normal(0, 1, n)
    o = oc.Outcomes.build(np.arange(n), rng.integers(0, 40, n), labels, scores)
    a = evaluate(o, n_boot=200, seed=1)
    b = evaluate(o, n_boot=200, seed=1)
    assert a == b
    for est in a.values():
        assert est.lo <= est.point <= est.hi
        assert est.n_groups == 40


def test_paired_delta_refuses_mismatched_items() -> None:
    a = oc.Outcomes.build(["x", "y", "z"], [0, 0, 1], [1, 0, 0], [0.9, 0.1, 0.2])
    b = oc.Outcomes.build(["x", "y", "w"], [0, 0, 1], [1, 0, 0], [0.9, 0.1, 0.2])
    with pytest.raises(MismatchedItems, match="different items"):
        paired_delta(a, b, n_boot=10)


def test_paired_delta_refuses_different_labels() -> None:
    a = oc.Outcomes.build(["x", "y"], [0, 1], [1, 0], [0.9, 0.1])
    b = oc.Outcomes.build(["y", "x"], [1, 0], [1, 0], [0.9, 0.1])
    with pytest.raises(MismatchedItems, match="labels"):
        paired_delta(a, b, n_boot=10)


def test_paired_delta_of_a_run_with_itself_is_zero_and_order_free() -> None:
    rng = np.random.default_rng(1)
    n = 500
    labels = (rng.random(n) < 0.1).astype(int)
    scores = rng.random(n) + labels
    ids = np.arange(n)
    a = oc.Outcomes.build(ids, ids % 10, labels, scores)
    perm = rng.permutation(n)
    b = oc.Outcomes.build(ids[perm], (ids % 10)[perm], labels[perm], scores[perm])
    for est in paired_delta(a, b, n_boot=50).values():
        assert est.point == pytest.approx(0.0)
        assert est.lo == pytest.approx(0.0) and est.hi == pytest.approx(0.0)


def test_unknown_labels_never_enter_a_metric() -> None:
    with pytest.raises(oc.UnknownLabel):
        oc.Outcomes.build(["a", "b"], [1, 1], [1, -1], [0.2, 0.3])
    with pytest.raises(oc.UnknownLabel):
        oc.Outcomes.build(["a", "b"], [1, 1], ["1", "unknown"], [0.2, 0.3])


def test_outcomes_round_trip(tmp_path: Path) -> None:
    o = oc.Outcomes.build(["a", "b"], [3, 4], [1, 0], [0.7, 0.2])
    meta = oc.RunMeta(run="r1", dataset="toy", split="t", arm="R", model="xgb", hops=0, seed=0)
    path = oc.write(o, meta, tmp_path)
    back = oc.read(path)
    assert np.array_equal(back.item_id, o.item_id) and np.array_equal(back.score, o.score)
    assert oc.read_meta(path) == meta
