"""AUPRC, recall at a false-positive rate, recall at an alert budget; bootstrap over groups.

Every metric is written for weighted items, sorted once. A bootstrap resample over groups is
then a vector of weights (how many times each item's group was drawn), so a thousand
resamples of a million items cost a thousand linear passes, not a thousand sorts.

Ties are scored as a random order would score them in expectation: a block of equal scores is
flagged or not as a whole when computing curves, and an alert budget that cuts through a block
takes the block's share of positives in proportion to the part it takes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from consort.eval.outcomes import Outcomes

FloatArray = npt.NDArray[np.float64]


class MismatchedItems(Exception):
    """A paired delta was asked of two runs that did not score identical items."""


@dataclass(frozen=True)
class Ranked:
    """Items sorted by descending score, with the boundaries of each block of equal scores."""

    label: FloatArray
    group_code: npt.NDArray[np.int64]
    n_groups: int
    block_end: npt.NDArray[np.int64]  # exclusive end index of each tie block
    order: npt.NDArray[np.int64]  # ranked position -> original position

    @classmethod
    def of(cls, labels: npt.ArrayLike, scores: npt.ArrayLike, groups: npt.ArrayLike) -> Ranked:
        s = np.asarray(scores, dtype=np.float64)
        order = np.argsort(-s, kind="stable")
        s_sorted = s[order]
        codes, inverse = np.unique(np.asarray(groups)[order], return_inverse=True)
        block_end = np.r_[np.flatnonzero(np.diff(s_sorted) != 0) + 1, s.shape[0]]
        return cls(
            label=np.asarray(labels, dtype=np.float64)[order],
            group_code=inverse.astype(np.int64),
            n_groups=int(codes.shape[0]),
            block_end=block_end.astype(np.int64),
            order=order.astype(np.int64),
        )

    def blocks(self, weights: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Weighted positives and negatives in each tie block."""
        tp = np.add.reduceat(weights * self.label, np.r_[0, self.block_end[:-1]])
        fp = np.add.reduceat(weights * (1.0 - self.label), np.r_[0, self.block_end[:-1]])
        return tp, fp


Metric = Callable[[Ranked, FloatArray], float]


def average_precision(r: Ranked, w: FloatArray) -> float:
    tp, fp = r.blocks(w)
    pos = tp.sum()
    if pos == 0:
        return float("nan")
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    precision = ctp / np.maximum(ctp + cfp, 1e-300)
    return float(np.sum(tp / pos * precision))


def recall_at_fpr(fpr: float) -> Metric:
    def metric(r: Ranked, w: FloatArray) -> float:
        tp, fp = r.blocks(w)
        pos, neg = tp.sum(), fp.sum()
        if pos == 0 or neg == 0:
            return float("nan")
        ok = np.cumsum(fp) / neg <= fpr + 1e-12
        if not ok.any():
            return 0.0
        return float(np.cumsum(tp)[ok].max() / pos)

    metric.__name__ = f"recall_at_fpr_{fpr}"
    return metric


def recall_at_budget(budget: float) -> Metric:
    """Recall when the top `budget` share of items (by weight) are flagged for review."""

    def metric(r: Ranked, w: FloatArray) -> float:
        tp, fp = r.blocks(w)
        pos = tp.sum()
        if pos == 0:
            return float("nan")
        size = tp + fp
        k = budget * size.sum()
        before = np.r_[0.0, np.cumsum(size)[:-1]]
        take = np.clip((k - before) / np.maximum(size, 1e-300), 0.0, 1.0)
        return float(np.sum(take * tp) / pos)

    metric.__name__ = f"recall_at_budget_{budget}"
    return metric


def budget_ceiling(budget: float) -> Metric:
    """The most recall any ranking could reach at the budget: min(1, alerts / positives).

    Reported beside recall at a budget, since a budget smaller than the positive rate caps
    recall well below 1 for every arm alike (Elliptic's labelled test set is about 6% illicit,
    so a 1% budget caps recall near 0.17)."""

    def metric(r: Ranked, w: FloatArray) -> float:
        pos = float(np.sum(w * r.label))
        if pos == 0:
            return float("nan")
        return float(min(1.0, budget * float(np.sum(w)) / pos))

    metric.__name__ = f"budget_ceiling_{budget}"
    return metric


def recall_at_budget_among(budget: float, member: npt.NDArray[np.bool_]) -> Metric:
    """Recall at an alert budget over all items, counted only among the positives that are
    members (one laundering pattern, say). The budget is the same queue for everyone."""

    def metric(r: Ranked, w: FloatArray) -> float:
        m = member[r.order].astype(np.float64)
        starts = np.r_[0, r.block_end[:-1]]
        size = np.add.reduceat(w, starts)
        tp_m = np.add.reduceat(w * r.label * m, starts)
        pos_m = tp_m.sum()
        if pos_m == 0:
            return float("nan")
        k = budget * size.sum()
        before = np.r_[0.0, np.cumsum(size)[:-1]]
        take = np.clip((k - before) / np.maximum(size, 1e-300), 0.0, 1.0)
        return float(np.sum(take * tp_m) / pos_m)

    metric.__name__ = f"recall_at_budget_{budget}_among"
    return metric


METRICS: dict[str, Metric] = {
    "auprc": average_precision,
    "recall_at_1pct_fpr": recall_at_fpr(0.01),
    "recall_at_0.5pct_budget": recall_at_budget(0.005),
    "recall_at_1pct_budget": recall_at_budget(0.01),
    "ceiling_at_0.5pct_budget": budget_ceiling(0.005),
    "ceiling_at_1pct_budget": budget_ceiling(0.01),
}


@dataclass(frozen=True)
class Estimate:
    point: float
    lo: float
    hi: float
    n_items: int
    n_groups: int
    n_boot: int

    def as_text(self, digits: int = 3) -> str:
        return f"{self.point:.{digits}f} ({self.lo:.{digits}f} to {self.hi:.{digits}f})"


def group_weights(
    group_code: npt.NDArray[np.int64], n_groups: int, rng: np.random.Generator
) -> FloatArray:
    """One bootstrap resample over groups, as item weights. Every item in a group gets the
    number of times its group was drawn, so a group is never split."""
    counts = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
    return counts[group_code].astype(np.float64)


def _interval(
    point: float, samples: FloatArray, n: int, g: int, n_boot: int, alpha: float
) -> Estimate:
    finite = samples[np.isfinite(samples)]
    lo, hi = np.quantile(finite, [alpha / 2, 1 - alpha / 2]) if finite.size else (np.nan, np.nan)
    return Estimate(point, float(lo), float(hi), n, g, n_boot)


def evaluate(
    o: Outcomes,
    metrics: dict[str, Metric] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Estimate]:
    metrics = metrics or METRICS
    r = Ranked.of(o.label, o.score, o.group)
    ones = np.ones(len(o))
    points = {k: m(r, ones) for k, m in metrics.items()}
    rng = np.random.default_rng(seed)
    samples = {k: np.empty(n_boot) for k in metrics}
    for b in range(n_boot):
        w = group_weights(r.group_code, r.n_groups, rng)
        for k, m in metrics.items():
            samples[k][b] = m(r, w)
    return {k: _interval(points[k], samples[k], len(o), r.n_groups, n_boot, alpha) for k in metrics}


def align(a: Outcomes, b: Outcomes) -> tuple[Outcomes, Outcomes]:
    """The two runs in the same item order, or MismatchedItems if they scored different
    items, or the same items with different labels or groups."""
    if len(a) != len(b) or not np.array_equal(np.sort(a.item_id), np.sort(b.item_id)):
        only_a = np.setdiff1d(a.item_id, b.item_id).shape[0]
        only_b = np.setdiff1d(b.item_id, a.item_id).shape[0]
        raise MismatchedItems(
            f"runs scored different items ({only_a} only in A, {only_b} only in B)"
        )
    ia, ib = np.argsort(a.item_id), np.argsort(b.item_id)
    a2 = Outcomes(a.item_id[ia], a.group[ia], a.label[ia], a.score[ia])
    b2 = Outcomes(b.item_id[ib], b.group[ib], b.label[ib], b.score[ib])
    if not np.array_equal(a2.label, b2.label):
        raise MismatchedItems("runs carry different labels for the same items")
    if not np.array_equal(a2.group, b2.group):
        raise MismatchedItems("runs carry different groups for the same items")
    return a2, b2


def paired_delta(
    a: Outcomes,
    b: Outcomes,
    metrics: dict[str, Metric] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Estimate]:
    """B minus A on identical items, both scored on the same group resample each time.

    The budget ceilings are left out by default: they depend only on the labels, which
    identical items share, so their delta is zero by construction."""
    metrics = metrics or {k: m for k, m in METRICS.items() if not k.startswith("ceiling_")}
    a2, b2 = align(a, b)
    ra = Ranked.of(a2.label, a2.score, a2.group)
    rb = Ranked.of(b2.label, b2.score, b2.group)
    # Each run's Ranked sorts its own way; the group codes are the same codes per item, so a
    # resample is drawn once as group counts and mapped into each run's order.
    rng = np.random.default_rng(seed)
    ones = np.ones(len(a2))
    points = {k: m(rb, ones) - m(ra, ones) for k, m in metrics.items()}
    samples = {k: np.empty(n_boot) for k in metrics}
    for i in range(n_boot):
        counts = np.bincount(rng.integers(0, ra.n_groups, ra.n_groups), minlength=ra.n_groups)
        wa = counts[ra.group_code].astype(np.float64)
        wb = counts[rb.group_code].astype(np.float64)
        for k, m in metrics.items():
            samples[k][i] = m(rb, wb) - m(ra, wa)
    return {
        k: _interval(points[k], samples[k], len(a2), ra.n_groups, n_boot, alpha) for k in metrics
    }
