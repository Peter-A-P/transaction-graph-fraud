"""The TabFM context sampler: every positive kept, size respected, deterministic by seed."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from consort.models.tabfm import ContextTooSmall, context_sample, context_swap_schedule


def _data(
    n: int = 10_000, rate: float = 0.02
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    rng = np.random.default_rng(5)
    return (rng.random(n) < rate).astype(int), rng.integers(1, 35, n)


def test_every_positive_is_kept_and_the_size_is_exact() -> None:
    y, g = _data()
    idx = context_sample(y, g, 2000, seed=1)
    assert idx.shape[0] == 2000 and np.unique(idx).shape[0] == 2000
    assert set(np.flatnonzero(y == 1)) <= set(idx.tolist())


def test_deterministic_by_seed() -> None:
    y, g = _data()
    assert np.array_equal(context_sample(y, g, 1500, seed=3), context_sample(y, g, 1500, seed=3))
    assert not np.array_equal(
        context_sample(y, g, 1500, seed=3), context_sample(y, g, 1500, seed=4)
    )


def test_negatives_follow_the_groups() -> None:
    y, g = _data()
    idx = context_sample(y, g, 3000, seed=0)
    neg = idx[y[idx] == 0]
    full = np.bincount(g[y == 0], minlength=35) / (y == 0).sum()
    got = np.bincount(g[neg], minlength=35) / neg.shape[0]
    assert np.abs(full - got).max() < 0.002


def test_too_many_positives_is_refused() -> None:
    y = np.ones(100, dtype=int)
    with pytest.raises(ContextTooSmall):
        context_sample(y, np.zeros(100), 50, seed=0)


def test_small_tables_are_taken_whole() -> None:
    y, g = _data(n=300)
    assert context_sample(y, g, 1000, seed=0).shape[0] == 300


def test_context_swap_uses_only_earlier_steps() -> None:
    sched = context_swap_schedule([43, 44], window=4)
    assert sched == {43: [39, 40, 41, 42], 44: [40, 41, 42, 43]}
