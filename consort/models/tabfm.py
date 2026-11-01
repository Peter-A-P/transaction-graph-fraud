"""The TabFM column: Google's zero-shot tabular foundation model on arm R's and arm G's tables.

TabFM is in-context: it is not trained, it is handed labelled rows as context and asked for
probabilities on query rows. Its context is bounded by memory, so it sees a sample of the
training rows of a stated size that keeps every positive (fraud is rare, and a uniform sample
would hand it a handful) and fills the rest with negatives drawn evenly across split groups.
The size is a reported parameter; the limit is a finding.

The model is imported lazily (the `tabfm` extra) and its weights, under the TabFM
Non-Commercial License v1.0, are never committed, bundled or exported; see docs/models.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

# The model card's guidance is up to 500 features; wider tables are subsampled per ensemble
# member by TabFM itself, which this wrapper refuses rather than let happen silently.
MAX_FEATURES = 500
TABFM_VERSION = "1.0.1"
TABFM_WEIGHTS = "google/tabfm-1.0.0-pytorch"


class ContextTooSmall(Exception):
    """The context size cannot hold every positive."""


def context_sample(
    labels: npt.ArrayLike, groups: npt.ArrayLike, size: int, seed: int
) -> npt.NDArray[np.int64]:
    """Indices of a context of exactly `size` rows (or every row, if fewer) holding every
    positive, with negatives spread across groups as evenly as their counts allow.
    Deterministic in the seed; returned sorted."""
    y = np.asarray(labels)
    g = np.asarray(groups)
    n = y.shape[0]
    if size >= n:
        return np.arange(n, dtype=np.int64)
    pos = np.flatnonzero(y == 1)
    if pos.shape[0] > size:
        raise ContextTooSmall(f"{pos.shape[0]} positives do not fit a context of {size}")
    rng = np.random.default_rng(seed)
    neg = np.flatnonzero(y != 1)
    need = size - pos.shape[0]
    # Allocate negatives to groups in proportion to each group's negatives, then fill any
    # remainder from the largest fractional shares, so the total is exact.
    values, inverse = np.unique(g[neg], return_inverse=True)
    counts = np.bincount(inverse, minlength=values.shape[0])
    share = counts / counts.sum() * need
    alloc = np.minimum(np.floor(share).astype(np.int64), counts)
    short = need - int(alloc.sum())
    if short:
        frac = np.where(alloc < counts, share - alloc, -1.0)
        for k in np.argsort(-frac, kind="stable")[:short]:
            alloc[k] += 1
    chosen = [
        rng.choice(neg[inverse == k], int(alloc[k]), replace=False) for k in range(values.shape[0])
    ]
    picked = np.concatenate([pos, *chosen]).astype(np.int64)
    return np.sort(picked)


def context_swap_schedule(test_steps: list[int], window: int) -> dict[int, list[int]]:
    """For each test step s, the `window` steps immediately before it, whose labels are
    assumed to have arrived by then. XGBoost retrained on the same rows uses the same map."""
    return {s: list(range(s - window, s)) for s in test_steps}


@dataclass
class TabFMScorer:
    """Thin wrapper over tabfm.TabFMClassifier with every setting stated."""

    context_size: int
    seed: int
    n_estimators: int = 32
    checkpoint: Path | None = None
    device: str | None = None
    _clf: Any = None

    def fit(self, X: pd.DataFrame, y: npt.ArrayLike, groups: npt.ArrayLike) -> TabFMScorer:
        if X.shape[1] > MAX_FEATURES:
            raise ValueError(f"{X.shape[1]} features; TabFM is optimised for {MAX_FEATURES}")
        idx = context_sample(y, groups, self.context_size, self.seed)
        import tabfm
        from tabfm import tabfm_v1_0_0_pytorch

        model = tabfm_v1_0_0_pytorch.load(
            checkpoint_path=str(self.checkpoint) if self.checkpoint else None, device=self.device
        )
        self._clf = tabfm.TabFMClassifier(
            model=model,
            n_estimators=self.n_estimators,
            # Set explicitly: the default (None) uses every context row, and the README and
            # the code disagree on what the default is.
            max_num_rows=self.context_size,
            random_state=self.seed,
        )
        self._clf.fit(X.iloc[idx], np.asarray(y)[idx])
        return self

    def predict(self, X: pd.DataFrame, batch: int = 10_000) -> npt.NDArray[np.float64]:
        if self._clf is None:
            raise RuntimeError("fit before predict")
        classes = list(self._clf.classes_)
        col = classes.index(1)
        parts = [
            np.asarray(self._clf.predict_proba(X.iloc[i : i + batch]))[:, col]
            for i in range(0, len(X), batch)
        ]
        return np.concatenate(parts).astype(np.float64)
