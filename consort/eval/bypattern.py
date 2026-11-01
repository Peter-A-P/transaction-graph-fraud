"""Recall by laundering pattern: of each pattern's laundering transfers, the share an alert
budget over all transfers reaches. One queue for every pattern, as a team would run it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from consort.eval.metrics import Estimate, Metric, evaluate, recall_at_budget_among
from consort.eval.outcomes import Outcomes


def recall_by_pattern(
    o: Outcomes,
    pattern_of: pd.Series,
    budget: float = 0.01,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Estimate]:
    """`pattern_of` maps item_id to a pattern name (NONE for no named pattern)."""
    patterns = pattern_of.reindex(o.item_id).to_numpy()
    if pd.isna(patterns).any():
        raise ValueError("an outcome item has no pattern entry")
    metrics: dict[str, Metric] = {}
    for p in sorted(set(patterns[o.label == 1].tolist())):
        metrics[p] = recall_at_budget_among(budget, np.asarray(patterns == p))
    return evaluate(o, metrics=metrics, n_boot=n_boot, seed=seed)
