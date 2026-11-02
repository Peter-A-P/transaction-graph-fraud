"""The serving parity test, 09's idea: features computed on the serving path equal the
features computed in batch, for the same node on the same graph."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from consort.graph.neighbourhood import Edges, EdgeSource, fetch, local_features
from consort.graph.registry import FeatureRegistry

__all__ = ["Edges", "Mismatch", "check"]


@dataclass(frozen=True)
class Mismatch:
    node: int
    column: str
    batch: float
    serving: float


def check(
    source: EdgeSource,
    registry: FeatureRegistry,
    node_columns: pd.DataFrame,
    batch: pd.DataFrame,
    nodes: Sequence[int],
    hops: int,
    cap: int,
    counters: bool = False,
) -> list[Mismatch]:
    """Every (node, column) where the serving value differs from the batch value. Exact
    equality: the fetch is designed to reproduce the batch arithmetic, not approximate it."""
    out: list[Mismatch] = []
    for v in nodes:
        edges = fetch(source, v, hops, cap, counters=counters)
        row = local_features(
            edges, v, registry, node_columns, hops, cap, stored=batch, counters=counters
        )
        want = batch.iloc[v]
        for col in batch.columns:
            a, b = float(want[col]), float(row[col])
            if not (a == b or (np.isnan(a) and np.isnan(b))):
                out.append(Mismatch(v, col, a, b))
    return out
