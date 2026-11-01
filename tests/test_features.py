"""Graph features on hand-built graphs with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from consort.graph.build import from_edges
from consort.graph.features import default_registry


def _table(src: list[int], dst: list[int], n: int, hops: int, cap: int = 0) -> pd.DataFrame:
    g = from_edges(n, src, dst, amount=np.ones(len(src)), t=np.arange(len(src)) * 3600.0)
    nodes = pd.DataFrame({"x": np.arange(n, dtype=float)})
    return default_registry(["x"], ("amount", "t")).compute(g, nodes, hops=hops, hub_cap=cap)


def test_hop_zero_computes_nothing() -> None:
    assert _table([0, 1], [1, 2], 3, hops=0).shape[1] == 0


def test_three_cycle_is_seen_by_every_member() -> None:
    t = _table([0, 1, 2, 3], [1, 2, 0, 0], 4, hops=2)
    assert list(t["cycle3"]) == [1, 1, 1, 0]


def test_a_path_is_not_a_cycle() -> None:
    t = _table([0, 1], [1, 2], 3, hops=2)
    assert t["cycle3"].sum() == 0


def test_fan_in_of_five() -> None:
    t = _table([1, 2, 3, 4, 5], [0, 0, 0, 0, 0], 6, hops=1)
    assert t.loc[0, "deg_in"] == 5 and t.loc[0, "uniq_in"] == 5
    assert t.loc[0, "amt_in_sum"] == 5 and t.loc[0, "deg_out"] == 0


def test_scatter_gather_width() -> None:
    # 0 scatters to 1, 2, 3, which all gather into 4.
    t = _table([0, 0, 0, 1, 2, 3], [1, 2, 3, 4, 4, 4], 5, hops=2)
    assert t.loc[0, "scatter_gather_width"] == 3
    assert t.loc[4, "gather_scatter_width"] == 3
    assert t.loc[1, "scatter_gather_width"] == 0


def test_reciprocal_counterparty() -> None:
    t = _table([0, 1, 0], [1, 0, 2], 3, hops=1)
    assert list(t["reciprocal"]) == [1, 1, 0]


def test_iterated_neighbour_mean_on_a_path() -> None:
    # Path 0 - 1 - 2 (directions do not matter to the neighbourhood), x = 0, 1, 2.
    t = _table([0, 1], [1, 2], 3, hops=2)
    assert list(t["nbr_mean_x_h1"]) == [1.0, 1.0, 1.0]
    assert list(t["nbr_max_x_h1"]) == [1.0, 2.0, 1.0]
    assert list(t["nbr_max_x_h2"]) == [2.0, 1.0, 2.0]


def test_hub_cap_keeps_the_most_recent_edges() -> None:
    # Node 0 receives from 1..6 in that order; a cap of 2 keeps 5 and 6.
    g = from_edges(7, [1, 2, 3, 4, 5, 6], [0] * 6)
    nbrs = g.neighbours(cap=2)
    assert sorted(nbrs[0].indices.tolist()) == [5, 6]
    # Each sender keeps its single edge, so it still sees node 0.
    assert nbrs[1].indices.tolist() == [0]


def test_degrees_are_exact_under_the_cap() -> None:
    t = _table([1, 2, 3, 4, 5, 6], [0] * 6, 7, hops=1, cap=2)
    assert t.loc[0, "deg_in"] == 6
    assert t.loc[0, "nbr_mean_x_h1"] == pytest.approx(5.5)


def test_velocity_counts_per_hour_of_span() -> None:
    # Three out-edges an hour apart: a two-hour span, 1.5 an hour.
    t = _table([0, 0, 0], [1, 2, 3], 4, hops=1)
    assert t.loc[0, "vel_out_per_h"] == pytest.approx(1.5)


def test_pagerank_sums_to_n() -> None:
    t = _table([0, 1, 2, 3], [1, 2, 0, 0], 4, hops=2)
    assert t["pagerank"].sum() == pytest.approx(4.0)
