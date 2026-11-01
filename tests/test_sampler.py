"""Arm N's sampler sees the same neighbourhood as arm G's cap and the serving fetch."""

from __future__ import annotations

import numpy as np

from consort.graph.build import from_edges
from consort.graph.sampler import CappedSampler, local_ids


def _star_and_tail() -> tuple[int, list[int], list[int]]:
    # 1..6 pay hub 0 in that order; 6 then pays 7, and 7 pays 8.
    src = [1, 2, 3, 4, 5, 6, 6, 7]
    dst = [0, 0, 0, 0, 0, 0, 7, 8]
    return 9, src, dst


def test_hop_zero_is_the_seeds_alone() -> None:
    n, s, d = _star_and_tail()
    b = CappedSampler(from_edges(n, s, d), cap=2).batch([0], 0)
    assert b.nodes.tolist() == [0] and b.edge_index.shape == (2, 0)


def test_one_hop_keeps_the_most_recent_under_the_cap() -> None:
    n, s, d = _star_and_tail()
    g = from_edges(n, s, d)
    b = CappedSampler(g, cap=2).batch([0], 1)
    assert b.nodes[0] == 0 and sorted(b.nodes.tolist()) == [0, 5, 6]
    assert sorted(g.neighbours(cap=2)[0].indices.tolist()) == [5, 6]


def test_two_hops_reach_along_kept_edges_only() -> None:
    n, s, d = _star_and_tail()
    b = CappedSampler(from_edges(n, s, d), cap=2).batch([0], 2)
    assert sorted(b.nodes.tolist()) == [0, 5, 6, 7]
    # Every batch edge joins two batch nodes, in the graph's direction.
    pairs = {(int(b.nodes[u]), int(b.nodes[v])) for u, v in b.edge_index.T}
    assert pairs == {(5, 0), (6, 0), (6, 7)}


def test_seed_order_and_local_ids() -> None:
    n, s, d = _star_and_tail()
    b = CappedSampler(from_edges(n, s, d), cap=10).batch([7, 0, 7], 1)
    assert b.n_seeds == 2 and b.nodes[:2].tolist() == [7, 0]
    assert local_ids(b, [0, 7]).tolist() == [1, 0]


def test_deterministic() -> None:
    rng = np.random.default_rng(0)
    g = from_edges(50, rng.integers(0, 50, 400), rng.integers(0, 50, 400))
    smp = CappedSampler(g, cap=3)
    a, b = smp.batch([1, 2, 3], 2), smp.batch([1, 2, 3], 2)
    assert np.array_equal(a.nodes, b.nodes) and np.array_equal(a.edge_id, b.edge_id)
