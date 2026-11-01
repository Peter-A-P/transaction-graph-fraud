"""Occlusion explanations and the faithfulness number on a graph with a known answer."""

from __future__ import annotations

import numpy as np

from consort.explain.faithfulness import Explanation, explain, faithfulness
from consort.graph.build import from_edges
from consort.graph.neighbourhood import Edges, MemorySource


def test_occlusion_ranks_the_neighbour_the_score_depends_on() -> None:
    # Node 0 has neighbours 1..5; the score counts edges to node 3 only.
    g = from_edges(6, [1, 2, 3, 4, 5, 3], [0, 0, 0, 0, 0, 0])

    def score(edges: Edges, centre: int) -> float:
        return float(np.sum((edges.src == 3) & (edges.dst == centre)))

    e = explain(MemorySource(g), 0, hops=1, cap=10, score=score)
    assert e.neighbours[0] == 3 and e.importance[0] == 2.0
    assert sorted(e.neighbours) == [1, 2, 3, 4, 5]


def test_faithfulness_against_the_random_baseline() -> None:
    illicit = np.zeros(10, dtype=bool)
    illicit[[3, 7]] = True
    good = [
        Explanation(0, 1.0, [3, 1, 2, 4], [1, 0, 0, 0]),
        Explanation(5, 1.0, [7, 6, 8, 9], [1, 0, 0, 0]),
    ]
    f = faithfulness(good, illicit, k=1, n_random=400)
    assert f.n_graded == 2 and f.explained_hit_rate == 1.0
    assert abs(f.random_hit_rate - 0.25) < 0.06
    bad = [Explanation(0, 1.0, [1, 2, 4, 3], [1, 0, 0, 0])]
    assert faithfulness(bad, illicit, k=1).explained_hit_rate == 0.0


def test_nodes_with_nothing_to_find_are_not_graded() -> None:
    f = faithfulness([Explanation(0, 1.0, [1, 2], [0, 0])], np.zeros(3, dtype=bool), k=1)
    assert f.n_graded == 0
