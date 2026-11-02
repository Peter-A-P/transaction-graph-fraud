"""Explanation faithfulness, scored rather than shown.

An explanation of a flag names the neighbours that drove it. Here a neighbour's importance is
its occlusion effect: fetch the flagged node's neighbourhood (the serving fetch), drop one
direct neighbour and every edge it carries, rescore, and take the fall in score. The method is
the same for every arm, since it needs only a function from a neighbourhood to a score, so the
arms' explanations are compared on one footing.

Faithfulness is a number: among flagged nodes with at least one known-illicit direct
neighbour, the share whose top-k neighbours by importance include one, against the share for
k neighbours drawn at random. Labels are read here only to grade explanations, never to make
them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from consort.graph.neighbourhood import Edges, EdgeSource, fetch

ScoreFn = Callable[[Edges, int], float]


def _without(edges: Edges, node: int) -> Edges:
    keep = (edges.src != node) & (edges.dst != node)
    return Edges(
        edges.edge_id[keep],
        edges.src[keep],
        edges.dst[keep],
        edges.order[keep],
        {k: v[keep] for k, v in edges.attrs.items()},
    )


@dataclass(frozen=True)
class Explanation:
    node: int
    score: float
    neighbours: list[int]  # direct neighbours, most important first
    importance: list[float]


def explain(source: EdgeSource, node: int, hops: int, cap: int, score: ScoreFn) -> Explanation:
    edges = fetch(source, node, hops, cap)
    base = score(edges, node)
    direct = np.union1d(edges.dst[edges.src == node], edges.src[edges.dst == node])
    direct = direct[direct != node]
    imp = np.array([base - score(_without(edges, int(u)), node) for u in direct])
    order = np.argsort(-imp, kind="stable")
    return Explanation(node, base, direct[order].tolist(), imp[order].tolist())


PairScoreFn = Callable[[Edges, Edges], float]


def explain_transfer(
    source: EdgeSource, transfer: int, src: int, dst: int, hops: int, cap: int, score: PairScoreFn
) -> Explanation:
    """A transfer's explanation (IBM): the direct neighbours of either account, each occluded
    from both accounts' fetched neighbourhoods at once. `transfer` names the explanation."""
    es, ed = fetch(source, src, hops, cap), fetch(source, dst, hops, cap)
    base = score(es, ed)
    direct = np.union1d(
        np.union1d(es.dst[es.src == src], es.src[es.dst == src]),
        np.union1d(ed.dst[ed.src == dst], ed.src[ed.dst == dst]),
    )
    direct = direct[(direct != src) & (direct != dst)]
    imp = np.array([base - score(_without(es, int(u)), _without(ed, int(u))) for u in direct])
    order = np.argsort(-imp, kind="stable")
    return Explanation(transfer, base, direct[order].tolist(), imp[order].tolist())


@dataclass(frozen=True)
class Faithfulness:
    k: int
    n_graded: int
    explained_hit_rate: float
    random_hit_rate: float


def graded(
    explanations: Sequence[Explanation],
    illicit: npt.NDArray[np.bool_],
    k: int,
    seed: int = 0,
    n_random: int = 100,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per graded node (more than k direct neighbours, one of them known illicit): its id, whether its top k
    hold one (0 or 1), and the chance that k random neighbours do. Bootstrapping these over
    groups gives faithfulness its interval."""
    rng = np.random.default_rng(seed)
    nodes, hit, rand = [], [], []
    for e in explanations:
        nb = np.array(e.neighbours, dtype=np.int64)
        # A node with k or fewer neighbours has nothing to rank: any k of them is all of them.
        if nb.size <= k or not illicit[nb].any():
            continue
        nodes.append(e.node)
        hit.append(float(illicit[nb[:k]].any()))
        draws = [
            illicit[rng.choice(nb, min(k, nb.size), replace=False)].any() for _ in range(n_random)
        ]
        rand.append(float(np.mean(draws)))
    return np.array(nodes, dtype=np.int64), np.array(hit), np.array(rand)


def faithfulness(
    explanations: Sequence[Explanation],
    illicit: npt.NDArray[np.bool_],
    k: int,
    seed: int = 0,
    n_random: int = 100,
) -> Faithfulness:
    """Hit rate of the top-k explanatory neighbours against random k neighbours, over the
    explained nodes with more than k neighbours, one of them known illicit to find."""
    rng = np.random.default_rng(seed)
    hits, rand, graded = 0.0, 0.0, 0
    for e in explanations:
        nb = np.array(e.neighbours, dtype=np.int64)
        # A node with k or fewer neighbours has nothing to rank: any k of them is all of them.
        if nb.size <= k or not illicit[nb].any():
            continue
        graded += 1
        hits += float(illicit[nb[:k]].any())
        draws = [
            illicit[rng.choice(nb, min(k, nb.size), replace=False)].any() for _ in range(n_random)
        ]
        rand += float(np.mean(draws))
    if graded == 0:
        return Faithfulness(k, 0, float("nan"), float("nan"))
    return Faithfulness(k, graded, hits / graded, rand / graded)
