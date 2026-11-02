"""The k-hop neighbourhood fetch: one algorithm, run over any edge store.

To score one node live, fetch every edge of the node itself (its degrees, amounts and
velocity are exact counts), then expand: each node within k-1 hops contributes its `cap` most
recent out-edges and `cap` most recent in-edges, and the nodes those reach form the next
frontier. Because the cap keeps the most recent edges, a node's sampled neighbours in the
fetched subgraph are exactly its sampled neighbours in the whole graph, for every node that was
expanded; so arm G's registry, run on the fetched subgraph, gives the node the same features
the batch run gave it. consort.serve.parity checks that, number for number.

PageRank is global, so it is never recomputed from a neighbourhood: the serving path reads it
from a stored column, as it would read any batch-maintained node property.

With `counters`, the node's own exact counts (degrees, amounts, velocity) are read from stored
columns too, as counters a feature service updates on every write, and the fetch starts from
the node's capped edges instead of its whole history. The features are the same; what a call
fetches is bounded by the cap even at a hub, whose whole history can run to tens of thousands
of edges (consort.serve.bench measures both).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from consort.graph.build import Graph, from_edges
from consort.graph.registry import FeatureRegistry

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
STORED = ("pagerank",)
# The node's exact counts over all its edges (consort.graph.features: degrees, amounts,
# velocity), which `counters` reads from storage instead of the fetched edges.
COUNTERS = (
    "deg_out",
    "deg_in",
    "uniq_out",
    "uniq_in",
    "reciprocal",
    "amt_out_sum",
    "amt_in_sum",
    "amt_out_max",
    "amt_in_max",
    "amt_pass_through",
    "vel_out_per_h",
    "vel_in_per_h",
)


@dataclass(frozen=True)
class Edges:
    edge_id: IntArray
    src: IntArray
    dst: IntArray
    order: IntArray
    attrs: dict[str, FloatArray]

    @staticmethod
    def concat(parts: Sequence[Edges]) -> Edges:
        """Union of edge sets, deduplicated by edge id, in edge-id order."""
        eid = np.concatenate([p.edge_id for p in parts])
        _, first = np.unique(eid, return_index=True)

        def cat(get: str) -> IntArray:
            return np.concatenate([getattr(p, get) for p in parts])[first]

        names = parts[0].attrs.keys()
        return Edges(
            edge_id=eid[first],
            src=cat("src"),
            dst=cat("dst"),
            order=cat("order"),
            attrs={k: np.concatenate([p.attrs[k] for p in parts])[first] for k in names},
        )


class EdgeSource(Protocol):
    def incident(self, node: int) -> Edges:
        """Every edge with the node at either end."""
        ...

    def capped(self, nodes: IntArray, cap: int) -> Edges:
        """For each node, its `cap` most recent out-edges and `cap` most recent in-edges."""
        ...


class MemorySource:
    """The in-process store: the edges indexed by each end, most recent first, as an adjacency
    list a feature service would hold in memory. A lookup costs the node's degree, not the
    graph's size (the first serving run, 2026-11-02, scanned every edge per call)."""

    def __init__(self, graph: Graph) -> None:
        self.g = graph
        idx = np.arange(graph.n_edges, dtype=np.int64)
        # Per end: edge ids grouped by node, most recent first, ties by larger edge index, the
        # order Graph.sampled keeps them in.
        self._by: dict[str, tuple[IntArray, IntArray]] = {}
        for end, owner in (("out", graph.src), ("in", graph.dst)):
            perm = np.lexsort((-idx, -graph.order, owner)).astype(np.int64)
            ptr = np.r_[0, np.cumsum(np.bincount(owner, minlength=graph.n_nodes))].astype(np.int64)
            self._by[end] = (ptr, perm)

    def _edges(self, idx: IntArray) -> Edges:
        g = self.g
        return Edges(
            idx.astype(np.int64),
            g.src[idx],
            g.dst[idx],
            g.order[idx],
            {k: v[idx] for k, v in g.edge_attrs.items()},
        )

    def _take(self, nodes: IntArray, cap: int | None) -> IntArray:
        parts = []
        for ptr, perm in self._by.values():
            starts = ptr[nodes]
            lens = ptr[nodes + 1] - starts
            if cap is not None:
                lens = np.minimum(lens, max(cap, 0))
            total = int(lens.sum())
            offsets = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
            parts.append(perm[np.repeat(starts, lens) + offsets])
        return np.unique(np.concatenate(parts))

    def incident(self, node: int) -> Edges:
        return self._edges(self._take(np.array([node], dtype=np.int64), None))

    def capped(self, nodes: IntArray, cap: int) -> Edges:
        return self._edges(self._take(np.asarray(nodes, dtype=np.int64), cap))


def fetch(source: EdgeSource, centre: int, hops: int, cap: int, counters: bool = False) -> Edges:
    first = source.capped(np.empty(0, np.int64), cap) if counters else source.incident(centre)
    parts = [first]
    seen = {centre}
    frontier = np.array([centre], dtype=np.int64)
    for _ in range(hops):
        got = source.capped(frontier, cap)
        parts.append(got)
        reached = np.union1d(got.src, got.dst)
        frontier = np.array(sorted(set(reached.tolist()) - seen), dtype=np.int64)
        seen.update(frontier.tolist())
        if frontier.size == 0:
            break
    return Edges.concat(parts)


def local_features(
    edges: Edges,
    centre: int,
    registry: FeatureRegistry,
    node_columns: pd.DataFrame,
    hops: int,
    cap: int,
    stored: pd.DataFrame | None = None,
    counters: bool = False,
) -> pd.Series:
    """The centre's feature row, computed by the registry on the fetched subgraph. Local ids
    keep the global order, so sparse sums run in the same order as in the batch run."""
    nodes = np.union1d(np.union1d(edges.src, edges.dst), [centre])
    local = np.searchsorted(nodes, edges.src), np.searchsorted(nodes, edges.dst)
    g = from_edges(len(nodes), local[0], local[1], order=edges.order, **edges.attrs)
    table = registry.compute(g, node_columns.iloc[nodes].reset_index(drop=True), hops, cap)
    row = table.iloc[int(np.searchsorted(nodes, centre))].copy()
    if stored is not None:
        for col in STORED + (COUNTERS if counters else ()):
            if col in row.index:
                row[col] = stored.iloc[centre][col]
    return row
