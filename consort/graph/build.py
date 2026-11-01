"""One graph, from one edge table, for every arm and for serving.

A Graph is directed and may hold parallel edges (the IBM set has many transfers between the
same two accounts). Nodes are 0..n-1 in a fixed order; the loaders map dataset identifiers to
that order once. Edges carry an order key (a timestamp, or the edge's position in the source
file where there is none) so that the hub cap keeps the same neighbours in batch and in the
serving fetch: the most recent `cap` edges per node and direction, ties broken by edge index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class Graph:
    n_nodes: int
    src: IntArray
    dst: IntArray
    # Recency order of each edge; larger is more recent. Elliptic: file position.
    order: IntArray
    # Edge attributes by name (IBM: amount paid, payment format code). May be empty.
    edge_attrs: dict[str, FloatArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        m = self.src.shape[0]
        if self.dst.shape != (m,) or self.order.shape != (m,):
            raise ValueError("src, dst and order must have one entry per edge")
        for name, values in self.edge_attrs.items():
            if values.shape != (m,):
                raise ValueError(f"edge attribute {name!r} has shape {values.shape}")
        if m and (self.src.min() < 0 or max(self.src.max(), self.dst.max()) >= self.n_nodes):
            raise ValueError("edge endpoint outside 0..n_nodes-1")

    @property
    def n_edges(self) -> int:
        return int(self.src.shape[0])

    def capped_edges(self, cap: int) -> npt.NDArray[np.bool_]:
        """Mask of edges kept when each node keeps at most `cap` most recent out-edges and
        `cap` most recent in-edges. An edge survives only if both of its endpoints keep it."""
        if cap <= 0:
            return np.ones(self.n_edges, dtype=bool)
        return _keep_recent(self.src, self.order, cap) & _keep_recent(self.dst, self.order, cap)

    def adjacency(self, cap: int = 0, weight: str | None = None) -> sp.csr_matrix:
        """Directed adjacency, A[u, v] = number (or total weight) of kept edges u -> v."""
        keep = self.capped_edges(cap)
        data = np.ones(int(keep.sum())) if weight is None else self.edge_attrs[weight][keep]
        A = sp.csr_matrix(
            (data, (self.src[keep], self.dst[keep])), shape=(self.n_nodes, self.n_nodes)
        )
        A.sum_duplicates()
        return A

    @cached_property
    def components(self) -> IntArray:
        """Weakly connected component of each node, the bootstrap group on synthetic data."""
        A = self.adjacency()
        _, labels = sp.csgraph.connected_components(A, directed=True, connection="weak")
        return labels.astype(np.int64)


def _keep_recent(owner: IntArray, order: IntArray, cap: int) -> npt.NDArray[np.bool_]:
    """For each owner, keep its `cap` edges with the largest order; ties by larger index."""
    m = owner.shape[0]
    idx = np.arange(m, dtype=np.int64)
    # Sort by owner, then most recent first, then larger edge index first.
    perm = np.lexsort((-idx, -order, owner))
    sorted_owner = owner[perm]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_owner)) + 1]
    rank = idx - np.repeat(starts, np.diff(np.r_[starts, m]))
    keep = np.zeros(m, dtype=bool)
    keep[perm[rank < cap]] = True
    return keep


def from_edges(
    n_nodes: int,
    src: npt.ArrayLike,
    dst: npt.ArrayLike,
    order: npt.ArrayLike | None = None,
    **edge_attrs: npt.ArrayLike,
) -> Graph:
    s = np.asarray(src, dtype=np.int64)
    o = np.arange(s.shape[0], dtype=np.int64) if order is None else np.asarray(order, np.int64)
    return Graph(
        n_nodes=n_nodes,
        src=s,
        dst=np.asarray(dst, dtype=np.int64),
        order=o,
        edge_attrs={k: np.asarray(v, dtype=np.float64) for k, v in edge_attrs.items()},
    )
