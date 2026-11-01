"""One graph, from one edge table, for every arm and for serving.

A Graph is directed and may hold parallel edges (the IBM set has many transfers between the
same two accounts). Nodes are 0..n-1 in a fixed order; the loaders map dataset identifiers to
that order once. Edges carry an order key (a timestamp, or the edge's position in the source
file where there is none) so that the hub cap keeps the same neighbours in batch and in the
serving fetch: each node keeps its most recent `cap` edges in each direction, ties broken by
the later edge index.
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

    def sampled(self, cap: int) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """The hub cap as neighbour sampling: (out, in) masks of the edges each node keeps,
        its `cap` most recent out-edges and its `cap` most recent in-edges. A cap of 0 keeps
        everything. The serving fetch reproduces this with ORDER BY order DESC LIMIT cap."""
        if cap <= 0:
            every = np.ones(self.n_edges, dtype=bool)
            return every, every
        return _keep_recent(self.src, self.order, cap), _keep_recent(self.dst, self.order, cap)

    def out_matrix(self, cap: int = 0) -> sp.csr_matrix:
        """Binary, deduplicated: M[u, v] = 1 if v is among u's kept out-neighbours."""
        keep, _ = self.sampled(cap)
        return _binary(self.src[keep], self.dst[keep], self.n_nodes)

    def in_matrix(self, cap: int = 0) -> sp.csr_matrix:
        """Binary, deduplicated: M[u, v] = 1 if v is among u's kept in-neighbours."""
        _, keep = self.sampled(cap)
        return _binary(self.dst[keep], self.src[keep], self.n_nodes)

    def neighbours(self, cap: int = 0) -> sp.csr_matrix:
        """Undirected sampled neighbourhood of each node (row), self-loops removed."""
        M = (self.out_matrix(cap) + self.in_matrix(cap)).tocsr()
        M.setdiag(0)
        M.eliminate_zeros()
        M.data[:] = 1.0
        return M

    def adjacency(self, weight: str | None = None) -> sp.csr_matrix:
        """Directed, uncapped: A[u, v] = number (or total weight) of edges u -> v."""
        data = np.ones(self.n_edges) if weight is None else self.edge_attrs[weight]
        A = sp.csr_matrix((data, (self.src, self.dst)), shape=(self.n_nodes, self.n_nodes))
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


def _binary(rows: IntArray, cols: IntArray, n: int) -> sp.csr_matrix:
    M = sp.csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(n, n))
    M.sum_duplicates()
    M.data[:] = 1.0
    return M


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
