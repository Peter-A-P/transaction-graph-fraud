"""Arm N's neighbour sampler: the same recency cap as arm G and the serving fetch.

PyTorch Geometric's samplers draw neighbours at random and need a native extension. This one
keeps, for every node, its `cap` most recent edges in each direction (Graph.sampled), so the
GNN at hop k sees the same neighbourhood arm G aggregates over and the serving path fetches:
the hop curve then compares models on identical neighbourhoods, and a GNN's live fetch is the
one already measured. It is deterministic, so a run is reproducible without a sampler seed.

A batch of seed nodes expands k times along kept edges; the batch's edges are every kept edge
leaving an expanded node (distance < k) to a node in the batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from consort.graph.build import Graph

IntArray = npt.NDArray[np.int64]


def _gather(indptr: IntArray, nodes: IntArray) -> tuple[IntArray, IntArray]:
    """CSR positions of every entry in the rows `nodes`, and the row each came from."""
    starts, ends = indptr[nodes], indptr[nodes + 1]
    lens = ends - starts
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    owner = np.repeat(nodes, lens)
    offsets = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
    return np.repeat(starts, lens) + offsets, owner


@dataclass(frozen=True)
class Batch:
    nodes: IntArray  # global ids; the seeds first, in the order given
    edge_index: IntArray  # 2 x E, local ids, direction as in the graph
    edge_id: IntArray  # global edge ids
    n_seeds: int


class CappedSampler:
    def __init__(self, graph: Graph, cap: int) -> None:
        out_keep, in_keep = graph.sampled(cap)
        kept = np.flatnonzero(out_keep | in_keep)
        # Undirected incidence over kept edges: each kept edge is listed under the endpoint(s)
        # that kept it, so a node's list is exactly its sampled neighbours.
        owner = np.concatenate([graph.src[out_keep], graph.dst[in_keep]])
        other = np.concatenate([graph.dst[out_keep], graph.src[in_keep]])
        eid = np.concatenate([np.flatnonzero(out_keep), np.flatnonzero(in_keep)])
        order = np.argsort(owner, kind="stable")
        self.indptr = np.r_[0, np.cumsum(np.bincount(owner, minlength=graph.n_nodes))].astype(
            np.int64
        )
        self.other = other[order].astype(np.int64)
        self.eid = eid[order].astype(np.int64)
        self.g = graph
        self.n_kept = int(kept.shape[0])

    def batch(self, seeds: npt.ArrayLike, hops: int) -> Batch:
        seeds_a = np.asarray(seeds, dtype=np.int64)
        uniq_seeds = list(dict.fromkeys(seeds_a.tolist()))
        seen = set(uniq_seeds)
        ordered = list(uniq_seeds)
        frontier = np.array(uniq_seeds, dtype=np.int64)
        expanded: list[IntArray] = []
        for _ in range(hops):
            if frontier.size == 0:
                break
            expanded.append(frontier)
            pos, _ = _gather(self.indptr, frontier)
            new = [v for v in dict.fromkeys(self.other[pos].tolist()) if v not in seen]
            seen.update(new)
            ordered.extend(new)
            frontier = np.array(new, dtype=np.int64)
        nodes = np.array(ordered, dtype=np.int64)
        if expanded:
            pos, _ = _gather(self.indptr, np.concatenate(expanded))
            eids = np.unique(self.eid[pos])
        else:
            eids = np.empty(0, np.int64)
        sorter = np.argsort(nodes)
        src, dst = self.g.src[eids], self.g.dst[eids]
        inside = np.isin(src, nodes) & np.isin(dst, nodes)
        eids, src, dst = eids[inside], src[inside], dst[inside]
        local = np.stack(
            [
                sorter[np.searchsorted(nodes, src, sorter=sorter)],
                sorter[np.searchsorted(nodes, dst, sorter=sorter)],
            ]
        )
        return Batch(
            nodes=nodes, edge_index=local.astype(np.int64), edge_id=eids, n_seeds=len(uniq_seeds)
        )


def local_ids(batch: Batch, ids: npt.ArrayLike) -> IntArray:
    """Local positions of global node ids that are in the batch."""
    sorter = np.argsort(batch.nodes)
    a = np.asarray(ids, dtype=np.int64)
    return sorter[np.searchsorted(batch.nodes, a, sorter=sorter)].astype(np.int64)
