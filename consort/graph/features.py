"""Arm G's graph features, to a stated hop count, under a hub cap. None reads a label.

Hop 1 is what a node's own edges and immediate counterparties show: degrees, distinct
counterparties, amounts in and out, how fast the edges came, how many counterparties also pay
back, and the mean and max of the neighbours' row features. Hop h > 1 iterates the neighbour
aggregates once more per hop (the mean of the neighbours' hop h-1 means, the max of their hop
h-1 maxes), which is what a GraphSAGE layer sees at depth h without the learned weights. Hop 2
adds the shapes that need a second step: directed 3-cycles through the node, the widest
scatter-gather leaving it and the widest gather-scatter arriving at it, and PageRank.

Counts over the node's own edges (degrees, amounts, velocity) are exact; everything that
reaches past them goes through the hub cap, so that a hub cannot make two hops explode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from consort.graph.registry import FeatureContext, FeatureRegistry, FeatureSpec

FloatArray = npt.NDArray[np.float64]

# Rows of a sparse product computed at a time, so that A @ A under the cap stays in memory.
CHUNK = 50_000


def _row_counts(M: sp.csr_matrix) -> FloatArray:
    return np.diff(M.indptr).astype(np.float64)


def degrees(ctx: FeatureContext) -> Mapping[str, FloatArray]:
    g = ctx.graph
    n = g.n_nodes
    out_m, in_m = g.out_matrix(), g.in_matrix()
    return {
        "deg_out": np.bincount(g.src, minlength=n).astype(np.float64),
        "deg_in": np.bincount(g.dst, minlength=n).astype(np.float64),
        "uniq_out": _row_counts(out_m),
        "uniq_in": _row_counts(in_m),
        # Counterparties that both send to and receive from the node: the two-step cycle.
        "reciprocal": _row_counts(out_m.multiply(in_m).tocsr()),
    }


def amounts(ctx: FeatureContext) -> Mapping[str, FloatArray]:
    g = ctx.graph
    n = g.n_nodes
    a = g.edge_attrs["amount"]
    out_sum = np.bincount(g.src, weights=a, minlength=n)
    in_sum = np.bincount(g.dst, weights=a, minlength=n)
    out_max = np.zeros(n)
    in_max = np.zeros(n)
    np.maximum.at(out_max, g.src, a)
    np.maximum.at(in_max, g.dst, a)
    return {
        "amt_out_sum": out_sum,
        "amt_in_sum": in_sum,
        "amt_out_max": out_max,
        "amt_in_max": in_max,
        # Money that passes through: the smaller of in and out over the larger.
        "amt_pass_through": np.minimum(in_sum, out_sum)
        / np.maximum(np.maximum(in_sum, out_sum), 1e-9),
    }


def velocity(ctx: FeatureContext) -> Mapping[str, FloatArray]:
    """Edges per hour of the node's active span, in and out; the span floored at an hour."""
    g = ctx.graph
    n = g.n_nodes
    t = g.edge_attrs["t"]
    out: dict[str, FloatArray] = {}
    for name, end in (("out", g.src), ("in", g.dst)):
        first = np.full(n, np.inf)
        last = np.full(n, -np.inf)
        np.minimum.at(first, end, t)
        np.maximum.at(last, end, t)
        count = np.bincount(end, minlength=n).astype(np.float64)
        span_h = np.where(count > 0, np.maximum((last - first) / 3600.0, 1.0), 1.0)
        out[f"vel_{name}_per_h"] = count / span_h
    return out


def _mean_agg(N: sp.csr_matrix, x: FloatArray) -> FloatArray:
    deg = _row_counts(N)
    return np.asarray(N @ x) / np.maximum(deg, 1.0)


def _max_agg(N: sp.csr_matrix, x: FloatArray) -> FloatArray:
    out = np.zeros(N.shape[0])
    has = np.diff(N.indptr) > 0
    rows = np.repeat(np.arange(N.shape[0]), np.diff(N.indptr))
    vals = x[N.indices]
    res = np.full(N.shape[0], -np.inf)
    np.maximum.at(res, rows, vals)
    out[has] = res[has]
    return out


def neighbour_aggregates(columns: Sequence[str]) -> FeatureSpec:
    def derive(ctx: FeatureContext) -> Mapping[str, FloatArray]:
        N = ctx.graph.neighbours(ctx.hub_cap)
        out: dict[str, FloatArray] = {}
        for c in columns:
            mean_prev = max_prev = ctx.column(c)
            for h in range(1, ctx.hops + 1):
                mean_prev = _mean_agg(N, mean_prev)
                max_prev = _max_agg(N, max_prev)
                out[f"nbr_mean_{c}_h{h}"] = mean_prev
                out[f"nbr_max_{c}_h{h}"] = max_prev
        return out

    return FeatureSpec(
        "neighbour_aggregates",
        tuple(columns),
        1,
        derive,
        "Mean and max of the neighbours' row features, iterated once per hop, under the cap",
    )


def _chunked_rows(
    left: sp.csr_matrix, right: sp.csr_matrix, reduce: str, mask: sp.csr_matrix | None
) -> FloatArray:
    """Per-row reduction of left @ right, computed CHUNK rows at a time. With mask, the
    product is first multiplied elementwise by mask's rows."""
    n = left.shape[0]
    out = np.zeros(n)
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        P = (left[start:stop] @ right).tocsr()
        if mask is not None:
            P = P.multiply(mask[start:stop]).tocsr()
        else:
            # Drop the 2-paths that return to their start: u -> x -> u is not a shape.
            C = P.tocoo()
            off = C.col != C.row + start
            P = sp.csr_matrix((C.data[off], (C.row[off], C.col[off])), shape=C.shape)
        if reduce == "sum":
            out[start:stop] = np.asarray(P.sum(axis=1)).ravel()
        else:
            out[start:stop] = P.max(axis=1).toarray().ravel()
    return out


def motifs(ctx: FeatureContext) -> Mapping[str, FloatArray]:
    g = ctx.graph
    out_m = g.out_matrix(ctx.hub_cap)
    in_m = g.in_matrix(ctx.hub_cap)
    return {
        # Directed 3-cycles u -> v -> w -> u: two out-steps, closed by one of u's in-edges.
        "cycle3": _chunked_rows(out_m, out_m, "sum", in_m),
        # The most distinct intermediaries on 2-paths from u to any one other node: u scatters
        # to many who gather into one.
        "scatter_gather_width": _chunked_rows(out_m, out_m, "max", None),
        # The same arriving at u: many sources each paying intermediaries who all pay u.
        "gather_scatter_width": _chunked_rows(in_m, in_m, "max", None),
    }


def pagerank(
    ctx: FeatureContext, damping: float = 0.85, iters: int = 50
) -> Mapping[str, FloatArray]:
    """On the full directed graph, power iteration; dangling mass spread uniformly. A batch
    property: the serving path reads it as a stored column rather than recomputing it."""
    A = ctx.graph.adjacency()
    n = A.shape[0]
    out_w = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, out_w, out=np.zeros(n), where=out_w > 0)
    P = sp.diags(inv) @ A
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        dangling = r[out_w == 0].sum()
        r = damping * (P.T @ r + dangling / n) + (1 - damping) / n
    return {"pagerank": r * n}


def default_registry(row_columns: Sequence[str], edge_attrs: Sequence[str] = ()) -> FeatureRegistry:
    """The arm G registry for a dataset whose row features are `row_columns` and whose edges
    carry `edge_attrs` (IBM: amount and t; Elliptic: none)."""
    reg = FeatureRegistry()
    reg.register(
        FeatureSpec("degrees", (), 1, degrees, "Degrees, distinct counterparties, reciprocity")
    )
    if "amount" in edge_attrs:
        reg.register(FeatureSpec("amounts", (), 1, amounts, "Amounts in and out, sum and max"))
    if "t" in edge_attrs:
        reg.register(FeatureSpec("velocity", (), 1, velocity, "Edges per hour of active span"))
    reg.register(neighbour_aggregates(row_columns))
    reg.register(
        FeatureSpec("motifs", (), 2, motifs, "3-cycles, scatter-gather and gather-scatter width")
    )
    reg.register(FeatureSpec("pagerank", (), 2, pagerank, "PageRank on the full graph"))
    return reg
