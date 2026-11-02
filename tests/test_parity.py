"""Serving parity: for ten accounts at one and two hops, the features the fetch path
computes equal the batch features exactly, from memory and from DuckDB."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from consort.data import ibm
from consort.data.tasks import IBM_NODE_COLUMNS, ibm_node_table
from consort.graph.features import default_registry
from consort.graph.neighbourhood import MemorySource, fetch
from consort.serve import parity
from consort.serve.duckdb_store import DuckDBSource
from tests.toy import write_ibm


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> ibm.IBM:
    root: Path = tmp_path_factory.mktemp("ibm")
    return ibm.load(write_ibm(root, n_background=4000, n_cycles=40), "T")


@pytest.mark.parametrize("hops", [1, 2])
@pytest.mark.parametrize("backend", ["memory", "duckdb"])
def test_serving_features_equal_batch_features(toy: ibm.IBM, hops: int, backend: str) -> None:
    g = toy.graph
    cap = 5  # small, so that the cap binds on most accounts
    reg = default_registry(IBM_NODE_COLUMNS, ("amount", "t"))
    nodes = ibm_node_table(g)
    batch = reg.compute(g, nodes, hops, cap)
    source = MemorySource(g) if backend == "memory" else DuckDBSource.from_graph(g)
    # The ten busiest accounts, where the cap matters most, and ten at random.
    busiest = np.argsort(-np.bincount(g.src, minlength=g.n_nodes))[:10]
    rng = np.random.default_rng(0)
    chosen = [*busiest.tolist(), *rng.choice(g.n_nodes, 10, replace=False).tolist()]
    bad = parity.check(source, reg, nodes, batch, chosen, hops, cap)
    assert bad == []


def test_the_cap_bounds_the_fetch(toy: ibm.IBM) -> None:
    g = toy.graph
    hub = int(np.argmax(np.bincount(g.src, minlength=g.n_nodes)))
    small = fetch(MemorySource(g), hub, 2, 3)
    large = fetch(MemorySource(g), hub, 2, 50)
    assert small.edge_id.shape[0] < large.edge_id.shape[0]


def test_duckdb_and_memory_fetch_the_same_edges(toy: ibm.IBM) -> None:
    g = toy.graph
    mem, db = MemorySource(g), DuckDBSource.from_graph(g)
    for v in (0, 5, 77):
        a, b = fetch(mem, v, 2, 4), fetch(db, v, 2, 4)
        assert np.array_equal(a.edge_id, b.edge_id)


def test_parity_catches_a_fetch_one_hop_short(toy: ibm.IBM) -> None:
    """The negative control: batch features at two hops against a serving path that fetches
    one. If this passed, the parity test would not be testing anything."""
    g = toy.graph
    reg = default_registry(IBM_NODE_COLUMNS, ("amount", "t"))
    nodes = ibm_node_table(g)
    batch = reg.compute(g, nodes, 2, 5)

    class ShortSource(MemorySource):
        def capped(self, frontier: npt.NDArray[np.int64], cap: int) -> parity.Edges:
            got = super().capped(frontier, cap)
            # Drop every edge that reaches past the first frontier's own edges.
            keep = np.isin(got.src, frontier) & np.isin(got.dst, frontier) | (frontier.size == 1)
            return parity.Edges(
                got.edge_id[keep],
                got.src[keep],
                got.dst[keep],
                got.order[keep],
                {k: v[keep] for k, v in got.attrs.items()},
            )

    busiest = np.argsort(-np.bincount(g.src, minlength=g.n_nodes))[:10].tolist()
    bad = parity.check(ShortSource(g), reg, nodes, batch, busiest, 2, 5)
    assert {m.column for m in bad} & {"scatter_gather_width", "nbr_mean_log_deg_in_h2"}


def test_stores_keep_the_same_edges_when_recency_ties() -> None:
    # IBM's timestamps tie often (minute resolution): every store must break ties by the
    # larger edge id, as Graph.sampled does, or the cap keeps different edges.
    from consort.graph.build import _keep_recent, from_edges

    rng = np.random.default_rng(3)
    n, m = 60, 3000
    g = from_edges(n, rng.integers(0, n, m), rng.integers(0, n, m), order=rng.integers(0, 4, m))
    mem, db = MemorySource(g), DuckDBSource.from_graph(g)
    out_keep, in_keep = _keep_recent(g.src, g.order, 3), _keep_recent(g.dst, g.order, 3)
    for _ in range(20):
        nodes = rng.choice(n, 5, replace=False)
        ref = np.flatnonzero((out_keep & np.isin(g.src, nodes)) | (in_keep & np.isin(g.dst, nodes)))
        assert np.array_equal(mem.capped(nodes, 3).edge_id, ref)
        assert np.array_equal(db.capped(nodes, 3).edge_id, ref)
    for v in range(0, n, 7):
        ref = np.flatnonzero((g.src == v) | (g.dst == v))
        assert np.array_equal(mem.incident(v).edge_id, ref)
