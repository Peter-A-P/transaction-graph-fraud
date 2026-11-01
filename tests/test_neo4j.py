"""The Neo4j store fetches exactly what the memory store fetches. Runs only against a live
server named by NEO4J_URI (with NEO4J_USER and NEO4J_PASSWORD); skipped otherwise."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from consort.data import ibm
from consort.graph.neighbourhood import MemorySource, fetch
from tests.toy import write_ibm

URI = os.environ.get("NEO4J_URI")
pytestmark = pytest.mark.skipif(URI is None, reason="no Neo4j server (NEO4J_URI unset)")


def test_neo4j_fetch_equals_memory_fetch(tmp_path: Path) -> None:
    neo4j = pytest.importorskip("neo4j")
    from consort.serve.neo4j_store import Neo4jSource

    data = ibm.load(write_ibm(tmp_path, n_background=800, n_cycles=8), "T")
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"])
    with neo4j.GraphDatabase.driver(URI, auth=auth) as driver:
        db = Neo4jSource.load(driver, data.graph)
        mem = MemorySource(data.graph)
        for v in (0, 3, 50):
            a, b = fetch(mem, v, 2, 4), fetch(db, v, 2, 4)
            assert np.array_equal(a.edge_id, b.edge_id)
            assert np.array_equal(a.attrs["amount"], b.attrs["amount"])
