"""The columnar store: the edges in DuckDB, one table per end, fetched by indexed lookups and
a window.

The module is named duckdb_store rather than duckdb so that it never shadows the library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import numpy.typing as npt
import pandas as pd

from consort.graph.build import Graph
from consort.graph.neighbourhood import Edges

IntArray = npt.NDArray[np.int64]


class DuckDBSource:
    def __init__(self, con: duckdb.DuckDBPyConnection, attrs: tuple[str, ...]) -> None:
        self.con = con
        self.attrs = attrs
        self.cols = ("edge_id", "src", "dst", "ord", *attrs)
        cols = ", ".join(self.cols)
        ecols = ", ".join(f"e.{c}" for c in self.cols)
        self._incident = (
            f"SELECT {cols} FROM out_edges WHERE src = $n "
            f"UNION ALL SELECT {cols} FROM in_edges WHERE dst = $n AND src <> $n"
        )
        # A node's `cap` most recent edges at one end are those whose sequence number there
        # exceeds its degree less the cap: a range read, where a window over the node's
        # edges cost twice as long per call.
        recent = (
            "SELECT {ecols} FROM {table} e JOIN {deg} d ON e.{end} = d.node "
            "WHERE d.node = ANY($nodes) AND e.{end} = ANY($nodes) AND e.seq > d.deg - $cap"
        )
        self._capped = (
            recent.format(ecols=ecols, table="out_edges", deg="out_deg", end="src")
            + " UNION "
            + recent.format(ecols=ecols, table="in_edges", deg="in_deg", end="dst")
        )

    @classmethod
    def from_graph(cls, graph: Graph, path: Path | str = ":memory:") -> DuckDBSource:
        con = duckdb.connect(str(path))
        attrs = tuple(sorted(graph.edge_attrs))
        df = pd.DataFrame(
            {
                "edge_id": np.arange(graph.n_edges, dtype=np.int64),
                "src": graph.src,
                "dst": graph.dst,
                "ord": graph.order,
                **{a: graph.edge_attrs[a] for a in attrs},
            }
        )
        con.register("edges_df", df)
        # One table per end, each sorted by its owner and then recency, so that a lookup by
        # owner reads a few contiguous row groups and the ART index finds them (one table
        # sorted by src left every in-edge lookup to a scan, in the first serving runs on
        # 2026-11-02). Each edge carries its sequence number at that end, oldest first, ties
        # by edge id, as Graph.sampled orders them: append-only, so a new edge takes the next
        # number and no stored row changes. The degree tables are the counters a write bumps.
        for table, deg, end in (("out_edges", "out_deg", "src"), ("in_edges", "in_deg", "dst")):
            con.execute(
                f"CREATE OR REPLACE TABLE {table} AS SELECT *, row_number() OVER "
                f"(PARTITION BY {end} ORDER BY ord, edge_id) AS seq FROM edges_df "
                f"ORDER BY {end}, seq DESC"
            )
            con.execute(f"CREATE INDEX IF NOT EXISTS {table}_{end} ON {table} ({end})")
            con.execute(
                f"CREATE OR REPLACE TABLE {deg} AS SELECT {end} AS node, count(*) AS deg "
                f"FROM edges_df GROUP BY {end} ORDER BY node"
            )
            con.execute(f"CREATE INDEX IF NOT EXISTS {deg}_node ON {deg} (node)")
        con.unregister("edges_df")
        return cls(con, attrs)

    def _edges(self, got: dict[str, npt.NDArray[np.generic]]) -> Edges:
        order = np.argsort(np.asarray(got["edge_id"]), kind="stable")

        def col(name: str, dtype: type[np.generic]) -> npt.NDArray[Any]:
            return np.asarray(got[name], dtype=dtype)[order]

        return Edges(
            edge_id=col("edge_id", np.int64),
            src=col("src", np.int64),
            dst=col("dst", np.int64),
            order=col("ord", np.int64),
            attrs={a: col(a, np.float64) for a in self.attrs},
        )

    def incident(self, node: int) -> Edges:
        return self._edges(self.con.execute(self._incident, {"n": node}).fetchnumpy())

    def capped(self, nodes: IntArray, cap: int) -> Edges:
        if len(nodes) == 0 or cap <= 0:
            return self._edges({c: np.empty(0) for c in self.cols})
        params = {"nodes": [int(v) for v in nodes], "cap": cap}
        return self._edges(self.con.execute(self._capped, params).fetchnumpy())
