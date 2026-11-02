"""The columnar store: the edge table in DuckDB, fetched by indexed lookups and a window.

The module is named duckdb_store rather than duckdb so that it never shadows the library.
"""

from __future__ import annotations

from pathlib import Path

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
        cols = ", ".join(("edge_id", "src", "dst", "ord", *attrs))
        self._incident = f"SELECT {cols} FROM edges WHERE src = $n UNION ALL SELECT {cols} FROM edges WHERE dst = $n AND src <> $n"
        ranked = (
            "SELECT {cols}, row_number() OVER (PARTITION BY {end} ORDER BY ord DESC, edge_id DESC) AS rn "
            "FROM edges WHERE {end} IN (SELECT unnest($nodes))"
        )
        self._capped = (
            f"SELECT {cols} FROM ({ranked.format(cols=cols, end='src')}) WHERE rn <= $cap "
            f"UNION SELECT {cols} FROM ({ranked.format(cols=cols, end='dst')}) WHERE rn <= $cap"
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
        # Sorted by source so that the zone maps prune a lookup by src; an ART index on dst
        # serves the in-edge lookups.
        con.execute("CREATE OR REPLACE TABLE edges AS SELECT * FROM edges_df ORDER BY src, ord")
        con.unregister("edges_df")
        con.execute("CREATE INDEX IF NOT EXISTS edges_src ON edges (src)")
        con.execute("CREATE INDEX IF NOT EXISTS edges_dst ON edges (dst)")
        return cls(con, attrs)

    def _edges(self, df: pd.DataFrame) -> Edges:
        df = df.sort_values("edge_id")
        return Edges(
            edge_id=df["edge_id"].to_numpy(np.int64),
            src=df["src"].to_numpy(np.int64),
            dst=df["dst"].to_numpy(np.int64),
            order=df["ord"].to_numpy(np.int64),
            attrs={a: df[a].to_numpy(np.float64) for a in self.attrs},
        )

    def incident(self, node: int) -> Edges:
        return self._edges(self.con.execute(self._incident, {"n": node}).df())

    def capped(self, nodes: IntArray, cap: int) -> Edges:
        params = {"nodes": [int(v) for v in nodes], "cap": cap}
        return self._edges(self.con.execute(self._capped, params).df())
