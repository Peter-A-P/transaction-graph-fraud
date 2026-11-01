"""The graph-database store: the same fetch in Cypher, against Neo4j.

Accounts are (:Account {id}) nodes and transfers are [:SENT {edge_id, ord, ...}]
relationships. The capped expansion orders each node's relationships by recency and keeps
the first `cap`, in each direction, which is the rule every arm and the DuckDB store use. A
uniqueness constraint on Account.id gives the index the lookups start from.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from consort.graph.build import Graph
from consort.graph.neighbourhood import Edges

IntArray = npt.NDArray[np.int64]

INCIDENT = """
MATCH (a:Account {id: $n})-[t:SENT]-()
RETURN DISTINCT t.edge_id AS edge_id, startNode(t).id AS src, endNode(t).id AS dst, t.ord AS ord, t AS rel
"""
CAPPED = """
UNWIND $nodes AS n
MATCH (a:Account {id: n})
CALL (a) {
  MATCH (a)-[t:SENT]->() WITH t ORDER BY t.ord DESC LIMIT $cap RETURN t
  UNION
  MATCH (a)<-[t:SENT]-() WITH t ORDER BY t.ord DESC LIMIT $cap RETURN t
}
RETURN DISTINCT t.edge_id AS edge_id, startNode(t).id AS src, endNode(t).id AS dst, t.ord AS ord, t AS rel
"""


class Neo4jSource:
    def __init__(self, driver: Any, attrs: tuple[str, ...], database: str = "neo4j") -> None:
        self.driver = driver
        self.attrs = attrs
        self.database = database

    @classmethod
    def load(
        cls, driver: Any, graph: Graph, batch: int = 50_000, database: str = "neo4j"
    ) -> Neo4jSource:
        attrs = tuple(sorted(graph.edge_attrs))
        with driver.session(database=database) as s:
            s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")
            s.run(
                "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE"
            )
            for i in range(0, graph.n_nodes, batch):
                s.run(
                    "UNWIND $ids AS i CREATE (:Account {id: i})",
                    ids=list(range(i, min(i + batch, graph.n_nodes))),
                )
            for i in range(0, graph.n_edges, batch):
                j = min(i + batch, graph.n_edges)
                rows = [
                    {
                        "e": k,
                        "s": int(graph.src[k]),
                        "d": int(graph.dst[k]),
                        "o": int(graph.order[k]),
                        **{a: float(graph.edge_attrs[a][k]) for a in attrs},
                    }
                    for k in range(i, j)
                ]
                props = ", ".join(f"{a}: r.{a}" for a in attrs)
                sep = ", " if props else ""
                s.run(
                    "UNWIND $rows AS r MATCH (a:Account {id: r.s}), (b:Account {id: r.d}) "
                    f"CREATE (a)-[:SENT {{edge_id: r.e, ord: r.o{sep}{props}}}]->(b)",
                    rows=rows,
                )
        return cls(driver, attrs, database)

    def _edges(self, records: list[Any]) -> Edges:
        df = pd.DataFrame(
            [
                {
                    "edge_id": r["edge_id"],
                    "src": r["src"],
                    "dst": r["dst"],
                    "ord": r["ord"],
                    **{a: r["rel"][a] for a in self.attrs},
                }
                for r in records
            ],
            columns=["edge_id", "src", "dst", "ord", *self.attrs],
        ).sort_values("edge_id")
        return Edges(
            edge_id=df["edge_id"].to_numpy(np.int64),
            src=df["src"].to_numpy(np.int64),
            dst=df["dst"].to_numpy(np.int64),
            order=df["ord"].to_numpy(np.int64),
            attrs={a: df[a].to_numpy(np.float64) for a in self.attrs},
        )

    def incident(self, node: int) -> Edges:
        records, _, _ = self.driver.execute_query(INCIDENT, n=node, database_=self.database)
        return self._edges(records)

    def capped(self, nodes: IntArray, cap: int) -> Edges:
        records, _, _ = self.driver.execute_query(
            CAPPED, nodes=[int(v) for v in nodes], cap=cap, database_=self.database
        )
        return self._edges(records)
