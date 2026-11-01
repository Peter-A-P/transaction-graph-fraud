# Serving

How a live score is fetched and computed, how the benchmark times it, and what has and has
not been measured.

## The path

To score one account live: fetch its neighbourhood, compute arm G's features on it, score.

1. **The account's own edges**, all of them, so that its degrees, amounts and velocity are
   exact counts.
2. **The capped expansion**, once per hop: every account within the hop budget contributes its
   `cap` most recent outgoing and `cap` most recent incoming transfers. This is the same
   recency rule arm G's batch features and arm N's sampler use (`consort/graph/build.py`,
   `Graph.sampled`), which is why the parity test can demand exact equality.
3. **Features**: the same registry as the batch run, on the fetched subgraph, with node ids
   kept in global order so that sparse sums run in the same order. PageRank is global, so it
   is read from a stored column, as a batch-maintained property would be.
4. **Score**: the arm's booster, or its ONNX export.

## The stores

- **DuckDB** (`consort/serve/duckdb_store.py`): one `edges` table sorted by source, with ART
  indexes on source and destination. The account's own edges are two indexed lookups; the
  capped expansion is a `row_number() OVER (PARTITION BY src ORDER BY ord DESC)` window over
  the frontier's edges, in each direction.
- **Neo4j** (`consort/serve/neo4j_store.py`): `(:Account {id})` nodes with a uniqueness
  constraint, `[:SENT]` relationships carrying the edge id, recency and amounts. The capped
  expansion is a `CALL` subquery per frontier account, ordering its relationships by recency
  with `LIMIT $cap` in each direction.

## Parity

`tests/test_parity.py` takes the ten busiest accounts (where the cap binds hardest) and ten at
random on a toy graph, at one and two hops with a cap of 5, and requires every serving feature
to equal the batch feature exactly, from memory and from DuckDB. A negative control fetches one
hop short and must be caught. `tests/test_neo4j.py` requires the Neo4j fetch to return exactly
the memory fetch's edges; it ran green against `neo4j:5.26-community` in Docker on 2026-10-31
and is skipped wherever `NEO4J_URI` is unset (CI).

## The benchmark

`consort/serve/bench.py` times whole calls (fetch, features, score) per store, hop count and
cap, after a warm-up, and reports p50, p99, mean edges fetched and cost per million scores:
the machine's hourly price over one sequential worker's throughput at a stated utilisation.
Every result carries the machine name, host, CPU, core count, price and price date.

**Nothing timed is reported from the laptop.** A result is reportable only when the benchmark
was run with an explicit machine and price and `reportable` set; the report refuses any other
(`NotReportable`). No reportable timing exists yet.
