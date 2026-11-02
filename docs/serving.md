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
(`NotReportable`).

## The account's own counts

An account's degrees, amounts and velocity are exact counts over all its edges, so the first
design fetched its whole history on every call. On IBM a transfer often touches a hub, and the
first run fetched 9,600 edges per call on average. With `counters` the fetch reads those counts
from stored columns, as counters a service updates on every write, and starts from the
account's capped edges, so a call fetches at most the cap's worth per expanded account; the
parity test runs both ways and demands the same features. The benchmark times both, and the
README's hop curve quotes the counters design.

## The first run, superseded (2026-11-02)

The first reportable run, on a Secure Cloud L4, timed every store at about 300 ms per call at
the median and is not reported, for two reasons found in its numbers. The in-memory store
answered each lookup by scanning all 5 million edges, so it measured a scan, not a store; it
now indexes edges by each end. And DuckDB and Neo4j broke ties in recency differently from
the batch features (IBM's timestamps are to the minute and tie often), so under a binding cap
they could keep different edges; both now break ties by the larger edge id, and a test with
tied timestamps fails the old queries. Its results are kept outside the repository.

## Running it

    NAME=consort-serve-1 CLOUD=SECURE SERVE=1 RUNS_FILE=ops/serve-runs.txt bash ops/pod-run.sh

On a Secure Cloud pod, so the machine named beside the timings can be rented again: Neo4j 5.26
Community installed natively (`ops/serve-setup.sh`; a pod is itself a container), arm G's
boosters at one and two hops exported to ONNX, then `consort bench`. On IBM the scored item is
a transfer, so one call fetches both its accounts' neighbourhoods from the store, computes arm
G's features for each with the same registry as the batch run, and scores the transfer's row
through the ONNX export, on 1,000 transfers drawn at random (seed 0) after 20 warm-up calls.
The pod's name, card, cloud, hourly price and the date it was read are written with every
result in `results/serving.json`; `consort report` reads only results marked reportable, which
only a run on the rented machine is.

