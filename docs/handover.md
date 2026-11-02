# The handover to the real-time platform

What this project hands to
[Real-Time Fraud & Risk Decisioning Platform](https://github.com/Peter-A-P/realtime-risk-platform)
(project 09), and what it does not. 09 owns the stream, the feature store and the decision;
this project hands over a scorer, the features it reads, and a measured cost of serving them.

## The scorer

    uv run consort train G --dataset ibm-HI-Small --hops 1 --hub-cap 50 --onnx

writes `checkpoints/<run>.onnx` and its sidecar `checkpoints/<run>.features.json`:

- **The ONNX file** has the shape 09's `OnnxModel` loads. It has one float32 input of shape
  (n, k), a `probabilities` output of shape (n, 2) with the fraud score in column 1, and it
  was converted by onnxmltools at target opset 15, as 09's own export is. Its scores match
  the booster's to within 1e-5 (`tests/test_export.py`), inside 09's 1e-4 parity tolerance.
  The same file test checks the input, output and domains.
- **The sidecar** is the contract: the k column names in input order, the arm, the hop count
  and the hub cap they were computed at, and the input and output names. 09 builds its input
  vector in that order; a column it cannot serve is a column it does not have, and the model
  is not handed over for that stream.

Only arms R and G are exported. Arm N is not handed over: on no dataset did it beat arm G
(README). TabFM's weights licence forbids redistributing the model or derivatives of it
(docs/models.md).

## The feature view

The features are computed by one registry (`consort/graph/features.py`), in batch and on the
serving path, and nothing in it reads a label (`consort/graph/registry.py`). At one hop, for
each account:

| Family | Columns | Read from |
|---|---|---|
| Degrees | edges out and in, distinct counterparties out and in, counterparties who also pay back | the account's own edges, exact |
| Amounts | sum and max of amounts out and in, pass-through | the account's own edges, exact |
| Velocity | edges per hour of active span, out and in | the account's own edges, exact |
| Neighbour aggregates | mean and max of each row feature over the counterparties | the capped edges: the 50 most recent out and 50 most recent in |

Two hops add the neighbour aggregates once more, directed 3-cycles, the widest scatter-gather
and gather-scatter, and PageRank. PageRank is global, so it is a stored column refreshed in
batch, never computed per call.

The exact counts are **counters**: a write to the edge store bumps them, and a call reads them
instead of fetching the account's whole history. That is the design the README's serving table
recommends. The parity test (`tests/test_parity.py`) holds the served features equal to the
batch features, number for number, with and without counters, from memory and from DuckDB.

## What it costs

From the README's serving table (a Secure Cloud RTX 4090 pod at US$0.89 an hour, read
2026-11-02, one thread per call): at one hop with counters, a whole call takes 4.4 ms at
the median in memory, 10.2 ms in DuckDB and 16.5 ms in Neo4j. That covers both accounts of a
transfer fetched, the features computed and the ONNX score. Two hops cost two to four and a
half times as much and added nothing measurable to arm G on IBM. One hop sits inside a
decision's budget; two hops are better run overnight, if at all.

## What does not carry over

- **The weights.** The booster is trained on IBM's synthetic transfers between bank accounts.
  09's events are card transactions with devices and merchants, a different graph with
  different row features. What carries over is the method, the interface and the cost. 09
  retrains on its own entity graph and runs the result through its own champion and
  challenger comparison.
- **The node types.** The graph (`consort/graph/build.py`) and the registry are written for
  one node type and directed edges. 09's graph is heterogeneous (cards, devices, merchants), so
  typed edges and features per node type would come first, and they are not built here.
- **The wiring.** Putting the fetch inside 09's stream, and the counters inside its store, is
  09's work (PLAN.md, out of scope).
