# Architecture

The figures behind [PLAN.md](../PLAN.md), drawn before the first line of code so the design
cannot be shaped to fit the result. Three views: the system, one account through the three
arms, and the evaluation design with its two guards. Nothing in them is measured yet; the
module names are the plan's.

## The system

![Public transaction graphs are fetched with their licences quoted and built once into DuckDB edge tables and PyTorch Geometric objects; temporal inductive splits pass a leakage test on every build; three arms and a foundation-model column score the same accounts; one evaluator that does not know which model scored writes the intervals; the explanations, the serving benchmark, the gate and the report follow, and the ONNX scorer and feature view are handed to project 09.](architecture/system.svg)

Two public graphs, never committed, come in: Elliptic's Bitcoin transaction graph and IBM's
synthetic bank transfers with their pattern files (`docs/data.md`). The fetch step keeps
loaders, checksums and the licences quoted, and tests the join of the generator's pattern
labels against its own counts (`consort.data`). The graph is built once, into DuckDB edge
tables and PyTorch Geometric objects from one source (`consort.graph.build`), and split by
time, inductively, with the shift split at Elliptic's step 43, the random split run once for
the leakage gap, and the leakage test on every build (`consort.data.splits`,
`consort.data.leakage`). From the split graphs in `data/graphs` and `data/features`, three
arms score every account: R on the row alone, G on the row plus graph features, N a graph
neural network, with the TabFM column run on R's and G's feature tables; N and TabFM run on a
rented spot GPU of the 24 GB class (`consort.models`). One evaluator computes area under the
precision-recall curve, recall at a false-positive rate and at an alert budget, bootstrapped
over groups with paired deltas (`consort.eval`), and writes one outcomes shape for every arm
to `data/runs`. From there the explanations and their faithfulness number and the case files
(`consort.explain`), the serving benchmark over Neo4j and DuckDB with the parity test
(`consort.serve`) and the 03 gate's paired comparison per dataset are produced, and the report
writes the tables, the figures, the case files and the static site at consort.peterparker.ca
(`consort.report`, `consort.site`). The ONNX export of the trained scorer and the feature view
go to project 09 (`consort.models.export`).

## One account through the three arms

![One account with its row features and its k-hop neighbourhood, capped at a stated number of neighbours per hop, is scored three ways: gradient boosting on the row alone, gradient boosting on the row plus features computed from the neighbourhood with no feature allowed to read a label, and a graph neural network with neighbour sampling; the tabular foundation model scores the first two feature tables from a stratified context that keeps every positive; one evaluator reads every score the same way.](architecture/three-arms.svg)

One account in the test split, with its row features and its k-hop neighbourhood fetched by
one implementation that batch and serving share, with a hub cap per hop and its label hidden
(`consort.graph.neighbourhood`), goes to every arm. R is XGBoost on the account's own
features. G is XGBoost on the row plus degree, counterparties, neighbour aggregates, motif
scores and PageRank from the feature registry, which refuses any derivation that reads a label
(`consort.graph.features`). N is GraphSAGE or GAT with sampled neighbours, trained inductively,
with the directed-multigraph adaptations behind flags. The TabFM column scores R's and G's
tables zero-shot from a stratified context that keeps every positive. The evaluator reads
every score the same way, by pattern and by hop as well as overall (`consort.eval`), and
writes outcomes per account with intervals over groups and paired deltas to `data/runs`.

## Evaluation design

![Elliptic's 49 time steps give the temporal inductive split, the random split run once for the leakage gap, the unknown-label rule and the shift split at step 43 with the context-swap schedule; the synthetic set gives the serving benchmark, the hop curve and recall by generator pattern; the leakage test guards every split and the parity test guards the serving path; every claim is bootstrapped over groups before the faithfulness number, the gate and the cost table are written.](architecture/evaluation.svg)

Which split carries which claim, and what the two guards prove. Elliptic's 49 time steps give
the temporal inductive split that carries the real-data headline, the random split run once so
the leakage gap is a published number, the rule that unknown-label nodes pass messages and
are never counted, and the shift split, trained on steps 1 to 34 and tested on 39 to 42 and on
43 to 49, with the context-swap schedule run step by step. The synthetic set gives the serving
benchmark over Neo4j and DuckDB, the hop curve with the gain beside the fetch latency, and
recall by the generator's patterns. The leakage test (`consort.data.leakage`) guards every
split: no test-step row in a training table and no feature that reads a label. The serving
parity test (`consort.serve.parity`) asserts on ten accounts at one and two hops that the
fetch path computes the batch features. Every claim's outcomes are bootstrapped over time
steps or components with 95% intervals and paired deltas on identical items, and from that come
the explanation faithfulness number against random neighbours, the 03 gate's comparison of G
and N against R with a stated delta, and the cost per million scores per hop and per store with
its price date.
