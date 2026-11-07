# Plan: Fraud Known by the Company It Keeps

**Written:** 2026-10-31, before the first line of code; amended in place where the build
proved it wrong, each amendment dated and explained. **Status:** built; the results are in the
README. **Build:** three weeks of evenings and weekends plus one of slack; the weeks in
section 6 are relative, not dated. **Package:** `consort`. **Repository:**
`transaction-graph-fraud`.
**Fed by:** 09 (the fraud framing, the alert-budget view of a review queue, the ONNX
challenger interface and the offline-online parity test it established), 03 (the gate runs the
paired comparison between arms with intervals from 02), 04 (only if a model ever writes a case
narrative; no model vendor is in the plan). **Feeds:** 09 (a challenger scorer exported to
ONNX, and graph features as a feature view for its store), a later serving
project (a trained model with a measured neighbourhood fetch to put on a cluster).

This is the portfolio's first project on a graph: it scores an account by the company it
keeps, and publishes when that beats scoring the account alone, which laundering shapes only
the graph can see, how far guilt travels, what happens when the world shifts, and what a live
score costs per hop.

---

## 1. What this produces

A fraud model that scores the account alone misses the fraud that lives in the links between
accounts. This project measures how much, on millions of real and synthetic transactions: a
graph neural network against gradient boosting on graph features against the plain row model,
which laundering shapes each one catches, how many hops of neighbours it takes, what happens
after a dark market shuts down, and what a live score costs per hop in milliseconds. A fourth
model, Google's zero-shot tabular foundation model, sits beside gradient boosting in the two
tabular arms, so the question "do these new models see relational structure" gets a number
too.

The numbers a stranger can check:

| Number | What it shows |
|---|---|
| Area under the precision-recall curve, recall at one percent false positives, and recall at a fixed alert budget (the top 0.5 and 1 percent of accounts flagged) per arm, per dataset, with 95% bootstrap intervals and paired deltas on identical items | Does the graph help, and by how much, on real and on synthetic data, in the terms an investigation team uses: how much of the fraud a fixed review capacity reaches |
| Recall by laundering pattern (fan-in, fan-out, cycle, scatter-gather, gather-scatter, bipartite, stack, random) per arm on the synthetic set, whose generator labels every pattern | Which shapes a row model cannot see at all, and which the graph catches; the figure that travels |
| The hop curve: every arm at zero, one, two and three hops of neighbourhood, with the gain per hop and the fetch latency per hop | How far guilt travels, and where the gain stops paying for the fetch; the number 09 needs for a serving decision |
| The shift table: trained before Elliptic's time step 43, the dark-market shutdown, tested before and after it, per arm; and the context-swap experiment: the foundation model given only the most recent steps as context against gradient boosting retrained on the same rows | Which arm degrades least when the world changes, and whether swapping context recovers faster than retraining |
| The leakage gap: the random split most papers use against the temporal, inductive split, same model, per dataset; and the neighbour-label-rate feature run once to show what it leaks | Why published graph fraud numbers are often too good, measured |
| Explanation faithfulness: for flagged accounts, the share whose top explanatory neighbours include a known illicit account, against a random-neighbour baseline; one case file per flagged account on the demo | Whether an investigator can read the flag, or only trust it |
| Serving latency p50 and p99 for a one-hop and a two-hop neighbourhood fetch plus scoring, from a graph database and from a columnar store, and cost per million scores | Whether this runs behind a decision or as an overnight batch, and which store to use |
| The foundation-model column: zero-shot on row features and on graph features at a stated context size, with its inference cost per thousand scores | Whether a tabular foundation model sees relational structure, and what it costs to ask |

## 2. Design decisions

### 2.1 Three arms and one scorer-agnostic evaluator

| Arm | What it is | Why it is here |
|---|---|---|
| R. Row only | Gradient boosting (XGBoost) on the account's or transaction's own features: amounts, counts, velocities, the dataset's node features | The model most organisations run; the baseline every gain is measured against |
| G. Graph features | The same gradient boosting on the row features plus features computed from the neighbourhood at a stated hop count: in- and out-degree, unique counterparties, amount in and out, neighbour feature aggregates (mean, max) at one and two hops, PageRank and a simple cycle and fan motif score from a pattern library, temporal velocity over edges. Never a feature derived from neighbour labels (section 10, candidate 1) | The cheap way to use the graph, and IBM's own result that it can beat a GNN; if it wins, the README says so, and the serving cost says why it matters |
| N. Graph neural network | GraphSAGE with neighbour sampling in PyTorch Geometric, trained inductively; GAT as the second architecture; the directed-multigraph adaptations from the Multi-GNN paper (reverse message passing, ego identifiers, port numbering) as an ablation on the synthetic set | The method gap this project closes; the arm that can see a cycle without being told what a cycle is |

A fourth model column, Google's TabFM (zero-shot, in-context, released 2026-06-30, weights
under its non-commercial licence, code Apache 2.0), is run on arm R's and arm G's feature
tables. Its context is bounded, so it sees a stratified sample of the training rows at a
stated size that keeps every positive, never the full table; the sample size is a reported
parameter and the limit is a finding, not a footnote. It is a column, not an arm: the GNN is
the arm.

Every arm scores the same accounts in the same split and writes outcomes to one file shape, so
the evaluator never knows which model produced a score. Intervals are bootstrapped over
accounts grouped by time step (Elliptic) or by connected component (synthetic), because
neighbours are not independent; paired deltas refuse two runs that did not score identical
items, 06's and 16's grader rule carried over.

### 2.2 Three datasets, each carrying a claim

- **Elliptic (Bitcoin transactions).** Real, labelled by a company whose business is this,
  with 49 time steps and a known regime change at step 43 when a dark market closed. Carries
  the real-data headline, the shift table and the leakage gap. Its 157,205 unknown-label
  nodes pass messages and never count in a metric.
- **IBM AML synthetic transactions (HI-Small and LI-Small; HI-Medium if the GPU budget
  allows).** Synthetic bank transfers from a generator that labels eight laundering patterns.
  Carries recall by pattern, the hop curve at scale, the directed-multigraph ablation and the
  serving benchmark. The README says in its first table that these patterns are the
  generator's: recall by pattern measures what the generator plants, not what a bank sees.
- **Elliptic++ actors (wallet addresses), as a stretch.** Real, 822,942 addresses with
  labels, no licence stated by its authors. Fetched only after an email to them is answered;
  otherwise its row stays empty and says why.
  *Amended 2026-11-01:* both authors agreed the same day (docs/data.md). The published wallet
  features are lifetime aggregates over all 49 steps, repeated on every step's row, and the
  address-to-address edges carry no step, so neither is used. Instead every wallet's features
  are rebuilt from the transaction-level files, as of a step, from transactions in that step
  and earlier only. An item is a labelled address, scored at the end of the step it first
  appears in and assigned to the partition of that step (Elliptic's 1 to 34, 35 to 38, 39 to
  49 and the same slices), so test addresses are new addresses: the split is inductive by
  construction. Its row features are what its first step's transactions say about it (how
  many it sent from and received in, and those transactions' sizes, fees, amounts and
  counterparty counts; per-address amounts are not in the data). Its graph is the
  input-to-output address pairs of every transaction up to that step, each edge dated by its
  transaction's step, and its neighbours' features are theirs as of the same step. Arm N
  follows the same step-by-step graphs. The published wallet features are kept out of every
  arm; a leakage test requires an address's features to be unchanged when later steps'
  transactions are added.
- **DGraph-Fin, if reachable.** A real consumer-lending graph of 3.7 million users with fraud
  labels; its host did not resolve on 2026-10-31. A loader stub is kept.

*Amended during the build (2026-10-31).* On IBM the item scored is the transfer, as the
generator labels transfers and names patterns by transfer; accounts are the graph's nodes, and
a transfer's graph features are its two accounts'. The alert budget is therefore a share of
transfers on IBM and of transactions on Elliptic. The IBM split is by day, cut where the
cumulative share of transfers passes 60 and 80 percent. A transfer's graph features are
computed on every transfer from the days before its own: what a feature view refreshed each
night holds when the transfer arrives. *Amended 2026-11-01:* they were first computed on every
transfer up to the end of the partition's window, which hands a test transfer its accounts'
later activity; arm G's first HI-Small run under that view had AUPRC 0.716 against arm R's
0.212, and a leakage test now shows the window view changing an earlier transfer's features
when later days are added and the prior-days view not. The window view is kept to publish
that gap beside the random split's, and arm N's IBM graph follows the same rule.
*Amended 2026-11-01:* every IBM result is also read on a "without-tail" test slice. After
normal activity stops, the generator's laundering attempts run on for several days, which
are then nearly all laundering (HI-Small's days 10 to 17: 1,128 transfers, 655 laundering, 41%
of the test set's; LI-Small has the same tail). A test day is in the tail when it holds under
1% of the median day's transfers (the tail days hold 3 to 396, a median day about 207,000 on
HI-Small and 283,000 on LI-Small). Both readings are published; the without-tail one is the
one that resembles a bank's day. Bootstrap groups on IBM are
hours by default rather than connected components: the transfer graph is expected to be one
giant component, which would leave too few groups to resample. The component grouping is
implemented and is reported beside the hour grouping once the data shows the component sizes.

### 2.3 Splits are temporal and inductive, and the random split is run once

The headline split is by time: train on earlier steps, validate on the next, test on the
latest. It is inductive: test accounts and their edges are absent from the training graph, and
at test time messages may pass through test-time neighbours but no label of a test account is
ever visible to a feature or a loss. The random node split that most of the literature uses is
run once per dataset, same model, and the gap is published as a number, because that gap is
the reason graph fraud papers overstate. A leakage test runs on every build: no test-step row
in any training table, and no feature whose name or derivation touches a label.

### 2.4 The alert budget is the business number

An investigation team reviews a fixed number of alerts a day. Beside area under the
precision-recall curve, every arm reports recall at an alert budget of 0.5 and 1 percent of
accounts, which is what a director can read as "of the fraud, how much does my team reach".
This is 09's expected-loss queue seen from the other side, and it is the number that decides
whether the graph pays.

Each budget is reported with its ceiling, the most recall any ranking could reach at it
(alerts over positives, capped at 1). Elliptic's labelled test set is about 6% illicit, so a
1% budget caps every arm near 0.17 there and arm R already sits at it; the ceiling stops that
reading as a weak model. (Amended 2026-11-01, after arm R's first Elliptic run.)

### 2.5 Hops are measured on both axes

Every arm runs at zero, one, two and three hops: arm G's features computed to that depth, arm
N's sampler bounded to it. The gain per hop is reported beside the neighbourhood fetch latency
per hop from section 2.7, so the curve says where the gain stops paying for the fetch. Hub
accounts are capped at a stated number of sampled neighbours per hop and the cap is reported.

*Amended during the build (2026-10-31).* The cap is a recency cap: every node keeps its `cap`
most recent edges in each direction (a timestamp on IBM, file order on Elliptic). Arm G's
features, arm N's sampler (`consort/graph/sampler.py`) and the serving fetch all use this one
rule, so the three see identical neighbourhoods and the parity test can demand exact equality.
Arm N does not use PyTorch Geometric's random neighbour samplers: they need `pyg-lib` or
`torch-sparse`, which have no Windows wheels for the current torch, and random sampling would
have given N a different neighbourhood from G's at the same hop and cap, confounding the hop
curve. Degrees, amounts and velocity remain exact counts over all of a node's edges.

*Amended during the build (2026-11-02).* On IBM, arm N trains each epoch on every illicit
training transfer and a fresh fifth of the licit ones, with the loss reweighted to the full
ratio; validation and test score every transfer. HI-Small has 3.2 million training transfers,
and at two hops the sampler's batch of 2,048 transfers reaches some 50,000 accounts, so a full
epoch is about 1,600 such batches; the first full run took four hours at one hop. Arm G's
booster still sees every row. The share is recorded in each run's notes.

### 2.6 The shift, and the context swap

On Elliptic, every arm is trained on steps 1 to 34, validated on 35 to 38, and tested on 39 to
42 (before the shutdown) and 43 to 49 (after), with intervals. That is the published failure
mode of the original paper, measured on three arms rather than two. Then the context-swap
experiment: TabFM is given the most recent n steps before each test step as its context and
scored step by step, against XGBoost retrained on the same rows at the same cadence and against
the frozen models. Whether swapping context recovers faster than retraining is the question
09 would ask of an in-context model, and the answer costs nothing beyond inference.

### 2.7 Serving is benchmarked on the whole path, from fetch to score

A live score on a graph has to fetch the neighbourhood first. The benchmark runs the full path
for a one-hop and a two-hop neighbourhood on the synthetic graph: fetch from Neo4j (Cypher) and
from DuckDB (a self-join on the edge table), compute arm G's features or arm N's sampled
subgraph, score, with p50 and p99 and cost per million scores at a stated utilisation. A
parity test, 09's idea, asserts that the features computed on the serving path equal the
features computed in batch for the same account on the same graph. If the answer is "two hops
is an overnight job", the README says so above the fold.

### 2.8 Explanations are scored, not shown

Arm N's flags are explained by a subgraph attribution method (GNNExplainer or a gradient-based
equivalent in PyTorch Geometric); arm G's by SHAP on the graph features. Faithfulness is a
number: among flagged accounts, the share whose top-k explanatory neighbours include a known
illicit account, against the same share for k random neighbours. A case file per flagged
account (the account, its top neighbours, the features that drove the score, the pattern if
the generator planted one) is written as JSON and rendered on the demo. No language model
writes a narrative; that is deferred and would route through 04.

*Amended during the build (2026-10-31).* The neighbour importance that faithfulness grades is
occlusion for every arm: fetch the flagged node's neighbourhood with the serving path, remove
one direct neighbour and its edges, rescore, and rank neighbours by the fall
(`consort/explain/faithfulness.py`). One method for all arms means the faithfulness numbers
compare the arms rather than the attribution methods, and arm G gets neighbour-level
explanations, which SHAP on aggregated features cannot give it. SHAP (XGBoost's own TreeSHAP
contributions) still supplies the case file's "features that drove the score", and
GNNExplainer stays available as a second reading of arm N if the occlusion number needs a
check.

### 2.9 The demo shows a ring, and models nothing

graph.peterparker.ca is a static site on Azure Static Web Apps, the 01, 08 and 16 pattern: an
explorer of one planted laundering ring from the synthetic set, with each arm's scores on its
accounts; the recall-by-pattern figure; the hop curve; the shift chart; the case-file viewer.
Nothing is modelled in the browser and nothing is uploaded. Only synthetic accounts are drawn;
Elliptic is shown in tables only, since its licence forbids redistribution of the data.

### 2.10 Out of scope, on purpose

- Graph research: new architectures, temporal graph networks, heterogeneous giant graphs.
  Elliptic2's 49 million nodes and its subgraph task are deferred, not attempted.
- Streaming integration. 09 owns real time; this project hands it a scorer and a feature
  view and measures the fetch, nothing more.
- A language model anywhere in the headline. No narratives, no model grading another.
- A general anti-money-laundering product: sanctions screening, name matching, case
  management. The plan rejected screening for 05 on the same grounds.
- Any data that is not public, or any design that is not from public sources (section 9).

## 3. Data

| Source | Size | What it gives | Access and terms |
|---|---|---|---|
| Elliptic data set (Elliptic, 2019; Kaggle `ellipticco/elliptic-data-set`) | 203,769 transaction nodes, 234,355 edges, 166 features, 49 time steps; 4,545 illicit, 42,019 licit, 157,205 unknown | The real-data headline, the shift table, the leakage gap | CC BY-NC-ND 4.0 (Kaggle licence field, read 2026-10-31). Loaders fetch it with a Kaggle token the reader supplies; nothing is redistributed, no derived dataset is published, numbers and code are. Quoted in `docs/data.md` before the first fetch |
| IBM Transactions for Anti Money Laundering (Altman et al., 2023; Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`) | HI-Small: 515,088 accounts, 5,078,345 transactions; LI-Small: 705,907 and 6,924,055; HI-Medium about 32 million transactions; eight pattern labels on the laundering transactions, with most laundering transactions tied to no named pattern | Recall by pattern, the hop curve at scale, the directed-multigraph ablation, the serving benchmark, the demo | Community Data License Agreement, Sharing, Version 1.0 (Kaggle licence field, read 2026-10-31): use, results and redistribution with attribution permitted |
| Elliptic++ actors (Elmougy and Liu, KDD 2023; `git-disl/EllipticPlusPlus`) | 822,942 wallet addresses, 2,868,964 address-to-address edges, 56 features, 49 time steps; 14,266 illicit, 251,088 licit | A second real graph, at the account rather than transaction level | No licence file and no terms stated (read 2026-10-31); a citation request only. Used only if the authors answer an email confirming research use and publication of results; the email goes on day one |
| DGraph-Fin (Huang et al., NeurIPS 2022) | 3,700,550 users, 4,300,999 edges, 17 features, fraud labels on about a million users | A real consumer-lending graph, inductive at scale | Host `dgraph.xinye.com` did not resolve on 2026-10-31; terms unread. Loader stub kept; used only if the host and its terms allow |

Nothing raw is committed; loaders verify recorded checksums and write `data/graphs/` with a
manifest. The Kaggle sources need a reader's own API token, which the README says in one line
with the command. The pattern labels in the IBM set come from the generator's pattern file,
joined to transactions by the published identifiers; the join is tested against the generator's
own counts.

## 4. Methods and architecture

See `docs/architecture.md` for the diagrams. The package:

```
consort/
  data/        fetch.py (Kaggle and GitHub loaders with checksums), elliptic.py, ibm.py, ellipticpp.py,
               dgraph.py (stub), manifest.py, splits.py (temporal, inductive; the random split for the
               leakage gap; the shift split on Elliptic), leakage.py (the build-time test)
  graph/       build.py (edge tables in DuckDB and PyG Data objects from one source), features.py
               (graph features to a stated hop, hub cap, no label-derived feature by construction),
               patterns.py (the generator's pattern labels joined to accounts; motif scores),
               neighbourhood.py (k-hop fetch, one implementation used by batch and serving)
  models/      row.py (XGBoost on row features), graphfeat.py (XGBoost on row plus graph features),
               gnn.py (GraphSAGE and GAT with neighbour sampling; the directed-multigraph adaptations
               behind flags), tabfm.py (stratified context sampling that keeps every positive,
               zero-shot scoring, the context-swap schedule), export.py (ONNX for 09)
  eval/        outcomes.py (one file shape for every arm), metrics.py (AUPRC, recall at a false-positive
               rate, recall at an alert budget; bootstrap over groups; paired deltas that refuse
               mismatched items), bypattern.py, hops.py, shift.py, leak.py
  explain/     subgraph.py (GNNExplainer or gradient attribution), shap.py, faithfulness.py, casefile.py
  serve/       neo4j.py (Cypher fetch), duckdb.py (self-join fetch), parity.py (serving features equal
               batch features), bench.py (p50, p99 per hop, cost per million scores)
  gate/        eval specs and outcome files for the 03 gate; the paired comparison per dataset
  report/      tables, the pattern figure, the hop curve, the shift chart, the site's JSON
  site/        the static explorer: the ring, the figures, the case files; built by consort site
  cli.py       consort data fetch | build | split | features | train <arm> | score | eval | hops | shift |
               explain | serve bench | gate | report | site
tests/         leakage (no test-step row in training tables; no feature touches a label); the pattern
               labeller against hand-built toy graphs (a 3-cycle, a fan-in of five, a scatter-gather);
               metric arithmetic on a synthetic score stream with a known answer; bootstrap grouping
               keeps a component together; serving parity for ten accounts at one and two hops;
               the TabFM context sampler keeps every positive and respects the size; ONNX export
               reproduces the scores; paired deltas refuse mismatched items
docs/          architecture.md, data.md (licences quoted, counts, the join test), models.md (versions
               and dates, the TabFM licence), patterns.md (what each generator pattern is), serving.md,
               rejected.md
```

### Tests that matter

The leakage test, written in week 1 before the first feature, as 09 did: it fails if any
training table holds a row from a test step, and it fails if any feature's derivation reads a
label column, by construction of the feature registry. The pattern labeller on toy graphs with
known answers, so a cycle is a cycle before any metric is trusted. The metric arithmetic on a
synthetic score stream whose AUPRC and recall at budget are known in closed form. The bootstrap
grouping test: resampling never splits a component. The serving parity test: for ten accounts,
the features the fetch path computes equal the batch features bit for bit. The TabFM sampler
test: every positive kept, size respected, deterministic by seed. The export test: ONNX scores
match the trained model's on a thousand rows.

## 5. Interfaces with the rest of the portfolio

| Project | What crosses the line |
|---|---|
| 09 | `consort.models.export` writes an ONNX scorer in the shape 09's challenger interface takes; `consort.graph.features` is written as a feature view so 09's store could compute it on its entity graph. 09's offline-online parity test is reused as the serving parity test here. Nothing from 09's internal design beyond its public interfaces |
| 03 | The gate's `outcomes_file` adapter: one spec per dataset and hop, the paired comparison of arm G and arm N against arm R with a stated delta, the ledger record written to this repository; a fourth real use of the adapter after 06, 07 and 16 |
| 02 | Interval and paired-delta conventions, through 03 |
| 04 | Not in the plan. If a case narrative is ever generated, it routes through `boundary` with `data_class=public`, which the synthetic accounts are |
| A later serving project | A trained model, an ONNX export, a measured fetch path and a load client: the inputs that project needs on day one |

## 6. Week by week

Relative weeks of evenings and weekends, in order.

| Week | Built | Done when |
|---|---|---|
| 1 | Licences quoted in `docs/data.md`; the Elliptic++ email sent; loaders with checksums for Elliptic and IBM HI-Small and LI-Small; the graph build into DuckDB and PyG from one source; temporal inductive splits, the shift split, the random split; **the leakage test, before the first feature**; the pattern join tested against the generator's counts; arm R; the graph feature registry and arm G at one and two hops; the evaluator with bootstrap over groups and paired deltas; the leakage gap; `docs/patterns.md` | Arms R and G filled with intervals on Elliptic and both IBM sets; recall by pattern for R and G; the leakage gap published |
| 2 | GPU price table and provider decision from 06's rule; a pod watchdog that deletes an idle pod (16's lesson); GraphSAGE and GAT with sampling on a rented GPU, three seeds, inductive; the hop curve for all arms at zero to three hops; the directed-multigraph ablation on HI-Small; the shift table on Elliptic for all arms; TabFM on arm R's and arm G's tables at a stated context size; the context-swap experiment; arm N's recall by pattern | Arm N and the TabFM column filled; the hop curve, the shift table and the context-swap chart in the README with intervals |
| 3 | Explanations and the faithfulness number; case files; the serving benchmark on Neo4j and DuckDB with the parity test, p50, p99 and cost per million scores; ONNX export and the feature view for 09; the 03 gate per dataset; the static site; README tables from `consort report`; `docs/rejected.md`; `v0.1.0`; repository public; graph.peterparker.ca live; the status flip logged on peterparker.ca | Every number in section 1 in the README with its interval; the site live |
| slack | One week | |

First to drop if behind: Elliptic++ (a stretch already), then HI-Medium, then the
directed-multigraph ablation, then GAT (GraphSAGE stays), then Neo4j (DuckDB carries the
serving number alone, and the README says so). None is in the definition of done. Not
droppable: three arms on Elliptic and HI-Small, recall by pattern, the hop curve, the shift
table, the leakage gap, the serving latency on at least one store, and the TabFM column.

## 7. Cost

Prices as of 2026-10-31, from what 16 recorded on 2026-10-28: RunPod RTX 3090 and 4090 class
at US$0.27 to 0.50 an hour, the 48 GB A6000 at US$0.59. No model vendor is called. Neo4j runs
as a container on the shared VPS or the rented machine; DuckDB is a library; the site is on
the Azure Static Web Apps free tier.

| Item | Basis | US$ | CA$ |
|---|---|---:|---:|
| GNN training | GraphSAGE and GAT on Elliptic (minutes a run) and on HI-Small and LI-Small with sampling (about an hour a run on the 24 GB class); three seeds, two architectures, four hop depths with reuse, the ablation: about 16 runs, about 14 GPU hours | 6 | 8 |
| HI-Medium, if run | Two runs on the 48 GB class, about 4 hours | 2 | 3 |
| TabFM inference | Two feature tables, three datasets, the step-by-step context swap on Elliptic; about 4 GPU hours | 2 | 3 |
| Serving benchmark | The same rented machine for the fetch-and-score runs, about 3 hours; Neo4j container free | 1 | 2 |
| Development, evaluation and explanation runs | Local CPU for Elliptic and the tabular arms; about 4 rented hours for arm N's explanations | 2 | 3 |
| Pods left idle | The watchdog from week 2; a reserve of | 4 | 5 |
| Object storage for checkpoints, one month | | 2 | 3 |
| Demo site | Azure Static Web Apps free tier | 0 | 0 |
| **Total** | | **19** | **27** |

Budgeted **CA$20 to 40**. If the GPU time runs over, HI-Medium goes first, then GAT, then the
ablation. *Spent, 2026-11-04:* about US$33 (about CA$45 at 1.35 CAD/USD) over seven pods (a smoke
test, five queues, the serving benchmark), CA$5 over the cap: IBM's arm N ran four hours on
the CPU of the first pod before its sampler was fixed, and TabFM on Elliptic++'s graph
features hung two pods before it finished on a later one. HI-Medium was not run. Each pod is
in `docs/gpu.md`.

## 8. Handover and reuse

`consort` v0.1.0 at the end of week 3. 09 can import the ONNX scorer and the feature view.
The evaluator, the feature registry and the neighbourhood fetch are dataset-agnostic: a fourth
graph is a loader and a manifest. A later serving project would start from the trained models,
the fetch path and the load client. `docs/patterns.md` and the pattern labeller are the part a
reader interested in the topic takes away, with the figure.

## 9. Risks

| Risk | Handling |
|---|---|
| Elliptic's licence (CC BY-NC-ND) | No redistribution, no derived dataset published, no Elliptic account drawn on the demo; loaders fetch from Kaggle with the reader's token; results and code published, which the licence does not restrict. Quoted in `docs/data.md` on day one |
| Elliptic++ has no licence | Email on day one; used only on a written answer; otherwise its row says "not run, no licence". *Answered 2026-11-01:* both authors agree to the plan as asked (research use, aggregate results and code published, data never redistributed); the terms and the answer are in `docs/data.md` |
| DGraph-Fin's host is down | Loader stub only; nothing in the definition of done depends on it |
| The synthetic patterns are the generator's | Said in the first table of the README and in `docs/patterns.md`; recall by pattern is reported as what the generator plants; the real-data numbers carry the headline |
| Elliptic's unknown labels | Unknown nodes pass messages and never enter a metric; stated in section 2.2 and tested |
| Hubs make two hops explode | A per-hop neighbour cap, reported; the latency table shows the uncapped p99 on a sample so the cap's cost is visible |
| The GNN does not beat graph features plus XGBoost | Published either way through the gate; that is the field's open question and the README's first finding whichever way it falls |
| TabFM's bounded context | A stratified sample at a stated size with every positive kept; the size is a parameter in the table and the limit is a finding; weights non-commercial, which the cost column notes as "not deployable by a bank" |
| Pods left running | A watchdog that deletes a pod idle for thirty minutes, written before the first rental |
| The laptop tempts a shortcut | Elliptic and the tabular arms may train on it; nothing timed is reported from it. Every latency number comes from the rented machine or the VPS |
| Public sources only | Crypto transactions and synthetic bank transfers, far from any public programme; public data and public problem statements only; no non-public architecture, features, thresholds or code. If the build drifts toward a non-public design, stop and raise it |

## 10. Candidates for the rejected approach

1. **A neighbour-label-rate feature.** The share of an account's neighbours that are labelled
   illicit, added to arm G. Expected: a large jump on the random split and a collapse on the
   temporal inductive split, because at test time the labels it reads are the ones being
   predicted. The clearest leakage demonstration in the field, and the reason section 2.3 is
   written the way it is.
2. **A GNN beats graph features plus gradient boosting everywhere.** Arm N against arm G,
   paired, every dataset and hop. Expected: not on Elliptic (the original paper's result, a
   random forest ahead of the GCN), and yes on the cycle and scatter-gather patterns of the
   synthetic set, where features built for fan shapes cannot see a cycle. If G wins
   everywhere, the README says the graph is a feature-engineering problem at a hundredth of the
   serving cost.
3. **More hops always help.** Expected: the gain saturates at two hops on every dataset while
   the fetch latency grows by an order of magnitude per hop, so three hops is never worth it
   on this data.
4. **A tabular foundation model on raw rows matches gradient boosting.** Expected: close on
   area under the precision-recall curve at tens to hundreds of times the inference cost, with
   no view of the graph until the graph features are handed to it; the interesting part is
   whether its context swap recovers after step 43 faster than retraining.

Whichever produces the clearest evidence becomes `docs/rejected.md`.

## 11. Definition of done

- [x] Elliptic and both IBM sets fetched by loaders with checksums, licences quoted in `docs/data.md`, the pattern join tested against the generator's counts, the Elliptic++ email sent and its answer recorded
- [x] Temporal inductive splits with the leakage test on every build; the random-split gap published per dataset
- [x] Three arms on Elliptic and HI-Small (LI-Small too unless dropped), identical items, intervals over groups, paired deltas; AUPRC, recall at one percent false positives, recall at 0.5 and 1 percent alert budgets
- [x] Recall by laundering pattern per arm on the synthetic set, with the figure
- [x] The hop curve at zero to three hops per arm, gain and fetch latency side by side
- [x] The shift table on Elliptic per arm, and the context-swap chart
- [x] The TabFM column on both tabular arms at a stated context size, with its cost per thousand scores
- [x] Explanation faithfulness per arm against the random baseline; case files on the demo
- [x] Serving latency p50 and p99 at one and two hops from at least one store with the parity test passing; cost per million scores, prices dated
- [x] ONNX export and the feature view for 09, with the export test passing
- [x] The 03 gate's paired comparison per dataset
- [x] graph.peterparker.ca live with the ring explorer, the figures and the case-file viewer
- [x] One rejected approach documented with evidence (`docs/rejected.md`)
- [x] Repository public, `v0.1.0` tagged, the status flip logged on peterparker.ca

## 12. Deferred

| Deferred | Kept so the door stays open |
|---|---|
| Elliptic2: 49 million clusters, 122 thousand labelled subgraphs, the subgraph task | The loader shape and the evaluator take a subgraph label; a new loader and a subgraph head |
| Temporal graph networks | The edge table carries timestamps; the PyG Data objects are built per step |
| Heterogeneous graphs (accounts, devices, merchants), as in 09's generator | Typed edges and per-type features would come first: as built (2026-11-02), `graph/build.py` and the registry hold one node type and directed edges, which every dataset here needed, so typed edge tables were not written (docs/handover.md) |
| A language-model case narrative through 04 | The case file is already JSON; a narrative is a rendering of it |
| Live scoring inside 09's stream | The ONNX scorer and the feature view are the handover; wiring is 09's |
| DGraph-Fin and Elliptic++ as full rows | Loader stubs with their terms recorded |
| HI-Medium and HI-Large tiers | The loaders take a tier name; GPU hours are the only cost |
| A write-up: "when does the graph pay" | The pattern figure, the hop curve and the shift chart are its three figures |

## 13. Name

**Chosen 2026-10-31: Fraud Known by the Company It Keeps.** The idiom as people say it, with
one word swapped; "known" is the outcome and "the company it keeps" is the mechanism in plain
words. The one-liner carries the magnitude and the thesis sentence. Weighed against "The Fraud
a Single Transaction Cannot Show" and "Fraud That Hides in the Company It Keeps", which name
the problem rather than the result, and against a magnitude-in-the-name form ("Fraud Invisible
in One Transaction, Caught Among a Million"), rejected because a magnitude in a name can read
as a requirement. Repository `transaction-graph-fraud`, package `consort` (to keep company
with), demo `graph.peterparker.ca`. The demo was first to share the package's name,
so infrastructure names travelled together; on 2026-11-04, before its DNS record went in, it
was renamed, since a visitor does not know the package and "graph" says what the page is about,
as the other project pages' addresses do (fraud, privacy, gate).
