# Fraud Known by the Company It Keeps

A fraud model that scores the account alone misses the fraud that lives in the links between
accounts. This project measures how much, on millions of real and synthetic transactions: a
graph neural network against gradient boosting on graph features against the plain row model,
which laundering shapes each one catches, how many hops of neighbours it takes, what happens
after a dark market shuts down, and what a live score costs per hop in milliseconds.

Most fraud models look at one account or one transaction at a time. Money laundering is a
shape: money fanning in, cycling round, scattering and gathering, and none of that is visible
in a single row. This project puts three ways of scoring on the same accounts, in the same
time-ordered split, and reports what each one catches, with confidence intervals. It adds a
zero-shot tabular foundation model as a fourth column, so the question of whether those models
see relational structure gets a number too, and it ends where an engineer's question starts:
what a neighbourhood costs to fetch and score when the decision is live.

## Result

Partly measured: arms R and G on all four datasets, single seed, on the laptop; arm N and the
TabFM column need the rented GPU and are still empty. Every table below is written by
`consort report` from `results/summary.json`; none is edited by hand. Every number is a point
estimate with its 95% bootstrap interval, resampled over groups (time steps on Elliptic, first
steps on Elliptic++, hours on IBM), never over single items.

**Does the graph help?** The headline hop count for arms G and N is the one with the best
validation AUPRC, never chosen on test. "Minus R" is a paired difference on identical items.
An alert budget flags that share of all test items; its ceiling is the most recall any ranking
could reach at it (alerts over positives), which binds on the real sets, where 4 to 6% of
labelled test items are illicit. IBM rows leave out the generator's tail, days at the end that
are nearly all laundering (PLAN.md 2.2); the leakage table below reads all test days.

<!-- report:main -->
| Dataset | Arm | AUPRC | AUPRC minus R, same items | Recall at 1% FPR | Recall at 0.5% alert budget | Recall at 1% alert budget |
|---|---|---|---|---|---|---|
| Elliptic (real, transactions) | R, row only | 0.694 (0.373 to 0.835) |  | 0.621 (0.317 to 0.765) | 0.086 (0.057 to 0.157); ceiling 0.086 | 0.172 (0.115 to 0.297); ceiling 0.172 |
| Elliptic (real, transactions) | G, graph features, 2 hops | 0.718 (0.406 to 0.851) | 0.023 (0.007 to 0.047) | 0.643 (0.333 to 0.789) | 0.086 (0.057 to 0.156); ceiling 0.086 | 0.172 (0.115 to 0.293); ceiling 0.172 |
| Elliptic (real, transactions) | N, graph neural network (sage), 2 hops | 0.583 (0.225 to 0.758) | -0.112 (-0.256 to -0.034) | 0.481 (0.210 to 0.631) | 0.085 (0.055 to 0.118); ceiling 0.086 | 0.156 (0.108 to 0.214); ceiling 0.172 |
| Elliptic (real, transactions) | TabFM-R, 0 hops | 0.698 (0.376 to 0.834) | 0.004 (-0.030 to 0.051) | 0.582 (0.293 to 0.728) | 0.086 (0.057 to 0.155); ceiling 0.086 | 0.172 (0.115 to 0.277); ceiling 0.172 |
| Elliptic++ (real, wallets) | R, row only | 0.389 (0.276 to 0.522) |  | 0.292 (0.191 to 0.445) | 0.108 (0.073 to 0.173); ceiling 0.110 | 0.207 (0.139 to 0.306); ceiling 0.221 |
| Elliptic++ (real, wallets) | G, graph features, 3 hops | 0.595 (0.396 to 0.791) | 0.207 (0.069 to 0.388) | 0.477 (0.275 to 0.695) | 0.110 (0.073 to 0.178); ceiling 0.110 | 0.219 (0.147 to 0.336); ceiling 0.221 |
| Elliptic++ (real, wallets) | N, graph neural network (sage), 2 hops | 0.418 (0.301 to 0.556) | 0.029 (-0.012 to 0.075) | 0.310 (0.185 to 0.483) | 0.107 (0.070 to 0.174); ceiling 0.110 | 0.202 (0.128 to 0.318); ceiling 0.221 |
| Elliptic++ (real, wallets) | TabFM-R, 0 hops | 0.280 (0.173 to 0.399) | -0.109 (-0.175 to -0.053) | 0.229 (0.137 to 0.350) | 0.069 (0.041 to 0.109); ceiling 0.110 | 0.132 (0.088 to 0.201); ceiling 0.221 |
| IBM AML HI-Small (synthetic), without the tail | R, row only | 0.088 (0.068 to 0.112) |  | 0.487 (0.442 to 0.523) | 0.373 (0.338 to 0.408); ceiling 1.000 | 0.474 (0.435 to 0.514); ceiling 1.000 |
| IBM AML HI-Small (synthetic), without the tail | G, graph features, 1 hop | 0.377 (0.335 to 0.417) | 0.289 (0.257 to 0.317) | 0.720 (0.680 to 0.752) | 0.589 (0.551 to 0.624); ceiling 1.000 | 0.708 (0.672 to 0.739); ceiling 1.000 |
| IBM AML HI-Small (synthetic), without the tail | N, graph neural network (sage), 1 hop | 0.060 (0.048 to 0.075) | -0.028 (-0.047 to -0.010) | 0.449 (0.404 to 0.488) | 0.298 (0.265 to 0.334); ceiling 1.000 | 0.432 (0.393 to 0.471); ceiling 1.000 |
| IBM AML LI-Small (synthetic), without the tail | R, row only | 0.019 (0.013 to 0.027) |  | 0.268 (0.237 to 0.312) | 0.171 (0.143 to 0.202); ceiling 1.000 | 0.266 (0.234 to 0.309); ceiling 1.000 |
| IBM AML LI-Small (synthetic), without the tail | G, graph features, 1 hop | 0.094 (0.068 to 0.126) | 0.075 (0.052 to 0.105) | 0.527 (0.484 to 0.575) | 0.421 (0.384 to 0.464); ceiling 1.000 | 0.525 (0.480 to 0.570); ceiling 1.000 |
| IBM AML LI-Small (synthetic), without the tail | N, graph neural network | not run yet | | | | |
<!-- /report:main -->

**Which shapes only the graph can see.** Recall at a 1% alert budget by laundering pattern on
IBM HI-Small, all test days: of each pattern's laundering transfers, the share one shared queue
reaches. The patterns are the generator's; this table measures what it plants, not what a bank
sees, and about a fifth of the attempts behind each name are too small to show their shape
([docs/patterns.md](docs/patterns.md)).

<!-- report:patterns -->
| Pattern | R | G | N |
|---|---|---|---|
| Fan-in | 0.740 (0.667 to 0.832) | 0.959 (0.910 to 0.992) | 0.488 (0.388 to 0.579) |
| Fan-out | 0.809 (0.725 to 0.885) | 0.870 (0.805 to 0.922) | 0.481 (0.385 to 0.583) |
| Cycle | 0.697 (0.604 to 0.811) | 0.838 (0.752 to 0.895) | 0.525 (0.430 to 0.615) |
| Scatter-gather | 0.728 (0.683 to 0.780) | 0.900 (0.861 to 0.944) | 0.515 (0.454 to 0.576) |
| Gather-scatter | 0.709 (0.656 to 0.776) | 0.948 (0.924 to 0.969) | 0.530 (0.473 to 0.586) |
| Bipartite | 0.651 (0.544 to 0.758) | 0.803 (0.716 to 0.891) | 0.515 (0.404 to 0.627) |
| Stack | 0.718 (0.621 to 0.797) | 0.880 (0.824 to 0.943) | 0.547 (0.446 to 0.663) |
| Random | 0.583 (0.483 to 0.670) | 0.845 (0.767 to 0.919) | 0.571 (0.463 to 0.691) |
| No named pattern | 0.124 (0.093 to 0.164) | 0.402 (0.361 to 0.442) | 0.299 (0.258 to 0.342) |
<!-- /report:patterns -->

**How far guilt travels.** AUPRC by hop count (hop 0 of arm G is arm R), beside the cost of
fetching that neighbourhood live. Fetch latency comes only from the rented machine or the VPS,
never from the laptop.

<!-- report:hops -->
| Dataset | Hops | G, AUPRC | N, AUPRC | Fetch p50 (ms) | Fetch p99 (ms) |
|---|---|---|---|---|---|
| Elliptic (real, transactions) | 0 | 0.694 (0.373 to 0.835) | 0.595 (0.299 to 0.745) | not measured yet | not measured yet |
| Elliptic (real, transactions) | 1 | 0.714 (0.408 to 0.846) | 0.613 (0.295 to 0.767) | not measured yet | not measured yet |
| Elliptic (real, transactions) | 2 | 0.718 (0.406 to 0.851) | 0.583 (0.225 to 0.758) | not measured yet | not measured yet |
| Elliptic (real, transactions) | 3 | 0.720 (0.409 to 0.851) | 0.564 (0.234 to 0.722) | not measured yet | not measured yet |
| Elliptic++ (real, wallets) | 0 | 0.389 (0.276 to 0.522) | 0.319 (0.173 to 0.502) | not measured yet | not measured yet |
| Elliptic++ (real, wallets) | 1 | 0.515 (0.349 to 0.687) | 0.382 (0.244 to 0.566) | not measured yet | not measured yet |
| Elliptic++ (real, wallets) | 2 | 0.595 (0.399 to 0.794) | 0.418 (0.301 to 0.556) | not measured yet | not measured yet |
| Elliptic++ (real, wallets) | 3 | 0.595 (0.396 to 0.791) | not run yet | not measured yet | not measured yet |
| IBM AML HI-Small (synthetic) | 0 | 0.088 (0.068 to 0.112) | not run yet | not measured yet | not measured yet |
| IBM AML HI-Small (synthetic) | 1 | 0.377 (0.335 to 0.417) | 0.060 (0.048 to 0.075) | not measured yet | not measured yet |
| IBM AML HI-Small (synthetic) | 2 | 0.380 (0.336 to 0.422) | not run yet | not measured yet | not measured yet |
| IBM AML HI-Small (synthetic) | 3 | not run yet | not run yet | not measured yet | not measured yet |
| IBM AML LI-Small (synthetic) | 0 | 0.019 (0.013 to 0.027) | not run yet | not measured yet | not measured yet |
| IBM AML LI-Small (synthetic) | 1 | 0.094 (0.068 to 0.126) | not run yet | not measured yet | not measured yet |
| IBM AML LI-Small (synthetic) | 2 | 0.110 (0.083 to 0.142) | not run yet | not measured yet | not measured yet |
| IBM AML LI-Small (synthetic) | 3 | not run yet | not run yet | not measured yet | not measured yet |
<!-- /report:hops -->

**When the world shifts.** Trained on steps 1 to 34, tested before and after the dark-market
shutdown at step 43. The recovery column (TabFM's context swap, retraining) is still to run.

<!-- report:shift -->
| Dataset | Arm | Before step 43 | From step 43 | From step 43, minus R |
|---|---|---|---|---|
| Elliptic (real, transactions) | R, row only | 0.887 (0.812 to 0.957) | 0.039 (0.026 to 0.072) |  |
| Elliptic (real, transactions) | G, graph features | 0.896 (0.828 to 0.964) | 0.054 (0.035 to 0.113) | 0.015 (0.002 to 0.047) |
| Elliptic (real, transactions) | N, graph neural network | 0.805 (0.706 to 0.869) | 0.038 (0.027 to 0.068) | -0.001 (-0.020 to 0.016) |
| Elliptic (real, transactions) | TabFM-R | 0.877 (0.786 to 0.947) | 0.067 (0.038 to 0.140) | 0.028 (0.003 to 0.068) |
| Elliptic++ (real, wallets) | R, row only | 0.589 (0.517 to 0.699) | 0.283 (0.143 to 0.410) |  |
| Elliptic++ (real, wallets) | G, graph features | 0.697 (0.640 to 0.758) | 0.542 (0.247 to 0.834) | 0.259 (0.055 to 0.512) |
| Elliptic++ (real, wallets) | N, graph neural network | 0.663 (0.593 to 0.784) | 0.290 (0.124 to 0.432) | 0.007 (-0.042 to 0.093) |
| Elliptic++ (real, wallets) | TabFM-R | 0.477 (0.385 to 0.587) | 0.183 (0.072 to 0.308) | -0.100 (-0.173 to -0.019) |
<!-- /report:shift -->

**The leakage gap.** The same arm under the temporal, inductive split this project reports;
under the random split most papers use; and, on IBM, with graph features that could see later
days, the leak this build found in itself and fixed (PLAN.md 2.2). All test days.

<!-- report:leakage -->
| Dataset | Arm | Temporal split, AUPRC | Random split, AUPRC | Features that saw later days, AUPRC |
|---|---|---|---|---|
| Elliptic (real, transactions) | R, row only | 0.694 (0.373 to 0.835) | 0.985 (0.972 to 0.993) | not applicable |
| Elliptic (real, transactions) | G, graph features | 0.718 (0.406 to 0.851) | 0.987 (0.976 to 0.994) | not applicable |
| Elliptic++ (real, wallets) | R, row only | 0.389 (0.276 to 0.522) | 0.965 (0.944 to 0.979) | not applicable |
| Elliptic++ (real, wallets) | G, graph features | 0.595 (0.396 to 0.791) | 0.981 (0.970 to 0.989) | not applicable |
| IBM AML HI-Small (synthetic) | R, row only | 0.212 (0.172 to 0.278) | 0.086 (0.073 to 0.105) | not applicable |
| IBM AML HI-Small (synthetic) | G, graph features | 0.563 (0.519 to 0.612) | 0.488 (0.448 to 0.526) | 0.716 (0.672 to 0.764) |
| IBM AML LI-Small (synthetic) | R, row only | 0.033 (0.025 to 0.046) | not run yet | not applicable |
| IBM AML LI-Small (synthetic) | G, graph features | 0.156 (0.124 to 0.196) | not run yet | not applicable |
<!-- /report:leakage -->

**Can an investigator read why?** For each run's alert queue (its top 1% of test items; on
IBM a random 300 of it), every direct neighbour of a flagged item is occluded in turn on the
served neighbourhood and the item rescored through the same exported model; the neighbours
whose removal drops the score most are its explanation. On IBM a transfer's neighbours are its
two accounts', on the graph of the days before it. Among flagged items with more than k
neighbours (with k or fewer, any k is all of them) and at least one known-illicit neighbour,
how often the top k hold one, against k neighbours drawn at random. Intervals over time steps
on Elliptic and hours on IBM. Elliptic's own aggregated features, and IBM's degree and volume
counts, are fixed per node, so occlusion moves only the neighbourhood features this project
computes. Most of Elliptic's flagged transactions have one or two neighbours, so few qualify.

<!-- report:faithfulness -->
| Run | k | Flagged | Graded | Top-k holds a known-illicit neighbour | k random neighbours do | Lift |
|---|---|---|---|---|---|---|
| elliptic.temporal.G.h1.cap50.s0 | 1 | 124 | 14 | 0.571 (0.500 to 1.000) | 0.516 (0.450 to 0.530) | 0.055 (-0.021 to 0.550) |
| elliptic.temporal.G.h2.cap50.s0 | 1 | 124 | 12 | 0.417 (0.000 to 1.000) | 0.505 (0.500 to 0.540) | -0.088 (-0.540 to 0.480) |
<!-- /report:faithfulness -->

**What a live score costs.** One call scores one IBM transfer from scratch: both accounts'
neighbourhoods fetched from the store, arm G's features computed on them by the batch run's own
registry, and the transfer scored through the ONNX export; 1,000 transfers at random after 20
warm-up calls, one worker. An account's own counts (degrees, amounts, velocity) are either
computed from its whole fetched history or read from counters a service would keep; the
features are identical either way ([docs/serving.md](docs/serving.md)), the fetch is not. Cost
per million scores assumes the machine busy half the hour.

<!-- report:serving -->
| Dataset | Store | Account's own counts | Hops | p50 (ms) | p99 (ms) | Edges fetched, mean | US$ per million scores | Machine, price (date) |
|---|---|---|---|---|---|---|---|---|
| not run yet | | | | | | | | |
<!-- /report:serving -->

## What this does not do

It does not score anyone's real accounts. The real data is a published, anonymised Bitcoin
transaction graph; the bank transfers are synthetic, from a generator whose laundering
patterns are its own, and the recall-by-pattern table says so in its caption. It does not run
inside a payment stream; it hands a scorer and a feature view to a platform that does. It does
not use a language model to explain a flag; the explanations are attributions scored against
a baseline, and the case files are data. And it does not claim that a graph neural network is
the answer: the point of the three arms is to find out when the graph pays, and the plan names
in advance the result that would say it does not.

## How it works

Three arms score the same accounts in the same time-ordered, inductive split: gradient
boosting on the row, gradient boosting on the row plus features computed from the
neighbourhood at a stated hop count, and a graph neural network with neighbour sampling. One
evaluator reads every arm's outcomes without knowing which model wrote them, bootstraps over
time steps or components rather than accounts, and refuses to compare runs that scored
different items. The design, the splits, the leakage test, the serving benchmark and the cost
arithmetic are in [PLAN.md](PLAN.md); the figures are in
[docs/architecture.md](docs/architecture.md).

## Running it

Python 3.13 and [uv](https://docs.astral.sh/uv/). The datasets are fetched with your own
Kaggle API token (accept each dataset's terms on its Kaggle page once); nothing raw is in this
repository, and every fetched file is checked against the committed checksum manifest.

```
uv sync --extra gnn --extra export
uv run consort data fetch elliptic
uv run consort data fetch ibm-HI-Small
uv run consort data check elliptic
uv run consort patterns --tier HI-Small
uv run consort train R --dataset elliptic
uv run consort train G --dataset elliptic --hops 2 --hub-cap 50
uv run consort train-gnn --dataset elliptic --hops 2 --hub-cap 25
uv run consort eval data/runs/elliptic.temporal.G.h2.cap50.s0.outcomes.parquet \
  --against data/runs/elliptic.temporal.R.h0.cap0.s0.outcomes.parquet
uv run pytest
```

The licences, as quoted before the first fetch, are in [docs/data.md](docs/data.md); model
versions and settings in [docs/models.md](docs/models.md); the laundering patterns in
[docs/patterns.md](docs/patterns.md); the serving path and its parity test in
[docs/serving.md](docs/serving.md).

## Part of a portfolio

One of seventeen projects on [peterparker.ca](https://peterparker.ca), each of which ends in a
measured result a stranger can reproduce from its repository with one command. This one hands
a challenger scorer and a feature view to
[Real-Time Fraud & Risk Decisioning Platform](https://github.com/Peter-A-P/realtime-risk-platform)
and runs its paired comparisons through
[AI Release Gate](https://github.com/Peter-A-P/ai-release-gate).

## How this was built

The design, the methodology and the judgement calls are Peter Parker's. AI coding assistants
were used for implementation. Every number in this repository is reproducible with one command,
and the plan was written before the first line of code so that the result could not be shaped
to fit.
