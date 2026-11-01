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

Nothing is measured yet. The plan was written before the first line of code
([PLAN.md](PLAN.md)); these are the tables the build fills, by `consort report`.

**Does the graph help?** Per dataset, 95% bootstrap intervals over groups, paired deltas on
identical accounts.

| Dataset | Arm | AUPRC | Recall at 1% FPR | Recall at 0.5% alert budget | Recall at 1% alert budget |
|---|---|---|---|---|---|
| Elliptic (real) | R, row only | | | | |
| Elliptic (real) | G, graph features | | | | |
| Elliptic (real) | N, graph neural network | | | | |
| Elliptic (real) | TabFM on R's and G's features | | | | |
| IBM AML HI-Small (synthetic) | R, G, N, TabFM | | | | |
| IBM AML LI-Small (synthetic) | R, G, N, TabFM | | | | |

**Which shapes only the graph can see.** Recall by laundering pattern on the synthetic set. The
patterns are the generator's; this table measures what it plants, not what a bank sees.

| Pattern | R | G | N |
|---|---|---|---|
| Fan-in | | | |
| Fan-out | | | |
| Cycle | | | |
| Scatter-gather | | | |
| Gather-scatter | | | |
| Bipartite | | | |
| Stack | | | |
| Random | | | |

**How far guilt travels.** Gain per hop beside the cost of fetching it.

| Hops | G, AUPRC | N, AUPRC | Fetch p50 (ms) | Fetch p99 (ms) |
|---|---|---|---|---|
| 0 | | | | |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

**When the world shifts.** Elliptic, trained before time step 43 (the dark-market shutdown),
tested before and after it; and the context swap.

| Arm | Before step 43 | After step 43 | Recovery by context swap or retraining |
|---|---|---|---|
| R | | | |
| G | | | |
| N | | | |
| TabFM, recent-context | | | |

**The leakage gap, the explanation faithfulness and the serving cost** follow in the same
shape, each with its interval and its price date.

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
