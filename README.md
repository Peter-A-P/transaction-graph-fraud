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
<!-- /report:main -->

**Which shapes only the graph can see.** Recall at a 1% alert budget by laundering pattern on
IBM HI-Small, all test days: of each pattern's laundering transfers, the share one shared queue
reaches. The patterns are the generator's; this table measures what it plants, not what a bank
sees, and about a fifth of the attempts behind each name are too small to show their shape
([docs/patterns.md](docs/patterns.md)).

<!-- report:patterns -->
<!-- /report:patterns -->

**How far guilt travels.** AUPRC by hop count (hop 0 of arm G is arm R), beside the cost of
fetching that neighbourhood live. Fetch latency comes only from the rented machine or the VPS,
never from the laptop.

<!-- report:hops -->
<!-- /report:hops -->

**When the world shifts.** Trained on steps 1 to 34, tested before and after the dark-market
shutdown at step 43. The recovery column (TabFM's context swap, retraining) is still to run.

<!-- report:shift -->
<!-- /report:shift -->

**The leakage gap.** The same arm under the temporal, inductive split this project reports;
under the random split most papers use; and, on IBM, with graph features that could see later
days, the leak this build found in itself and fixed (PLAN.md 2.2). All test days.

<!-- report:leakage -->
<!-- /report:leakage -->

**Can an investigator read why?** For each run's alert queue (its top 1% of test items), every
direct neighbour of a flagged transaction is occluded in turn on the served neighbourhood and
the transaction rescored through the same exported model; the neighbours whose removal drops
the score most are its explanation. Among flagged transactions with at least one known-illicit
neighbour, how often the top k hold one, against k neighbours drawn at random. Intervals over
time steps. Elliptic's own aggregated features are fixed per transaction, so occlusion moves
only the features this project computes.

<!-- report:faithfulness -->
<!-- /report:faithfulness -->

The serving cost per million scores follows in the same shape, with its price date, once the
rented machine has run.

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
