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

Measured on all four datasets: arms R and G on the laptop, arm N, the TabFM column and the
serving benchmark on rented machines. Every table reads seed 0. Every table below is written by
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
| Elliptic (real, transactions) | G, graph features, 3 hops | 0.720 (0.409 to 0.851) | 0.025 (0.008 to 0.051) | 0.639 (0.333 to 0.785) | 0.086 (0.057 to 0.157); ceiling 0.086 | 0.172 (0.115 to 0.293); ceiling 0.172 |
| Elliptic (real, transactions) | N, graph neural network (sage), 2 hops | 0.583 (0.225 to 0.758) | -0.112 (-0.256 to -0.034) | 0.481 (0.210 to 0.631) | 0.085 (0.055 to 0.118); ceiling 0.086 | 0.156 (0.108 to 0.214); ceiling 0.172 |
| Elliptic (real, transactions) | TabFM-G, 1 hop | 0.776 (0.491 to 0.891) | 0.082 (0.009 to 0.193) | 0.637 (0.345 to 0.787) | 0.086 (0.057 to 0.154); ceiling 0.086 | 0.172 (0.115 to 0.277); ceiling 0.172 |
| Elliptic (real, transactions) | TabFM-R, 0 hops | 0.698 (0.376 to 0.833) | 0.003 (-0.030 to 0.051) | 0.582 (0.293 to 0.728) | 0.086 (0.057 to 0.155); ceiling 0.086 | 0.172 (0.115 to 0.276); ceiling 0.172 |
| Elliptic++ (real, wallets) | R, row only | 0.389 (0.276 to 0.522) |  | 0.292 (0.191 to 0.445) | 0.108 (0.073 to 0.173); ceiling 0.110 | 0.207 (0.139 to 0.306); ceiling 0.221 |
| Elliptic++ (real, wallets) | G, graph features, 3 hops | 0.595 (0.396 to 0.791) | 0.207 (0.069 to 0.388) | 0.477 (0.275 to 0.695) | 0.110 (0.073 to 0.178); ceiling 0.110 | 0.219 (0.147 to 0.336); ceiling 0.221 |
| Elliptic++ (real, wallets) | N, graph neural network (sage), 2 hops | 0.418 (0.301 to 0.556) | 0.029 (-0.012 to 0.075) | 0.310 (0.185 to 0.483) | 0.107 (0.070 to 0.174); ceiling 0.110 | 0.202 (0.128 to 0.318); ceiling 0.221 |
| Elliptic++ (real, wallets) | TabFM-G, 1 hop | 0.413 (0.275 to 0.547) | 0.024 (-0.016 to 0.096) | 0.344 (0.230 to 0.482) | 0.105 (0.070 to 0.167); ceiling 0.110 | 0.202 (0.134 to 0.312); ceiling 0.221 |
| Elliptic++ (real, wallets) | TabFM-R, 0 hops | 0.280 (0.173 to 0.399) | -0.109 (-0.175 to -0.053) | 0.229 (0.137 to 0.350) | 0.069 (0.041 to 0.109); ceiling 0.110 | 0.132 (0.088 to 0.201); ceiling 0.221 |
| IBM AML HI-Small (synthetic), without the tail | R, row only | 0.088 (0.068 to 0.112) |  | 0.487 (0.442 to 0.523) | 0.373 (0.338 to 0.408); ceiling 1.000 | 0.474 (0.435 to 0.514); ceiling 1.000 |
| IBM AML HI-Small (synthetic), without the tail | G, graph features, 1 hop | 0.377 (0.335 to 0.417) | 0.289 (0.257 to 0.317) | 0.720 (0.680 to 0.752) | 0.589 (0.551 to 0.624); ceiling 1.000 | 0.708 (0.672 to 0.739); ceiling 1.000 |
| IBM AML HI-Small (synthetic), without the tail | N, graph neural network (sage+rmp+ego), 2 hops | 0.083 (0.066 to 0.104) | -0.006 (-0.021 to 0.010) | 0.586 (0.546 to 0.626) | 0.365 (0.334 to 0.402); ceiling 1.000 | 0.566 (0.518 to 0.606); ceiling 1.000 |
| IBM AML LI-Small (synthetic), without the tail | R, row only | 0.019 (0.013 to 0.027) |  | 0.268 (0.237 to 0.312) | 0.171 (0.143 to 0.202); ceiling 1.000 | 0.266 (0.234 to 0.309); ceiling 1.000 |
| IBM AML LI-Small (synthetic), without the tail | G, graph features, 1 hop | 0.094 (0.068 to 0.126) | 0.075 (0.052 to 0.105) | 0.527 (0.484 to 0.575) | 0.421 (0.384 to 0.464); ceiling 1.000 | 0.525 (0.480 to 0.570); ceiling 1.000 |
| IBM AML LI-Small (synthetic), without the tail | N, graph neural network (sage), 2 hops | 0.025 (0.020 to 0.031) | 0.006 (-0.002 to 0.013) | 0.439 (0.401 to 0.480) | 0.304 (0.268 to 0.339); ceiling 1.000 | 0.437 (0.396 to 0.473); ceiling 1.000 |
<!-- /report:main -->

What the table says:

- **Graph features pay where the laundering is a shape.** Arm G adds 0.29 AUPRC on IBM
  HI-Small, 0.08 on LI-Small and 0.21 on Elliptic++, each interval clear of zero. On Elliptic,
  whose own features already aggregate each transaction's neighbours, it adds 0.025.
- **The graph neural network never beats the graph features**, on any set. On Elliptic it is
  worse than the row model (-0.11). On IBM its AUPRC is the row model's, yet its 1% alert
  queue reaches more laundering: project 03's gate passes it at +6.9 points of recall on
  HI-Small and +14.0 on LI-Small ([docs/gate.md](docs/gate.md)). It ranks the top of the queue
  better than the rest of it.
- **TabFM, handed arm G's table, scores Elliptic best of all** (0.776, 0.08 above the row
  model), with no training. On Elliptic++'s wallets it does not: handed the row it is 0.11
  below the row model, and handed arm G's table it ties the row model (0.413, +0.024, the
  interval spanning zero) and sits 0.18 below arm G. How many training rows it is shown
  matters as much as the table: at 10,000 context rows instead of 30,000 the same run scored
  0.233. Zero-shot is not a safe default on a set it has not been tried on.
- **On the real sets, the alert budget cannot separate the arms.** At 1% every arm sits at or
  near its ceiling, so the AUPRC column is where they differ.

**How far another seed moves it.** The tables read seed 0. Arm N's headline runs were repeated
at seeds 1 and 2. Arm N moves by about 0.12 AUPRC from one seed to another on both real
sets, as much as some of the differences above; at its best seed it is still below the row
model on Elliptic and below arm G on Elliptic++. On IBM's synthetic sets it barely moves (under
0.01), and every seed stays far below arm G.

<!-- report:seeds -->
| Dataset | Run | Hops | AUPRC by seed | Spread (largest minus smallest) |
|---|---|---|---|---|
| Elliptic (real, transactions) | N-sage | 2 | seed 0: 0.583; seed 1: 0.536; seed 2: 0.657 | 0.121 |
| Elliptic++ (real, wallets) | N-sage | 2 | seed 0: 0.418; seed 1: 0.301; seed 2: 0.406 | 0.117 |
| IBM AML HI-Small (synthetic) | N-sage+rmp+ego | 2 | seed 0: 0.083; seed 1: 0.081; seed 2: 0.076 | 0.007 |
| IBM AML LI-Small (synthetic) | N-sage | 2 | seed 0: 0.025; seed 1: 0.027; seed 2: 0.025 | 0.002 |
<!-- /report:seeds -->

**Which shapes only the graph can see.** Recall at a 1% alert budget by laundering pattern on
IBM HI-Small, all test days: of each pattern's laundering transfers, the share one shared queue
reaches. The patterns are the generator's; this table measures what it plants, not what a bank
sees, and about a fifth of the attempts behind each name are too small to show their shape
([docs/patterns.md](docs/patterns.md)).

<!-- report:patterns -->
| Pattern | R | G | N |
|---|---|---|---|
| Fan-in | 0.740 (0.667 to 0.832) | 0.959 (0.910 to 0.992) | 0.797 (0.723 to 0.879) |
| Fan-out | 0.809 (0.725 to 0.885) | 0.870 (0.805 to 0.922) | 0.756 (0.676 to 0.840) |
| Cycle | 0.697 (0.604 to 0.811) | 0.838 (0.752 to 0.895) | 0.727 (0.632 to 0.828) |
| Scatter-gather | 0.728 (0.683 to 0.780) | 0.900 (0.861 to 0.944) | 0.782 (0.730 to 0.852) |
| Gather-scatter | 0.709 (0.656 to 0.776) | 0.948 (0.924 to 0.969) | 0.795 (0.746 to 0.839) |
| Bipartite | 0.651 (0.544 to 0.758) | 0.803 (0.716 to 0.891) | 0.651 (0.540 to 0.753) |
| Stack | 0.718 (0.621 to 0.797) | 0.880 (0.824 to 0.943) | 0.658 (0.548 to 0.743) |
| Random | 0.583 (0.483 to 0.670) | 0.845 (0.767 to 0.919) | 0.655 (0.536 to 0.782) |
| No named pattern | 0.124 (0.093 to 0.164) | 0.402 (0.361 to 0.442) | 0.294 (0.249 to 0.345) |
<!-- /report:patterns -->

Arm G reaches more of every named pattern than the row model, most of all fan-in (96% against
74%) and gather-scatter (95% against 71%), the shapes defined by many accounts meeting at one.
Arm N sits between the two on most patterns and below the row model on fan-out and stack.

**How far guilt travels.** AUPRC by hop count (hop 0 of arm G is arm R), beside the cost of
fetching that neighbourhood live. Fetch latency comes only from the rented machine or the VPS,
never from the laptop. Only IBM's synthetic transfers are served; the real sets are not, so
their fetch columns say so.

<!-- report:hops -->
| Dataset | Hops | G, AUPRC | N, AUPRC | Fetch p50 (ms) | Fetch p99 (ms) |
|---|---|---|---|---|---|
| Elliptic (real, transactions) | 0 | 0.694 (0.373 to 0.835) | 0.595 (0.299 to 0.745) | not served | not served |
| Elliptic (real, transactions) | 1 | 0.714 (0.408 to 0.846) | 0.613 (0.295 to 0.767) | not served | not served |
| Elliptic (real, transactions) | 2 | 0.718 (0.406 to 0.851) | 0.583 (0.225 to 0.758) | not served | not served |
| Elliptic (real, transactions) | 3 | 0.720 (0.409 to 0.851) | 0.564 (0.234 to 0.722) | not served | not served |
| Elliptic++ (real, wallets) | 0 | 0.389 (0.276 to 0.522) | 0.319 (0.173 to 0.502) | not served | not served |
| Elliptic++ (real, wallets) | 1 | 0.515 (0.349 to 0.687) | 0.382 (0.244 to 0.566) | not served | not served |
| Elliptic++ (real, wallets) | 2 | 0.595 (0.399 to 0.794) | 0.418 (0.301 to 0.556) | not served | not served |
| Elliptic++ (real, wallets) | 3 | 0.595 (0.396 to 0.791) | not run yet | not served | not served |
| IBM AML HI-Small (synthetic) | 0 | 0.088 (0.068 to 0.112) | not run yet | no fetch | no fetch |
| IBM AML HI-Small (synthetic) | 1 | 0.377 (0.335 to 0.417) | 0.068 (0.056 to 0.083) | memory 4.4; duckdb 10.2; neo4j 16.5 | memory 5.0; duckdb 11.2; neo4j 134.3 |
| IBM AML HI-Small (synthetic) | 2 | 0.380 (0.336 to 0.422) | 0.072 (0.059 to 0.089) | memory 10.1; duckdb 45.9; neo4j 48.0 | memory 12.1; duckdb 125.9; neo4j 348.4 |
| IBM AML HI-Small (synthetic) | 3 | not run yet | not run yet | not measured | not measured |
| IBM AML LI-Small (synthetic) | 0 | 0.019 (0.013 to 0.027) | not run yet | no fetch | no fetch |
| IBM AML LI-Small (synthetic) | 1 | 0.094 (0.068 to 0.126) | 0.029 (0.024 to 0.036) | not measured | not measured |
| IBM AML LI-Small (synthetic) | 2 | 0.110 (0.083 to 0.142) | 0.025 (0.020 to 0.031) | not measured | not measured |
| IBM AML LI-Small (synthetic) | 3 | not run yet | not run yet | not measured | not measured |
<!-- /report:hops -->

On Elliptic one hop is all the graph features need, and arm N is best at one hop and worse
with each hop after. Elliptic++'s wallets gain up to two hops. On IBM the second hop adds nothing measurable
to arm G and costs two to four and a half times as much to fetch, so one hop, from counters, is
the live design: under 20 ms at p50 in every store tested.

**When the world shifts.** Trained on steps 1 to 34, tested before and after the dark-market
shutdown at step 43. The last two Elliptic rows recover with labels: before each test step,
the row model is retrained, and TabFM's context replaced, on the four steps just before it,
labels the frozen arms never see (PLAN.md 2.6). Same rows for both.

<!-- report:shift -->
| Dataset | Arm | Before step 43 | From step 43 | From step 43, minus R |
|---|---|---|---|---|
| Elliptic (real, transactions) | R, row only | 0.887 (0.812 to 0.957) | 0.039 (0.026 to 0.072) |  |
| Elliptic (real, transactions) | G, graph features | 0.897 (0.831 to 0.963) | 0.056 (0.035 to 0.122) | 0.017 (0.002 to 0.051) |
| Elliptic (real, transactions) | N, graph neural network | 0.805 (0.706 to 0.869) | 0.038 (0.027 to 0.068) | -0.001 (-0.020 to 0.016) |
| Elliptic (real, transactions) | TabFM-G | 0.908 (0.820 to 0.969) | 0.172 (0.093 to 0.363) | 0.133 (0.054 to 0.302) |
| Elliptic (real, transactions) | TabFM-R | 0.877 (0.786 to 0.947) | 0.067 (0.038 to 0.140) | 0.028 (0.003 to 0.068) |
| Elliptic (real, transactions) | R, retrained each test step on the latest labels | 0.891 (0.800 to 0.943) | 0.311 (0.137 to 0.542) | 0.272 (0.107 to 0.474) |
| Elliptic (real, transactions) | TabFM-R, context swapped each test step to the latest labels | 0.929 (0.842 to 0.979) | 0.649 (0.094 to 0.915) | 0.610 (0.056 to 0.847) |
| Elliptic++ (real, wallets) | R, row only | 0.589 (0.517 to 0.699) | 0.283 (0.143 to 0.410) |  |
| Elliptic++ (real, wallets) | G, graph features | 0.697 (0.640 to 0.758) | 0.542 (0.247 to 0.834) | 0.259 (0.055 to 0.512) |
| Elliptic++ (real, wallets) | N, graph neural network | 0.663 (0.593 to 0.784) | 0.290 (0.124 to 0.432) | 0.007 (-0.042 to 0.093) |
| Elliptic++ (real, wallets) | TabFM-G | 0.616 (0.553 to 0.714) | 0.302 (0.134 to 0.494) | 0.019 (-0.038 to 0.146) |
| Elliptic++ (real, wallets) | TabFM-R | 0.477 (0.385 to 0.587) | 0.183 (0.072 to 0.308) | -0.100 (-0.173 to -0.019) |
<!-- /report:shift -->

After the shutdown the frozen arms collapse on Elliptic: AUPRC falls from about 0.9 to under
0.07, graph or not. The exception is TabFM handed arm G's table, which holds 0.17. Recovering with the
latest labels works: retraining the row model brings it to 0.31, and swapping TabFM's context
to the same rows brings it to 0.65. The swap costs only inference. Its interval is wide (0.09
to 0.92) and overlaps retraining's, so it is the better recovery on this test, not a proven
one. Elliptic++'s wallets fall less, and there arm G keeps more than half its AUPRC.

**The leakage gap.** The same arm under the temporal, inductive split this project reports;
under the random split most papers use; and, on IBM, with graph features that could see later
days, the leak this build found in itself and fixed (PLAN.md 2.2). All test days.

<!-- report:leakage -->
| Dataset | Arm | Temporal split, AUPRC | Random split, AUPRC | Features that saw later days, AUPRC |
|---|---|---|---|---|
| Elliptic (real, transactions) | R, row only | 0.694 (0.373 to 0.835) | 0.985 (0.972 to 0.993) | not applicable |
| Elliptic (real, transactions) | G, graph features | 0.720 (0.409 to 0.851) | 0.987 (0.976 to 0.994) | not applicable |
| Elliptic++ (real, wallets) | R, row only | 0.389 (0.276 to 0.522) | 0.965 (0.944 to 0.979) | not applicable |
| Elliptic++ (real, wallets) | G, graph features | 0.595 (0.396 to 0.791) | 0.981 (0.970 to 0.989) | not applicable |
| IBM AML HI-Small (synthetic) | R, row only | 0.212 (0.172 to 0.278) | 0.086 (0.073 to 0.105) | not applicable |
| IBM AML HI-Small (synthetic) | G, graph features | 0.563 (0.519 to 0.612) | 0.488 (0.448 to 0.526) | 0.716 (0.672 to 0.764) |
| IBM AML LI-Small (synthetic) | R, row only | 0.033 (0.025 to 0.046) | 0.021 (0.017 to 0.029) | not applicable |
| IBM AML LI-Small (synthetic) | G, graph features | 0.156 (0.124 to 0.196) | 0.159 (0.131 to 0.191) | not applicable |
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
| ibm-HI-Small.temporal.G.h1.cap50.s0 | 1 | 300 | 101 | 0.465 (0.366 to 0.580) | 0.340 (0.290 to 0.397) | 0.126 (0.035 to 0.221) |
| ibm-HI-Small.temporal.G.h1.cap50.s0 | 3 | 300 | 88 | 0.670 (0.554 to 0.785) | 0.649 (0.584 to 0.714) | 0.021 (-0.095 to 0.126) |
| ibm-HI-Small.temporal.G.h1.cap50.s0 | 5 | 300 | 67 | 0.851 (0.761 to 0.940) | 0.766 (0.697 to 0.835) | 0.085 (0.010 to 0.166) |
<!-- /report:faithfulness -->

On IBM HI-Small the occlusion ranking carries some signal: the single most important neighbour
is an account the generator used for laundering 47% of the time, against 34% for a neighbour
drawn at random, a lift of 13 points (4 to 22). It is weaker at three neighbours, where the
interval spans zero. An investigator gets a better-than-random pointer, not a reliable one.
On Elliptic too few flagged transactions have more than one neighbour to say anything.

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
| IBM AML HI-Small (synthetic) | duckdb | stored counters | 1 | 10.2 | 11.2 | 94 | 5.04 | RunPod SECURE NVIDIA GeForce RTX 4090, US$0.89/h (2026-11-02) |
| IBM AML HI-Small (synthetic) | memory | stored counters | 1 | 4.4 | 5.0 | 94 | 2.19 | RunPod SECURE NVIDIA GeForce RTX 4090, US$0.89/h (2026-11-02) |
| IBM AML HI-Small (synthetic) | neo4j | stored counters | 1 | 16.5 | 134.3 | 94 | 11.26 | RunPod SECURE NVIDIA GeForce RTX 4090, US$0.89/h (2026-11-02) |
| IBM AML HI-Small (synthetic) | duckdb | stored counters | 2 | 45.9 | 125.9 | 545 | 25.70 | RunPod SECURE NVIDIA GeForce RTX 4090, US$0.89/h (2026-11-02) |
| IBM AML HI-Small (synthetic) | memory | stored counters | 2 | 10.1 | 12.1 | 545 | 5.07 | RunPod SECURE NVIDIA GeForce RTX 4090, US$0.89/h (2026-11-02) |
| IBM AML HI-Small (synthetic) | neo4j | stored counters | 2 | 48.0 | 348.4 | 545 | 37.53 | RunPod SECURE NVIDIA GeForce RTX 4090, US$0.89/h (2026-11-02) |
| IBM AML HI-Small (synthetic) | duckdb | fetched whole | 1 | 11.6 | 114.5 | 9,608 | 8.48 | RunPod SECURE NVIDIA GeForce RTX 4090, US$0.89/h (2026-11-02) |
| IBM AML HI-Small (synthetic) | memory | fetched whole | 1 | 3.8 | 105.8 | 9,608 | 4.62 | RunPod SECURE NVIDIA GeForce RTX 4090, US$0.89/h (2026-11-02) |
| IBM AML HI-Small (synthetic) | neo4j | fetched whole | 1 | 22.0 | 9549.2 | 9,608 | 271.93 | RunPod SECURE NVIDIA GeForce RTX 4090, US$0.89/h (2026-11-02) |
| IBM AML HI-Small (synthetic) | duckdb | fetched whole | 2 | 46.5 | 298.2 | 9,971 | 30.49 | RunPod SECURE NVIDIA GeForce RTX 4090, US$0.89/h (2026-11-02) |
| IBM AML HI-Small (synthetic) | memory | fetched whole | 2 | 9.5 | 181.8 | 9,971 | 9.42 | RunPod SECURE NVIDIA GeForce RTX 4090, US$0.89/h (2026-11-02) |
| IBM AML HI-Small (synthetic) | neo4j | fetched whole | 2 | 53.8 | 9911.4 | 9,971 | 302.00 | RunPod SECURE NVIDIA GeForce RTX 4090, US$0.89/h (2026-11-02) |
<!-- /report:serving -->

TabFM is not in that table: its licence rules out serving it. For scale, it scored Elliptic's
test rows at 287 ms each on a RunPod NVIDIA L4 at US$0.59 an hour (2026-11-03), with no fetch,
about US$94 per million on the same half-busy convention. That is one timed run, so it has no
interval ([docs/models.md](docs/models.md)).

## What this does not do

It does not score anyone's real accounts. The real data is a published, anonymised Bitcoin
transaction graph; the bank transfers are synthetic, from a generator whose laundering
patterns are its own, and the recall-by-pattern table says so in its caption. It does not run
inside a payment stream; it hands a scorer and a feature view to a platform that does. It does
not use a language model to explain a flag; the explanations are attributions scored against
a baseline, and the case files are data. And it does not claim that a graph neural network is
the answer: the point of the three arms is to find out when the graph pays, and the plan names
in advance the result that would say it does not.

Where the result is weakest:

- **Arm N rests on one seed per table**, and a second seed moves it by about 0.12 AUPRC on the
  real sets. It is below arm G at every seed run, and below the row model on Elliptic at every
  seed; against the row model on Elliptic++ it is not settled.
- **The largest gains are on synthetic data.** Arm G's 0.29 on IBM HI-Small measures a
  generator's laundering, which plants shapes on purpose. On the real wallets of Elliptic++
  the gain is 0.21, but its interval runs from 0.07 to 0.39.
- **Elliptic's test is small after step 43.** The seven steps after the shutdown hold 169
  illicit transactions, two of them in step 46. Every interval in the shift table is wide,
  and the context swap's lead over retraining is not proven.
- **Explanations are only somewhat better than random.** The occlusion ranking's lift is 13
  points at one neighbour on IBM, and not measurable at three.

## What was tried and rejected

**Scoring an account by how many of its neighbours are already known to be illicit.** It is
guilt by association in one column, and published graph fraud results often lean on it. Here
it costs arm G 0.019 AUPRC on Elliptic's time-ordered split (0.008 to 0.033): a test
transaction's neighbours sit in its own time step, whose labels are not yet known, so the
column is empty exactly where it is needed. On a random split it adds 0.002, its interval
touching zero, because a random split already gives the model the test step's own
transactions to learn from. The uplift it shows in the literature comes from the split, not
the column ([docs/rejected.md](docs/rejected.md)).

## How it works

Three arms score the same accounts in the same time-ordered, inductive split: gradient
boosting on the row, gradient boosting on the row plus features computed from the
neighbourhood at a stated hop count, and a graph neural network with neighbour sampling. One
evaluator reads every arm's outcomes without knowing which model wrote them, bootstraps over
time steps or components rather than accounts, and refuses to compare runs that scored
different items. The design, the splits, the leakage test, the serving benchmark and the cost
arithmetic are in [PLAN.md](PLAN.md); the figures are in
[docs/architecture.md](docs/architecture.md).

## The demo page

Live at https://graph.peterparker.ca, styled like the other project pages on peterparker.ca,
with a short film of the ring (`video/`):
the results above as interactive charts with their intervals, and one laundering ring from
IBM's synthetic set as each arm scored it, with arm G's case file for every transfer it
flagged. The ring is chosen by a rule that reads no score: the scatter-gather attempt with the
most transfers on the test days. Its accounts are named by their place in the ring. Only
synthetic accounts are drawn; Elliptic appears only as aggregates. Every number on it is read
from `site/data.json`, which `uv run consort site` writes from `results/`, the gate's ledger
and the IBM HI-Small runs; how it is hosted is in [docs/deploy.md](docs/deploy.md).

## Data and citations

Three public datasets, each fetched by the reader and never redistributed here; their terms are
quoted in [docs/data.md](docs/data.md).

- **Elliptic** (CC BY-NC-ND 4.0): Mark Weber, Giacomo Domeniconi, Jie Chen, Daniel Karl I.
  Weidele, Claudio Bellei, Tom Robinson and Charles E. Leiserson. "Anti-Money Laundering in
  Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics". KDD
  Workshop on Anomaly Detection in Finance, 2019. https://arxiv.org/abs/1908.02591
- **Elliptic++** (used with both authors' written permission, for research and aggregate
  results only): Youssef Elmougy and Ling Liu. 2023. "Demystifying Fraudulent Transactions and
  Illicit Nodes in the Bitcoin Network for Financial Forensics". In Proceedings of the 29th ACM
  SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23).
  https://doi.org/10.1145/3580305.3599803
- **IBM Transactions for Anti Money Laundering** (CDLA Sharing 1.0): Erik Altman, Jovan
  Blanusa, Luc von Niederhausern, Beni Egressy, Andreea Anghel and Kubilay Atasu. "Realistic
  Synthetic Financial Transactions for Anti-Money Laundering Models". NeurIPS Datasets and
  Benchmarks, 2023. https://arxiv.org/abs/2306.16424

The code is under the MIT licence ([LICENSE](LICENSE)); the licence does not reach the
datasets or the TabFM weights, which keep their own terms.

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
[docs/serving.md](docs/serving.md); what is handed to the real-time platform, and what is not, in
[docs/handover.md](docs/handover.md).

## Part of a portfolio

One of seventeen projects on [peterparker.ca](https://peterparker.ca), each of which ends in a
measured result a stranger can reproduce from its repository with one command. This one hands
a challenger scorer and a feature view to
[Real-Time Fraud & Risk Decisioning Platform](https://github.com/Peter-A-P/realtime-risk-platform)
and runs its paired comparisons through
[AI Release Gate](https://github.com/Peter-A-P/ai-release-gate).

## How this was built

I designed this project and made the calls in it. I wrote the plan and drew the three
architecture diagrams before the first line of code, so that the result could not be shaped to
fit, and wherever the build proved the plan wrong I amended it in place, with the date and the
reason. The leakage test came before the first feature. I chose to split every dataset by time
and to run the usual random split once, only to publish the gap it opens. My own build had a leak
of the same kind, graph features computed with sight of later days, and it was fixed once found.
I measured the business question as an alert queue, how much of the laundering a fixed team
reaches, because that is the number that decides whether a graph is worth funding.

I used an AI coding assistant (Claude Code) for implementation and drafting, which is now
standard practice in software, data science and machine learning work. Every number in this
repository is reproducible with one command.
