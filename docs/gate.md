# The 03 gate

Project 03's release gate decides whether a challenger arm is non-inferior to arm R, through
its `outcomes_file` adapter. The question it is asked is the plan's business number: for each
illicit test item, does the 1% alert queue reach it? So each side file holds one verdict per
illicit test item, paired across the arms, and the gate's paired test is a test on recall at
the 1% alert budget, with a margin of 3 points.

    uv run consort gate-export --baseline data/runs/<R run>.outcomes.parquet \
        --candidate data/runs/<G or N run>.outcomes.parquet --suite <dataset>
    cd ../03-ai-release-gate && uv run gate compare --spec <17>/data/gate/<suite>/spec.yaml \
        --baseline <17>/data/gate/<suite>/R.json --candidate <17>/data/gate/<suite>/G.json \
        --ledger <17>/gate/ledger.jsonl

- The queue is the top 1% of all test items by score, ties broken by item id, so every item
  gets a definite verdict (`consort/gate/export.py`).
- Side files name items, so they stay under `data/gate/`, ignored: Elliptic's licence forbids
  redistributing derivatives, and Elliptic++ was granted for aggregates only. The gate's
  ledger record, committed in `gate/ledger.jsonl`, holds the decision, the aggregates and the
  SHA-256 of each side file, so anyone who rebuilds the files can check the decision was made
  on them.
- The gate resamples items; its interval is its own. The README's intervals resample groups.
- Batch scores are not timed calls, so the side files mark every call uncosted and carry no
  latency; the serving benchmark prices a live score.

## Decisions

| Date (UTC) | Suite | Baseline | Candidate | Recall at 1% budget, baseline | Candidate | Difference | Verdict |
|---|---|---|---|---|---|---|---|
| 2026-11-02 | ibm-hi-small | R | G, 1 hop, cap 50 | 57.9% (55.5 to 60.3) | 78.6% (76.7 to 80.6) | +20.8 points (+18.6 to +23.2) | pass |
| 2026-11-02 | ibm-li-small | R | G, 1 hop, cap 50 | 33.2% | 58.4% | +25.2 points (+21.6 to +28.8) | pass |
| 2026-11-02 | elliptic | R | G, 2 hops, cap 50 | 17.3% | 17.3% | +0.0 points (-3.1 to +3.1) | block |
| 2026-11-02 | elliptic | R | N, GraphSAGE, 2 hops | 17.3% | 15.6% | -1.7 points (-4.7 to +1.5) | block |
| 2026-11-02 | ellipticpp | R | G, 3 hops, cap 50 | 20.7% | 21.9% | +1.1 points (-0.1 to +2.4) | pass |
| 2026-11-02 | ellipticpp | R | N, GraphSAGE, 2 hops | 20.7% | 20.2% | -0.5 points (-1.7 to +0.7) | pass |

Each candidate is the arm's headline run, its hop count (and for arm N its architecture)
chosen on validation AUPRC. Arm N on IBM waits for its rerun (docs/gpu.md).

Read the verdicts as what they are: non-inferiority at a 3-point margin, not improvement.
"block" on Elliptic means the gate cannot rule out a 3-point drop, not that one was seen: the
1% queue there holds about 167 transactions against 717 illicit ones, so no ranking can reach
more than 23% of them, and both arms sit near that ceiling. The same holds on Elliptic++ (719
alerts against 3,259 illicit addresses, a ceiling of 22%), where the larger test set narrows
the interval enough for both challengers to pass. On the real sets the 1% budget asks too
little to separate the arms; the README's AUPRC column is where they differ. On IBM, where
the budget reaches most of the laundering, arm G's gain is twenty points or more.

On every decision so far the gate's recall at the budget equals the evaluator's to the third
decimal (0.579 and 0.786 on HI-Small), two independent implementations agreeing.
