# Working notes for Claude Code

This repository is Fraud Known by the Company It Keeps, package `consort`: measuring when
scoring an account by its neighbourhood beats scoring it alone, on a real Bitcoin transaction
graph and on synthetic bank transfers with labelled laundering patterns; three arms (row only,
graph features, graph neural network), a tabular foundation model as a fourth column, the hop
curve, the shift at Elliptic's step 43, the leakage gap, explanation faithfulness and the cost
of a live neighbourhood fetch. The plan is in [PLAN.md](PLAN.md); the diagrams in
[docs/architecture.md](docs/architecture.md). Status: building since 2026-10-31. The package, the leakage test, the evaluator, the
loaders, arms R, G and N, the TabFM wrapper, the serving fetch with its parity test, the ONNX
export and the faithfulness metric are built and tested on toy graphs; no dataset has been
fetched (that needs a Kaggle token), so nothing is measured.

## Read first

- [README.md](README.md): what this is and, once there are any, the result tables.
- [PLAN.md](PLAN.md): the design. Do not deviate from it silently; if something in it turns
  out wrong, change the plan in the same commit as the code and say why in the commit
  message.
- `docs/data.md`, `docs/models.md`, `docs/patterns.md` and `docs/serving.md` once they exist:
  the licences as quoted, the model versions with dates, what each generator pattern is, and
  how the serving benchmark was run.

## Engineering standard

- Python 3.13. Typed throughout; `mypy --strict` and `ruff` clean in CI.
- Tests that fail meaningfully: the leakage test, the pattern labeller on toy graphs, the
  metric arithmetic against closed-form answers, the bootstrap grouping, the serving parity
  test, the context sampler, the ONNX export round-trip.
- `pyproject.toml` with pinned major versions and a comment saying why for each pin.
- Docs ship in the same commit as the change.
- Never commit credentials, a Kaggle token, a Neo4j password, or anything from `.env`.

## Rules specific to this repository

- **The split is temporal and inductive.** The random split is run once, to publish the gap,
  and never carries a headline number.
- **No feature reads a label.** The feature registry refuses a derivation that touches a label
  column; the neighbour-label-rate feature exists only in `docs/rejected.md`'s experiment.
- **The leakage test runs on every build**, and it was written before the first feature.
- **Nothing raw is committed.** Loaders fetch and verify checksums; licences are quoted in
  `docs/data.md` before the first fetch. Elliptic is never redistributed and never drawn on
  the demo; only synthetic accounts are drawn.
- **Unknown-label nodes pass messages and never enter a metric.**
- **One outcomes file shape for every arm.** The evaluator does not know which model scored.
- **Intervals are bootstrapped over groups** (time steps or components), never over accounts,
  and paired deltas refuse runs that scored different items.
- **Nothing timed is reported from the laptop.** Latency comes from the rented machine or the
  VPS, with the machine named.
- **Every reported number carries a confidence interval**, and every price carries its date.
- **No model vendor is called.** If a case narrative is ever generated, it goes through the
  portfolio gateway with the public data class, and the synthetic accounts are the only ones
  that qualify.
- **Plain punctuation** in everything written here: no em-dashes or other typographic dashes,
  straight quotes only.

## What goes in the README

The README opens with the one-liner, the results tables and the honest limitation, before any
installation instructions. The report command fills the tables; do not hand-edit them. Record
one approach that was tried and rejected, with the evidence, once the work has produced it.
