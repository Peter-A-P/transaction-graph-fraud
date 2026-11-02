# What was tried and rejected

## The neighbour-label-rate feature

**The idea.** Add to arm G the share of an item's neighbours that are already labelled illicit.
Guilt by association, in one column, and the most direct way to say "known by the company it
keeps". Many published graph fraud results lean on a feature like it, or on a split that lets
a model learn it implicitly.

**Why it was expected to fail.** On a random split, a test transaction's neighbours are mostly
training transactions, whose labels the feature may read; on a temporal, inductive split they
are in the same test step, unlabelled at scoring time, so the feature is empty exactly where it
would be needed (PLAN.md 10). The feature registry refuses any derivation that reads a label,
so this one is built outside it (`consort/eval/leak.py`), reads only the labels training may
see, never counts an item's own label, and exists only for this experiment.

**The evidence.** Arm G at one hop on Elliptic, with and without the feature, under both
splits; 95% intervals over time steps, the difference paired on identical transactions.
Written by `consort report`.

<!-- report:rejected -->
| Split | Arm G, AUPRC | Arm G with the neighbour-label rate, AUPRC | Difference, same items |
|---|---|---|---|
| Temporal, inductive | 0.714 (0.408 to 0.846) | 0.695 (0.381 to 0.837) | -0.019 (-0.033 to -0.008) |
| Random | 0.987 (0.976 to 0.994) | 0.990 (0.981 to 0.995) | 0.002 (-0.000 to 0.006) |
<!-- /report:rejected -->

**What it showed.** On the temporal, inductive split the feature costs arm G 0.019 of AUPRC
(0.008 to 0.033): a test transaction's neighbours sit in its own step, whose labels training
may not read, so the column is empty at test time while the booster learned to lean on it in
training, where it was full. On the random split it adds 0.002 (0.000 to 0.006), its
interval touching zero: arm G there already scores 0.987, because a random split hands the model
the test step's own transactions to learn from. The random split's 0.27 of AUPRC over the
temporal one is the leak that matters (the leakage table in the README); the feature only
names it. It is rejected: on the split this project reports, it makes the model worse, and
the uplift that motivates it in the literature comes from the split, not the column.
