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
<!-- /report:rejected -->
