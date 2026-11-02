# Models

Every model this project runs, its version, where it came from and on what date, and the
settings it ran with. A number in the README names the run; the run names the model here.

## Arms R and G: XGBoost

- **Version:** xgboost 3.4.1 (resolved in `uv.lock`, 2026-10-31).
- **Settings, identical for both arms** (`consort/models/boost.py`): binary logistic,
  histogram trees, depth 6, learning rate 0.05, row and column subsample 0.8, minimum child
  weight 1, up to 2,000 rounds with early stopping after 100 rounds without a gain in
  validation AUPRC. No per-arm tuning: the difference between R and G is the features.
- **Seeds:** 0, 1, 2. The random split's assignment uses one fixed seed (20261031).

## Arm N: graph neural network

GraphSAGE (`SAGEConv`) and GAT (`GATConv`, 4 heads) in PyTorch Geometric 2.x, in
`consort/models/gnn.py`.

- **Neighbourhood:** the recency cap of `consort/graph/sampler.py`, the same neighbourhood arm
  G aggregates over and the serving path fetches; not PyG's random samplers (PLAN.md 2.5).
- **Shape:** a linear input layer, one message-passing layer per hop with a residual and layer
  norm, hidden width 64, dropout 0.2. At hop 0 it is a multilayer perceptron on the row. On
  Elliptic and Elliptic++ each node is classified; on IBM a transfer is classified from its two
  accounts' embeddings and its own row.
- **Training:** AdamW, learning rate 3e-3, weight decay 1e-5, batches of 2,048, at most 40
  epochs, early stopping on validation AUPRC with a patience of 6 (the rule arms R and G use),
  the positive class weighted by the training ratio. Elliptic++ and IBM batches never mix steps
  or days, each scored on its own graph as of then. On IBM each epoch keeps every illicit
  transfer and a fresh fifth of the licit ones, the weight scaled to match (PLAN.md 2.5).
- **Ablation flags (IBM):** reverse message passing (separate weights along and against an
  edge) and ego identifiers (a marker on the nodes being classified), after Egressy et al.,
  Multi-GNN (2024). Port numbering is not implemented.
- **Variant choice:** like the hop count, the architecture in a headline row is the one with
  the best validation AUPRC; the hop curve is GraphSAGE's.

## The TabFM column

- **What:** Google's TabFM, a zero-shot in-context tabular foundation model. Repository
  https://github.com/google-research/tabfm; announced 2026-06-30
  (https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/).
  No technical report is published (the repository's FAQ says so).
- **Version:** the `tabfm` package 1.0.1 (PyPI, uploaded 2026-07-21), with the weights
  `google/tabfm-1.0.0-pytorch` on Hugging Face, the set its loader names. A
  `google/tabfm-1.1.0-pytorch` weight set (2026-07-31) exists with no matching code release on
  PyPI as of 2026-10-31, so it is not used.
- **Licences:** code Apache 2.0. Weights under the **TabFM Non-Commercial License v1.0**
  (https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE): use for testing,
  evaluation or research only; the model and its derivatives may not be redistributed;
  outputs are not derivatives. So the weights are fetched at run time and never committed or
  bundled, no TabFM model is exported to ONNX or handed to 09, and the cost column says "not
  deployable by a bank" beside every TabFM number.
- **Settings** (`consort/models/tabfm.py`): `TabFMClassifier` with 32 ensemble members,
  `max_num_rows` set explicitly to the context size (the README and the code disagree on the
  default), the seed as `random_state`, default bfloat16. Tables wider than 500 features are
  refused rather than silently subsampled by the model.
- **Context:** a sample of the training rows of a stated size that keeps every positive and
  spreads the negatives across split groups in proportion; deterministic by seed. The size is
  a parameter in every table it appears in.
- **Queries** are scored in batches of 1,000: 10,000 at once ran a 24 GB RTX 4090 out of
  memory on arm G's table for Elliptic (10,000 context rows), 2026-11-02.
- **Known limits:** no published GPU memory figure; the model card says memory grows with the
  number of context rows. Results are non-deterministic by the package's own declaration, so
  the column reports three seeds.
