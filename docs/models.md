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

Not built yet. GraphSAGE with neighbour sampling and GAT in PyTorch Geometric 2.x; the
directed-multigraph adaptations (reverse message passing, ego identifiers, port numbering) as
an ablation on the synthetic set.

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
- **Known limits:** no published GPU memory figure; the model card says memory grows with the
  number of context rows. Results are non-deterministic by the package's own declaration, so
  the column reports three seeds.
