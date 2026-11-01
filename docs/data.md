# Data

Every source this project reads, its terms as quoted before the first fetch, what is fetched,
how it is verified, and what is never published. Written 2026-10-31, before any loader ran.

Nothing raw is committed. `consort data fetch <source>` downloads into `data/raw/<source>/`,
verifies each file against the checksum recorded in `manifests/<source>.json` (written on the
first fetch and committed, so every later fetch is checked against it), and refuses to go on if
a file differs.

## Elliptic data set

- **Source:** Elliptic, 2019. Kaggle `ellipticco/elliptic-data-set`. Weber et al., "Anti-Money
  Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial
  Forensics", KDD Workshop on Anomaly Detection in Finance, 2019.
- **Licence:** CC BY-NC-ND 4.0 (the Kaggle dataset page's licence field, read 2026-10-31).
  Deed: https://creativecommons.org/licenses/by-nc-nd/4.0/. Legal code:
  https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en. Quoted from the legal code,
  Section 2(a)(1):

  > Subject to the terms and conditions of this Public License, the Licensor hereby grants You
  > a worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable license to exercise
  > the Licensed Rights in the Licensed Material to:
  > 1. reproduce and Share the Licensed Material, in whole or in part, for NonCommercial
  >    purposes only; and
  > 2. produce and reproduce, but not Share, Adapted Material for NonCommercial purposes only.

  Section 1, the two definitions that decide what this repository may do:

  > Adapted Material means material subject to Copyright and Similar Rights that is derived
  > from or based upon the Licensed Material and in which the Licensed Material is translated,
  > altered, arranged, transformed, or otherwise modified in a manner requiring permission
  > under the Copyright and Similar Rights held by the Licensor.

  > NonCommercial means not primarily intended for or directed towards commercial advantage or
  > monetary compensation.

- **What that means here.** The data is fetched by each reader with their own Kaggle token
  and is never redistributed. Feature tables, splits with row identifiers, outcome files and
  checkpoints built from it are Adapted Material and are produced and kept locally, never
  shared: they live under `data/` and `checkpoints/`, which are gitignored, and the site never
  draws an Elliptic transaction. What is published is code and aggregate numbers (metrics with
  intervals), which carry no part of the data. The checksum manifest holds file names, sizes
  and SHA-256 digests only.
- **Files:** `elliptic_txs_features.csv` (no header; transaction id, time step, 93 local
  features, 72 aggregated features), `elliptic_txs_classes.csv` (`txId,class`; `1` illicit,
  `2` licit, `unknown`), `elliptic_txs_edgelist.csv` (`txId1,txId2`).
- **Counts the loader checks:** 203,769 transactions, 234,355 edges, 49 time steps; 4,545
  illicit, 42,019 licit, 157,205 unknown.
- **Row features.** The 72 aggregated features are Elliptic's own one-hop neighbour
  aggregates. They are neighbourhood information, so arm R (row only) reads the time step and
  the 93 local features only; the aggregated 72 are a graph feature and enter arm G with the
  rest. Arm R on all 165 is reported once, labelled, because it is the number the original
  paper published.

## IBM Transactions for Anti Money Laundering

- **Source:** Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money
  Laundering Models", NeurIPS Datasets and Benchmarks, 2023 (arXiv 2306.16424). Kaggle
  `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`.
- **Licence:** Community Data License Agreement, Sharing, Version 1.0 (the Kaggle dataset
  page's licence field, read 2026-10-31). Text: https://cdla.dev/sharing-1-0/. Quoted:

  > 2.1 Subject to the conditions set forth in Section 3 of this Agreement, Data Provider(s)
  > hereby grant(s) to You a worldwide, non-exclusive, irrevocable (except as provided in
  > Section 5) right to: (a) Use Data; and (b) Publish Data.

  > 1.11 "Results" means the outcomes or outputs that You obtain from Your Computational Use of
  > Data. Results shall not include more than a *de minimis* portion of the Data on which the
  > Computational Use is based.

  > 3.5 This Agreement imposes no obligations or restrictions on Your Use or Publication of
  > Results.

  > (c) If You Publish Data You Receive, You must preserve all credit or attribution to the
  > Data Provider(s). Such retained credit or attribution includes any of the following to the
  > extent they exist in Data as You have Received it: legal notices or metadata;
  > identification of the Data Provider(s); or hyperlinks to Data to the extent it is
  > practical to do so.

  > 3.3 If You Publish Data You Receive, You must do so under an unmodified form of this
  > Agreement and include the text of this Agreement, the name of this Agreement and/or a
  > hyperlink or other method reasonably likely to provide a copy of the text of this
  > Agreement.

- **What that means here.** Metrics are Results and carry no obligation. The demo draws one
  planted laundering ring from this set: a handful of synthetic accounts and their transfers,
  which is Publishing Data, so the site carries the credit to Altman et al. and IBM, a link to
  the dataset, the name of the agreement and a link to its text, as 3.1(c) and 3.3 require.
- **Tiers:** HI-Small (515,088 accounts, 5,078,345 transactions) and LI-Small (705,907 and
  6,924,055) in the definition of done; HI-Medium (about 32 million) only if the GPU budget
  allows. Each tier is `<tier>_Trans.csv` and `<tier>_Patterns.txt`.
- **Transactions file:** one row per transfer, header `Timestamp, From Bank, Account, To Bank,
  Account, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format,
  Is Laundering`. An account is identified by its bank and account number together.
- **Patterns file:** blocks of transactions in the same column order, each opened by a line
  `BEGIN LAUNDERING ATTEMPT - <PATTERN>...` and closed by `END LAUNDERING ATTEMPT - <PATTERN>`.
  The eight patterns are fan-out, fan-in, cycle, random, bipartite, stack, scatter-gather and
  gather-scatter. Most laundering transactions in the transactions file belong to no block;
  they are reported as "no named pattern", not dropped. The literal header and block format
  were written from the dataset's published description and are confirmed against the files
  on the first fetch; `tests/test_ibm.py` holds the fixture the parser is tested on.
- **The pattern join** matches a pattern block's transfer to the transfers file on every
  column but the label (bank codes compared as numbers, since one file zero-pads them and the
  other does not; amounts compared in cents). Every pattern transfer must be found, or the
  loader refuses (`PatternJoinFailed`). Where the transfers file holds identical duplicate
  rows, all of them carry the block's pattern, since nothing distinguishes them; a transfer
  named by two blocks keeps the first. The loader reports both counts on the first fetch.
- **Items and nodes.** Transfers are the items scored, as the generator labels them; accounts
  (bank and account number together) are the nodes. A transfer's row features are its own
  (amount in dollars, format, currencies, same bank, self transfer, hour, weekday); its graph
  features are its two accounts'.
- **Dollar amounts.** Amounts are converted to US dollars at rates the data implies: the
  median ratio of amount received to amount paid on cross-currency transfers landing in
  dollars (or leaving them, for a currency with none landing). The rates are written to the
  build manifest.

## Elliptic++ actors (stretch)

- **Source:** Elmougy and Liu, KDD 2023; `git-disl/EllipticPlusPlus`.
- **Licence:** none stated; no licence file and no terms (read 2026-10-31), a citation request
  only. **Not fetched** until the authors answer an email confirming research use and the
  publication of results. Until then its row in every table reads "not run, no licence".

## DGraph-Fin (stub)

- **Source:** Huang et al., NeurIPS 2022. Host `dgraph.xinye.com` did not resolve on
  2026-10-31 and its terms are unread. A loader stub raises with that message; nothing in the
  definition of done depends on it.

## Unknown labels

Elliptic's 157,205 unknown-label transactions are nodes in the graph: they pass messages to arm
N and contribute to arm G's neighbourhood features as structure and as their own features.
They never enter a training loss and never enter a metric. The evaluator refuses an outcome
row whose label is unknown (`tests/test_outcomes.py`).

## Fetching

A reader needs a Kaggle API token (`kaggle.json` in `~/.kaggle/`, or `KAGGLE_USERNAME` and
`KAGGLE_KEY`), and must accept each dataset's terms on its Kaggle page once. Then:

```
uv run consort data fetch elliptic
uv run consort data fetch ibm --tier HI-Small
```
