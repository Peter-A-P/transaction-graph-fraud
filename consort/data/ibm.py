"""IBM's synthetic AML transfers: accounts as nodes, transfers as edges, and as the items.

Every transfer is an item to score (laundering or not), which is how the generator labels
them and how its pattern file names them. A node is an account, identified by bank and
account number together. Amounts are converted to US dollars at rates the data itself
implies (the median ratio of received to paid on cross-currency transfers that land in
dollars), so that an amount feature means the same thing across currencies.

The pattern file is parsed into blocks, one per laundering attempt, and each block's
transfers are joined to the transfers file on every column. A pattern transfer that is not
found stops the load; a laundering transfer in no block is "no named pattern".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from consort.graph.build import Graph, from_edges

COLUMNS = [
    "timestamp",
    "from_bank",
    "from_acct",
    "to_bank",
    "to_acct",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]
PATTERNS = (
    "FAN-OUT",
    "FAN-IN",
    "CYCLE",
    "RANDOM",
    "BIPARTITE",
    "STACK",
    "SCATTER-GATHER",
    "GATHER-SCATTER",
)
NO_PATTERN = "NONE"
BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z-]+)")
END = re.compile(r"^END LAUNDERING ATTEMPT")
DOLLAR = "US Dollar"
JOIN_KEYS = [
    "timestamp",
    "from_bank",
    "from_acct",
    "to_bank",
    "to_acct",
    "amount_received_c",
    "receiving_currency",
    "amount_paid_c",
    "payment_currency",
    "payment_format",
]


class PatternJoinFailed(Exception):
    """A transfer in the pattern file has no match in the transfers file."""


def read_transfers(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    if len(df.columns) != len(COLUMNS):
        raise ValueError(f"{path}: {len(df.columns)} columns, expected {len(COLUMNS)}")
    df.columns = COLUMNS
    return _typed(df)


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], format="%Y/%m/%d %H:%M")
    # Banks are numeric codes that one file may zero-pad and another not.
    out["from_bank"] = out["from_bank"].astype(np.int64)
    out["to_bank"] = out["to_bank"].astype(np.int64)
    out["amount_received"] = out["amount_received"].astype(np.float64)
    out["amount_paid"] = out["amount_paid"].astype(np.float64)
    out["is_laundering"] = out["is_laundering"].astype(np.int8)
    out["amount_received_c"] = (out["amount_received"] * 100).round().astype(np.int64)
    out["amount_paid_c"] = (out["amount_paid"] * 100).round().astype(np.int64)
    return out


@dataclass(frozen=True)
class Block:
    pattern: str
    attempt: int
    rows: list[list[str]]


def read_patterns(path: Path) -> list[Block]:
    blocks: list[Block] = []
    current: Block | None = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := BEGIN.match(line):
            if current is not None:
                raise ValueError(f"{path}: BEGIN inside an open block ({current.pattern})")
            name = m.group(1).rstrip("-:")
            if name not in PATTERNS:
                raise ValueError(f"{path}: unknown pattern {name!r}")
            current = Block(name, len(blocks), [])
        elif END.match(line):
            if current is None:
                raise ValueError(f"{path}: END with no open block")
            blocks.append(current)
            current = None
        else:
            if current is None:
                raise ValueError(f"{path}: transfer line outside a block: {line[:60]}")
            fields = line.split(",")
            if len(fields) != len(COLUMNS):
                raise ValueError(f"{path}: {len(fields)} fields in a pattern line")
            current.rows.append(fields)
    if current is not None:
        raise ValueError(f"{path}: block {current.pattern} never closed")
    return blocks


def join_patterns(transfers: pd.DataFrame, blocks: list[Block]) -> pd.DataFrame:
    """Pattern name and attempt number per transfer: NONE and -1 where no block names it.
    Raises if any pattern transfer is unmatched. Identical duplicate transfers both carry the
    block's label, since nothing distinguishes them."""
    rows = [(b.pattern, b.attempt, *r) for b in blocks for r in b.rows]
    pat = pd.DataFrame(rows, columns=["pattern", "attempt", *COLUMNS])
    pat = _typed(pat)
    keyed = transfers[JOIN_KEYS].reset_index(names="row")
    merged = pat.merge(keyed, on=JOIN_KEYS, how="left")
    missing = int(merged["row"].isna().sum())
    if missing:
        raise PatternJoinFailed(
            f"{missing} of {len(pat)} pattern transfers not found in the transfers file"
        )
    # A transfer in two blocks keeps the first; the count is reported by the caller.
    merged = merged.sort_values(["row", "attempt"]).drop_duplicates("row")
    out = pd.DataFrame({"pattern": NO_PATTERN, "attempt": -1}, index=transfers.index)
    out.loc[merged["row"].astype(np.int64).to_numpy(), "pattern"] = merged["pattern"].to_numpy()
    out.loc[merged["row"].astype(np.int64).to_numpy(), "attempt"] = merged["attempt"].to_numpy()
    return out


def dollar_rates(t: pd.DataFrame) -> dict[str, float]:
    """US dollars per unit of each currency, implied by the data."""
    rates = {DOLLAR: 1.0}
    to_usd = t[(t["receiving_currency"] == DOLLAR) & (t["payment_currency"] != DOLLAR)]
    ratio = to_usd["amount_received"] / to_usd["amount_paid"]
    for cur, r in ratio.groupby(to_usd["payment_currency"]).median().items():
        rates[str(cur)] = float(r)
    from_usd = t[(t["payment_currency"] == DOLLAR) & (t["receiving_currency"] != DOLLAR)]
    inv = from_usd["amount_paid"] / from_usd["amount_received"]
    for cur, r in inv.groupby(from_usd["receiving_currency"]).median().items():
        rates.setdefault(str(cur), float(r))
    return rates


@dataclass(frozen=True)
class IBM:
    tier: str
    transfers: pd.DataFrame  # one row per transfer, in file order; row i is graph edge i
    accounts: pd.DataFrame  # bank, acct; row i is graph node i
    graph: Graph
    rates: dict[str, float]

    @property
    def row_columns(self) -> list[str]:
        return ROW_COLUMNS


ROW_COLUMNS = [
    "log_amount_usd",
    "payment_format_code",
    "payment_currency_code",
    "receiving_currency_code",
    "cross_currency",
    "same_bank",
    "self_transfer",
    "hour",
    "weekday",
]


def build(tier: str, transfers: pd.DataFrame, patterns: pd.DataFrame | None) -> IBM:
    t = transfers.reset_index(drop=True)
    rates = dollar_rates(t)
    unknown = sorted(set(t["payment_currency"]) - set(rates))
    if unknown:
        raise ValueError(f"no implied dollar rate for {unknown}")
    t["amount_usd"] = t["amount_paid"] * t["payment_currency"].map(rates)

    keys_from = t["from_bank"].astype(str) + ":" + t["from_acct"]
    keys_to = t["to_bank"].astype(str) + ":" + t["to_acct"]
    uniq = np.unique(np.concatenate([keys_from.to_numpy(), keys_to.to_numpy()]))
    t["src"] = np.searchsorted(uniq, keys_from.to_numpy())
    t["dst"] = np.searchsorted(uniq, keys_to.to_numpy())
    bank, _, acct = np.char.partition(uniq.astype(str), ":").T
    accounts = pd.DataFrame({"bank": bank.astype(np.int64), "acct": acct})

    secs = (t["timestamp"] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    # Edge recency: timestamp, then file position; file position alone breaks ties.
    order = np.lexsort((np.arange(len(t)), secs))
    rank = np.empty(len(t), dtype=np.int64)
    rank[order] = np.arange(len(t))
    graph = from_edges(
        len(accounts), t["src"], t["dst"], order=rank, amount=t["amount_usd"], t=secs
    )

    t["log_amount_usd"] = np.log1p(t["amount_usd"])
    for col, code in (
        ("payment_format", "payment_format_code"),
        ("payment_currency", "payment_currency_code"),
        ("receiving_currency", "receiving_currency_code"),
    ):
        t[code] = pd.Categorical(t[col]).codes.astype(np.float64)
    t["cross_currency"] = (t["payment_currency"] != t["receiving_currency"]).astype(np.float64)
    t["same_bank"] = (t["from_bank"] == t["to_bank"]).astype(np.float64)
    t["self_transfer"] = (t["src"] == t["dst"]).astype(np.float64)
    t["hour"] = t["timestamp"].dt.hour.astype(np.float64)
    t["weekday"] = t["timestamp"].dt.weekday.astype(np.float64)
    t["t"] = secs
    if patterns is not None:
        t["pattern"] = patterns["pattern"].to_numpy()
        t["attempt"] = patterns["attempt"].to_numpy()
    return IBM(tier=tier, transfers=t, accounts=accounts, graph=graph, rates=rates)


def load(root: Path, tier: str) -> IBM:
    transfers = read_transfers(root / f"{tier}_Trans.csv")
    patterns = join_patterns(transfers, read_patterns(root / f"{tier}_Patterns.txt"))
    return build(tier, transfers, patterns)
