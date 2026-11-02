"""Elliptic++ actors: wallet addresses, with features rebuilt step by step from transactions.

The published wallet features are lifetime aggregates over all 49 steps, repeated on every
step's row, and the address-to-address edges carry no step (docs/data.md), so neither is used.
From the transaction-level files instead:

- each address's incidence: the transactions it sent from (an input) or received in (an
  output), each with that transaction's step;
- each address's first step, the step it is scored at and split by;
- the graph: for every transaction, an edge from each input address to each output address,
  dated by the transaction's step;
- an address's features as of step s: what its transactions in steps up to s say about it.
  Per-address amounts are not in the data, so these are counts and the sizes, fees, amounts
  and counterparty counts of the transactions it took part in.

Addresses are the nodes, in the sorted order of the classes file. Labels: 1 illicit, 0 licit,
-1 unknown (unknown addresses pass messages and never enter a metric).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from consort.graph.build import Graph, from_edges

COUNTS = {"addresses": 822_942, "illicit": 14_266, "licit": 251_088, "steps": 49}

# Per-transaction columns summarised over an address's transactions, on a log scale.
TX_COLUMNS = ["total_BTC", "fees", "size", "num_input_addresses", "num_output_addresses"]
ROLES = ("sent", "recv")
NODE_COLUMNS = [f"log_n_{r}" for r in ROLES] + [
    f"{r}_mean_log_{c}" for r in ROLES for c in TX_COLUMNS
]


class CountMismatch(Exception):
    """The fetched files do not hold the counts the paper publishes."""


@dataclass(frozen=True)
class EllipticPP:
    addresses: npt.NDArray[np.str_]  # node i is addresses[i]
    label: npt.NDArray[np.int8]
    first_step: npt.NDArray[np.int64]
    inc_addr: npt.NDArray[np.int64]  # one row per (address, transaction, role)
    inc_role: npt.NDArray[np.int8]  # 0 sent from (input), 1 received in (output)
    inc_step: npt.NDArray[np.int64]
    inc_tx: npt.NDArray[np.float64]  # log1p of TX_COLUMNS for the row's transaction
    edge_src: npt.NDArray[np.int64]
    edge_dst: npt.NDArray[np.int64]
    edge_step: npt.NDArray[np.int64]
    edge_amount: npt.NDArray[np.float64]  # the transaction's total BTC

    @property
    def n(self) -> int:
        return int(self.addresses.shape[0])

    def node_table(self, keep: npt.NDArray[np.bool_]) -> pd.DataFrame:
        """NODE_COLUMNS for every address from the incidence rows in `keep`."""
        cols: dict[str, npt.NDArray[np.float64]] = {}
        sums: dict[str, npt.NDArray[np.float64]] = {}
        for code, role in enumerate(ROLES):
            m = keep & (self.inc_role == code)
            count = np.bincount(self.inc_addr[m], minlength=self.n).astype(np.float64)
            cols[f"log_n_{role}"] = np.log1p(count)
            for j, c in enumerate(TX_COLUMNS):
                total = np.bincount(self.inc_addr[m], weights=self.inc_tx[m, j], minlength=self.n)
                sums[f"{role}_mean_log_{c}"] = total / np.maximum(count, 1.0)
        return pd.DataFrame({**cols, **sums})[NODE_COLUMNS]

    def row_table(self) -> pd.DataFrame:
        """Each address's features from its first step's transactions: what is known about it
        when it is scored."""
        return self.node_table(self.inc_step == self.first_step[self.inc_addr])

    def graph_upto(self, step: int) -> Graph:
        keep = self.edge_step <= step
        # Recency for the hub cap: step, then file position.
        order = np.lexsort((np.arange(int(keep.sum())), self.edge_step[keep]))
        rank = np.empty(order.shape[0], dtype=np.int64)
        rank[order] = np.arange(order.shape[0])
        return from_edges(
            self.n,
            self.edge_src[keep],
            self.edge_dst[keep],
            order=rank,
            amount=self.edge_amount[keep],
        )


def _index(addresses: npt.NDArray[np.str_], values: pd.Series, what: str) -> npt.NDArray[np.int64]:
    v = values.to_numpy().astype(str)
    pos = np.searchsorted(addresses, v)
    pos = np.minimum(pos, addresses.shape[0] - 1)
    missing = addresses[pos] != v
    if missing.any():
        raise CountMismatch(f"{int(missing.sum())} addresses in {what} are not in the classes file")
    out: npt.NDArray[np.int64] = pos.astype(np.int64)
    return out


def load(actors: Path, txs: Path, check_counts: bool = True) -> EllipticPP:
    classes = pd.read_csv(actors / "wallets_classes.csv")
    classes = classes.sort_values("address", kind="stable").reset_index(drop=True)
    addresses = classes["address"].to_numpy().astype(str)
    cls = classes["class"].to_numpy()
    label = np.where(cls == 1, 1, np.where(cls == 2, 0, -1)).astype(np.int8)

    tx = pd.read_csv(txs / "txs_features.csv", usecols=["txId", "Time step", *TX_COLUMNS])
    tx = tx.set_index("txId")
    tx_log = np.log1p(np.nan_to_num(tx[TX_COLUMNS].to_numpy(np.float64), nan=0.0).clip(min=0))

    def incidence(path: Path, addr_col: str, role: int) -> pd.DataFrame:
        e = pd.read_csv(path)
        rows = tx.index.get_indexer(pd.Index(e["txId"]))
        if (rows < 0).any():
            raise CountMismatch(
                f"{int((rows < 0).sum())} transactions in {path.name} not in txs_features"
            )
        return pd.DataFrame(
            {"addr": _index(addresses, e[addr_col], path.name), "tx": rows, "role": role}
        )

    sent = incidence(txs / "AddrTx_edgelist.csv", "input_address", 0)
    recv = incidence(txs / "TxAddr_edgelist.csv", "output_address", 1)
    inc = pd.concat([sent, recv], ignore_index=True)
    step = tx["Time step"].to_numpy(np.int64)
    inc_step = step[inc["tx"].to_numpy()]

    first = np.full(addresses.shape[0], np.iinfo(np.int64).max)
    np.minimum.at(first, inc["addr"].to_numpy(), inc_step)
    if (first == np.iinfo(np.int64).max).any():
        raise CountMismatch(
            f"{int((first == np.iinfo(np.int64).max).sum())} addresses in no transaction"
        )

    pairs = sent.merge(recv, on="tx", suffixes=("_in", "_out"))
    total_btc = tx["total_BTC"].to_numpy(np.float64)

    data = EllipticPP(
        addresses=addresses,
        label=label,
        first_step=first,
        inc_addr=inc["addr"].to_numpy(np.int64),
        inc_role=inc["role"].to_numpy(np.int8),
        inc_step=inc_step,
        inc_tx=tx_log[inc["tx"].to_numpy()],
        edge_src=pairs["addr_in"].to_numpy(np.int64),
        edge_dst=pairs["addr_out"].to_numpy(np.int64),
        edge_step=step[pairs["tx"].to_numpy()],
        edge_amount=np.nan_to_num(total_btc[pairs["tx"].to_numpy()], nan=0.0),
    )
    if check_counts:
        got = {
            "addresses": data.n,
            "illicit": int((label == 1).sum()),
            "licit": int((label == 0).sum()),
            "steps": int(np.unique(step).shape[0]),
        }
        if got != COUNTS:
            raise CountMismatch(f"Elliptic++ counts {got}, published {COUNTS}")
    return data
