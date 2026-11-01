"""Elliptic: 203,769 Bitcoin transactions over 49 time steps, a fifth of them labelled.

Items are transactions (nodes). Labels: 1 illicit, 0 licit, -1 unknown; unknown nodes stay
in the graph and never reach a loss or a metric. The 72 aggregated features are Elliptic's own
one-hop neighbour aggregates, so they are graph information and arm R does not read them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from consort.graph.build import Graph, from_edges

N_LOCAL = 93
N_AGG = 72
LOCAL = [f"local_{i:03d}" for i in range(1, N_LOCAL + 1)]
AGG = [f"agg_{i:03d}" for i in range(1, N_AGG + 1)]

EXPECTED = {
    "nodes": 203_769,
    "edges": 234_355,
    "illicit": 4_545,
    "licit": 42_019,
    "unknown": 157_205,
}


class CountMismatch(Exception):
    """The files do not hold the counts the dataset's authors published."""


@dataclass(frozen=True)
class Elliptic:
    nodes: pd.DataFrame  # tx_id, step, label, LOCAL..., AGG...; row i is graph node i
    graph: Graph

    @property
    def row_columns(self) -> list[str]:
        """Arm R's columns: the time step and the 93 local features."""
        return ["step", *LOCAL]


def load(root: Path, check_counts: bool = True) -> Elliptic:
    base = root / "elliptic_bitcoin_dataset"
    feats = pd.read_csv(base / "elliptic_txs_features.csv", header=None)
    if feats.shape[1] != 2 + N_LOCAL + N_AGG:
        raise CountMismatch(
            f"features file has {feats.shape[1]} columns, expected {2 + N_LOCAL + N_AGG}"
        )
    feats.columns = ["tx_id", "step", *LOCAL, *AGG]
    classes = pd.read_csv(base / "elliptic_txs_classes.csv", dtype={"txId": np.int64, "class": str})
    label_map = {"1": 1, "2": 0, "unknown": -1}
    if not set(classes["class"]).issubset(label_map):
        raise CountMismatch(
            f"unexpected class values {sorted(set(classes['class']) - set(label_map))}"
        )
    labels = classes.set_index("txId")["class"].map(label_map)
    nodes = feats.assign(label=feats["tx_id"].map(labels))
    if nodes["label"].isna().any():
        raise CountMismatch(f"{int(nodes['label'].isna().sum())} transactions have no class row")
    nodes["label"] = nodes["label"].astype(np.int8)
    nodes["step"] = nodes["step"].astype(np.int64)
    nodes = nodes[["tx_id", "step", "label", *LOCAL, *AGG]].reset_index(drop=True)

    edges = pd.read_csv(base / "elliptic_txs_edgelist.csv")
    index = pd.Series(np.arange(len(nodes)), index=nodes["tx_id"].to_numpy())
    src = index.reindex(edges["txId1"].to_numpy()).to_numpy()
    dst = index.reindex(edges["txId2"].to_numpy()).to_numpy()
    if np.isnan(src.astype(float)).any() or np.isnan(dst.astype(float)).any():
        raise CountMismatch("an edge names a transaction absent from the features file")
    graph = from_edges(len(nodes), src.astype(np.int64), dst.astype(np.int64))

    if check_counts:
        got = {
            "nodes": len(nodes),
            "edges": graph.n_edges,
            "illicit": int((nodes["label"] == 1).sum()),
            "licit": int((nodes["label"] == 0).sum()),
            "unknown": int((nodes["label"] == -1).sum()),
        }
        if got != EXPECTED:
            raise CountMismatch(f"counts {got} differ from the published {EXPECTED}")
    return Elliptic(nodes=nodes, graph=graph)
