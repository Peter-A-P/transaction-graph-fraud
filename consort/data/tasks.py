"""A dataset as a scoring task: items, their row features, and their graph features on demand.

Every arm sees a task through this one interface, so every arm scores the same items in the
same split. An item has an id, a split group (the time step or day that decides its
partition), a bootstrap group (what the evaluator resamples), a label (1, 0, or -1 for
unknown) and row features. Graph features are computed per partition on the graph that
partition may see:

- Elliptic: edges join transactions in the same time step only, so the full graph is already
  inductive; a test step's transactions are absent from every training step's neighbourhood.
  Item i is node i.
- IBM: a transfer's graph features are its sender's and its receiver's, computed on every
  transfer from the days before its own ("prior-days", the default): what a feature view
  refreshed each night would hold when the transfer arrives. Item i is transfer i. The other
  view, "window", computes them on every transfer up to the end of the partition's window,
  which hands a test transfer its accounts' later activity; it inflated arm G's AUPRC on
  HI-Small from the first run and is kept only to publish that gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from consort.data import elliptic as ell
from consort.data import ibm as ibm_mod
from consort.data.splits import ELLIPTIC_TEMPORAL, PARTITIONS, Partition, Split, temporal_split
from consort.graph.build import Graph, from_edges
from consort.graph.features import default_registry

NodeFeatureCache = dict[tuple[str, int, int], pd.DataFrame]


@dataclass
class Task:
    name: str
    items: pd.DataFrame  # item_id, split_group, boot_group, label, *row_columns
    row_columns: list[str]
    split: Split
    _cache: NodeFeatureCache = field(default_factory=dict, repr=False)

    def partition(self) -> npt.NDArray[np.str_]:
        return self.split.assign(self.items["split_group"].to_numpy())

    def labelled(self) -> npt.NDArray[np.bool_]:
        return (self.items["label"] >= 0).to_numpy()

    def graph_features(self, partition: Partition, hops: int, hub_cap: int) -> pd.DataFrame:
        raise NotImplementedError


@dataclass
class EllipticTask(Task):
    data: ell.Elliptic | None = None

    def graph_features(self, partition: Partition, hops: int, hub_cap: int) -> pd.DataFrame:
        assert self.data is not None
        key = ("all", hops, hub_cap)
        if key not in self._cache:
            reg = default_registry(ell.LOCAL)
            computed = reg.compute(self.data.graph, self.data.nodes, hops, hub_cap)
            # Elliptic's own one-hop aggregates are graph features too, from hop 1.
            agg = self.data.nodes[ell.AGG] if hops >= 1 else pd.DataFrame(index=computed.index)
            self._cache[key] = pd.concat([computed, agg.reset_index(drop=True)], axis=1).astype(
                np.float32
            )
        return self._cache[key]


def elliptic_task(data: ell.Elliptic, split: Split = ELLIPTIC_TEMPORAL) -> EllipticTask:
    n = data.nodes
    items = pd.DataFrame(
        {
            "item_id": n["tx_id"].astype(str),
            "split_group": n["step"],
            "boot_group": n["step"],
            "label": n["label"],
        }
    )
    items = pd.concat([items, n[data.row_columns].astype(np.float32)], axis=1)
    return EllipticTask(
        name="elliptic", items=items, row_columns=data.row_columns, split=split, data=data
    )


# Node columns aggregated over neighbours on the IBM graph: an account has no features of its
# own beyond what its transfers say, so these are its window's volume statistics.
IBM_NODE_COLUMNS = ["log_deg_in", "log_deg_out", "log_amt_in", "log_amt_out"]


def ibm_node_table(g: Graph) -> pd.DataFrame:
    n = g.n_nodes
    a = g.edge_attrs["amount"]
    return pd.DataFrame(
        {
            "log_deg_in": np.log1p(np.bincount(g.dst, minlength=n)),
            "log_deg_out": np.log1p(np.bincount(g.src, minlength=n)),
            "log_amt_in": np.log1p(np.bincount(g.dst, weights=a, minlength=n)),
            "log_amt_out": np.log1p(np.bincount(g.src, weights=a, minlength=n)),
        }
    )


FeatureView = Literal["prior-days", "window"]


@dataclass
class IBMTask(Task):
    data: ibm_mod.IBM | None = None
    window_end: dict[Partition, int] = field(default_factory=dict)
    view: FeatureView = "prior-days"
    _prior: dict[tuple[int, int], tuple[pd.DataFrame, set[int]]] = field(
        default_factory=dict, repr=False
    )

    def window_graph(self, partition: Partition) -> Graph:
        """Every transfer whose split group is at or before the partition's last group."""
        assert self.data is not None
        return self._graph_upto(self.items["split_group"].to_numpy() <= self.window_end[partition])

    def _graph_upto(self, keep: npt.NDArray[np.bool_]) -> Graph:
        assert self.data is not None
        g = self.data.graph
        return from_edges(
            g.n_nodes,
            g.src[keep],
            g.dst[keep],
            order=g.order[keep],
            **{k: v[keep] for k, v in g.edge_attrs.items()},
        )

    def _node_features(self, g: Graph, hops: int, hub_cap: int) -> pd.DataFrame:
        reg = default_registry(IBM_NODE_COLUMNS, ("amount", "t"))
        return reg.compute(g, ibm_node_table(g), hops, hub_cap).astype(np.float32)

    def _per_transfer(self, node: pd.DataFrame, rows: npt.NDArray[np.int64]) -> pd.DataFrame:
        assert self.data is not None
        t = self.data.transfers
        src = node.iloc[t["src"].to_numpy()[rows]].add_prefix("src_").reset_index(drop=True)
        dst = node.iloc[t["dst"].to_numpy()[rows]].add_prefix("dst_").reset_index(drop=True)
        return pd.concat([src, dst], axis=1)

    def graph_features(self, partition: Partition, hops: int, hub_cap: int) -> pd.DataFrame:
        if self.view == "window":
            return self._window_features(partition, hops, hub_cap)
        return self._prior_day_features(partition, hops, hub_cap)

    def _window_features(self, partition: Partition, hops: int, hub_cap: int) -> pd.DataFrame:
        key = (partition, hops, hub_cap)
        if key not in self._cache:
            node = self._node_features(self.window_graph(partition), hops, hub_cap)
            self._cache[key] = self._per_transfer(node, np.arange(len(self.items)))
        return self._cache[key]

    def _prior_day_features(self, partition: Partition, hops: int, hub_cap: int) -> pd.DataFrame:
        """Rows of every day up to the partition's last are filled; later rows stay NaN and
        belong to no partition that asks for this table."""
        day = self.items["split_group"].to_numpy()
        if (hops, hub_cap) not in self._prior:
            self._prior[(hops, hub_cap)] = (pd.DataFrame(), set())
        table, done = self._prior[(hops, hub_cap)]
        for d in sorted(set(np.unique(day[day <= self.window_end[partition]]).tolist()) - done):
            rows = np.flatnonzero(day == d)
            node = self._node_features(self._graph_upto(day < d), hops, hub_cap)
            block = self._per_transfer(node, rows)
            if table.empty:
                table = pd.DataFrame(
                    np.nan, index=range(len(self.items)), columns=block.columns, dtype=np.float32
                )
            table.iloc[rows] = block.to_numpy()
            done.add(d)
        self._prior[(hops, hub_cap)] = (table, done)
        return table


BootGroup = Literal["hour", "component", "day"]
TAIL_SHARE = 0.01


def ibm_task(
    data: ibm_mod.IBM,
    fractions: tuple[float, float] = (0.6, 0.8),
    boot_group: BootGroup = "hour",
    view: FeatureView = "prior-days",
) -> IBMTask:
    """Split by day so that the cumulative share of transfers is about 60, 20 and 20 percent;
    the days are recorded in the split, so the cut is reproducible and visible."""
    t = data.transfers
    day = ((t["t"] - t["t"].min()) // 86_400).astype(np.int64).to_numpy()
    counts = np.bincount(day)
    share = np.cumsum(counts) / counts.sum()
    days = np.arange(counts.shape[0])
    cut1 = int(np.searchsorted(share, fractions[0]))
    cut2 = max(int(np.searchsorted(share, fractions[1])), cut1 + 1)
    test_days = days[cut2 + 1 :]
    # The generator's tail: after normal activity stops, its laundering attempts run on for a
    # few days, so those days are nearly all laundering (HI-Small's days 10 to 17: 1,128
    # transfers, 655 laundering, 41% of the test set's). A day is in the tail when it holds
    # under 1% of the median day's transfers; the gap is wide, so the threshold is not delicate.
    tail = test_days[counts[test_days] < TAIL_SHARE * np.median(counts)]
    split = temporal_split(
        f"ibm-{data.tier}-temporal",
        train=days[: cut1 + 1].tolist(),
        val=days[cut1 + 1 : cut2 + 1].tolist(),
        test=test_days.tolist(),
        test_slices={"without-tail": sorted(set(test_days.tolist()) - set(tail.tolist()))},
    )
    empty = [p for p in PARTITIONS if not split.groups[p]]
    if empty:
        raise ValueError(f"the day cut leaves {empty} empty; {counts.shape[0]} days in the data")
    if boot_group == "hour":
        boot = ((t["t"] - t["t"].min()) // 3600).astype(np.int64).to_numpy()
    elif boot_group == "day":
        boot = day
    else:
        boot = data.graph.components[t["src"].to_numpy()]
    items = pd.DataFrame(
        {
            "item_id": np.arange(len(t)).astype(str),
            "split_group": day,
            "boot_group": boot,
            "label": t["is_laundering"].astype(np.int8),
        }
    )
    items = pd.concat([items, t[data.row_columns].astype(np.float32)], axis=1)
    return IBMTask(
        name=f"ibm-{data.tier}",
        items=items,
        row_columns=data.row_columns,
        split=split,
        data=data,
        window_end={p: max(split.groups[p]) for p in PARTITIONS},
        view=view,
    )
