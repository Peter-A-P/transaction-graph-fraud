"""The leakage gap, and the neighbour-label-rate feature that exists only to be rejected.

The feature registry refuses any feature that reads a label. This module builds one anyway,
outside the registry, for one experiment (docs/rejected.md): the share of an item's labelled
neighbours that are illicit, reading only the labels training is allowed to see. On a random
split a test item's neighbours are often training items, so the feature looks powerful. On the
temporal, inductive split a test item's neighbours are in test steps too, their labels are not
visible, and the feature is empty exactly where it is needed. The gap between the two is the
point; nothing here is ever used for a headline number.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from consort.data.tasks import EllipticTask, IBMTask, Task
from consort.graph.build import Graph

FloatArray = npt.NDArray[np.float64]


def _node_rate(
    g: Graph, node_pos: FloatArray, node_seen: FloatArray
) -> tuple[FloatArray, FloatArray]:
    N = g.neighbours()
    seen = np.asarray(N @ node_seen)
    pos = np.asarray(N @ node_pos)
    rate = np.divide(pos, seen, out=np.full(g.n_nodes, np.nan), where=seen > 0)
    return rate, seen


def neighbour_label_rate(task: Task, visible: npt.NDArray[np.bool_]) -> pd.DataFrame:
    labels = task.items["label"].to_numpy()
    if isinstance(task, EllipticTask):
        assert task.data is not None
        g = task.data.graph
        rate, seen = _node_rate(g, (visible & (labels == 1)).astype(float), visible.astype(float))
        return pd.DataFrame({"leak_nbr_rate": rate, "leak_nbr_seen": seen})
    if isinstance(task, IBMTask):
        assert task.data is not None
        g = task.data.graph
        n = g.n_nodes
        # Per account: visible laundering transfers touching it over visible transfers.
        pos = np.bincount(g.src, weights=(visible & (labels == 1)), minlength=n) + np.bincount(
            g.dst, weights=(visible & (labels == 1)), minlength=n
        )
        seen = np.bincount(g.src, weights=visible, minlength=n) + np.bincount(
            g.dst, weights=visible, minlength=n
        )
        # The item's own label is never counted in its own feature.
        own = visible.astype(float)
        own_pos = (visible & (labels == 1)).astype(float)
        out = {}
        for end, idx in (("src", g.src), ("dst", g.dst)):
            s = seen[idx] - own
            p = pos[idx] - own_pos
            out[f"leak_{end}_rate"] = np.divide(
                p, s, out=np.full(idx.shape[0], np.nan), where=s > 0
            )
        return pd.DataFrame(out)
    raise TypeError(f"no neighbour-label-rate for {type(task).__name__}")
