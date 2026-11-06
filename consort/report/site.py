"""The demo site's data: one planted laundering ring from IBM's synthetic set, scored by every
arm, with arm G's case files; and the aggregate results the figures draw.

Only synthetic accounts are drawn (PLAN.md 2.9). Elliptic and Elliptic++ appear only as
aggregates from results/summary.json, which the README already publishes. Accounts are named
by their place in the ring (A1, A2, ...), never by the generator's bank and account numbers.

The ring is chosen by a rule that reads no score: the attempt of the named pattern with the
most transfers on the test days without the tail, ties to the earlier attempt. So the page
cannot have been picked to flatter an arm.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from consort.data.tasks import IBM_NODE_COLUMNS, IBMTask, ibm_node_table
from consort.eval import outcomes as oc
from consort.explain.faithfulness import explain_transfer
from consort.explain.run import _pair_score_fn
from consort.graph.features import default_registry
from consort.graph.neighbourhood import MemorySource
from consort.serve.bench import onnx_scorer

TAIL = "without-tail"


def pick_ring(task: IBMTask, pattern: str = "SCATTER-GATHER") -> int:
    """The attempt of `pattern` with the most transfers on the test days without the tail."""
    assert task.data is not None
    t = task.data.transfers
    day = task.items["split_group"].to_numpy()
    shown = np.isin(day, task.split.test_slices[TAIL])
    of = (t["pattern"] == pattern).to_numpy()
    if not (of & shown).any():
        raise ValueError(f"no {pattern} transfer on the test days")
    counts = pd.Series(shown[of].astype(int)).groupby(t["attempt"].to_numpy()[of]).sum()
    best = counts[counts == counts.max()]
    return int(best.index.min())


def ranks(o: oc.Outcomes) -> dict[str, float]:
    """Each test item's place in the run's ranking, as the share of all test items scored at
    or above it (1 / n for the top item, 1.0 for the last), ties by item id as `queue` breaks
    them. The scores are not calibrated, and differ in range from arm to arm (the graph neural
    network, trained with the rare class weighted up, puts most items above 0.99); the rank is
    what decides the alert queue, and it reads the same for every arm."""
    order = np.lexsort((o.item_id, -o.score))
    n = len(o)
    return {str(i): (k + 1) / n for k, i in enumerate(o.item_id[order])}


def queue(o: oc.Outcomes, budget: float) -> set[str]:
    """The run's alert queue: its top `budget` share of all test items, ties by item id (the
    gate's queue, consort.gate.export)."""
    k = max(1, round(budget * len(o)))
    order = np.lexsort((o.item_id, -o.score))
    return {str(i) for i in o.item_id[order[:k]]}


def ring(
    task: IBMTask,
    attempt: int,
    runs: Mapping[str, Path],
    model: Path,
    hops: int,
    cap: int,
    budget: float = 0.01,
    k: int = 5,
) -> dict[str, Any]:
    """The ring's accounts and transfers, each arm's score and verdict on every scored
    transfer, and a case file per transfer arm G flagged: the score, and the direct neighbours
    whose occlusion drops it most, through the ONNX export that would serve it."""
    assert task.data is not None
    t = task.data.transfers
    rows = np.flatnonzero((t["attempt"] == attempt).to_numpy())
    rows = rows[np.argsort(t["t"].to_numpy()[rows], kind="stable")]
    src, dst = t["src"].to_numpy(np.int64), t["dst"].to_numpy(np.int64)
    members: dict[int, str] = {}
    for i in rows:
        for v in (int(src[i]), int(dst[i])):
            members.setdefault(v, f"A{len(members) + 1}")
    item = task.items["item_id"].astype(str).to_numpy()
    day = task.items["split_group"].to_numpy()
    shown = set(task.split.test_slices[TAIL])
    secs = t["t"].to_numpy(np.float64)
    usd = t["amount_usd"].to_numpy(np.float64)
    laundering = t["is_laundering"].to_numpy() == 1
    t0 = float(secs[rows].min())
    deg = np.bincount(src, minlength=task.data.graph.n_nodes) + np.bincount(
        dst, minlength=task.data.graph.n_nodes
    )

    scores: dict[str, dict[str, float]] = {}
    flagged: dict[str, set[str]] = {}
    top: dict[str, dict[str, float]] = {}
    for arm, path in runs.items():
        o = oc.read(path)
        pick = np.isin(o.item_id, item[rows])
        ids = [str(i) for i in o.item_id[pick]]
        scores[arm] = dict(zip(ids, o.score[pick].tolist(), strict=True))
        flagged[arm] = queue(o, budget) & set(scores[arm])
        everyone = ranks(o)
        top[arm] = {i: everyone[i] for i in ids}

    transfers = []
    for i in rows:
        iid = str(item[i])
        scored = iid in scores.get("R", {})
        transfers.append(
            {
                "from": members[int(src[i])],
                "to": members[int(dst[i])],
                "usd": round(float(usd[i]), 2),
                "hours": round((float(secs[i]) - t0) / 3600.0, 1),
                "day": int(day[i]),
                "scored": scored and int(day[i]) in shown,
                "laundering": bool(laundering[i]),
                "arms": {
                    arm: {
                        "score": round(s[iid], 4),
                        "top": round(top[arm][iid], 6),
                        "flagged": iid in flagged[arm],
                    }
                    for arm, s in scores.items()
                    if iid in s
                },
            }
        )

    cases = _cases(task, rows, item, members, flagged.get("G", set()), model, hops, cap, k)
    accounts = [{"name": name, "transfers_in_data": int(deg[v])} for v, name in members.items()]
    return {"attempt": attempt, "accounts": accounts, "transfers": transfers, "cases": cases}


def _cases(
    task: IBMTask,
    rows: npt.NDArray[np.int64],
    item: npt.NDArray[Any],
    members: dict[int, str],
    flagged: set[str],
    model: Path,
    hops: int,
    cap: int,
    k: int,
) -> list[dict[str, Any]]:
    assert task.data is not None
    t = task.data.transfers
    src, dst = t["src"].to_numpy(np.int64), t["dst"].to_numpy(np.int64)
    secs = t["t"].to_numpy(np.float64)
    day = task.items["split_group"].to_numpy().astype(np.int64)
    label = task.items["label"].to_numpy() == 1
    laundering = np.zeros(task.data.graph.n_nodes, dtype=bool)
    laundering[src[label]] = True
    laundering[dst[label]] = True
    x = task.items[task.row_columns].reset_index(drop=True)
    reg = default_registry(IBM_NODE_COLUMNS, ("amount", "t"))
    score = onnx_scorer(model)
    outside: dict[int, str] = {}
    out = []
    todo = [int(i) for i in rows if str(item[i]) in flagged]
    for d in sorted({int(day[i]) for i in todo}):
        g = task._graph_upto(day < d)
        node_columns = ibm_node_table(g)
        stored = reg.compute(g, node_columns, hops, cap)
        source = MemorySource(g)
        for i in (i for i in todo if day[i] == d):
            s_, d_ = int(src[i]), int(dst[i])
            fn = _pair_score_fn(x.iloc[i], s_, d_, reg, node_columns, stored, hops, cap, score)
            e = explain_transfer(source, i, s_, d_, hops, cap, fn)
            top = []
            for nb, imp in zip(e.neighbours[:k], e.importance[:k], strict=False):
                name = members.get(nb) or outside.setdefault(nb, f"outside {len(outside) + 1}")
                top.append(
                    {
                        "account": name,
                        "in_ring": nb in members,
                        "in_any_laundering": bool(laundering[nb]),
                        "score_drop": round(float(imp), 4),
                    }
                )
            out.append(
                {
                    "transfer": f"{members[s_]} to {members[d_]}",
                    "hours": round((float(secs[i]) - float(secs[rows].min())) / 3600, 1),
                    "score": round(float(e.score), 4),
                    "neighbours": len(e.neighbours),
                    "top": top,
                }
            )
    return out
