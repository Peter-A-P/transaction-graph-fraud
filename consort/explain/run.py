"""Faithfulness for arm G: occlusion through the serving fetch, scored by the arm's exported
booster, graded against known-illicit neighbours, with intervals over time steps (Elliptic) or
hours (IBM).

The flagged nodes are the run's alert queue: its top-scored test transactions at the 1% alert
budget. Each is explained by occluding its direct neighbours one at a time on the fetched
neighbourhood and rescoring through the same ONNX export that would serve it. Elliptic's own
aggregated features (part of arm G's table from one hop) are the dataset's, fixed per
transaction, so occlusion moves only the features this project computes; the README says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel

from consort.data import elliptic as ell
from consort.data.tasks import IBM_NODE_COLUMNS, EllipticTask, IBMTask, ibm_node_table
from consort.eval import outcomes as oc
from consort.explain.faithfulness import (
    Explanation,
    PairScoreFn,
    ScoreFn,
    explain,
    explain_transfer,
    graded,
)
from consort.graph.features import default_registry
from consort.graph.neighbourhood import Edges, MemorySource, local_features
from consort.graph.registry import FeatureRegistry
from consort.serve.bench import TransferScore, onnx_scorer

Triple = tuple[float, float, float]


class FaithfulnessResult(BaseModel):
    dataset: str
    run: str
    k: int
    flagged: int
    graded: int
    n_groups: int
    explained: Triple
    random: Triple
    lift: Triple  # explained minus random, the same nodes in every resample


def _boot(
    values: dict[str, npt.NDArray[np.float64]],
    groups: npt.NDArray[np.int64],
    n_boot: int,
    seed: int,
) -> dict[str, Triple]:
    uniq, code = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    out: dict[str, Triple] = {}
    samples = {k: np.empty(n_boot) for k in values}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[code]
        for k, v in values.items():
            samples[k][b] = float(np.sum(w * v)) / max(float(np.sum(w)), 1.0)
    for k, v in values.items():
        lo, hi = np.quantile(samples[k], [0.025, 0.975])
        out[k] = (round(float(v.mean()), 4), round(float(lo), 4), round(float(hi), 4))
    return out


def elliptic_score_fn(task: EllipticTask, model: Path, hops: int, cap: int) -> ScoreFn:
    assert task.data is not None
    data = task.data
    reg = default_registry(ell.LOCAL)
    node_columns = data.nodes[ell.LOCAL].reset_index(drop=True)
    stored = reg.compute(data.graph, node_columns, hops, cap)
    rows = task.items[task.row_columns].reset_index(drop=True)
    agg = data.nodes[ell.AGG].reset_index(drop=True)
    score = onnx_scorer(model)

    def fn(edges: Edges, v: int) -> float:
        g = local_features(edges, v, reg, node_columns, hops, cap, stored=stored)
        parts = [rows.iloc[v], g] + ([agg.iloc[v]] if hops >= 1 else [])
        return score(pd.concat(parts).to_frame().T)

    return fn


def _results(
    name: str,
    run: str,
    exps: list[Explanation],
    illicit: npt.NDArray[np.bool_],
    group: npt.NDArray[np.int64],
    ks: tuple[int, ...],
    n_boot: int,
) -> list[FaithfulnessResult]:
    out = []
    for k in ks:
        nodes, hit, rand = graded(exps, illicit, k)
        if nodes.size == 0:
            continue
        est = _boot({"e": hit, "r": rand, "d": hit - rand}, group[nodes], n_boot, seed=k)
        out.append(
            FaithfulnessResult(
                dataset=name,
                run=run,
                k=k,
                flagged=len(exps),
                graded=int(nodes.size),
                n_groups=int(np.unique(group[nodes]).size),
                explained=est["e"],
                random=est["r"],
                lift=est["d"],
            )
        )
    return out


def run_faithfulness(
    task: EllipticTask,
    outcomes: Path,
    model: Path,
    hops: int,
    cap: int,
    ks: tuple[int, ...] = (1, 3, 5),
    budget: float = 0.01,
    n_boot: int = 1000,
) -> list[FaithfulnessResult]:
    assert task.data is not None
    o, meta = oc.read(outcomes), oc.read_meta(outcomes)
    node_of = {item: i for i, item in enumerate(task.items["item_id"].tolist())}
    n_flag = max(1, round(budget * len(o)))
    top = np.argsort(-o.score, kind="stable")[:n_flag]
    flagged = [node_of[i] for i in o.item_id[top].tolist()]
    fn = elliptic_score_fn(task, model, hops, cap)
    src = MemorySource(task.data.graph)
    exps = [explain(src, v, hops, cap, fn) for v in flagged]
    illicit = (task.items["label"] == 1).to_numpy()
    step = task.items["split_group"].to_numpy().astype(np.int64)
    return _results(task.name, meta.run, exps, illicit, step, ks, n_boot)


def _pair_score_fn(
    row: pd.Series,
    src: int,
    dst: int,
    reg: FeatureRegistry,
    node_columns: pd.DataFrame,
    stored: pd.DataFrame,
    hops: int,
    cap: int,
    score: TransferScore,
) -> PairScoreFn:
    """One transfer's live score from its two accounts' fetched neighbourhoods, as the serving
    benchmark computes it (consort.serve.bench.run_transfers)."""

    def fn(es: Edges, ed: Edges) -> float:
        parts = [row]
        for end, v, e in (("src", src, es), ("dst", dst, ed)):
            f = local_features(e, v, reg, node_columns, hops, cap, stored=stored)
            parts.append(f.add_prefix(f"{end}_"))
        return score(pd.concat(parts).to_frame().T)

    return fn


def run_faithfulness_ibm(
    task: IBMTask,
    outcomes: Path,
    model: Path,
    hops: int,
    cap: int,
    ks: tuple[int, ...] = (1, 3, 5),
    budget: float = 0.01,
    sample: int = 300,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[FaithfulnessResult]:
    """IBM: a sample of the run's 1% alert queue, each transfer explained on the graph of the
    days before its own (the features it was scored with), by occluding a direct neighbour of
    either account. A neighbour counts as known illicit if the generator marks any transfer it
    sent or received as laundering: labels grade the explanation and never make it."""
    assert task.data is not None
    o, meta = oc.read(outcomes), oc.read_meta(outcomes)
    row_of = {item: i for i, item in enumerate(task.items["item_id"].tolist())}
    n_flag = max(1, round(budget * len(o)))
    queue = np.array([row_of[i] for i in o.item_id[np.argsort(-o.score, kind="stable")[:n_flag]]])
    rng = np.random.default_rng(seed)
    flagged = np.sort(rng.choice(queue, min(sample, queue.size), replace=False))
    t = task.data.transfers
    src, dst = t["src"].to_numpy(np.int64), t["dst"].to_numpy(np.int64)
    label = task.items["label"].to_numpy() == 1
    illicit = np.zeros(task.data.graph.n_nodes, dtype=bool)
    illicit[src[label]] = True
    illicit[dst[label]] = True
    day = task.items["split_group"].to_numpy().astype(np.int64)
    rows = task.items[task.row_columns].reset_index(drop=True)
    reg = default_registry(IBM_NODE_COLUMNS, ("amount", "t"))
    score = onnx_scorer(model)
    exps: list[Explanation] = []
    for d in np.unique(day[flagged]):
        g = task._graph_upto(day < d)
        node_columns = ibm_node_table(g)
        stored = reg.compute(g, node_columns, hops, cap)
        source = MemorySource(g)
        for i in flagged[day[flagged] == d]:
            s_, d_ = int(src[i]), int(dst[i])
            fn = _pair_score_fn(
                rows.iloc[int(i)], s_, d_, reg, node_columns, stored, hops, cap, score
            )
            exps.append(explain_transfer(source, int(i), s_, d_, hops, cap, fn))
    # Intervals over hours, the evaluator's IBM groups: a queue spans only a few test days.
    hour = task.items["boot_group"].to_numpy().astype(np.int64)
    return _results(task.name, meta.run, exps, illicit, hour, ks, n_boot)
