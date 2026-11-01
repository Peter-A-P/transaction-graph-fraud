"""The serving benchmark: fetch, features and score, timed per call, per hop, per store.

Each call scores one node from scratch: the neighbourhood fetch from the store, arm G's
registry on the fetched subgraph, and the model. p50 and p99 are over calls; cost per million
scores is the machine's hourly price over the throughput one worker sustains at the stated
utilisation. Every result names the machine, its price and the price's date.

Nothing timed is reported from the laptop: a result is reportable only when it was run with
an explicit machine name and price, and the report refuses any other.
"""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Callable, Sequence
from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel

from consort.graph.neighbourhood import EdgeSource, fetch, local_features
from consort.graph.registry import FeatureRegistry

ScoreRow = Callable[[pd.Series], float]


class Machine(BaseModel):
    name: str
    usd_per_hour: float
    price_date: date
    host: str = platform.node()
    cpu: str = platform.processor()
    cores: int = os.cpu_count() or 0
    reportable: bool = False


class BenchResult(BaseModel):
    store: str
    hops: int
    hub_cap: int
    calls: int
    p50_ms: float
    p99_ms: float
    mean_ms: float
    mean_edges_fetched: float
    utilisation: float
    usd_per_million: float
    machine: Machine


def cost_per_million(mean_ms: float, usd_per_hour: float, utilisation: float) -> float:
    """One worker, sequential calls: scores per hour = 3.6e6 ms / mean_ms, scaled by the share
    of the hour the machine is busy."""
    scores_per_hour = 3_600_000.0 / mean_ms * utilisation
    return usd_per_hour / scores_per_hour * 1_000_000.0


def run(
    store: str,
    source: EdgeSource,
    registry: FeatureRegistry,
    node_columns: pd.DataFrame,
    stored: pd.DataFrame,
    score: ScoreRow,
    nodes: Sequence[int],
    hops: int,
    hub_cap: int,
    machine: Machine,
    utilisation: float = 0.5,
    warmup: int = 20,
) -> BenchResult:
    def call(v: int) -> int:
        edges = fetch(source, v, hops, hub_cap)
        row = local_features(edges, v, registry, node_columns, hops, hub_cap, stored=stored)
        score(row)
        return int(edges.edge_id.shape[0])

    for v in nodes[:warmup]:
        call(v)
    times, sizes = [], []
    for v in nodes:
        t0 = time.perf_counter()
        sizes.append(call(v))
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = np.array(times)
    mean = float(ms.mean())
    return BenchResult(
        store=store,
        hops=hops,
        hub_cap=hub_cap,
        calls=len(nodes),
        p50_ms=float(np.quantile(ms, 0.5)),
        p99_ms=float(np.quantile(ms, 0.99)),
        mean_ms=mean,
        mean_edges_fetched=float(np.mean(sizes)),
        utilisation=utilisation,
        usd_per_million=cost_per_million(mean, machine.usd_per_hour, utilisation),
        machine=machine,
    )


class NotReportable(Exception):
    """A timing that may not appear in a published table."""


def check_reportable(result: BenchResult) -> None:
    if not result.machine.reportable:
        raise NotReportable(
            f"{result.store} h{result.hops} was timed on {result.machine.name!r} "
            "without --reportable; nothing timed is reported from the laptop"
        )
