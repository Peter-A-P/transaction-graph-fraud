from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from consort.data import ibm
from consort.data.tasks import IBM_NODE_COLUMNS, ibm_node_table
from consort.graph.features import default_registry
from consort.graph.neighbourhood import MemorySource
from consort.serve import bench
from tests.toy import write_ibm


def test_cost_per_million_arithmetic() -> None:
    # 1 ms a score, fully busy: 3.6 million scores an hour; at US$0.36 an hour, US$0.10 a million.
    assert bench.cost_per_million(1.0, 0.36, 1.0) == pytest.approx(0.10)
    assert bench.cost_per_million(1.0, 0.36, 0.5) == pytest.approx(0.20)


def test_bench_runs_and_laptop_timings_are_refused(tmp_path: Path) -> None:
    data = ibm.load(write_ibm(tmp_path, n_background=1500, n_cycles=10), "T")
    g = data.graph
    reg = default_registry(IBM_NODE_COLUMNS, ("amount", "t"))
    nodes = ibm_node_table(g)
    stored = reg.compute(g, nodes, 2, 5)
    machine = bench.Machine(name="laptop", usd_per_hour=0.0, price_date=date(2026, 10, 31))
    res = bench.run(
        "memory",
        MemorySource(g),
        reg,
        nodes,
        stored,
        lambda row: float(row.sum()),
        list(range(30)),
        hops=2,
        hub_cap=5,
        machine=machine,
        warmup=5,
    )
    assert res.calls == 30 and res.p99_ms >= res.p50_ms > 0
    with pytest.raises(bench.NotReportable):
        bench.check_reportable(res)
    assert np.isfinite(res.mean_edges_fetched)


def test_transfer_bench_scores_through_the_onnx_export(tmp_path: Path) -> None:
    from consort.data.tasks import ibm_task
    from consort.models.run import run_tabular

    data = ibm.load(write_ibm(tmp_path, n_background=1500, n_cycles=10), "T")
    task = ibm_task(data, boot_group="day")
    run_tabular(task, "G", 1, 5, seed=0, out_dir=tmp_path / "runs", onnx_dir=tmp_path / "m")
    (model,) = (tmp_path / "m").glob("*.onnx")
    g = data.graph
    reg = default_registry(IBM_NODE_COLUMNS, ("amount", "t"))
    nodes = ibm_node_table(g)
    stored = reg.compute(g, nodes, 1, 5)
    t = data.transfers
    machine = bench.Machine(name="laptop", usd_per_hour=0.0, price_date=date(2026, 11, 1))
    res = {
        counters: bench.run_transfers(
            "memory",
            "ibm-T",
            MemorySource(g),
            reg,
            nodes,
            stored,
            bench.onnx_scorer(model),
            t.iloc[:25].reset_index(drop=True),
            data.row_columns,
            1,
            5,
            machine,
            warmup=3,
            counters=counters,
        )
        for counters in (False, True)
    }
    assert res[False].unit == "transfer" and res[False].calls == 25 and res[False].p50_ms > 0
    assert (res[False].centre, res[True].centre) == ("history", "counters")
    # Stored counters bound the fetch by the cap: at most 2 x 5 edges per account.
    assert res[True].mean_edges_fetched <= min(20, res[False].mean_edges_fetched)
