"""The report: summaries from outcome files, baselines, slices, the headline hop chosen on
validation, and README tables written only between their markers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from consort.data import elliptic
from consort.data.splits import ELLIPTIC_TEMPORAL
from consort.data.tasks import elliptic_task
from consort.eval import outcomes as oc
from consort.models.run import run_tabular
from consort.report import summary as sm
from consort.report import tables as tb
from tests.toy import write_elliptic


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("rep")
    task = elliptic_task(elliptic.load(write_elliptic(root), check_counts=False))
    out = root / "runs"
    run_tabular(task, "R", 0, 0, seed=0, out_dir=out)
    for h in (1, 2):
        run_tabular(task, "G", h, 10, seed=0, out_dir=out)
    return out


def _summaries(runs: Path, cache: Path) -> list[sm.RunSummary]:
    metas = {
        p.name.removesuffix(".meta.json"): oc.RunMeta.model_validate_json(p.read_text())
        for p in runs.glob("*.meta.json")
    }
    ctx = sm.Context(ELLIPTIC_TEMPORAL)
    return [
        sm.summarise(runs / f"{n}.outcomes.parquet", metas, lambda _: ctx, cache, n_boot=40)
        for n in metas
    ]


def test_summaries_pair_against_r_and_read_the_shift_slices(runs: Path, tmp_path: Path) -> None:
    out = {s.arm + str(s.hops): s for s in _summaries(runs, tmp_path / "cache")}
    assert out["R0"].baseline is None
    assert out["G2"].baseline == "elliptic.temporal.R.h0.cap0.s0"
    assert set(out["G2"].slices) == {"before-43", "from-43"}
    assert set(out["G2"].slice_delta) == {"before-43", "from-43"}
    lo, hi = out["G2"].metrics["auprc"][1:]
    assert lo <= out["G2"].metrics["auprc"][0] <= hi


def test_summaries_are_cached_by_the_outcomes_file(runs: Path, tmp_path: Path) -> None:
    first = _summaries(runs, tmp_path / "cache")
    cached = list((tmp_path / "cache").glob("*.json"))
    assert len(cached) == len(first)
    for p in cached:  # mark every cached summary; a cache hit must return the mark
        s = sm.RunSummary.model_validate_json(p.read_text())
        s.metrics["auprc"] = (0.123, 0.1, 0.2)
        s.val_auprc = -1.0
        p.write_text(s.model_dump_json())
    again = _summaries(runs, tmp_path / "cache")
    assert {s.metrics["auprc"][0] for s in again} == {0.123}
    # The validation score is the meta file's, read afresh: a rerun can add it.
    assert [s.val_auprc for s in again] == [s.val_auprc for s in first]


def test_headline_hop_is_chosen_on_validation(runs: Path, tmp_path: Path) -> None:
    summaries = _summaries(runs, tmp_path / "cache")
    gs = sorted((s for s in summaries if s.arm == "G"), key=lambda s: s.hops)
    assert len(gs) >= 2
    # The toy graph is easy enough that every hop count can reach 1.0 on validation, and then
    # the pick is only the tie-break, so the validation scores are set here: more hops, worse.
    for i, s in enumerate(gs):
        s.val_auprc = 0.9 - 0.1 * i
    best, worst = gs[0], gs[-1]
    # Give the worse-on-validation run the better test score: the headline must not move.
    worst.metrics["auprc"] = (0.999, 0.99, 1.0)
    assert tb.headline(summaries, "elliptic", "G") is best
    # And the other way round, so the tie-break to fewer hops cannot pass it alone.
    best.val_auprc, worst.val_auprc = worst.val_auprc, best.val_auprc
    assert tb.headline(summaries, "elliptic", "G") is worst


def test_tables_fill_only_between_markers(runs: Path, tmp_path: Path) -> None:
    summaries = _summaries(runs, tmp_path / "cache")
    text = "intro\n<!-- report:main -->\nold\n<!-- /report:main -->\noutro\n"
    filled = tb.fill(text, {"main": tb.main_table(summaries)})
    assert filled.startswith("intro\n<!-- report:main -->\n| Dataset |")
    assert filled.endswith("\n<!-- /report:main -->\noutro\n")
    assert "old" not in filled
    assert tb.NOT_RUN in filled  # arm N and the datasets not run here say so
    with pytest.raises(ValueError):
        tb.fill("no markers", {"main": "x"})
    timed = [
        {
            "dataset": "ibm-HI-Small",
            "store": "duckdb",
            "centre": "counters",
            "hops": 1,
            "p50_ms": 4.25,
            "p99_ms": 9.0,
            "mean_edges_fetched": 180.0,
            "usd_per_million": 1.5,
            "machine": {"name": "RunPod L4", "usd_per_hour": 0.59, "price_date": "2026-11-02"},
        }
    ]
    row = tb.serving_table(timed).splitlines()[-1]
    assert "| duckdb | stored counters | 1 | 4.2 | 9.0 | 180 | 1.50 |" in row
    assert "US$0.59/h (2026-11-02)" in row
    # Adjacent markers, as a fresh README has them, fill too; and filling twice is stable.
    once = tb.fill("<!-- report:main -->\n<!-- /report:main -->\n", {"main": "x"})
    assert once == "<!-- report:main -->\nx\n<!-- /report:main -->\n"
    assert tb.fill(once, {"main": "x"}) == once
    assert np.isfinite(summaries[0].metrics["auprc"][0])
    # Seeds: one seed is no spread to report; a second seed of the same run is.
    assert tb.NOT_RUN in tb.seed_table(summaries)
    g = next(s for s in summaries if s.arm == "G")
    other = g.model_copy(update={"seed": 1, "metrics": {**g.metrics, "auprc": (0.5, 0.4, 0.6)}})
    row = tb.seed_table([*summaries, other]).splitlines()[-1]
    assert f"seed 0: {g.metrics['auprc'][0]:.3f}; seed 1: 0.500" in row
