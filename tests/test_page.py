"""The demo site's figures: the committed site/data.json holds exactly what the committed results
say, so a result that changes without `consort site` being rerun fails CI. The ring needs IBM's
data and the exported model, so only its shape is checked here."""

from __future__ import annotations

import json
from pathlib import Path

from consort.report import page

ROOT = Path(__file__).parents[1]


def test_the_committed_data_matches_the_committed_results() -> None:
    data = json.loads((ROOT / "site" / "data.json").read_text(encoding="utf8"))
    fresh = page.figures(ROOT)
    for key, value in fresh.items():
        assert data[key] == json.loads(json.dumps(value)), (
            f"site/data.json is stale in {key!r}: run consort site"
        )
    ring = data["ring"]
    assert ring["accounts"] and ring["transfers"]
    assert all(
        a["name"].startswith("A") for a in ring["accounts"]
    )  # named by place, never by number


def test_the_figures_read_the_slices_the_readme_reads() -> None:
    summary = [
        {
            "run": "ibm-HI-Small.temporal.G.h1.cap50.s0",
            "dataset": "ibm-HI-Small",
            "kind": "temporal",
            "arm": "G",
            "hops": 1,
            "seed": 0,
            "val_auprc": 0.5,
            "metrics": {"auprc": [0.6, 0.5, 0.7]},
            "slices": {page.TAIL: {"auprc": [0.4, 0.3, 0.5]}},
            "delta": {"auprc": [0.35, 0.3, 0.4]},
            "slice_delta": {page.TAIL: {"auprc": [0.29, 0.25, 0.32]}},
            "patterns": {},
        },
        {
            "run": "elliptic.random.R.h0.cap0.s0",
            "dataset": "elliptic",
            "kind": "random",
            "arm": "R",
            "hops": 0,
            "seed": 0,
            "metrics": {"auprc": [0.98, 0.97, 0.99]},
            "slices": {},
            "delta": {},
        },
        {
            "run": "elliptic.temporal.R.h0.cap0.s1",
            "dataset": "elliptic",
            "kind": "temporal",
            "arm": "R",
            "hops": 0,
            "seed": 1,
            "metrics": {"auprc": [0.7, 0.6, 0.8]},
            "slices": {},
        },
    ]
    ibm, rnd = page.aggregates(summary)  # seed 1 is left out
    # IBM's temporal runs: the without-tail slice and its paired delta, as the main table.
    assert ibm["auprc"] == [0.4, 0.3, 0.5] and ibm["delta_auprc"] == [0.29, 0.25, 0.32]
    assert ibm["auprc_all_days"] == [0.6, 0.5, 0.7]  # the leakage table reads all days
    assert rnd["kind"] == "random" and rnd["auprc"] == [0.98, 0.97, 0.99]


def test_the_gate_keeps_the_latest_decision_per_run() -> None:
    def rec(ts: str, diff: float) -> dict[str, object]:
        return {
            "ts_utc": ts,
            "baseline": {"label": "x.temporal.R.h0.cap0.s0"},
            "candidate": {"label": "x.temporal.G.h1.cap50.s0"},
            "suites": [
                {
                    "paired_items": 10,
                    "baseline_accuracy": 0.5,
                    "candidate_accuracy": 0.5 + diff,
                    "difference": diff,
                    "lo": diff - 0.1,
                    "hi": diff + 0.1,
                    "delta": 0.03,
                    "verdict": "pass",
                }
            ],
        }

    (only,) = page.gate([rec("2026-11-02T10:00:00Z", 0.2), rec("2026-11-02T09:00:00Z", 0.1)])
    assert only["difference"][0] == 0.2 and only["date"] == "2026-11-02"
