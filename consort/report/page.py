"""The demo page's figures: what `consort site` writes to site/data.json besides the ring.

Everything here is read from committed files (results/summary.json, results/faithfulness.json,
results/serving.json and the gate's ledger), so CI can rebuild it and fail when the committed
data.json has fallen behind them (tests/test_page.py). The ring needs IBM's data and the
exported model, so it is built in `consort.report.site` and checked on the laptop only.

Every figure keeps its interval: a metric is always [point, low, high].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TAIL = "without-tail"
METRICS = ("auprc", "recall_at_1pct_fpr", "recall_at_1pct_budget", "ceiling_at_1pct_budget")


def aggregates(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Seed-0 runs, temporal and random, each metric with its interval. IBM's temporal runs
    read the without-tail slice, as the README's main table does, and keep the all-days AUPRC
    for the leakage comparison, which reads all test days."""
    out = []
    for r in summary:
        if r["seed"] != 0:
            continue
        ibm = r["dataset"].startswith("ibm-") and r["kind"] == "temporal"
        m = r["slices"].get(TAIL, r["metrics"]) if ibm else r["metrics"]
        delta = (r.get("slice_delta") or {}).get(TAIL) if ibm else r.get("delta")
        out.append(
            {
                "dataset": r["dataset"],
                "kind": r["kind"],
                "arm": r["arm"],
                "hops": r["hops"],
                "val_auprc": r.get("val_auprc"),
                **{k: m.get(k) for k in METRICS},
                "auprc_all_days": r["metrics"].get("auprc"),
                "delta_auprc": (delta or {}).get("auprc"),
                "before_43": r["slices"].get("before-43", {}).get("auprc"),
                "from_43": r["slices"].get("from-43", {}).get("auprc"),
                "from_43_delta": (r.get("slice_delta") or {}).get("from-43", {}).get("auprc"),
                "patterns": r.get("patterns") or {},
            }
        )
    return out


def seeds(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every seed of a run that has more than one: the spread the page reports beside arm N."""
    runs: dict[str, list[dict[str, Any]]] = {}
    for r in summary:
        if r["kind"] == "temporal":
            runs.setdefault(r["run"].rsplit(".s", 1)[0], []).append(r)
    return [
        {
            "dataset": rs[0]["dataset"],
            "arm": rs[0]["arm"],
            "hops": rs[0]["hops"],
            "auprc": {
                str(r["seed"]): (
                    r["slices"].get(TAIL, r["metrics"])
                    if r["dataset"].startswith("ibm-")
                    else r["metrics"]
                )["auprc"][0]
                for r in sorted(rs, key=lambda r: int(r["seed"]))
            },
        }
        for rs in runs.values()
        if len(rs) > 1
    ]


def gate(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project 03's decisions: recall at the 1% alert budget, candidate against arm R, at a
    3-point non-inferiority margin. The latest decision per candidate run."""
    latest: dict[str, dict[str, Any]] = {}
    for rec in sorted(ledger, key=lambda r: r["ts_utc"]):
        s = rec["suites"][0]
        latest[rec["candidate"]["label"]] = {
            "date": rec["ts_utc"][:10],
            "run": rec["candidate"]["label"],
            "baseline": rec["baseline"]["label"],
            "items": s["paired_items"],
            "baseline_recall": s["baseline_accuracy"],
            "candidate_recall": s["candidate_accuracy"],
            "difference": [s["difference"], s["lo"], s["hi"]],
            "margin": s["delta"],
            "verdict": s["verdict"],
        }
    return list(latest.values())


def figures(root: Path) -> dict[str, Any]:
    """Everything on the page but the ring, from the committed results under `root`."""
    summary = json.loads((root / "results" / "summary.json").read_text())
    serving = root / "results" / "serving.json"
    ledger = root / "gate" / "ledger.jsonl"
    return {
        "runs": aggregates(summary),
        "seeds": seeds(summary),
        "faithfulness": json.loads((root / "results" / "faithfulness.json").read_text()),
        "serving": json.loads(serving.read_text()) if serving.exists() else [],
        "gate": gate([json.loads(x) for x in ledger.read_text().splitlines() if x.strip()])
        if ledger.exists()
        else [],
    }
