"""Side files and a spec for project 03's gate (its `outcomes_file` adapter).

The gate takes one boolean per item, paired across a baseline and a candidate. For a ranker
the boolean is the business question of PLAN.md 2.4: for each illicit test item, did the 1%
alert queue reach it? The items are the illicit test items, identical across arms, so the
gate's paired non-inferiority test is a test on recall at the 1% alert budget. The queue is
the top 1% of all test items by score, ties broken by item id so that every item gets a
definite verdict (the evaluator takes ties in expectation instead; the two agree when no
tie straddles the cut).

The gate resamples items, not groups; its decision is its own, on its own statistics. The
README's intervals are this project's, over groups.

Side files carry item ids, so for Elliptic (CC BY-NC-ND: no derivatives redistributed) and
Elliptic++ (aggregates only, by the authors' permission) they stay under data/, ignored; only
the gate's ledger record, which names each file by its SHA-256, is committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from consort.eval import outcomes as oc


def caught(o: oc.Outcomes, budget: float = 0.01) -> dict[str, bool]:
    """For each positive item: whether the top `budget` share of all items reaches it."""
    k = max(1, round(budget * len(o)))
    order = np.lexsort((o.item_id, -o.score))
    flagged = np.zeros(len(o), dtype=bool)
    flagged[order[:k]] = True
    pos = o.label == 1
    return {str(i): bool(f) for i, f in zip(o.item_id[pos], flagged[pos], strict=True)}


def side(path: Path, suite: str, budget: float = 0.01) -> dict[str, object]:
    o, meta = oc.read(path), oc.read_meta(path)
    return {
        "label": meta.run,
        "source": {"run_id": meta.run, "directory": str(path.parent.as_posix())},
        "suites": {
            suite: {
                "suite": suite,
                "outcomes": caught(o, budget),
                "ungradeable_items": 0,
                "calls": len(o),
                # Scores here are batch scores, not timed calls: latency is not measured on
                # this path and no call is costed (the serving benchmark prices a live score).
                "latency_p50_ms": 0.0,
                "cost_usd": 0.0,
                "uncosted_calls": len(o),
            }
        },
    }


def spec(name: str, suite: str, delta_points: float) -> str:
    return (
        "version: 1\n"
        f"name: {name}\n"
        f"delta_points: {delta_points}\n"
        "suites:\n"
        f"  - key: {suite}\n"
        f"    source: {{kind: outcomes_file, block: {suite}}}\n"
    )


def export(
    baseline: Path, candidates: list[Path], out: Path, suite: str, delta_points: float = 3.0
) -> list[Path]:
    """Write spec.yaml, the baseline's side file and each candidate's into `out`. The gate's
    names are lowercase (its spec schema), so the suite is too."""
    suite = suite.lower()
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.yaml").write_text(spec(f"consort-{suite}", suite, delta_points))
    written = []
    for path in [baseline, *candidates]:
        name = oc.read_meta(path).arm
        target = out / f"{name}.json"
        target.write_text(json.dumps(side(path, suite), indent=1) + "\n")
        written.append(target)
    return written
