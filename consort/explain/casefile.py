"""One case file per flagged account: the data an investigator reads, as JSON.

No language model writes anything here; a narrative, if one is ever wanted, is a rendering of
this file and goes through the portfolio gateway with the public data class (synthetic
accounts only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from consort.explain.faithfulness import Explanation


def casefile(
    e: Explanation,
    account: dict[str, Any],
    top_features: list[tuple[str, float]],
    pattern: str | None,
    k: int = 5,
) -> dict[str, Any]:
    return {
        "account": account,
        "score": e.score,
        "top_neighbours": [
            {"node": n, "importance": imp}
            for n, imp in zip(e.neighbours[:k], e.importance[:k], strict=False)
        ],
        "top_features": [{"feature": f, "contribution": c} for f, c in top_features],
        "planted_pattern": pattern,
    }


def write(case: dict[str, Any], directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(case, indent=2) + "\n")
    return path
