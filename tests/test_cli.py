from __future__ import annotations

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from consort.cli import app
from consort.eval import outcomes as oc


def _run(tmp_path: Path, name: str, scores: np.ndarray) -> Path:
    n = scores.shape[0]
    labels = (np.arange(n) % 10 == 0).astype(int)
    o = oc.Outcomes.build(np.arange(n), np.arange(n) % 5, labels, scores)
    meta = oc.RunMeta(run=name, dataset="toy", split="t", arm=name, model="m", hops=0, seed=0)
    return oc.write(o, meta, tmp_path)


def test_eval_prints_intervals_and_paired_deltas(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    a = _run(tmp_path, "a", rng.random(200))
    b = _run(tmp_path, "b", rng.random(200))
    result = CliRunner().invoke(app, ["eval", str(b), "--against", str(a), "--n-boot", "20"])
    assert result.exit_code == 0, result.output
    assert "auprc:" in result.output and "delta auprc:" in result.output
