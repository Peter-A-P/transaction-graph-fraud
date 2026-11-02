"""The ONNX export reproduces the trained model's scores on a thousand rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("onnxmltools")

from consort.models import boost
from consort.models.export import export, score_onnx


def test_onnx_scores_match_the_booster(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    cols = ["deg_in", "amt_out_sum", "nbr_mean_x_h1", "cycle3"]
    X = pd.DataFrame(rng.normal(size=(3000, 4)).astype(np.float32), columns=cols)
    y = ((X["deg_in"] + X["cycle3"] * 2 + rng.normal(0, 0.5, 3000)) > 1).astype(int)
    fitted = boost.fit(X[:2000], y[:2000], X[2000:2500], y[2000:2500], seed=0)
    path = export(fitted, tmp_path / "g.onnx", arm="G", hops=2, hub_cap=50)
    test = X[2000:3000]
    # Columns given out of order: the contract puts them back.
    got = score_onnx(path, test[cols[::-1]])
    want = fitted.predict(test)
    assert np.abs(got - want).max() < 1e-5


def test_the_export_has_the_shape_09_loads(tmp_path: Path) -> None:
    """09's OnnxModel loads a file with one float input of the sidecar's width and a
    `probabilities` output of two columns, fraud second; its own export targets opset 15 too.
    The handover holds only if the export keeps that shape (docs/handover.md)."""
    import onnx
    import onnxruntime as ort

    rng = np.random.default_rng(1)
    cols = ["deg_out", "amt_in_max", "nbr_max_x_h1"]
    X = pd.DataFrame(rng.normal(size=(600, 3)).astype(np.float32), columns=cols)
    y = (X["deg_out"] > 0.5).astype(int)
    fitted = boost.fit(X[:400], y[:400], X[400:], y[400:], seed=0)
    path = export(fitted, tmp_path / "g.onnx", arm="G", hops=1, hub_cap=50)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    # A tree ensemble: only the ML domain is imported, as in 09's own export at target opset 15.
    assert {o.domain for o in model.opset_import} <= {"", "ai.onnx", "ai.onnx.ml"}
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (inp,) = session.get_inputs()
    assert inp.type == "tensor(float)" and inp.shape[1] == len(cols)
    assert "probabilities" in [o.name for o in session.get_outputs()]
    out = session.run(["probabilities"], {inp.name: X.to_numpy(np.float32)[:5]})[0]
    assert out.shape == (5, 2) and np.allclose(out.sum(axis=1), 1.0, atol=1e-5)
