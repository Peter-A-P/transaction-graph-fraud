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
