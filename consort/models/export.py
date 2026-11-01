"""The ONNX handover to 09: an arm's booster as an ONNX scorer, with its feature contract.

The ONNX file takes one float32 input, `input`, of shape (n, k) with the columns in the order
the sidecar lists, and returns `probabilities` (n, 2); column 1 is the fraud score. The
sidecar is the feature view: the column names a caller must compute, the hop count and cap
they are computed at, and the arm. Only arms R and G are exported; TabFM's weights licence
forbids redistributing the model or its derivatives (docs/models.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from consort.models.boost import Fitted

OPSET = 15


def export(fitted: Fitted, path: Path, arm: str, hops: int, hub_cap: int) -> Path:
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    booster = fitted.booster[: fitted.best_iteration + 1]
    # onnxmltools reads features as f0..fk; the names live in the sidecar instead.
    booster.feature_names = None
    booster.feature_types = None
    model = onnxmltools.convert_xgboost(
        booster,
        initial_types=[("input", FloatTensorType([None, len(fitted.columns)]))],
        target_opset=OPSET,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())
    contract = {
        "arm": arm,
        "hops": hops,
        "hub_cap": hub_cap,
        "input": "input",
        "output": "probabilities",
        "score_column": 1,
        "columns": fitted.columns,
    }
    path.with_suffix(".features.json").write_text(json.dumps(contract, indent=2) + "\n")
    return path


def score_onnx(path: Path, X: pd.DataFrame) -> npt.NDArray[np.float64]:
    import onnxruntime as ort

    contract = json.loads(path.with_suffix(".features.json").read_text())
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = X[contract["columns"]].to_numpy(np.float32)
    out = session.run([contract["output"]], {contract["input"]: x})[0]
    return np.asarray(out[:, contract["score_column"]], dtype=np.float64)
