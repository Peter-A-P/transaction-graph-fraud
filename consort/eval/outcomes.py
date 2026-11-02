"""One outcomes file shape for every arm. The evaluator never learns which model scored.

An outcomes file is a parquet table with exactly the columns item_id, group, label, score,
beside a JSON sidecar that says what produced it. The evaluator reads the table only; the
sidecar is for the report, which needs to name the arm in a row label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel

COLUMNS = ("item_id", "group", "label", "score")


class UnknownLabel(Exception):
    """An outcome row carries a label that is neither 0 nor 1 (Elliptic's unknown, say)."""


class RunMeta(BaseModel):
    run: str
    dataset: str
    split: str
    arm: str
    model: str
    hops: int
    seed: int
    partition: str = "test"
    # AUPRC on the validation partition, from the fitted model: what a hop count or a variant
    # is chosen by, so no choice is made on test. None for runs written before it was recorded.
    val_auprc: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class Outcomes:
    item_id: npt.NDArray[np.str_]
    group: npt.NDArray[np.int64]
    label: npt.NDArray[np.int8]
    score: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        n = self.item_id.shape[0]
        for name in ("group", "label", "score"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"outcomes column {name} has the wrong length")
        if not np.isin(self.label, (0, 1)).all():
            raise UnknownLabel("outcome labels must be 0 or 1; unknown labels never enter a metric")
        if not np.isfinite(self.score).all():
            raise ValueError("outcome scores must be finite")
        if np.unique(self.item_id).shape[0] != n:
            raise ValueError("outcome item ids must be unique")

    def __len__(self) -> int:
        return int(self.item_id.shape[0])

    @classmethod
    def build(
        cls,
        item_id: npt.ArrayLike,
        group: npt.ArrayLike,
        label: npt.ArrayLike,
        score: npt.ArrayLike,
    ) -> Outcomes:
        lab = np.asarray(label)
        if lab.dtype.kind not in "iub":
            raise UnknownLabel(f"outcome labels must be integers, got dtype {lab.dtype}")
        return cls(
            item_id=np.asarray(item_id).astype(str),
            group=np.asarray(group, dtype=np.int64),
            label=lab.astype(np.int8),
            score=np.asarray(score, dtype=np.float64),
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"item_id": self.item_id, "group": self.group, "label": self.label, "score": self.score}
        )

    def subset(self, mask: npt.NDArray[np.bool_]) -> Outcomes:
        return Outcomes(self.item_id[mask], self.group[mask], self.label[mask], self.score[mask])


def write(outcomes: Outcomes, meta: RunMeta, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{meta.run}.outcomes.parquet"
    outcomes.frame().to_parquet(path, index=False)
    (directory / f"{meta.run}.meta.json").write_text(meta.model_dump_json(indent=2) + "\n")
    return path


def read(path: Path) -> Outcomes:
    df = pd.read_parquet(path)
    if tuple(df.columns) != COLUMNS:
        raise ValueError(f"{path}: outcome columns {list(df.columns)} are not {list(COLUMNS)}")
    return Outcomes.build(df["item_id"], df["group"], df["label"], df["score"])


def read_meta(path: Path) -> RunMeta:
    meta_path = path.with_name(path.name.replace(".outcomes.parquet", ".meta.json"))
    return RunMeta.model_validate(json.loads(meta_path.read_text()))
