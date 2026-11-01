"""Arm R: gradient boosting on the item's own row. The model most organisations run."""

from __future__ import annotations

import pandas as pd

from consort.data.splits import Partition
from consort.data.tasks import Task


def features(task: Task, partition: Partition, hops: int, hub_cap: int) -> pd.DataFrame:
    """The row columns only; hops and the cap do not apply and are refused if set."""
    if hops != 0:
        raise ValueError("arm R has no neighbourhood; run it at hop 0")
    return task.items[task.row_columns]
