"""Arm G: the same gradient boosting on the row plus graph features to a stated hop count."""

from __future__ import annotations

import pandas as pd

from consort.data.leakage import check_feature_columns
from consort.data.splits import Partition
from consort.data.tasks import Task


def features(task: Task, partition: Partition, hops: int, hub_cap: int) -> pd.DataFrame:
    graph = task.graph_features(partition, hops, hub_cap)
    X = pd.concat([task.items[task.row_columns].reset_index(drop=True), graph], axis=1)
    check_feature_columns(X.columns, label_column="label")
    return X
