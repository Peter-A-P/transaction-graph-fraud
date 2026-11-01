"""Train an arm on a task's training partition, and write its test outcomes.

The leakage guard runs on every training table before every fit. Unknown labels are dropped
from training and from the outcomes; they stay in the graph that features were computed on.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from consort.data.leakage import check_random_training_rows, check_training_rows
from consort.data.splits import PARTITIONS, Partition, random_split
from consort.data.tasks import Task
from consort.eval import outcomes as oc
from consort.models import boost, graphfeat, row

FeatureFn = Callable[[Task, Partition, int, int], pd.DataFrame]
# Columns added outside the registry, given the mask of items whose labels training may see.
# Only consort.eval.leak uses this, for the rejected-approach experiment.
ExtraFn = Callable[[Task, npt.NDArray[np.bool_]], pd.DataFrame]
ARMS: dict[str, FeatureFn] = {"R": row.features, "G": graphfeat.features}
SplitKind = Literal["temporal", "random"]
RANDOM_SPLIT_SEED = 20261031


def run_tabular(
    task: Task,
    arm: str,
    hops: int,
    hub_cap: int,
    seed: int,
    out_dir: Path,
    split_kind: SplitKind = "temporal",
    extra: ExtraFn | None = None,
    arm_label: str | None = None,
) -> Path:
    feats = ARMS[arm]
    labelled = task.labelled()
    items = task.items
    y = items["label"].to_numpy()

    if split_kind == "temporal":
        part = task.partition()
        tables = {p: feats(task, p, hops, hub_cap) for p in PARTITIONS}
    else:
        # The random split most papers use: items drawn at random from every time step, and
        # features from the whole graph. Run once per dataset, to publish the gap.
        part = np.full(len(items), "", dtype="<U5")
        part[labelled] = random_split(int(labelled.sum()), seed=RANDOM_SPLIT_SEED)
        whole = feats(task, "test", hops, hub_cap)
        tables = {"train": whole, "val": whole, "test": whole}

    masks = {p: (part == p) & labelled for p in PARTITIONS}
    if split_kind == "temporal":
        check_training_rows(items["split_group"].to_numpy()[masks["train"]], task.split, task.name)
    else:
        check_random_training_rows(part[masks["train"]], task.name)
    if extra is not None:
        added = extra(task, masks["train"])
        tables = {
            p: pd.concat([t.reset_index(drop=True), added], axis=1) for p, t in tables.items()
        }

    fitted = boost.fit(
        tables["train"][masks["train"]],
        y[masks["train"]],
        tables["val"][masks["val"]],
        y[masks["val"]],
        seed=seed,
    )
    test = masks["test"]
    scores = fitted.predict(tables["test"][test])
    outcomes = oc.Outcomes.build(
        items["item_id"].to_numpy()[test],
        items["boot_group"].to_numpy()[test],
        y[test],
        scores,
    )
    split_name = task.split.name if split_kind == "temporal" else f"{task.name}-random"
    label = arm_label or arm
    run = f"{task.name}.{split_kind}.{label}.h{hops}.cap{hub_cap}.s{seed}"
    meta = oc.RunMeta(
        run=run,
        dataset=task.name,
        split=split_name,
        arm=label,
        model="xgboost",
        hops=hops,
        seed=seed,
        notes=f"hub_cap={hub_cap}; best_iteration={fitted.best_iteration}; "
        f"features={len(fitted.columns)}",
    )
    return oc.write(outcomes, meta, out_dir)
