"""The TabFM column and the context swap, writing the same outcomes shape as every arm.

- `run_tabfm`: TabFM handed arm R's or arm G's table. Its context is a sample of the labelled
  training rows of a stated size (every positive kept); validation rows are scored to record
  validation AUPRC like any run; test rows are scored once. Arm label `TabFM-R` or `TabFM-G`.
- `run_context_swap` (Elliptic and Elliptic++, PLAN.md 2.6): each test step s is scored with a
  context drawn from the `window` steps just before s, whose labels are taken to have arrived
  by then; XGBoost is retrained on exactly the same rows at the same cadence. Both write one
  outcomes file over every test item, so the shift slices read them like any run. These are
  the only runs whose fitting rows come from test steps, always strictly earlier than the step
  scored, which `check_strictly_before` enforces in place of the split guard.

The scorer is injected (`make`), so the logic is tested without the model installed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from consort.data.leakage import check_training_rows
from consort.data.splits import PARTITIONS
from consort.data.tasks import Task
from consort.eval import outcomes as oc
from consort.eval.metrics import Ranked, average_precision
from consort.models import boost
from consort.models.run import ARMS
from consort.models.tabfm import context_sample, context_swap_schedule


class Scorer(Protocol):
    def fit(self, X: pd.DataFrame, y: npt.ArrayLike, groups: npt.ArrayLike) -> Scorer: ...
    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]: ...


MakeScorer = Callable[[int, int], Scorer]  # (context size, seed) -> an unfitted scorer


class LeakedStep(Exception):
    """A context row is not from a step strictly before the step it scores."""


def check_strictly_before(context_steps: npt.ArrayLike, step: int) -> None:
    s = np.asarray(context_steps)
    if s.size and int(s.max()) >= step:
        raise LeakedStep(f"context holds step {int(s.max())} for scoring step {step}")


def _auprc(y: npt.NDArray[np.int8], s: npt.NDArray[np.float64]) -> float:
    return average_precision(Ranked.of(y, s, np.zeros(len(y))), np.ones(len(y)))


def _write(
    task: Task,
    rows: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
    arm: str,
    model: str,
    hops: int,
    cap: int,
    seed: int,
    val_auprc: float | None,
    notes: str,
    out_dir: Path,
) -> Path:
    items = task.items
    outcomes = oc.Outcomes.build(
        items["item_id"].to_numpy()[rows],
        items["boot_group"].to_numpy()[rows],
        items["label"].to_numpy()[rows],
        scores,
    )
    run = f"{task.name}.temporal.{arm}.h{hops}.cap{cap}.s{seed}"
    meta = oc.RunMeta(
        run=run,
        dataset=task.name,
        split=task.split.name,
        arm=arm,
        model=model,
        hops=hops,
        seed=seed,
        val_auprc=val_auprc,
        notes=notes,
    )
    return oc.write(outcomes, meta, out_dir)


def run_tabfm(
    task: Task,
    features: str,
    hops: int,
    hub_cap: int,
    context_size: int,
    seed: int,
    out_dir: Path,
    make: MakeScorer,
) -> Path:
    feats = ARMS[features]
    part = task.partition()
    y = task.items["label"].to_numpy(np.int8)
    masks = {p: (part == p) & (y >= 0) for p in PARTITIONS}
    check_training_rows(task.items["split_group"].to_numpy()[masks["train"]], task.split, task.name)
    tables = {p: feats(task, p, hops, hub_cap) for p in PARTITIONS}
    groups = task.items["split_group"].to_numpy()
    scorer = make(context_size, seed).fit(
        tables["train"][masks["train"]].reset_index(drop=True),
        y[masks["train"]],
        groups[masks["train"]],
    )
    val = scorer.predict(tables["val"][masks["val"]].reset_index(drop=True))
    test_rows = np.flatnonzero(masks["test"]).astype(np.int64)
    # Scoring is timed alone (no table building, no fit), for TabFM's cost per thousand scores;
    # the time is read only from a run on the rented machine (docs/models.md).
    started = time.perf_counter()
    scores = scorer.predict(tables["test"][masks["test"]].reset_index(drop=True))
    predict_s = time.perf_counter() - started
    return _write(
        task,
        test_rows,
        scores,
        f"TabFM-{features}",
        "tabfm",
        hops,
        hub_cap,
        seed,
        _auprc(y[masks["val"]], val),
        f"context_size={context_size}; features={features}; "
        f"test_rows={test_rows.size}; test_predict_s={predict_s:.1f}",
        out_dir,
    )


def run_context_swap(
    task: Task,
    features: str,
    hops: int,
    hub_cap: int,
    context_size: int,
    window: int,
    seed: int,
    out_dir: Path,
    make: MakeScorer,
) -> tuple[Path, Path]:
    """TabFM with a sliding context, and XGBoost retrained on the same rows, step by step."""
    feats = ARMS[features]
    y = task.items["label"].to_numpy(np.int8)
    step = task.items["split_group"].to_numpy()
    lab = y >= 0
    # Every row's features from the table of the partition it belongs to.
    part = task.partition()
    table = pd.concat([feats(task, p, hops, hub_cap)[part == p] for p in PARTITIONS]).sort_index()
    test_steps = sorted(task.split.groups["test"])
    out = {"tabfm": np.full(len(y), np.nan), "xgb": np.full(len(y), np.nan)}
    for s, ctx_steps in context_swap_schedule(test_steps, window).items():
        ctx = np.flatnonzero(lab & np.isin(step, ctx_steps))
        query = np.flatnonzero(lab & (step == s))
        if query.size == 0:
            continue
        check_strictly_before(step[ctx], s)
        # The same rows for both: TabFM's context sample, which keeps every positive.
        keep = ctx[context_sample(y[ctx], step[ctx], context_size, seed)]
        Xc, yc = table.iloc[keep].reset_index(drop=True), y[keep]
        Xq = table.iloc[query].reset_index(drop=True)
        out["tabfm"][query] = make(context_size, seed).fit(Xc, yc, step[keep]).predict(Xq)
        # XGBoost retrained on those rows; the latest context step is its early-stopping set.
        last = step[keep] == max(ctx_steps)
        if not (yc[last] == 1).any() or not (yc[~last] == 1).any():
            last = np.ones(len(yc), dtype=bool)  # no positive to stop on: stop on the context
            fitted = boost.fit(Xc, yc, Xc, yc, seed=seed)
        else:
            fitted = boost.fit(Xc[~last], yc[~last], Xc[last], yc[last], seed=seed)
        out["xgb"][query] = fitted.predict(Xq)
    rows = np.flatnonzero(lab & np.isin(step, test_steps) & np.isfinite(out["tabfm"]))
    notes = f"context_size={context_size}; window={window}; features={features}"
    a = _write(
        task,
        rows,
        out["tabfm"][rows],
        f"TabFM-{features}-swap",
        "tabfm",
        hops,
        hub_cap,
        seed,
        None,
        notes,
        out_dir,
    )
    b = _write(
        task,
        rows,
        out["xgb"][rows],
        f"{features}-retrain",
        "xgboost",
        hops,
        hub_cap,
        seed,
        None,
        notes,
        out_dir,
    )
    return a, b
