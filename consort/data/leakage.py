"""The build-time leakage checks.

Two ways a graph fraud number gets too good: a test-step row reaches a training table, or a
feature reads a label (its own, or a neighbour's, which is the same thing one hop away). The
first is checked on every table a model is fitted on. The second is refused when a feature is
registered (consort.graph.registry), and checked again here on the finished table's columns.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from consort.data.splits import Split

# Every column, in any loader, that is or encodes the target. A feature may not declare one
# as an input, and a feature table may not carry one except as the declared label column.
LABEL_COLUMNS: frozenset[str] = frozenset(
    {"label", "class", "is_laundering", "Is Laundering", "illicit", "pattern", "pattern_id"}
)

# A feature whose name says it is about labels is refused whatever it declares, so that a
# neighbour-label-rate feature cannot be registered under an innocent-looking derivation.
LABEL_WORDS: tuple[str, ...] = ("label", "illicit", "licit", "launder", "fraud", "class", "pattern")


class SplitLeak(Exception):
    """A training table holds a row from a validation or test group."""


class LabelLeak(Exception):
    """A feature reads, or is named for, a label."""


def check_training_rows(row_groups: npt.ArrayLike, split: Split, table: str) -> None:
    """Refuse a training table holding any row whose group is outside the train partition."""
    g = np.asarray(row_groups)
    allowed = set(split.groups.get("train", []))
    bad = np.unique(g[~np.isin(g, list(allowed))])
    if bad.size:
        shown = ", ".join(str(v) for v in bad[:10])
        raise SplitLeak(f"{table}: training rows from groups outside train ({shown})")


def check_random_training_rows(row_partition: npt.ArrayLike, table: str) -> None:
    """The same refusal for the random split, whose assignment is per item."""
    p = np.asarray(row_partition)
    if np.any(p != "train"):
        raise SplitLeak(f"{table}: {int(np.sum(p != 'train'))} non-train rows in a training table")


def label_word(name: str) -> str | None:
    lowered = name.lower()
    return next((w for w in LABEL_WORDS if w in lowered), None)


def check_feature_inputs(feature: str, inputs: Iterable[str]) -> None:
    """Refuse a feature that declares a label column as an input, or is named for a label."""
    touched = sorted(set(inputs) & LABEL_COLUMNS)
    if touched:
        raise LabelLeak(f"feature {feature!r} reads label column(s) {touched}")
    word = label_word(feature)
    if word is not None:
        raise LabelLeak(f"feature {feature!r} is named for a label ({word!r})")


def check_feature_columns(columns: Iterable[str], label_column: str) -> None:
    """Refuse a feature table whose columns, the label aside, include a label column."""
    for c in columns:
        if c == label_column:
            continue
        if c in LABEL_COLUMNS or label_word(c) is not None:
            raise LabelLeak(f"feature table carries label-like column {c!r}")
