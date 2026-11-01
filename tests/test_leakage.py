"""The leakage test. Written before the first feature, and run on every build.

It fails if a training table holds a row from a validation or test step, if a feature can be
registered that reads a label, and if any feature in the default registry, computed on a toy
graph whose labels are planted to be perfectly predictable from a neighbour's label, carries
that signal.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from consort.data import ibm
from consort.data.leakage import (
    LabelLeak,
    SplitLeak,
    check_feature_columns,
    check_random_training_rows,
    check_training_rows,
)
from consort.data.splits import ELLIPTIC_TEMPORAL, random_split, temporal_split
from consort.data.tasks import IBMTask, ibm_task
from consort.graph.build import from_edges
from consort.graph.registry import FeatureContext, FeatureRegistry, FeatureSpec
from tests.toy import write_ibm

FloatArray = npt.NDArray[np.float64]


def test_training_table_with_a_test_step_row_is_refused() -> None:
    steps = np.array([1, 2, 3, 34, 39])
    with pytest.raises(SplitLeak, match="39"):
        check_training_rows(steps, ELLIPTIC_TEMPORAL, "arm-r-train")


def test_training_table_with_a_validation_step_row_is_refused() -> None:
    with pytest.raises(SplitLeak, match="35"):
        check_training_rows(np.array([10, 35]), ELLIPTIC_TEMPORAL, "arm-r-train")


def test_clean_training_table_passes() -> None:
    check_training_rows(np.arange(1, 35), ELLIPTIC_TEMPORAL, "arm-r-train")


def test_temporal_split_refuses_training_after_test() -> None:
    with pytest.raises(ValueError, match="trains on a group at or after"):
        temporal_split("bad", train=[1, 2, 10], val=[5], test=[6])


def test_temporal_split_refuses_overlap() -> None:
    with pytest.raises(ValueError, match="both"):
        temporal_split("bad", train=[1, 2], val=[2], test=[3])


def test_elliptic_shift_slices_cover_the_test_partition() -> None:
    slices = ELLIPTIC_TEMPORAL.test_slices
    assert sorted(slices["before-43"] + slices["from-43"]) == ELLIPTIC_TEMPORAL.groups["test"]
    assert min(slices["from-43"]) == 43


def test_random_split_training_rows_guard() -> None:
    part = random_split(100, seed=7)
    check_random_training_rows(part[part == "train"], "random-train")
    with pytest.raises(SplitLeak):
        check_random_training_rows(part, "random-train")


def _noop(ctx: FeatureContext) -> Mapping[str, FloatArray]:
    return {}


@pytest.mark.parametrize("label_col", ["label", "class", "Is Laundering", "is_laundering"])
def test_registry_refuses_a_feature_reading_a_label(label_col: str) -> None:
    reg = FeatureRegistry()
    with pytest.raises(LabelLeak):
        reg.register(FeatureSpec("innocent_mean", ("amount", label_col), 1, _noop, ""))


def test_registry_refuses_a_neighbour_label_rate_under_any_name_with_label_words() -> None:
    reg = FeatureRegistry()
    for name in ("neighbour_illicit_rate", "nbr_label_mean", "laundering_share"):
        with pytest.raises(LabelLeak):
            reg.register(FeatureSpec(name, ("amount",), 1, _noop, ""))


def test_a_derivation_cannot_read_an_undeclared_column() -> None:
    """Declaring only 'amount' and reading 'label' fails at compute time, and the label is
    not in the context at all, because the registry drops label columns first."""

    def sneaky(ctx: FeatureContext) -> Mapping[str, FloatArray]:
        return {"sneaky": ctx.column("label")}

    reg = FeatureRegistry()
    reg.register(FeatureSpec("sneaky", ("amount",), 1, sneaky, ""))
    g = from_edges(3, [0, 1], [1, 2])
    nodes = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "label": [0, 1, 0]})
    with pytest.raises(LabelLeak, match="undeclared"):
        reg.compute(g, nodes, hops=1, hub_cap=0)


def test_a_derivation_cannot_emit_a_label_named_column() -> None:
    def emits(ctx: FeatureContext) -> Mapping[str, FloatArray]:
        return {"nbr_illicit": np.zeros(ctx.graph.n_nodes)}

    reg = FeatureRegistry()
    reg.register(FeatureSpec("ok_name", ("amount",), 1, emits, ""))
    g = from_edges(2, [0], [1])
    with pytest.raises(LabelLeak, match="emitted"):
        reg.compute(g, pd.DataFrame({"amount": [1.0, 2.0]}), hops=1, hub_cap=0)


def test_feature_table_with_a_label_column_is_refused() -> None:
    check_feature_columns(["deg_in", "amount_out_sum", "label"], label_column="label")
    with pytest.raises(LabelLeak):
        check_feature_columns(["deg_in", "class", "label"], label_column="label")


def test_default_registry_carries_no_neighbour_label_signal() -> None:
    """A star of pairs: node 2k's label is a coin flip and node 2k+1 copies it, the two joined
    by an edge, with identical row features everywhere. Any feature that reads a neighbour's
    label would separate the classes; every real feature must be constant within each label
    class's distribution, so a perfect split is impossible."""
    from consort.graph.features import default_registry

    rng = np.random.default_rng(0)
    n_pairs = 200
    labels = np.repeat(rng.integers(0, 2, n_pairs), 2)
    src = np.arange(0, 2 * n_pairs, 2)
    g = from_edges(2 * n_pairs, src, src + 1, amount=np.ones(n_pairs))
    nodes = pd.DataFrame({"f0": np.ones(2 * n_pairs), "label": labels})
    table = default_registry(["f0"]).compute(g, nodes, hops=3, hub_cap=10)
    assert len(table.columns) > 0
    check_feature_columns(table.columns, label_column="label")
    for col in table.columns:
        values = table[col].to_numpy()
        # The pair structure is the same for every pair, so the only per-node difference is
        # sender against receiver, which is independent of the label.
        for role in (0, 1):
            v = values[role::2]
            assert np.allclose(v, v[0]), f"{col} varies across identical pairs"


def _ibm_with_late_burst(tmp_path: Path, burst: bool) -> IBMTask:
    """The toy set, optionally with a burst of transfers on its last day among accounts that
    traded earlier; a feature view that sees the future changes earlier transfers' features."""
    root = tmp_path / ("burst" if burst else "plain")
    root.mkdir()
    write_ibm(root, n_background=1500, n_cycles=10)
    if burst:
        with (root / "T_Trans.csv").open("a") as f:
            for i in range(60):
                a, b = i % 10, (i * 3 + 1) % 10
                f.write(
                    f"2022/09/20 23:{i % 60:02d},{a % 7 + 1},{a:08X},{b % 7 + 1},{b:08X},"
                    "500.00,US Dollar,500.00,US Dollar,ACH,0\n"
                )
    return ibm_task(ibm.load(root, "T"), boot_group="day")


@pytest.mark.parametrize("view", ["prior-days", "window"])
def test_ibm_graph_features_do_not_see_later_days(tmp_path: Path, view: str) -> None:
    plain, burst = _ibm_with_late_burst(tmp_path, False), _ibm_with_late_burst(tmp_path, True)
    assert plain.split == burst.split
    last = int(burst.items["split_group"].max())
    plain.view = burst.view = view  # type: ignore[assignment]
    a = plain.graph_features("test", 1, 50)
    b = burst.graph_features("test", 1, 50).iloc[: len(a)]
    test_rows = (plain.partition() == "test") & (plain.items["split_group"] < last).to_numpy()
    same = np.allclose(a[test_rows].to_numpy(), b[test_rows].to_numpy(), equal_nan=True)
    # The default view is causal; the window view is the leak it replaced, and this test must
    # be able to see it.
    assert same == (view == "prior-days")
