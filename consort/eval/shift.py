"""Test slices: one model's test outcomes read on named parts of the test partition (Elliptic's
before and after the shutdown at step 43; IBM without the generator's tail days)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from consort.data.splits import Split
from consort.eval.metrics import Estimate, evaluate
from consort.eval.outcomes import Outcomes


def by_slice(
    o: Outcomes,
    split: Split,
    n_boot: int = 1000,
    seed: int = 0,
    split_group: Mapping[str, int] | None = None,
) -> dict[str, dict[str, Estimate]]:
    """Every metric on each named test slice. An item's split group is looked up in
    `split_group` (item id to group) when given; otherwise the outcomes' bootstrap groups must
    be the split's groups (Elliptic: both are the time step)."""
    if not split.test_slices:
        raise ValueError(f"split {split.name} has no test slices")
    if split_group is None:
        groups_of = np.asarray(o.group)
    else:
        groups_of = np.array([split_group[i] for i in o.item_id.tolist()])
    out: dict[str, dict[str, Estimate]] = {}
    for name, groups in split.test_slices.items():
        mask = np.isin(groups_of, groups)
        if not mask.any():
            raise ValueError(f"no outcomes fall in slice {name}")
        out[name] = evaluate(o.subset(mask), n_boot=n_boot, seed=seed)
    return out
