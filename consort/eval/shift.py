"""The shift table: one model's test outcomes read before and after a regime change."""

from __future__ import annotations

import numpy as np

from consort.data.splits import Split
from consort.eval.metrics import Estimate, evaluate
from consort.eval.outcomes import Outcomes


def by_slice(
    o: Outcomes, split: Split, n_boot: int = 1000, seed: int = 0
) -> dict[str, dict[str, Estimate]]:
    """Every metric on each named test slice. Requires the outcomes' groups to be the split's
    groups (Elliptic: both are the time step)."""
    if not split.test_slices:
        raise ValueError(f"split {split.name} has no test slices")
    out: dict[str, dict[str, Estimate]] = {}
    for name, groups in split.test_slices.items():
        mask = np.isin(o.group, groups)
        if not mask.any():
            raise ValueError(f"no outcomes fall in slice {name}")
        out[name] = evaluate(o.subset(mask), n_boot=n_boot, seed=seed)
    return out
