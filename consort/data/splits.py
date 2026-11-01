"""Temporal, inductive splits; the shift split on Elliptic; the random split, run once.

A split assigns every item to a partition by its group (an Elliptic time step, an IBM time
bucket), never by the item itself, so that a test step is wholly absent from training. The
random split is the exception, and exists only to publish the gap it opens.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, model_validator

Partition = Literal["train", "val", "test"]
PARTITIONS: tuple[Partition, ...] = ("train", "val", "test")


class Split(BaseModel):
    """A named assignment of groups to partitions, with the kind of split it is."""

    name: str
    kind: Literal["temporal", "random"]
    # For a temporal split: the group values in each partition. For a random split these are
    # empty and the assignment is per item (see random_split).
    groups: dict[Partition, list[int]] = Field(default_factory=dict)
    # Test partitions may be reported in named slices (the shift table's before and after).
    test_slices: dict[str, list[int]] = Field(default_factory=dict)
    seed: int | None = None

    @model_validator(mode="after")
    def _disjoint(self) -> Split:
        seen: dict[int, Partition] = {}
        for part, values in self.groups.items():
            for v in values:
                if v in seen:
                    raise ValueError(f"group {v} is in both {seen[v]} and {part}")
                seen[v] = part
        test = set(self.groups.get("test", []))
        for name, values in self.test_slices.items():
            if not set(values) <= test:
                raise ValueError(f"test slice {name} holds groups outside the test partition")
        if self.kind == "temporal":
            train, val = self.groups.get("train", []), self.groups.get("val", [])
            later = val + list(test)
            if train and later and max(train) >= min(later):
                raise ValueError("a temporal split trains on a group at or after a test group")
        return self

    def assign(self, groups: npt.ArrayLike) -> npt.NDArray[np.str_]:
        """The partition of each item, by its group; '' for a group in no partition."""
        if self.kind != "temporal":
            raise TypeError("a random split assigns items, not groups; use random_split")
        g = np.asarray(groups)
        out = np.full(g.shape, "", dtype="<U5")
        for part, values in self.groups.items():
            out[np.isin(g, values)] = part
        return out


def steps(first: int, last: int) -> list[int]:
    """Inclusive range of time steps."""
    return list(range(first, last + 1))


def temporal_split(
    name: str,
    train: Iterable[int],
    val: Iterable[int],
    test: Iterable[int],
    test_slices: dict[str, list[int]] | None = None,
) -> Split:
    return Split(
        name=name,
        kind="temporal",
        groups={"train": list(train), "val": list(val), "test": list(test)},
        test_slices=test_slices or {},
    )


# Elliptic: one trained model per arm serves both the headline and the shift table. Train on
# steps 1 to 34, validate on 35 to 38, test on 39 to 49; the shift table reads the test
# partition in two slices, before the dark-market shutdown at step 43 and from it on.
ELLIPTIC_TEMPORAL = temporal_split(
    "elliptic-temporal",
    train=steps(1, 34),
    val=steps(35, 38),
    test=steps(39, 49),
    test_slices={"before-43": steps(39, 42), "from-43": steps(43, 49)},
)


def random_split(
    n_items: int, seed: int, fractions: tuple[float, float, float] = (0.7, 0.1, 0.2)
) -> npt.NDArray[np.str_]:
    """The random node split most papers use. Run once per dataset, to publish the gap."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to one")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    cut1 = round(fractions[0] * n_items)
    cut2 = cut1 + round(fractions[1] * n_items)
    out = np.empty(n_items, dtype="<U5")
    out[order[:cut1]] = "train"
    out[order[cut1:cut2]] = "val"
    out[order[cut2:]] = "test"
    return out


def time_buckets(
    timestamps: npt.NDArray[np.datetime64], bucket: np.timedelta64
) -> npt.NDArray[np.int64]:
    """Integer bucket index from the earliest timestamp, for datasets with continuous time."""
    t0 = timestamps.min()
    return ((timestamps - t0) // bucket).astype(np.int64)
