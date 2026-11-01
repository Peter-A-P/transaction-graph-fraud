"""The feature registry: a feature is a declared derivation, and it cannot see a label.

Every graph feature is registered with the node columns it reads. Registration refuses a
label column or a label-named feature (consort.data.leakage). At compute time the derivation
receives a FeatureContext holding only its declared columns, built from a node table that has
already had every label column removed, so a feature that tried to read a label would fail
rather than leak. The neighbour-label-rate feature therefore cannot be written here; it exists
only in consort.eval.leak, outside the registry, for the rejected-approach experiment.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from consort.data.leakage import LABEL_COLUMNS, LabelLeak, check_feature_inputs, label_word
from consort.graph.build import Graph

FloatArray = npt.NDArray[np.float64]


class FeatureContext:
    """What a derivation may see: the graph's structure and edge attributes, its hop budget
    and hub cap, and the node columns it declared. Nothing else."""

    def __init__(
        self, graph: Graph, columns: Mapping[str, FloatArray], hops: int, hub_cap: int
    ) -> None:
        self.graph = graph
        self._columns = dict(columns)
        self.hops = hops
        self.hub_cap = hub_cap

    def column(self, name: str) -> FloatArray:
        if name not in self._columns:
            raise LabelLeak(f"undeclared column {name!r} read by a feature derivation")
        return self._columns[name]


Derivation = Callable[[FeatureContext], Mapping[str, FloatArray]]


@dataclass(frozen=True)
class FeatureSpec:
    """One derivation. It may emit several named columns (one per aggregate, say); every
    emitted name is checked as it is produced."""

    name: str
    inputs: tuple[str, ...]
    min_hops: int
    derive: Derivation
    description: str


@dataclass
class FeatureRegistry:
    specs: dict[str, FeatureSpec] = field(default_factory=dict)

    def register(self, spec: FeatureSpec) -> FeatureSpec:
        check_feature_inputs(spec.name, spec.inputs)
        if spec.name in self.specs:
            raise ValueError(f"feature {spec.name!r} registered twice")
        self.specs[spec.name] = spec
        return spec

    def __iter__(self) -> Iterator[FeatureSpec]:
        return iter(self.specs.values())

    def compute(self, graph: Graph, nodes: pd.DataFrame, hops: int, hub_cap: int) -> pd.DataFrame:
        """Every registered feature whose minimum hop count is within budget, one row per node
        in graph order. Hop 0 computes nothing: the row features are arm R's, not these."""
        safe = nodes.drop(columns=[c for c in nodes.columns if c in LABEL_COLUMNS])
        out: dict[str, FloatArray] = {}
        for spec in self.specs.values():
            if spec.min_hops > hops:
                continue
            cols = {c: safe[c].to_numpy(dtype=np.float64) for c in spec.inputs}
            ctx = FeatureContext(graph, cols, hops=hops, hub_cap=hub_cap)
            for col, values in spec.derive(ctx).items():
                word = label_word(col)
                if word is not None:
                    raise LabelLeak(f"feature {spec.name!r} emitted label-named column {col!r}")
                if values.shape != (graph.n_nodes,):
                    raise ValueError(f"feature column {col!r} has shape {values.shape}")
                if col in out:
                    raise ValueError(f"feature column {col!r} emitted twice")
                out[col] = values
        return pd.DataFrame(out)
