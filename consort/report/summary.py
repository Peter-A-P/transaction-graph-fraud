"""Run summaries: every outcomes file read once into metrics with intervals, its test slices,
its paired delta against its baseline and, on IBM, recall by pattern.

A summary is cached under data/report/ by the outcomes file's SHA-256, so the bootstrap runs
once per run; results/summary.json collects them all and is what the README tables are built
from. Baselines: arm R for every arm on the same dataset, split and seed; the window-view run
against the prior-days run it was the leak of.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from consort.data.splits import Split
from consort.eval import outcomes as oc
from consort.eval.bypattern import recall_by_pattern
from consort.eval.metrics import Estimate, evaluate, paired_delta
from consort.eval.shift import by_slice

VERSION = 1  # bump when what a summary holds changes, to invalidate the cache
Triple = tuple[float, float, float]


class RunSummary(BaseModel):
    run: str
    dataset: str
    split: str
    kind: str  # temporal or random
    arm: str
    model: str
    hops: int
    cap: int
    seed: int
    val_auprc: float | None
    n_items: int
    n_groups: int
    n_boot: int
    metrics: dict[str, Triple]
    slices: dict[str, dict[str, Triple]] = {}
    baseline: str | None = None
    delta: dict[str, Triple] = {}
    slice_delta: dict[str, dict[str, Triple]] = {}
    patterns: dict[str, Triple] = {}


def _t(e: Estimate) -> Triple:
    return (round(e.point, 4), round(e.lo, 4), round(e.hi, 4))


def _ts(d: Mapping[str, Estimate]) -> dict[str, Triple]:
    return {k: _t(v) for k, v in d.items()}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cap(run: str) -> int:
    m = re.search(r"\.cap(\d+)\.", run)
    return int(m.group(1)) if m else 0


def baseline_of(meta: oc.RunMeta, available: Mapping[str, oc.RunMeta]) -> str | None:
    """The run this one is paired against, if it was run."""
    kind = "random" if meta.split.endswith("-random") else "temporal"
    if meta.arm == "R":
        return None
    if meta.arm.endswith("-window"):
        want = meta.run.replace(f".{meta.arm}.", f".{meta.arm.removesuffix('-window')}.")
    else:
        want = f"{meta.dataset}.{kind}.R.h0.cap0.s{meta.seed}"
    return want if want in available else None


class Context:
    """What a dataset contributes to its runs' summaries: its split, its items' split groups
    when the bootstrap groups are not the split's, and its pattern names."""

    def __init__(
        self,
        split: Split,
        split_group: Mapping[str, int] | None = None,
        pattern_of: Mapping[str, str] | None = None,
    ) -> None:
        self.split = split
        self.split_group = split_group
        self.pattern_of = pattern_of


def summarise(
    path: Path,
    metas: Mapping[str, oc.RunMeta],
    context: Callable[[str], Context | None],
    cache: Path,
    n_boot: int = 1000,
) -> RunSummary:
    meta = oc.read_meta(path)
    key = cache / f"{meta.run}.{_sha(path)[:16]}.v{VERSION}.b{n_boot}.json"
    base_name = baseline_of(meta, metas)
    if key.exists():
        cached = RunSummary.model_validate_json(key.read_text())
        if cached.baseline == base_name:
            return cached
    o = oc.read(path)
    kind = "random" if meta.split.endswith("-random") else "temporal"
    est = evaluate(o, n_boot=n_boot)
    s = RunSummary(
        run=meta.run,
        dataset=meta.dataset,
        split=meta.split,
        kind=kind,
        arm=meta.arm,
        model=meta.model,
        hops=meta.hops,
        cap=_cap(meta.run),
        seed=meta.seed,
        val_auprc=meta.val_auprc,
        n_items=len(o),
        n_groups=next(iter(est.values())).n_groups,
        n_boot=n_boot,
        metrics=_ts(est),
        baseline=base_name,
    )
    base = oc.read(path.with_name(f"{base_name}.outcomes.parquet")) if base_name else None
    if base is not None:
        s.delta = _ts(paired_delta(base, o, n_boot=n_boot))
    ctx = context(meta.dataset) if kind == "temporal" else None
    if ctx is not None and ctx.split.test_slices:
        for name, ests in by_slice(o, ctx.split, n_boot, split_group=ctx.split_group).items():
            s.slices[name] = _ts(ests)
            if base is not None:
                groups = ctx.split.test_slices[name]

                def inside(x: oc.Outcomes, groups: list[int] = groups) -> npt.NDArray[np.bool_]:
                    g = (
                        x.group
                        if ctx is None or ctx.split_group is None
                        else np.array([ctx.split_group[i] for i in x.item_id.tolist()])
                    )
                    return np.isin(g, groups)

                d = paired_delta(base.subset(inside(base)), o.subset(inside(o)), n_boot=n_boot)
                s.slice_delta[name] = _ts(d)
    if ctx is not None and ctx.pattern_of is not None:
        import pandas as pd

        series = pd.Series(ctx.pattern_of)
        s.patterns = _ts(recall_by_pattern(o, series, budget=0.01, n_boot=n_boot))
    cache.mkdir(parents=True, exist_ok=True)
    key.write_text(s.model_dump_json(indent=1))
    return s


def write_all(summaries: list[RunSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [json.loads(s.model_dump_json()) for s in sorted(summaries, key=lambda s: s.run)]
    path.write_text(json.dumps(data, indent=1) + "\n")


def read_all(path: Path) -> list[RunSummary]:
    return [RunSummary.model_validate(d) for d in json.loads(path.read_text())]
