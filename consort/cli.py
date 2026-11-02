"""consort: data fetch | data check | train | eval | patterns."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import typer

from consort import env
from consort.data import fetch as fetch_mod

if TYPE_CHECKING:
    from consort.data.tasks import BootGroup, FeatureView, Task

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data", help="Fetch and check the datasets.")

RUNS = Path("data/runs")


def _task(dataset: str, boot_group: str = "hour", view: str = "prior-days") -> Task:
    from consort.data import elliptic, ibm
    from consort.data.tasks import elliptic_task, ibm_task

    if boot_group not in ("hour", "day", "component"):
        raise typer.BadParameter(f"unknown bootstrap group {boot_group}")
    if view not in ("prior-days", "window"):
        raise typer.BadParameter(f"unknown feature view {view}")

    if dataset == "elliptic":
        return elliptic_task(elliptic.load(fetch_mod.RAW / "elliptic"))
    if dataset.startswith("ibm-"):
        tier = dataset.removeprefix("ibm-")
        data = ibm.load(fetch_mod.RAW / dataset, tier)
        return ibm_task(
            data, boot_group=cast("BootGroup", boot_group), view=cast("FeatureView", view)
        )
    raise typer.BadParameter(f"unknown dataset {dataset}")


@data_app.command("fetch")
def data_fetch(
    source: str = typer.Argument(..., help=f"One of {sorted(fetch_mod.SOURCES)}"),
) -> None:
    """Download with the reader's Kaggle token and verify against the committed manifest."""
    if source not in fetch_mod.SOURCES:
        raise typer.BadParameter(f"unknown source {source}")
    env.load()
    root = fetch_mod.fetch(source)
    typer.echo(f"{source}: verified in {root}")


@data_app.command("check")
def data_check(dataset: str) -> None:
    """Load a fetched dataset, check its published counts, and print its split."""
    task = _task(dataset)
    part = task.partition()
    lab = task.labelled()
    summary = {
        "dataset": task.name,
        "items": len(task.items),
        "labelled": int(lab.sum()),
        "positives": int((task.items["label"] == 1).sum()),
        "split": task.split.model_dump(),
        "partition_items": {p: int(((part == p) & lab).sum()) for p in ("train", "val", "test")},
        "partition_positives": {
            p: int(((part == p) & (task.items["label"] == 1)).sum())
            for p in ("train", "val", "test")
        },
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def patterns(tier: str = "HI-Small") -> None:
    """Classify every generator attempt by shape and print the agreement table."""
    from consort.data import ibm
    from consort.graph.patterns import classify

    root = fetch_mod.RAW / f"ibm-{tier}"
    blocks = ibm.read_patterns(root / f"{tier}_Patterns.txt")
    table: dict[str, Counter[str]] = {}
    for b in blocks:
        src = [f"{int(r[1])}:{r[2]}" for r in b.rows]
        dst = [f"{int(r[3])}:{r[4]}" for r in b.rows]
        order = sorted(range(len(b.rows)), key=lambda i: b.rows[i][0])
        shape = classify([src[i] for i in order], [dst[i] for i in order])
        table.setdefault(b.pattern, Counter())[shape] += 1
    typer.echo("| Pattern | Attempts | Classified as named | Most common other shape |")
    typer.echo("|---|---|---|---|")
    for name, counts in sorted(table.items()):
        total = sum(counts.values())
        same = counts.get(name, 0)
        others = [(s, c) for s, c in counts.most_common() if s != name]
        other = f"{others[0][0]} ({others[0][1]})" if others else "none"
        typer.echo(f"| {name} | {total} | {same} ({same / total:.0%}) | {other} |")


@app.command()
def train(
    arm: str = typer.Argument(..., help="R or G"),
    dataset: str = typer.Option(..., help="elliptic, ibm-HI-Small, ibm-LI-Small"),
    hops: int = 0,
    hub_cap: int = 50,
    seed: int = 0,
    split: str = typer.Option("temporal", help="temporal, or random (run once, for the gap)"),
    boot_group: str = "hour",
    out: Path = RUNS,
    onnx: bool = typer.Option(False, help="Also export the booster to checkpoints/ as ONNX"),
    feature_view: str = typer.Option(
        "prior-days",
        help="IBM: prior-days (each transfer sees earlier days only), or window (sees its "
        "whole window, later days included; run once, to publish the gap)",
    ),
) -> None:
    """Train arm R or G and write its test outcomes."""
    from consort.models.run import SplitKind, run_tabular

    task = _task(dataset, boot_group, feature_view)
    if split not in ("temporal", "random"):
        raise typer.BadParameter(f"unknown split {split}")
    kind = cast(SplitKind, split)
    path = run_tabular(
        task,
        arm,
        hops,
        hub_cap if hops else 0,
        seed,
        out,
        split_kind=kind,
        arm_label=f"{arm}-window" if feature_view == "window" and arm == "G" else None,
        onnx_dir=Path("checkpoints") if onnx else None,
    )
    typer.echo(str(path))


@app.command("train-gnn")
def train_gnn(
    dataset: str = typer.Option(..., help="elliptic, ibm-HI-Small, ibm-LI-Small"),
    arch: str = "sage",
    hops: int = 2,
    hub_cap: int = 25,
    seed: int = 0,
    epochs: int = 40,
    batch_size: int = 2048,
    reverse_mp: bool = False,
    ego_ids: bool = False,
    boot_group: str = "hour",
    out: Path = RUNS,
) -> None:
    """Train arm N (needs the gnn extra) and write its test outcomes."""
    from consort.models.gnn import GNNConfig, run

    if arch not in ("sage", "gat"):
        raise typer.BadParameter(f"unknown architecture {arch}")
    cfg = GNNConfig(
        arch="sage" if arch == "sage" else "gat",
        hops=hops,
        cap=hub_cap,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        reverse_mp=reverse_mp,
        ego_ids=ego_ids,
    )
    typer.echo(str(run(_task(dataset, boot_group), cfg, out)))


@app.command("eval")
def eval_cmd(
    outcomes: list[Path],
    against: Path | None = typer.Option(None, help="Baseline outcomes for a paired delta"),
    n_boot: int = 1000,
    slices: bool = typer.Option(
        False, help="Also read each named test slice (loads the dataset to find item groups)"
    ),
) -> None:
    """Metrics with grouped bootstrap intervals; with --against, paired deltas."""
    from consort.eval import outcomes as oc
    from consort.eval.metrics import evaluate, paired_delta
    from consort.eval.shift import by_slice

    base = oc.read(against) if against else None
    tasks: dict[str, Task] = {}
    for path in outcomes:
        o = oc.read(path)
        meta = oc.read_meta(path)
        typer.echo(f"{meta.run} ({len(o)} items)")
        for k, est in evaluate(o, n_boot=n_boot).items():
            typer.echo(f"  {k}: {est.as_text()}")
        if base is not None:
            for k, est in paired_delta(base, o, n_boot=n_boot).items():
                typer.echo(f"  delta {k}: {est.as_text()}")
        if slices:
            if meta.dataset not in tasks:
                tasks[meta.dataset] = _task(meta.dataset)
            task = tasks[meta.dataset]
            group = dict(zip(task.items["item_id"], task.items["split_group"].astype(int), strict=True))
            for name, ests in by_slice(o, task.split, n_boot, split_group=group).items():
                for k, est in ests.items():
                    typer.echo(f"  [{name}] {k}: {est.as_text()}")
                if base is not None:
                    groups = task.split.test_slices[name]
                    in_o = np.isin([group[i] for i in o.item_id.tolist()], groups)
                    in_b = np.isin([group[i] for i in base.item_id.tolist()], groups)
                    delta = paired_delta(base.subset(in_b), o.subset(in_o), n_boot=n_boot)
                    for k, est in delta.items():
                        typer.echo(f"  [{name}] delta {k}: {est.as_text()}")


@app.command()
def watchdog(
    self_pod: bool = typer.Option(
        False, "--self", help="On a pod: watch only this pod (RUNPOD_POD_ID) and delete it"
    ),
    idle_minutes: float = 30.0,
    max_hours: float = 8.0,
    poll_seconds: float = 60.0,
) -> None:
    """Delete this project's RunPod pods (names starting consort-) when idle or too old."""
    import os

    from consort.ops import watchdog as wd

    env.load()
    self_id = None
    if self_pod:
        self_id = os.environ.get("RUNPOD_POD_ID")
        if not self_id:
            raise typer.BadParameter("--self needs RUNPOD_POD_ID, which RunPod sets on a pod")
    policy = wd.Policy(idle_minutes=idle_minutes, max_hours=max_hours)
    wd.watch(wd.RunPod(wd.key_from_env()), policy, self_id=self_id, poll_s=poll_seconds)


def main() -> None:
    app()
