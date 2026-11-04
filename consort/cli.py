"""consort: data fetch | data check | train | eval | patterns."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import typer

from consort import env
from consort.data import fetch as fetch_mod

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

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
    if dataset == "ellipticpp":
        from consort.data import ellipticpp
        from consort.data.tasks import ellipticpp_task

        return ellipticpp_task(
            ellipticpp.load(fetch_mod.RAW / "ellipticpp", fetch_mod.RAW / "ellipticpp-txs")
        )
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


def _nlr(task: Task, visible: npt.NDArray[np.bool_]) -> pd.DataFrame:
    from consort.eval.leak import neighbour_label_rate

    return neighbour_label_rate(task, visible)


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
    neighbour_label_rate: bool = typer.Option(
        False,
        help="Add the neighbour-label-rate feature, built outside the registry, for the "
        "rejected-approach experiment only (docs/rejected.md); never a headline run",
    ),
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
        extra=_nlr if neighbour_label_rate else None,
        arm_label=(
            f"{arm}+nlr"
            if neighbour_label_rate
            else f"{arm}-window"
            if feature_view == "window" and arm == "G"
            else None
        ),
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
    neg_share: float = typer.Option(1.0, help="IBM: share of licit transfers per epoch"),
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
        neg_share=neg_share,
    )
    typer.echo(str(run(_task(dataset, boot_group), cfg, out)))


@app.command("train-tabfm")
def train_tabfm(
    dataset: str = typer.Option(...),
    features: str = typer.Option("R", help="R or G: whose table TabFM is handed"),
    hops: int = 0,
    hub_cap: int = 0,
    context_size: int = 10_000,
    seed: int = 0,
    swap_window: int = typer.Option(
        0, help="With a window of n steps: the context swap instead (Elliptic, Elliptic++)"
    ),
    device: str = typer.Option("", help="cuda or cpu; empty lets TabFM choose"),
    out: Path = RUNS,
) -> None:
    """The TabFM column (needs the tabfm extra), or with --swap-window the context swap."""
    from consort.models.tabfm import TabFMScorer
    from consort.models.tabfm_run import run_context_swap, run_tabfm

    if features not in ("R", "G"):
        raise typer.BadParameter("features must be R or G")

    def make(size: int, seed: int) -> TabFMScorer:
        return TabFMScorer(context_size=size, seed=seed, device=device or None)

    task = _task(dataset)
    cap = hub_cap if hops else 0
    if swap_window:
        for p in run_context_swap(
            task, features, hops, cap, context_size, swap_window, seed, out, make
        ):
            typer.echo(str(p))
    else:
        typer.echo(str(run_tabfm(task, features, hops, cap, context_size, seed, out, make)))


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
            group = dict(
                zip(task.items["item_id"], task.items["split_group"].astype(int), strict=True)
            )
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
def report(
    runs: Path = RUNS,
    out: Path = Path("results/summary.json"),
    readme: Path = Path("README.md"),
    n_boot: int = 1000,
    tables_only: bool = typer.Option(False, help="Rebuild the tables from the summary file"),
) -> None:
    """Summarise every run (cached by its outcomes file) and fill the README's tables."""
    from consort.data.splits import ELLIPTIC_TEMPORAL
    from consort.eval import outcomes as oc
    from consort.report import summary as sm
    from consort.report import tables as tb

    if not tables_only:
        metas = {
            p.name.removesuffix(".meta.json"): oc.RunMeta.model_validate_json(p.read_text())
            for p in sorted(runs.glob("*.meta.json"))
        }
        contexts: dict[str, sm.Context | None] = {}

        def context(dataset: str) -> sm.Context | None:
            if dataset not in contexts:
                if dataset in ("elliptic", "ellipticpp"):
                    # Bootstrap groups are the split's groups: the step, or the first step.
                    contexts[dataset] = sm.Context(ELLIPTIC_TEMPORAL)
                elif dataset.startswith("ibm-"):
                    from consort.data.tasks import IBMTask

                    task = _task(dataset)
                    assert isinstance(task, IBMTask) and task.data is not None
                    ids = task.items["item_id"].tolist()
                    t = task.data.transfers
                    contexts[dataset] = sm.Context(
                        task.split,
                        split_group=dict(zip(ids, task.items["split_group"].tolist(), strict=True)),
                        pattern_of=dict(zip(ids, t["pattern"].tolist(), strict=True)),
                    )
                else:
                    contexts[dataset] = None
            return contexts[dataset]

        done = []
        for name in metas:
            typer.echo(f"summarising {name}")
            path = runs / f"{name}.outcomes.parquet"
            done.append(sm.summarise(path, metas, context, Path("data/report"), n_boot))
        sm.write_all(done, out)
    summaries = sm.read_all(out)

    serving = Path("results/serving.json")
    timed = json.loads(serving.read_text()) if serving.exists() else []
    timed = [t for t in timed if t["machine"]["reportable"]]  # never a laptop timing

    def fetch(dataset: str, hops: int) -> tuple[str, str]:
        # The hop curve's latency is the counters design's, the one a service would run.
        rows = [
            t
            for t in timed
            if t["dataset"] == dataset
            and t["hops"] == hops
            and t.get("centre", "history") == "counters"
        ]
        if not rows:
            # The real sets are never served (Elliptic is never drawn or redistributed, and
            # the benchmark is IBM's); at zero hops there is no neighbourhood to fetch.
            why = (
                "not served"
                if dataset.startswith("elliptic")
                else "no fetch"
                if hops == 0
                else "not measured"
            )
            return (why, why)
        p = {q: "; ".join(f"{t['store']} {t[q]:.1f}" for t in rows) for q in ("p50_ms", "p99_ms")}
        return (p["p50_ms"], p["p99_ms"])

    faith = Path("results/faithfulness.json")
    tables = {
        "faithfulness": tb.faithfulness_table(
            json.loads(faith.read_text()) if faith.exists() else []
        ),
        "main": tb.main_table(summaries),
        "patterns": tb.pattern_table(summaries),
        "hops": tb.hop_table(summaries, fetch),
        "shift": tb.shift_table(summaries),
        "leakage": tb.leakage_table(summaries),
        "serving": tb.serving_table(timed),
        "seeds": tb.seed_table(summaries),
    }
    readme.write_text(tb.fill(readme.read_text(encoding="utf8"), tables), encoding="utf8")
    rejected = Path("docs/rejected.md")
    if rejected.exists():
        text = tb.fill(
            rejected.read_text(encoding="utf8"), {"rejected": tb.rejected_table(summaries)}
        )
        rejected.write_text(text, encoding="utf8")
    typer.echo(f"{len(summaries)} runs; tables written to {readme}")


@app.command()
def bench(
    dataset: str = "ibm-HI-Small",
    stores: str = "memory,duckdb,neo4j",
    hops: str = "1,2",
    hub_cap: int = 50,
    calls: int = 1000,
    centre: str = typer.Option(
        "history,counters", help="Fetch the account's whole history, or read its counters"
    ),
    models: Path = Path("checkpoints"),
    machine: str = typer.Option(..., help="The machine's name, as the README will print it"),
    usd_per_hour: float = typer.Option(..., help="Its price"),
    price_date: str = typer.Option(..., help="The date the price was read, YYYY-MM-DD"),
    reportable: bool = typer.Option(False, help="Only on the rented machine or the VPS"),
    threads: int = typer.Option(1, help="Threads one call may use (DuckDB, ONNX Runtime)"),
    out: Path = Path("results/serving.json"),
) -> None:
    """Time the live path per transfer (fetch both accounts, arm G's features, the ONNX
    score) per store and hop count; append to results/serving.json."""
    from datetime import date

    from consort.data.tasks import IBM_NODE_COLUMNS, IBMTask, ibm_node_table
    from consort.graph.features import default_registry
    from consort.graph.neighbourhood import EdgeSource, MemorySource
    from consort.serve import bench as bn

    env.load()
    task = _task(dataset)
    assert isinstance(task, IBMTask) and task.data is not None
    g = task.data.graph
    node_table = ibm_node_table(g)
    reg = default_registry(IBM_NODE_COLUMNS, ("amount", "t"))
    m = bn.Machine(
        name=machine,
        usd_per_hour=usd_per_hour,
        price_date=date.fromisoformat(price_date),
        threads=threads,
        reportable=reportable,
    )
    rng = np.random.default_rng(0)
    pick = rng.choice(len(task.data.transfers), size=calls, replace=False)
    transfers = task.data.transfers.iloc[pick].reset_index(drop=True)
    sources: dict[str, EdgeSource] = {}
    for name in stores.split(","):
        if name == "memory":
            sources[name] = MemorySource(g)
        elif name == "duckdb":
            from consort.serve.duckdb_store import DuckDBSource

            db = DuckDBSource.from_graph(g, Path("data/serve.duckdb"))
            db.con.execute(f"SET threads = {threads}")
            sources[name] = db
        elif name == "neo4j":
            import os

            from neo4j import GraphDatabase

            from consort.serve.neo4j_store import Neo4jSource

            driver = GraphDatabase.driver(
                os.environ["NEO4J_URI"],
                auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
            )
            typer.echo("loading the graph into Neo4j")
            sources[name] = Neo4jSource.load(driver, g)
        else:
            raise typer.BadParameter(f"unknown store {name}")
    results = []
    for h in (int(x) for x in hops.split(",")):
        model = models / f"{dataset}.temporal.G.h{h}.cap{hub_cap}.s0.onnx"
        score = bn.onnx_scorer(model, threads)
        stored = reg.compute(g, node_table, h, hub_cap)
        for (name, src), mode in itertools.product(sources.items(), centre.split(",")):
            r = bn.run_transfers(
                name,
                dataset,
                src,
                reg,
                node_table,
                stored,
                score,
                transfers,
                task.row_columns,
                h,
                hub_cap,
                m,
                counters=mode == "counters",
            )
            typer.echo(f"{name} {mode} h{h}: p50 {r.p50_ms:.1f} ms, p99 {r.p99_ms:.1f} ms")
            results.append(r)
    old = json.loads(out.read_text()) if out.exists() else []

    def key(d: dict[str, Any]) -> tuple[Any, ...]:
        return (
            d["dataset"],
            d["store"],
            d.get("centre", "history"),
            d["hops"],
            d["hub_cap"],
            d["machine"]["name"],
        )

    new = [json.loads(r.model_dump_json()) for r in results]
    keep = [d for d in old if key(d) not in {key(n) for n in new}]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(keep + new, indent=1) + "\n")


@app.command()
def site(
    out: Path = Path("site"),
    runs: Path = RUNS,
    model: Path = Path("checkpoints/ibm-HI-Small.temporal.G.h1.cap50.s0.onnx"),
    pattern: str = "SCATTER-GATHER",
) -> None:
    """The demo site: one synthetic ring scored by every arm, with arm G's case files, and the
    aggregate figures, from results/ and the IBM HI-Small runs; written to site/index.html."""
    import subprocess
    from datetime import UTC, datetime

    from consort.data.tasks import IBMTask
    from consort.report import host
    from consort.report import site as st
    from consort.report import tables as tb
    from consort.report.summary import RunSummary

    task = _task("ibm-HI-Small")
    assert isinstance(task, IBMTask)
    summary = json.loads(Path("results/summary.json").read_text())
    summaries = [RunSummary.model_validate(r) for r in summary]
    heads = {}
    for arm in ("R", "G", "N"):
        h = tb.headline(summaries, "ibm-HI-Small", arm)
        if h is not None:
            heads[arm] = runs / f"{h.run}.outcomes.parquet"
    attempt = st.pick_ring(task, pattern)
    contract = json.loads(model.with_suffix(".features.json").read_text())
    ring = st.ring(task, attempt, heads, model, contract["hops"], contract["hub_cap"])
    serving = Path("results/serving.json")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    data = {
        "runs": st.aggregates(summary),
        "ring": ring,
        "serving": json.loads(serving.read_text()) if serving.exists() else [],
        "built": datetime.now(UTC).strftime("%Y-%m-%d"),
        "commit": commit,
    }
    template = (Path(st.__file__).parent / "site.html").read_text(encoding="utf8")
    # Inside a script element, "</" must not appear in the data.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out.mkdir(parents=True, exist_ok=True)
    page = out / "index.html"
    html = template.replace("/*DATA*/null", payload)
    page.write_text(html, encoding="utf8")
    # The host's headers, its content security policy hashing this page's script and style.
    (out / "staticwebapp.config.json").write_text(
        json.dumps(host.config(html), indent=2) + "\n", encoding="utf8"
    )
    (out / "robots.txt").write_text("User-agent: *\nAllow: /\n", encoding="utf8")
    typer.echo(f"ring attempt {attempt}: {len(ring['transfers'])} transfers; wrote {page}")


@app.command()
def explain(
    run: Path = typer.Option(..., help="An arm G run's outcomes file (Elliptic or IBM)"),
    model: Path = typer.Option(..., help="That run's ONNX export (train --onnx)"),
    out: Path = Path("results/faithfulness.json"),
    sample: int = typer.Option(300, help="IBM: transfers drawn from the alert queue"),
    n_boot: int = 1000,
) -> None:
    """Explanation faithfulness for the run's alert queue: occlusion against random k."""
    from consort.data.tasks import EllipticTask, IBMTask
    from consort.eval import outcomes as oc
    from consort.explain.run import run_faithfulness, run_faithfulness_ibm

    meta = oc.read_meta(run)
    task = _task(meta.dataset)
    cap = int(meta.run.split(".cap")[1].split(".")[0])
    if isinstance(task, EllipticTask):
        results = run_faithfulness(task, run, model, meta.hops, cap, n_boot=n_boot)
    elif isinstance(task, IBMTask):
        results = run_faithfulness_ibm(
            task, run, model, meta.hops, cap, sample=sample, n_boot=n_boot
        )
    else:
        raise typer.BadParameter(f"no faithfulness run for {meta.dataset}")
    for r in results:
        typer.echo(f"k={r.k}: explained {r.explained}, random {r.random}, lift {r.lift}")
    old = json.loads(out.read_text()) if out.exists() else []
    keep = [d for d in old if d["run"] != meta.run]
    out.parent.mkdir(parents=True, exist_ok=True)
    new = [json.loads(r.model_dump_json()) for r in results]
    out.write_text(json.dumps(keep + new, indent=1) + "\n")


@app.command("gate-export")
def gate_export(
    baseline: Path = typer.Option(..., help="Arm R's outcomes file"),
    candidate: list[Path] = typer.Option(..., help="Each challenger's outcomes file"),
    suite: str = typer.Option(..., help="The suite's name, usually the dataset"),
    delta_points: float = 3.0,
    out: Path = Path("data/gate"),
) -> None:
    """Spec and side files for project 03's gate (recall at the 1% alert budget, paired)."""
    from consort.gate.export import export

    for p in export(baseline, candidate, out / suite, suite, delta_points):
        typer.echo(str(p))


@app.command("pod-up")
def pod_up(
    name: str = typer.Option(..., help="Must start with consort- (the watchdog's prefix)"),
    gpu: str = typer.Option("NVIDIA RTX A5000", help="A RunPod gpuTypes id"),
    cloud: str = "COMMUNITY",
) -> None:
    """Rent one pod with a public IP, wait for its SSH port, print id, host, port and price."""
    from consort.ops.pod import Pods, PodSpec
    from consort.ops.watchdog import key_from_env

    env.load()
    pods = Pods(key_from_env())
    pod_id, cost = pods.deploy(PodSpec(name, gpu, cloud=cloud))
    try:
        ssh = pods.wait_ssh(pod_id)
    except BaseException:
        pods.delete(pod_id)  # a pod that never came up is not left running
        raise
    typer.echo(json.dumps({"id": pod_id, "host": ssh.host, "port": ssh.port, "cost_per_hr": cost}))


@app.command("pod-down")
def pod_down(
    pod_id: str = typer.Argument("", help="The pod to terminate"),
    name: str = typer.Option("", help="Instead: terminate every pod with this exact name"),
) -> None:
    """Terminate a pod, or every pod with a name (the launcher's sweep on every exit)."""
    from consort.ops.watchdog import PREFIX, RunPod, key_from_env

    env.load()
    api = RunPod(key_from_env())
    if name:
        if not name.startswith(PREFIX):
            raise typer.BadParameter(f"only this project's pods ({PREFIX}...) are swept")
        ids = [s.pod_id for s in api.pods(0.0, include_starting=True) if s.name == name]
    else:
        ids = [pod_id] if pod_id else []
    for i in ids:
        api.delete(i)
        typer.echo(f"terminated {i}")


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
