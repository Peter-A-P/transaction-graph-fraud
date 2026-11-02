"""The README's result tables, built from results/summary.json and nothing else.

Each table sits in the README between `<!-- report:NAME -->` and `<!-- /report:NAME -->`;
`consort report` rewrites what is between the markers and leaves everything else alone. A
headline arm's hop count is the one with the best validation AUPRC, never chosen on test.
IBM rows read the "without-tail" slice (PLAN.md 2.2) unless a table says otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from consort.report.summary import RunSummary, Triple

DATASETS = {
    "elliptic": "Elliptic (real, transactions)",
    "ellipticpp": "Elliptic++ (real, wallets)",
    "ibm-HI-Small": "IBM AML HI-Small (synthetic)",
    "ibm-LI-Small": "IBM AML LI-Small (synthetic)",
}
TAIL = "without-tail"
NOT_RUN = "not run yet"
PATTERNS = [
    "FAN-IN",
    "FAN-OUT",
    "CYCLE",
    "SCATTER-GATHER",
    "GATHER-SCATTER",
    "BIPARTITE",
    "STACK",
    "RANDOM",
    "NONE",
]
PATTERN_NAMES = {
    "FAN-IN": "Fan-in",
    "FAN-OUT": "Fan-out",
    "CYCLE": "Cycle",
    "SCATTER-GATHER": "Scatter-gather",
    "GATHER-SCATTER": "Gather-scatter",
    "BIPARTITE": "Bipartite",
    "STACK": "Stack",
    "RANDOM": "Random",
    "NONE": "No named pattern",
}
ARM_NAMES = {"R": "R, row only", "G": "G, graph features", "N": "N, graph neural network"}


def fmt(t: Triple | None) -> str:
    if t is None:
        return NOT_RUN
    return f"{t[0]:.3f} ({t[1]:.3f} to {t[2]:.3f})"


def _metrics(s: RunSummary) -> dict[str, Triple]:
    return s.slices.get(TAIL, s.metrics) if s.dataset.startswith("ibm-") else s.metrics


def _delta(s: RunSummary) -> dict[str, Triple]:
    return s.slice_delta.get(TAIL, s.delta) if s.dataset.startswith("ibm-") else s.delta


def of_arm(run_arm: str, arm: str) -> bool:
    """Arm N's runs are tagged by variant (N-sage, N-gat+rmp+ego); the variant, like the hop
    count, is chosen on validation."""
    return run_arm == arm or (arm == "N" and run_arm.startswith("N-"))


def headline(runs: Sequence[RunSummary], dataset: str, arm: str) -> RunSummary | None:
    """The arm's temporal run on the dataset with the best validation AUPRC (seed 0)."""
    pool = [r for r in runs if r.dataset == dataset and of_arm(r.arm, arm) and r.kind == "temporal"]
    pool = [r for r in pool if r.seed == 0]
    if not pool:
        return None
    return max(pool, key=lambda r: (r.val_auprc is not None, r.val_auprc or 0.0, -r.hops))


def _arms(runs: Sequence[RunSummary], dataset: str) -> list[str]:
    seen = {
        "N" if r.arm.startswith("N-") else r.arm
        for r in runs
        if r.dataset == dataset and r.kind == "temporal"
    }
    order = ["R", "G", "N", *sorted(a for a in seen if a.startswith("TabFM"))]
    return [a for a in order if a in seen or a in ("R", "G", "N")]


def main_table(runs: Sequence[RunSummary]) -> str:
    lines = [
        "| Dataset | Arm | AUPRC | AUPRC minus R, same items | Recall at 1% FPR "
        "| Recall at 0.5% alert budget | Recall at 1% alert budget |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds, name in DATASETS.items():
        label = f"{name}, without the tail" if ds.startswith("ibm-") else name
        for arm in _arms(runs, ds):
            s = headline(runs, ds, arm)
            arm_name = ARM_NAMES.get(arm, arm)
            if s is None:
                lines.append(f"| {label} | {arm_name} | {NOT_RUN} | | | | |")
                continue
            m, d = _metrics(s), _delta(s)
            hop = f", {s.hops} hop{'s' if s.hops != 1 else ''}" if arm != "R" else ""
            if s.arm.startswith("N-"):
                hop = f" ({s.arm.removeprefix('N-')}){hop}"
            cell = {
                b: f"{fmt(m.get(f'recall_at_{b}_budget'))}; ceiling "
                f"{m.get(f'ceiling_at_{b}_budget', (float('nan'),))[0]:.3f}"
                for b in ("0.5pct", "1pct")
            }
            lines.append(
                f"| {label} | {arm_name}{hop} | {fmt(m.get('auprc'))} "
                f"| {fmt(d.get('auprc')) if d else ''} | {fmt(m.get('recall_at_1pct_fpr'))} "
                f"| {cell['0.5pct']} | {cell['1pct']} |"
            )
    return "\n".join(lines)


def pattern_table(runs: Sequence[RunSummary], dataset: str = "ibm-HI-Small") -> str:
    heads = {a: headline(runs, dataset, a) for a in ("R", "G", "N")}
    lines = ["| Pattern | R | G | N |", "|---|---|---|---|"]
    for p in PATTERNS:
        cells = [fmt(h.patterns.get(p)) if h and h.patterns else NOT_RUN for h in heads.values()]
        lines.append(f"| {PATTERN_NAMES[p]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def hop_table(runs: Sequence[RunSummary], fetch: Callable[[str, int], tuple[str, str]]) -> str:
    lines = [
        "| Dataset | Hops | G, AUPRC | N, AUPRC | Fetch p50 (ms) | Fetch p99 (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for ds, name in DATASETS.items():
        for h in range(4):
            cells = []
            for arm in ("G", "N"):
                # The curve is GraphSAGE's: the architecture every hop count was run with.
                arm_at = "R" if (arm == "G" and h == 0) else ("N-sage" if arm == "N" else arm)
                pool = [
                    r
                    for r in runs
                    if r.dataset == ds
                    and r.arm == arm_at
                    and r.kind == "temporal"
                    and r.hops == h
                    and r.seed == 0
                ]
                cells.append(fmt(_metrics(pool[0]).get("auprc")) if pool else NOT_RUN)
            p50, p99 = fetch(ds, h)
            lines.append(f"| {name} | {h} | {cells[0]} | {cells[1]} | {p50} | {p99} |")
    return "\n".join(lines)


def shift_table(runs: Sequence[RunSummary]) -> str:
    lines = [
        "| Dataset | Arm | Before step 43 | From step 43 | From step 43, minus R |",
        "|---|---|---|---|---|",
    ]
    for ds in ("elliptic", "ellipticpp"):
        for arm in _arms(runs, ds):
            s = headline(runs, ds, arm)
            name = ARM_NAMES.get(arm, arm)
            if s is None or not s.slices:
                lines.append(f"| {DATASETS[ds]} | {name} | {NOT_RUN} | | |")
                continue
            before = s.slices.get("before-43", {}).get("auprc")
            after = s.slices.get("from-43", {}).get("auprc")
            d = s.slice_delta.get("from-43", {}).get("auprc")
            lines.append(
                f"| {DATASETS[ds]} | {name} | {fmt(before)} | {fmt(after)} "
                f"| {fmt(d) if d else ''} |"
            )
    return "\n".join(lines)


def leakage_table(runs: Sequence[RunSummary]) -> str:
    lines = [
        "| Dataset | Arm | Temporal split, AUPRC | Random split, AUPRC "
        "| Features that saw later days, AUPRC |",
        "|---|---|---|---|---|",
    ]
    for ds, name in DATASETS.items():
        for arm in ("R", "G"):
            t = headline(runs, ds, arm)
            if t is None:
                continue
            rand = [r for r in runs if r.dataset == ds and r.arm == arm and r.kind == "random"]
            win = [
                r for r in runs if r.dataset == ds and r.arm == f"{arm}-window" and r.hops == t.hops
            ]
            lines.append(
                f"| {name} | {ARM_NAMES[arm]} | {fmt(t.metrics.get('auprc'))} "
                f"| {fmt(rand[0].metrics.get('auprc')) if rand else NOT_RUN} "
                f"| {fmt(win[0].metrics.get('auprc')) if win else 'not applicable'} |"
            )
    return "\n".join(lines)


def serving_table(timed: Sequence[dict[str, Any]]) -> str:
    """Every reportable serving result: one live score per call, on the machine named."""
    lines = [
        "| Dataset | Store | Account's own counts | Hops | p50 (ms) | p99 (ms) "
        "| Edges fetched, mean | US$ per million scores | Machine, price (date) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    if not timed:
        lines.append(f"| {NOT_RUN} | | | | | | | | |")
    centre = {"history": "fetched whole", "counters": "stored counters"}
    for t in sorted(
        timed, key=lambda t: (t["dataset"], t.get("centre", "history"), t["hops"], t["store"])
    ):
        m = t["machine"]
        lines.append(
            f"| {DATASETS.get(t['dataset'], t['dataset'])} | {t['store']} "
            f"| {centre[t.get('centre', 'history')]} | {t['hops']} | {t['p50_ms']:.1f} "
            f"| {t['p99_ms']:.1f} | {t['mean_edges_fetched']:,.0f} | {t['usd_per_million']:.2f} "
            f"| {m['name']}, US${m['usd_per_hour']:.2f}/h ({m['price_date']}) |"
        )
    return "\n".join(lines)


def faithfulness_table(results: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| Run | k | Flagged | Graded | Top-k holds a known-illicit neighbour "
        "| k random neighbours do | Lift |",
        "|---|---|---|---|---|---|---|",
    ]
    if not results:
        lines.append(f"| {NOT_RUN} | | | | | | |")
    for r in results:
        e, rnd, lift = (tuple(r[k]) for k in ("explained", "random", "lift"))
        lines.append(
            f"| {r['run']} | {r['k']} | {r['flagged']} | {r['graded']} "
            f"| {fmt(e)} | {fmt(rnd)} | {fmt(lift)} |"
        )
    return "\n".join(lines)


def rejected_table(runs: Sequence[RunSummary], dataset: str = "elliptic") -> str:
    """The neighbour-label-rate experiment: arm G with and without the feature, on both splits,
    the difference paired on identical items."""
    lines = [
        "| Split | Arm G, AUPRC | Arm G with the neighbour-label rate, AUPRC "
        "| Difference, same items |",
        "|---|---|---|---|",
    ]
    for kind, name in (("temporal", "Temporal, inductive"), ("random", "Random")):
        leak = [r for r in runs if r.dataset == dataset and r.arm == "G+nlr" and r.kind == kind]
        if not leak:
            lines.append(f"| {name} | {NOT_RUN} | | |")
            continue
        base = [r for r in runs if r.run == leak[0].baseline]
        lines.append(
            f"| {name} | {fmt(base[0].metrics.get('auprc')) if base else NOT_RUN} "
            f"| {fmt(leak[0].metrics.get('auprc'))} | {fmt(leak[0].delta.get('auprc'))} |"
        )
    return "\n".join(lines)


def fill(text: str, tables: dict[str, str]) -> str:
    for name, body in tables.items():
        # The markers may be adjacent (no table yet) or hold an earlier table.
        pattern = re.compile(
            rf"(<!-- report:{name} -->\n)(?:.*?\n)?(<!-- /report:{name} -->)", re.DOTALL
        )
        if not pattern.search(text):
            raise ValueError(f"the README has no report:{name} markers")
        text = pattern.sub(lambda m: f"{m.group(1)}{body}\n{m.group(2)}", text)  # noqa: B023
    return text
