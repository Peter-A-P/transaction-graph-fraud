"""Toy datasets in the published file formats, with signal planted where a test wants it."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.test_loaders import TRANS_HEADER


def write_ibm(
    root: Path,
    tier: str = "T",
    n_accounts: int = 400,
    n_background: int = 6000,
    n_cycles: int = 60,
    seed: int = 0,
) -> Path:
    """Background transfers between random accounts, and planted 3- and 4-cycles of laundering
    whose transfers look exactly like background ones row by row: same amounts, formats and
    hours. Only the closure of the cycle tells them apart."""
    rng = np.random.default_rng(seed)
    start = np.datetime64("2022-09-01T00:00")
    lines: list[str] = []
    blocks: list[str] = []

    def line(t: np.datetime64, a: int, b: int, amt: float, fmt: str, lab: int) -> str:
        ts = str(t.astype("datetime64[m]")).replace("-", "/").replace("T", " ")
        return f"{ts},{a % 7 + 1},{a:08X},{b % 7 + 1},{b:08X},{amt:.2f},US Dollar,{amt:.2f},US Dollar,{fmt},{lab}"

    formats = ["ACH", "Cheque", "Credit Card", "Wire"]
    for _ in range(n_background):
        a, b = rng.choice(n_accounts, 2, replace=False)
        t = start + np.timedelta64(int(rng.integers(0, 20 * 24 * 60)), "m")
        lines.append(
            line(t, int(a), int(b), float(rng.lognormal(7, 1)), formats[rng.integers(4)], 0)
        )
    for _ in range(n_cycles):
        k = int(rng.integers(3, 5))
        members = rng.choice(np.arange(n_accounts, 2 * n_accounts), k, replace=False)
        t = start + np.timedelta64(int(rng.integers(0, 19 * 24 * 60)), "m")
        block = [f"BEGIN LAUNDERING ATTEMPT - CYCLE:  Max {k} hops"]
        for i in range(k):
            t = t + np.timedelta64(int(rng.integers(5, 300)), "m")
            ln = line(
                t,
                int(members[i]),
                int(members[(i + 1) % k]),
                float(rng.lognormal(7, 1)),
                formats[rng.integers(4)],
                1,
            )
            lines.append(ln)
            block.append(ln)
        # The cycle's accounts also trade ordinarily, so degree alone does not give them away.
        for m in members:
            for _ in range(int(rng.integers(1, 4))):
                other = int(rng.integers(0, n_accounts))
                tt = start + np.timedelta64(int(rng.integers(0, 20 * 24 * 60)), "m")
                lines.append(
                    line(tt, int(m), other, float(rng.lognormal(7, 1)), formats[rng.integers(4)], 0)
                )
        block.append("END LAUNDERING ATTEMPT - CYCLE")
        blocks.append("\n".join(block))
    (root / f"{tier}_Trans.csv").write_text("\n".join([TRANS_HEADER, *lines]) + "\n")
    (root / f"{tier}_Patterns.txt").write_text("\n\n".join(blocks) + "\n")
    return root
