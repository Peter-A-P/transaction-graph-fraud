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


def write_elliptic(root: Path, per_step: int = 60, seed: int = 0) -> Path:
    """49 steps of transactions, edges only within a step. Illicit transactions come in
    connected chains whose own features look licit; a neighbour's features give them away."""
    from consort.data import elliptic

    rng = np.random.default_rng(seed)
    base = root / "elliptic_bitcoin_dataset"
    base.mkdir(parents=True, exist_ok=True)
    feats, classes, edges = [], ["txId,class"], ["txId1,txId2"]
    tx = 1000
    for step in range(1, 50):
        ids = list(range(tx, tx + per_step))
        tx += per_step
        illicit = set(ids[:6])
        for i in ids:
            f = rng.normal(size=elliptic.N_LOCAL + elliptic.N_AGG)
            # A marker on the illicit chain's hub only, never on its other members.
            if i == ids[0]:
                f[0] += 4.0
            feats.append(",".join([str(i), str(step), *(f"{v:.5f}" for v in f)]))
            lab = "1" if i in illicit else ("2" if rng.random() < 0.5 else "unknown")
            classes.append(f"{i},{lab}")
        for j in range(1, 6):
            edges.append(f"{ids[0]},{ids[j]}")
        for _ in range(per_step):
            a, b = rng.choice(ids[6:], 2, replace=False)
            edges.append(f"{a},{b}")
    (base / "elliptic_txs_features.csv").write_text("\n".join(feats) + "\n")
    (base / "elliptic_txs_classes.csv").write_text("\n".join(classes) + "\n")
    (base / "elliptic_txs_edgelist.csv").write_text("\n".join(edges) + "\n")
    return root


def write_ellipticpp(
    root: Path,
    n_addr: int = 300,
    n_tx: int = 900,
    steps: int = 12,
    seed: int = 0,
    late: int = 0,
) -> tuple[Path, Path]:
    """A small Elliptic++: transactions over `steps` steps with one to three inputs and outputs
    among `n_addr` addresses; illicit addresses trade with each other. `late` adds that many
    transactions in a step after the last, among the first twenty addresses."""
    rng = np.random.default_rng(seed)
    actors, txs = root / "actors", root / "txs"
    actors.mkdir(parents=True, exist_ok=True)
    txs.mkdir(parents=True, exist_ok=True)
    names = [f"a{i:05d}" for i in range(n_addr)]
    illicit = set(range(0, n_addr, 10))
    feats = ["txId,Time step,total_BTC,fees,size,num_input_addresses,num_output_addresses"]
    ins, outs = ["input_address,txId"], ["txId,output_address"]

    def tx(i: int, step: int, pool: list[int]) -> None:
        k_in, k_out = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        a_in = rng.choice(pool, k_in, replace=False)
        a_out = rng.choice(pool, k_out, replace=False)
        feats.append(
            f"{i},{step},{rng.lognormal(0, 1):.6f},0.0001,{rng.integers(200, 900)},{k_in},{k_out}"
        )
        ins.extend(f"{names[a]},{i}" for a in a_in)
        outs.extend(f"{i},{names[a]}" for a in a_out)

    # Addresses arrive over time, as real ones do, so every step has new addresses to score.
    for i in range(n_tx):
        step = 1 + i * steps // n_tx
        live = list(range(max(40, n_addr * step // steps)))  # at least 4 illicit to pick from
        pool = [a for a in live if a in illicit] if rng.random() < 0.15 else live
        tx(1000 + i, step, pool)
    for j in range(late):
        tx(5000 + j, steps + 1, list(range(20)))
    # Only addresses that took part in a transaction are listed, as in the real classes file.
    used = sorted(
        {int(r.split(",")[0][1:]) for r in ins[1:]} | {int(r.split(",")[1][1:]) for r in outs[1:]}
    )
    cls = ["address,class"] + [
        f"{names[i]},{1 if i in illicit else (2 if i % 3 else 3)}" for i in used
    ]
    (actors / "wallets_classes.csv").write_text("\n".join(cls) + "\n")
    (txs / "txs_features.csv").write_text("\n".join(feats) + "\n")
    (txs / "AddrTx_edgelist.csv").write_text("\n".join(ins) + "\n")
    (txs / "TxAddr_edgelist.csv").write_text("\n".join(outs) + "\n")
    return actors, txs
