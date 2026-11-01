"""The loaders against small files in each dataset's published format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from consort.data import elliptic, fetch, ibm

TRANS_HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering"
)
TRANSFERS = [
    "2022/09/01 00:20,10,8000EBD30,10,8000EBD30,3697.34,US Dollar,3697.34,US Dollar,Reinvestment,0",
    "2022/09/01 00:20,3208,8000F4580,1,8000F5340,0.01,US Dollar,0.01,US Dollar,Cheque,0",
    "2022/09/01 00:00,3209,8000F4670,3209,8000F4670,14675.57,US Dollar,14675.57,US Dollar,Reinvestment,0",
    "2022/09/01 02:02,12,8000F5030,12,8000F5030,2806.97,Euro,2806.97,Euro,Reinvestment,0",
    "2022/09/01 06:10,1,8000F5340,12,8000F5030,100.00,US Dollar,85.00,Euro,ACH,0",
    "2022/09/02 11:05,70,100428660,1124,800825340,8000.00,US Dollar,8000.00,US Dollar,ACH,1",
    "2022/09/02 11:15,1124,800825340,70,100428660,7000.00,US Dollar,7000.00,US Dollar,ACH,1",
    "2022/09/03 09:00,70,100428660,3208,8000F4580,500.00,US Dollar,500.00,US Dollar,Cash,1",
]
# The pattern file zero-pads bank codes; the join must not care.
PATTERNS_TXT = """BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 2 hops
2022/09/02 11:05,070,100428660,01124,800825340,8000.00,US Dollar,8000.00,US Dollar,ACH,1
2022/09/02 11:15,01124,800825340,070,100428660,7000.00,US Dollar,7000.00,US Dollar,ACH,1
END LAUNDERING ATTEMPT - CYCLE

"""


def _ibm_files(tmp_path: Path, patterns: str = PATTERNS_TXT) -> Path:
    (tmp_path / "T_Trans.csv").write_text("\n".join([TRANS_HEADER, *TRANSFERS]) + "\n")
    (tmp_path / "T_Patterns.txt").write_text(patterns)
    return tmp_path


def test_ibm_loads_accounts_edges_and_patterns(tmp_path: Path) -> None:
    d = ibm.load(_ibm_files(tmp_path), "T")
    t = d.transfers
    assert d.graph.n_edges == len(TRANSFERS)
    # Bank and account together: 8000F5340 at bank 1 is one account across two rows.
    assert d.graph.n_nodes == 7
    assert list(t["pattern"]) == ["NONE"] * 5 + ["CYCLE", "CYCLE", "NONE"]
    # The unnamed laundering transfer stays laundering, with no pattern.
    assert t.loc[7, "is_laundering"] == 1 and t.loc[7, "attempt"] == -1
    assert t.loc[0, "self_transfer"] == 1.0 and t.loc[4, "cross_currency"] == 1.0


def test_ibm_dollar_rates_are_implied_by_the_data(tmp_path: Path) -> None:
    d = ibm.load(_ibm_files(tmp_path), "T")
    assert d.rates["Euro"] == pytest.approx(100 / 85)
    assert d.transfers.loc[3, "amount_usd"] == pytest.approx(2806.97 * 100 / 85)


def test_ibm_edge_recency_follows_the_timestamp(tmp_path: Path) -> None:
    d = ibm.load(_ibm_files(tmp_path), "T")
    # Row 2 is the earliest transfer in the file.
    assert d.graph.order[2] == 0
    assert d.graph.order[7] == len(TRANSFERS) - 1


def test_ibm_pattern_join_refuses_an_unmatched_transfer(tmp_path: Path) -> None:
    bad = PATTERNS_TXT.replace("7000.00,US Dollar,7000.00", "7000.01,US Dollar,7000.01")
    with pytest.raises(ibm.PatternJoinFailed, match="1 of 2"):
        ibm.load(_ibm_files(tmp_path, bad), "T")


def test_ibm_pattern_parser_refuses_an_unknown_pattern(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown pattern"):
        ibm.load(_ibm_files(tmp_path, PATTERNS_TXT.replace("CYCLE", "SPIRAL")), "T")


def test_ibm_pattern_parser_refuses_an_unclosed_block(tmp_path: Path) -> None:
    unclosed = PATTERNS_TXT.replace("END LAUNDERING ATTEMPT - CYCLE\n", "")
    with pytest.raises(ValueError, match="never closed"):
        ibm.load(_ibm_files(tmp_path, unclosed), "T")


def _elliptic_files(tmp_path: Path) -> Path:
    base = tmp_path / "elliptic_bitcoin_dataset"
    base.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i, tx in enumerate([101, 102, 103, 104]):
        f = rng.normal(size=elliptic.N_LOCAL + elliptic.N_AGG)
        rows.append(",".join([str(tx), str(1 + i // 2), *(f"{v:.6f}" for v in f)]))
    (base / "elliptic_txs_features.csv").write_text("\n".join(rows) + "\n")
    (base / "elliptic_txs_classes.csv").write_text("txId,class\n101,1\n102,2\n103,unknown\n104,2\n")
    (base / "elliptic_txs_edgelist.csv").write_text("txId1,txId2\n101,102\n103,104\n")
    return tmp_path


def test_elliptic_loads_labels_and_edges(tmp_path: Path) -> None:
    d = elliptic.load(_elliptic_files(tmp_path), check_counts=False)
    assert list(d.nodes["label"]) == [1, 0, -1, 0]
    assert list(d.nodes["step"]) == [1, 1, 2, 2]
    assert d.graph.src.tolist() == [0, 2] and d.graph.dst.tolist() == [1, 3]
    assert "agg_001" not in d.row_columns and "local_093" in d.row_columns


def test_elliptic_count_check_refuses_a_partial_file(tmp_path: Path) -> None:
    with pytest.raises(elliptic.CountMismatch):
        elliptic.load(_elliptic_files(tmp_path))


def test_checksum_mismatch_is_refused(tmp_path: Path) -> None:
    src = fetch.Source("toy", "x/y", ("a.csv",), "none")
    (tmp_path / "a.csv").write_text("1,2\n")
    m = fetch.Manifest(
        source="toy",
        kaggle="x/y",
        licence="none",
        first_fetched="2026-10-31",
        files=[fetch.record(tmp_path, "a.csv")],
    )
    fetch.verify(src, tmp_path, m)
    (tmp_path / "a.csv").write_text("1,3\n")
    with pytest.raises(fetch.ChecksumMismatch):
        fetch.verify(src, tmp_path, m)
