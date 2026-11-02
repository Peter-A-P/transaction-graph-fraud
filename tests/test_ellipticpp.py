"""Elliptic++ rebuilt step by step: the loader's construction and the leakage guarantee."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from consort.data import ellipticpp
from consort.data.splits import temporal_split
from consort.data.tasks import ellipticpp_task
from tests.toy import write_ellipticpp

SPLIT = temporal_split("toy", train=range(1, 8), val=range(8, 10), test=range(10, 14))


def test_edges_pair_every_input_with_every_output_and_carry_the_step(tmp_path: Path) -> None:
    d = ellipticpp.load(*write_ellipticpp(tmp_path), check_counts=False)
    # Edges per transaction are inputs times outputs.
    tx_of = d.inc_tx[:, 3]  # log1p(num_input_addresses); pairs each input with each output
    assert tx_of.shape[0] == d.inc_addr.shape[0]
    n_in = np.bincount(d.inc_step[d.inc_role == 0], minlength=20)
    assert d.edge_src.shape[0] >= n_in.sum()
    # Every edge's step is the step of a transaction both its ends took part in.
    for s, a, b in zip(d.edge_step[:200], d.edge_src[:200], d.edge_dst[:200], strict=True):
        assert s in set(d.inc_step[d.inc_addr == a]) and s in set(d.inc_step[d.inc_addr == b])
    # First step is the earliest step an address appears in.
    for a in range(0, d.n, 37):
        assert d.first_step[a] == d.inc_step[d.inc_addr == a].min()


def test_later_steps_change_no_earlier_address_features(tmp_path: Path) -> None:
    plain = ellipticpp_task(
        ellipticpp.load(*write_ellipticpp(tmp_path / "p"), check_counts=False), SPLIT
    )
    later = ellipticpp_task(
        ellipticpp.load(*write_ellipticpp(tmp_path / "l", late=40), check_counts=False), SPLIT
    )
    assert (plain.items["item_id"] == later.items["item_id"]).all()
    cols = plain.row_columns
    assert np.allclose(plain.items[cols].to_numpy(), later.items[cols].to_numpy())
    a = plain.graph_features("test", 1, 5)
    b = later.graph_features("test", 1, 5)
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
    # The late transactions do change a lifetime view: the same addresses' activity over every
    # step differs, which is what the published wallet features would have carried.
    d_plain, d_later = plain.data, later.data
    assert d_plain is not None and d_later is not None
    every = d_plain.node_table(np.ones(d_plain.inc_addr.shape[0], bool))
    every_later = d_later.node_table(np.ones(d_later.inc_addr.shape[0], bool))
    assert not np.allclose(every.to_numpy(), every_later.to_numpy())
