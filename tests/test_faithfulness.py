"""Occlusion explanations and the faithfulness number on a graph with a known answer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from consort.explain.faithfulness import Explanation, explain, faithfulness
from consort.graph.build import from_edges
from consort.graph.neighbourhood import Edges, MemorySource


def test_occlusion_ranks_the_neighbour_the_score_depends_on() -> None:
    # Node 0 has neighbours 1..5; the score counts edges to node 3 only.
    g = from_edges(6, [1, 2, 3, 4, 5, 3], [0, 0, 0, 0, 0, 0])

    def score(edges: Edges, centre: int) -> float:
        return float(np.sum((edges.src == 3) & (edges.dst == centre)))

    e = explain(MemorySource(g), 0, hops=1, cap=10, score=score)
    assert e.neighbours[0] == 3 and e.importance[0] == 2.0
    assert sorted(e.neighbours) == [1, 2, 3, 4, 5]


def test_faithfulness_against_the_random_baseline() -> None:
    illicit = np.zeros(10, dtype=bool)
    illicit[[3, 7]] = True
    good = [
        Explanation(0, 1.0, [3, 1, 2, 4], [1, 0, 0, 0]),
        Explanation(5, 1.0, [7, 6, 8, 9], [1, 0, 0, 0]),
    ]
    f = faithfulness(good, illicit, k=1, n_random=400)
    assert f.n_graded == 2 and f.explained_hit_rate == 1.0
    assert abs(f.random_hit_rate - 0.25) < 0.06
    bad = [Explanation(0, 1.0, [1, 2, 4, 3], [1, 0, 0, 0])]
    assert faithfulness(bad, illicit, k=1).explained_hit_rate == 0.0


def test_nodes_with_nothing_to_find_are_not_graded() -> None:
    f = faithfulness([Explanation(0, 1.0, [1, 2], [0, 0])], np.zeros(3, dtype=bool), k=1)
    assert f.n_graded == 0


def test_arm_g_faithfulness_on_toy_elliptic_through_the_onnx_export(tmp_path: Path) -> None:
    from consort.data import elliptic
    from consort.data.tasks import elliptic_task
    from consort.explain.run import run_faithfulness
    from consort.models.run import run_tabular
    from tests.toy import write_elliptic

    task = elliptic_task(elliptic.load(write_elliptic(tmp_path), check_counts=False))
    out = run_tabular(task, "G", 1, 10, seed=0, out_dir=tmp_path / "r", onnx_dir=tmp_path / "m")
    (model,) = (tmp_path / "m").glob("*.onnx")
    res = run_faithfulness(task, out, model, 1, 10, ks=(1,), budget=0.5, n_boot=50)
    assert res and res[0].graded > 0
    lo, hi = res[0].lift[1:]
    assert lo <= res[0].lift[0] <= hi
    # The toy's flagged nodes sit in illicit chains, so random neighbours find one too; the
    # explanation must do at least as well.
    assert res[0].explained[0] >= res[0].random[0]


def test_transfer_occlusion_removes_a_neighbour_from_both_accounts() -> None:
    from consort.explain.faithfulness import explain_transfer

    # Transfer 0 -> 1; node 2 touches both accounts, node 3 only the payer.
    g = from_edges(4, [0, 2, 1, 3], [1, 0, 2, 0])

    def score(es: Edges, ed: Edges) -> float:
        return float(((es.src == 2) | (es.dst == 2)).sum() + ((ed.src == 2) | (ed.dst == 2)).sum())

    e = explain_transfer(MemorySource(g), 7, 0, 1, hops=1, cap=10, score=score)
    assert e.node == 7 and sorted(e.neighbours) == [2, 3]
    assert e.neighbours[0] == 2 and e.importance[0] == e.score > 0


def test_arm_g_faithfulness_on_toy_ibm_through_the_onnx_export(tmp_path: Path) -> None:
    from consort.data import ibm
    from consort.data.tasks import ibm_task
    from consort.explain.run import run_faithfulness_ibm
    from consort.models.run import run_tabular
    from tests.toy import write_ibm

    task = ibm_task(ibm.load(write_ibm(tmp_path, n_background=1500, n_cycles=15), "T"))
    out = run_tabular(task, "G", 1, 10, seed=0, out_dir=tmp_path / "r", onnx_dir=tmp_path / "m")
    (model,) = (tmp_path / "m").glob("*.onnx")
    res = run_faithfulness_ibm(task, out, model, 1, 10, ks=(1,), budget=0.1, sample=40, n_boot=50)
    assert res and 0 < res[0].graded <= res[0].flagged <= 40
    assert res[0].lift[1] <= res[0].lift[0] <= res[0].lift[2]
