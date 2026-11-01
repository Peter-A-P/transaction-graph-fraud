"""The pattern labeller on hand-built toy graphs with known answers."""

from __future__ import annotations

import pytest

from consort.graph.patterns import classify


def test_three_cycle() -> None:
    assert classify(["a", "b", "c"], ["b", "c", "a"]) == "CYCLE"


def test_two_disjoint_cycles_are_not_one_cycle() -> None:
    assert classify(["a", "b", "c", "d"], ["b", "a", "d", "c"]) != "CYCLE"


def test_fan_in_of_five() -> None:
    assert classify(["a", "b", "c", "d", "e"], ["h"] * 5) == "FAN-IN"


def test_fan_out_of_four() -> None:
    assert classify(["h"] * 4, ["a", "b", "c", "d"]) == "FAN-OUT"


def test_scatter_gather() -> None:
    assert (
        classify(["s", "s", "s", "a", "b", "c"], ["a", "b", "c", "t", "t", "t"]) == "SCATTER-GATHER"
    )


def test_gather_scatter() -> None:
    assert classify(["a", "b", "c", "h", "h"], ["h", "h", "h", "x", "y"]) == "GATHER-SCATTER"


def test_bipartite() -> None:
    assert classify(["a", "a", "b", "c"], ["x", "y", "x", "y"]) == "BIPARTITE"


def test_stack_of_three_layers() -> None:
    src = ["a", "a", "b", "b", "x", "x", "y", "y"]
    dst = ["x", "y", "x", "y", "p", "q", "p", "q"]
    assert classify(src, dst) == "STACK"


def test_random_walk() -> None:
    assert classify(["a", "b", "c", "d"], ["b", "c", "d", "e"]) == "RANDOM"


def test_a_cycle_with_a_chord_is_not_a_cycle() -> None:
    # Not in walk order either, so not RANDOM.
    assert classify(["a", "a", "b", "c"], ["b", "c", "c", "a"]) == "OTHER"


def test_repeated_transfers_do_not_change_the_shape() -> None:
    assert classify(["a", "b", "c", "a"], ["b", "c", "a", "b"]) == "CYCLE"


def test_empty_is_refused() -> None:
    with pytest.raises(ValueError):
        classify([], [])
