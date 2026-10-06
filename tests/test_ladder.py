import numpy as np
import pytest

from gap_maximizing_on_ladders import BOTTOM_RAIL, RUNG, TOP_RAIL, make_ladder


@pytest.mark.parametrize("n", range(1, 11))
def test_sizes_and_edge_order(n):
    lad = make_ladder(n)
    assert lad.B.shape == (3 * n - 2, 2 * n)
    assert len(lad.edges) == 3 * n - 2
    # n rungs, then n-1 top-rail edges, then n-1 bottom-rail edges
    expected_kind = [RUNG] * n + [TOP_RAIL] * (n - 1) + [BOTTOM_RAIL] * (n - 1)
    assert lad.kind.tolist() == expected_kind


@pytest.mark.parametrize("n", range(1, 11))
def test_positions(n):
    lad = make_ladder(n)
    assert lad.position[lad.kind == RUNG].tolist() == list(range(n))
    assert lad.position[lad.kind == TOP_RAIL].tolist() == list(range(n - 1))
    assert lad.position[lad.kind == BOTTOM_RAIL].tolist() == list(range(n - 1))


@pytest.mark.parametrize("n", range(1, 11))
def test_edges_match_graph(n):
    lad = make_ladder(n)
    assert {frozenset(e) for e in lad.edges} == {frozenset(e) for e in lad.graph.edges}
    assert len(set(lad.edges)) == len(lad.edges)  # no duplicates


@pytest.mark.parametrize("n", range(1, 11))
def test_incidence_rows(n):
    lad = make_ladder(n)
    for row, (u, v) in zip(lad.B, lad.edges):
        assert sorted(row[row != 0].tolist()) == [-1.0, 1.0]
        assert row[u] != 0 and row[v] != 0


@pytest.mark.parametrize("bad_n", [0, -1])
def test_invalid_n(bad_n):
    with pytest.raises(ValueError):
        make_ladder(bad_n)
