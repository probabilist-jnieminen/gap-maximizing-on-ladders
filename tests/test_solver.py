import networkx as nx
import numpy as np
import pytest

from gap_maximizing_on_ladders import (
    algebraic_connectivity,
    make_ladder,
    solve_gap_sdp,
    solve_ladder,
)

NS = range(1, 9)
TOL = 1e-6


def incidence(G):
    """Oriented incidence matrix with one row per edge, as in make_ladder."""
    return nx.incidence_matrix(G, oriented=True).toarray().T


# ---- known answers ---------------------------------------------------------

def test_single_rung():
    sol = solve_ladder(1)
    np.testing.assert_allclose(sol.w, [1.0], atol=TOL)
    assert sol.lambda2 == pytest.approx(2.0, abs=TOL)


def test_four_cycle():
    assert solve_ladder(2).lambda2 == pytest.approx(0.5, abs=TOL)


# For an edge-transitive graph the uniform weights are optimal: averaging any
# optimum over the graph's symmetries gives uniform weights, and by concavity
# of lambda_2 the average is at least as good.
BUILDERS = {"cycle": nx.cycle_graph, "complete": nx.complete_graph, "star": nx.star_graph}
EDGE_TRANSITIVE = (
    [("cycle", k) for k in range(3, 8)]
    + [("complete", k) for k in range(3, 7)]
    + [("star", k) for k in range(2, 6)]
)


@pytest.mark.parametrize("family, k", EDGE_TRANSITIVE)
def test_edge_transitive_graphs(family, k):
    B = incidence(BUILDERS[family](k))
    m = B.shape[0]
    expected = algebraic_connectivity(B, np.full(m, 1 / m))
    assert solve_gap_sdp(B).lambda2 == pytest.approx(expected, abs=TOL)


# ---- properties that must hold for every n ---------------------------------

@pytest.mark.parametrize("n", NS)
def test_feasible(n):
    w = solve_ladder(n).w
    assert w.shape == (3 * n - 2,)
    assert np.all(w >= 0)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("n", NS)
def test_sdp_value_matches_numpy(n):
    sol = solve_ladder(n)
    assert sol.lambda2 == pytest.approx(sol.lambda2_check, abs=TOL)


@pytest.mark.parametrize("n", NS)
def test_beats_feasible_points(n):
    """The optimum is at least lambda_2 of uniform and of random feasible weights."""
    lad = make_ladder(n)
    m = len(lad.edges)
    best = solve_gap_sdp(lad.B).lambda2
    candidates = [np.full(m, 1 / m)]
    candidates += list(np.random.default_rng(n).dirichlet(np.ones(m), size=20))
    for w in candidates:
        assert best >= algebraic_connectivity(lad.B, w) - TOL


@pytest.mark.parametrize("n", NS)
def test_trace_bound(n):
    """trace L(w) = 2, and the 2n-1 nonzero eigenvalues are >= lambda_2."""
    assert solve_ladder(n).lambda2 <= 2 / (2 * n - 1) + TOL


# ---- second solver ---------------------------------------------------------

@pytest.mark.parametrize("n", [2, 4, 6])
def test_clarabel_and_scs_agree(n):
    # SCS is a first-order method, accurate to about 1e-4 by default.
    assert solve_ladder(n, solver="SCS").lambda2 == pytest.approx(
        solve_ladder(n).lambda2, abs=1e-3
    )
