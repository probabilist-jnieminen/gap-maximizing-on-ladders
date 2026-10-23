"""Tests of the conjectured closed form (see gap_maximizing_on_ladders/closed_form.py).

The first group does not use the solver: it checks that the closed form is a
feasible point with the claimed value and spectrum. The second group checks
that the SDP solver finds the same optimum. None of this proves optimality.
"""

import numpy as np
import pytest

from gap_maximizing_on_ladders import (
    RUNG,
    algebraic_connectivity,
    closed_form_lambda2,
    closed_form_unit,
    closed_form_w,
    laplacian,
    make_ladder,
    solve_ladder,
)

# ---- no solver involved ------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 20, 50])
def test_closed_form_is_feasible_and_attains_value(n):
    lad = make_ladder(n)
    w = closed_form_w(n)
    assert w.shape == (3 * n - 2,)
    assert np.all(w > 0)
    assert w.sum() == pytest.approx(1.0, rel=1e-12)
    assert algebraic_connectivity(lad.B, w) == pytest.approx(closed_form_lambda2(n), rel=1e-10)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 20, 50])
def test_closed_form_spectrum(n):
    """Observed: spec L(w) = u * ({k(k+1)} and {k(k+1) + 2}), k = 0..n-1."""
    u = closed_form_unit(n)
    k = np.arange(n)
    expected = u * np.sort(np.concatenate([k * (k + 1), k * (k + 1) + 2]))
    got = np.linalg.eigvalsh(laplacian(make_ladder(n).B, closed_form_w(n)))
    np.testing.assert_allclose(got, expected, atol=1e-9 * u * n**2)


def test_closed_form_small_cases():
    np.testing.assert_allclose(closed_form_w(1), [1.0])
    np.testing.assert_allclose(closed_form_w(2), np.full(4, 0.25))   # uniform on the 4-cycle


def test_invalid_n():
    with pytest.raises(ValueError):
        closed_form_w(0)


# ---- solver against closed form ----------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 12, 20])
def test_solver_matches_closed_form(n):
    lad = make_ladder(n)
    sol = solve_ladder(n)
    lam = closed_form_lambda2(n)
    assert sol.lambda2 == pytest.approx(lam, rel=1e-6)
    assert algebraic_connectivity(lad.B, sol.w) == pytest.approx(lam, rel=1e-6)
    # Only the rails are compared edge by edge: the optimum is very flat in
    # the rung conductances, so the solver cannot pin those down well.
    rails = lad.kind != RUNG
    np.testing.assert_allclose(sol.w[rails], closed_form_w(n)[rails], rtol=1e-3)
