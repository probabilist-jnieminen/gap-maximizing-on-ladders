"""Conjectured closed-form optimum for the ladder problem (NOT proved).

Numerical solutions of the SDP for the ladder with n rungs and unit budget
sum(w) = 1 match the following formulas (checked for n up to 50, see
scripts/closed_form_table.py and tests/test_closed_form.py):

    rung conductances (all n of them)           u = 3 / (n (n^2 + 2))
    rail edge joining positions i and i+1
    (i = 1, ..., n-1; the same on both rails)   i (n - i) * u
    optimal value                               lambda_2* = 2u = 6 / (n (n^2 + 2))

Positions are 1-based in the formulas above. In code the position of a rail
edge is 0-based, as in ladder.py, so the edge at position p has multiplier
(p + 1) (n - 1 - p).
"""

import numpy as np


def closed_form_unit(n: int) -> float:
    """The rung conductance u = 3 / (n (n^2 + 2)) of the conjectured optimum."""

    return 3.0 / (n * (n**2 + 2))


def closed_form_lambda2(n: int) -> float:
    """Conjectured optimal algebraic connectivity 6 / (n (n^2 + 2))."""

    return 2.0 * closed_form_unit(n)


def closed_form_w(n: int) -> np.ndarray:
    """Conjectured optimal conductances, in the edge order of make_ladder(n).

    n rungs, then n-1 top-rail edges, then n-1 bottom-rail edges. The entries
    sum to 1 (the total is n (n^2 + 2) / 3 units).
    """

    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    p = np.arange(n - 1)          # 0-based position of a rail edge
    rail = (p + 1) * (n - 1 - p)  # multiple of u
    return closed_form_unit(n) * np.concatenate([np.ones(n), rail, rail])
