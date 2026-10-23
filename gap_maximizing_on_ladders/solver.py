"""Maximise the algebraic connectivity over a unit conductance budget (SDP).

Problem (P):   maximise   lambda_2(L(w))
               subject to w >= 0,  sum(w) = 1,

with L(w) = B^T diag(w) B. Let the columns of Q be an orthonormal basis of
the vectors orthogonal to the all-ones vector. The all-ones vector is always
in the kernel of L(w), so lambda_2(L(w)) = lambda_min(Q^T L(w) Q), and
lambda_min(M) >= t exactly when M - t I is PSD. Since L(w) is linear in w,
(P) is the semidefinite program

    maximise   t
    subject to Q^T L(w) Q - t I  PSD,   w >= 0,   sum(w) = 1.
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.linalg import null_space

from .ladder import make_ladder
from .laplacian import algebraic_connectivity

ACCEPTED_STATUSES = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

# Clarabel's default tolerances are about 1e-8. The optimiser w is only
# determined to roughly the square root of that, so ask for more.
# Only used for Clarabel; other solvers keep their own defaults.
CLARABEL_OPTS = dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=500)


@dataclass(frozen=True, eq=False)
class GapSolution:
    """Result of solve_gap_sdp."""

    w: np.ndarray          # optimal conductances, same edge order as the rows of B
    lambda2: float         # optimal value t of the SDP
    lambda2_check: float   # lambda_2(L(w)) recomputed with numpy from w
    status: str            # cvxpy status: "optimal" or "optimal_inaccurate"
    solver: str            # name of the solver that was used


def solve_gap_sdp(B: np.ndarray, solver: str = cp.CLARABEL, **solve_kwargs) -> GapSolution:
    """Maximise lambda_2(L(w)) over w >= 0, sum(w) = 1, for the graph with incidence matrix B.

    B is (edges x vertices), as returned by make_ladder. The default solver is
    Clarabel (an interior-point method, accurate to about 1e-8); pass
    solver="SCS" for a cross-check. Extra keyword arguments go to cvxpy's
    Problem.solve, e.g. verbose=True.
    """
    
    m, N = B.shape
    Q = null_space(np.ones((1, N)))   # N x (N-1), orthonormal columns
    C = B @ Q                         # so that Q^T L(w) Q = C^T diag(w) C

    # Solve with budget sum(w) = m (mean weight 1) so the numbers stay O(1).
    # lambda_2 is homogeneous of degree 1 in w, so the unit-budget solution is
    # w / m and the unit-budget optimal value is t / m.
    w = cp.Variable(m)
    t = cp.Variable()
    M = C.T @ cp.diag(w) @ C
    constraints = [
        0.5 * (M + M.T) - t * np.eye(N - 1) >> 0,
        w >= 0,
        cp.sum(w) == m,
    ]
    prob = cp.Problem(cp.Maximize(t), constraints)

    opts = {**CLARABEL_OPTS, **solve_kwargs} if solver == cp.CLARABEL else dict(solve_kwargs)
    prob.solve(solver=solver, **opts)

    w_val, t_val = w.value, t.value
    # (P) is always feasible and bounded, so anything else means solver trouble.
    if prob.status not in ACCEPTED_STATUSES or w_val is None or t_val is None:
        raise RuntimeError(f"solver {solver} returned status {prob.status!r}")

    # Remove solver noise: tiny negative weights become 0, then scale to unit budget.
    w_opt = np.clip(w_val, 0.0, None)
    w_opt /= w_opt.sum()

    return GapSolution(
        w=w_opt,
        lambda2=float(t_val) / m,
        lambda2_check=algebraic_connectivity(B, w_opt),
        status=prob.status,
        solver=prob.solver_stats.solver_name,
    )


def solve_ladder(n: int, solver: str = cp.CLARABEL, **solve_kwargs) -> GapSolution:
    """solve_gap_sdp for the ladder with n rungs; w follows make_ladder(n).edges."""
    
    return solve_gap_sdp(make_ladder(n).B, solver=solver, **solve_kwargs)
