"""Phase 1 check: solve the ladder SDP for n = 2..20 and print a summary table.

Run from the repo root:  python scripts/run_baseline.py
"""

import time

import numpy as np

from gap_maximizing_on_ladders import algebraic_connectivity, make_ladder, solve_gap_sdp

ZERO_TOL = 1e-5


def main():
    print(f"{'n':>3}  {'lambda2*':>10}  {'uniform':>10}  {'bound':>8}  "
          f"{'rel_err':>9}  {'w~0':>4}  {'status':>8}  {'secs':>5}")
    for n in range(2, 21):
        lad = make_ladder(n)
        m = len(lad.edges)

        start = time.perf_counter()
        sol = solve_gap_sdp(lad.B)
        secs = time.perf_counter() - start

        uniform = algebraic_connectivity(lad.B, np.full(m, 1 / m))
        bound = 2 / (2 * n - 1)
        rel_err = abs(sol.lambda2 - sol.lambda2_check) / sol.lambda2
        n_zero = int(np.sum(sol.w <= ZERO_TOL * sol.w.max()))   # edges the optimum switches off

        print(f"{n:>3}  {sol.lambda2:>10.6f}  {uniform:>10.6f}  {bound:>8.4f}  "
              f"{rel_err:>9.1e}  {n_zero:>4}  {sol.status:>8}  {secs:>5.2f}")


if __name__ == "__main__":
    main()
