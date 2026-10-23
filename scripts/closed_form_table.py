"""Compare the SDP solver with the conjectured closed form for n = 1..nmax.

Run from the repo root:
    python scripts/closed_form_table.py            # n = 1..30
    python scripts/closed_form_table.py --nmax 50  # slower: n = 50 takes about a minute

Prints a table and writes results/closed_form_errors.csv and
results/closed_form_errors.pdf/.png. Columns:
    lam_err    relative difference of the solver's optimal value from 6/(n(n^2+2))
    lam_w_err  the same for lambda_2 recomputed (numpy) from the solver's w
    rail_err   largest relative error of w over the rail edges
    rung_err   largest relative error of w over the rungs (the optimum is very
               flat in the rungs, so this is the noisiest one)
"""

import argparse
import csv
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gap_maximizing_on_ladders import (
    RUNG,
    closed_form_lambda2,
    closed_form_w,
    make_ladder,
    solve_gap_sdp,
)

FIELDS = ["n", "lambda2_closed_form", "lambda2_solver", "lam_err", "lam_w_err",
          "rail_err", "rung_err", "status", "secs"]


def main():
    parser = argparse.ArgumentParser(description="Solver versus closed form.")
    parser.add_argument("--nmax", type=int, default=30, help="largest n (default: 30)")
    parser.add_argument("--outdir", default="results", help="output folder (default: results)")
    args = parser.parse_args()

    # Clarabel often reports "optimal_inaccurate"; the status column shows it.
    warnings.filterwarnings("ignore", message="Solution may be inaccurate")

    rows = []
    print(f"{'n':>3}  {'lambda2':>11}  {'lam_err':>8}  {'lam_w_err':>9}  "
          f"{'rail_err':>8}  {'rung_err':>8}  {'status':>18}  {'secs':>6}")
    for n in range(1, args.nmax + 1):
        lad = make_ladder(n)
        lam, w_cf = closed_form_lambda2(n), closed_form_w(n)

        start = time.perf_counter()
        sol = solve_gap_sdp(lad.B)
        secs = time.perf_counter() - start

        err = np.abs(sol.w - w_cf) / w_cf
        rungs = lad.kind == RUNG
        row = dict(
            n=n, lambda2_closed_form=lam, lambda2_solver=sol.lambda2,
            lam_err=abs(sol.lambda2 - lam) / lam,
            lam_w_err=abs(sol.lambda2_check - lam) / lam,
            rail_err=err[~rungs].max(initial=0.0), rung_err=err[rungs].max(),
            status=sol.status, secs=secs,
        )
        rows.append(row)
        print(f"{n:>3}  {lam:>11.6g}  {row['lam_err']:>8.1e}  {row['lam_w_err']:>9.1e}  "
              f"{row['rail_err']:>8.1e}  {row['rung_err']:>8.1e}  {sol.status:>18}  {secs:>6.2f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "closed_form_errors.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Error against n, log scale (floor 1e-16 so exact zeros stay visible).
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for key, label, marker in [("lam_err", r"$\lambda_2$ (solver value)", "o"),
                               ("lam_w_err", r"$\lambda_2$ of solver $w$", "x"),
                               ("rail_err", "rail conductances, max", "s"),
                               ("rung_err", "rung conductances, max", "^")]:
        ax.semilogy(ns, np.maximum([r[key] for r in rows], 1e-16), marker + "-", ms=4, lw=1,
                    label=label)
    ax.set_xlabel("number of rungs n")
    ax.set_ylabel("relative error vs closed form")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "closed_form_errors.pdf")
    fig.savefig(outdir / "closed_form_errors.png", dpi=150)
    plt.close(fig)
    print(f"wrote {outdir / 'closed_form_errors.csv'} and closed_form_errors.pdf/.png")


if __name__ == "__main__":
    main()
