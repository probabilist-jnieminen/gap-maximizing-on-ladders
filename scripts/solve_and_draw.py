"""Solve the gap SDP for one ladder; save the conductances as CSV and drawings.

Run from the repo root, e.g. for n = 10 rungs:
    python scripts/solve_and_draw.py 10
This writes, in results/:
    ladder_n10.csv                    solved conductances, with the conjectured
                                      closed form and the relative error
    ladder_n10.pdf                    drawing of the solved ladder
    ladder_n10_vs_closed_form.pdf/png solver | closed form | relative error per edge
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gap_maximizing_on_ladders import (
    BOTTOM_RAIL,
    RUNG,
    TOP_RAIL,
    closed_form_lambda2,
    closed_form_w,
    conductance_labels,
    draw_comparison,
    draw_ladder,
    make_ladder,
    solve_gap_sdp,
)

KIND_NAME = {RUNG: "rung", TOP_RAIL: "top_rail", BOTTOM_RAIL: "bottom_rail"}


def main():
    parser = argparse.ArgumentParser(description="Solve and draw one ladder.")
    parser.add_argument("n", type=int, help="number of rungs")
    parser.add_argument("--outdir", default="results", help="output folder (default: results)")
    parser.add_argument("--zero-tol", type=float, default=1e-5,
                        help="edges with w <= zero_tol * max(w) count as zero (default: 1e-5)")
    args = parser.parse_args()

    lad = make_ladder(args.n)
    sol = solve_gap_sdp(lad.B)
    unit, ratios, _ = conductance_labels(sol.w, args.zero_tol)
    w_cf = closed_form_w(args.n)               # conjectured optimum, same edge order
    rel_err = np.abs(sol.w - w_cf) / w_cf
    rungs = lad.kind == RUNG

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"ladder_n{args.n}.csv"
    pdf_path = outdir / f"ladder_n{args.n}.pdf"
    cmp_stem = outdir / f"ladder_n{args.n}_vs_closed_form"

    # CSV: one row per edge, in the order of lad.edges. Full precision;
    # multiple_of_unit is 0 for edges counted as zero. closed_form is the
    # conjectured optimal conductance and rel_error = |conductance - closed_form| / closed_form.
    # Note that the unit is the smallest solved conductance, so it carries the
    # solver noise of that edge; closed_form is free of it.
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "position", "u", "v", "conductance", "multiple_of_unit",
                         "closed_form", "rel_error"])
        for (a, b), kind, pos, w_e, r, c, e in zip(
            lad.edges, lad.kind, lad.position, sol.w, ratios, w_cf, rel_err
        ):
            writer.writerow([KIND_NAME[kind], pos, a, b, w_e, r, c, e])

    # PDF drawing.
    plt.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts (cleaner PDF, text selectable)
    fig, ax = plt.subplots(figsize=(max(5.0, args.n + 0.5), 2.6))
    draw_ladder(lad, sol.w, ax=ax, zero_tol=args.zero_tol)
    ax.set_title(f"Gap-optimal conductances, n = {args.n}:   "
                 f"$\\lambda_2^\\star$ = {sol.lambda2:.6g}", loc="left", fontsize=11)
    caption = f"Labels: conductance in units of u = {unit:.4g} (smallest nonzero conductance)"
    if (ratios == 0).any():
        caption += "\nDashed edges: zero conductance"
    ax.text(0, 0, caption, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="0.35")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # Solver against the conjectured closed form.
    fig, axes = draw_comparison(lad, sol.w, w_cf, zero_tol=args.zero_tol)
    fig.suptitle(f"n = {args.n}:   $\\lambda_2^\\star$ = {sol.lambda2:.6g}   "
                 f"closed form {closed_form_lambda2(args.n):.6g}   "
                 f"max error: rails {rel_err[~rungs].max(initial=0):.1e}, rungs {rel_err[rungs].max():.1e}",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(cmp_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(cmp_stem.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"n = {args.n}: lambda2* = {sol.lambda2:.8g} "
          f"(numpy check {sol.lambda2_check:.8g}), status {sol.status}")
    print(f"closed form: lambda2 = {closed_form_lambda2(args.n):.8g}; max relative error of w: "
          f"rails {rel_err[~rungs].max(initial=0):.1e}, rungs {rel_err[rungs].max():.1e}")
    print(f"wrote {csv_path}, {pdf_path} and {cmp_stem.name}.pdf/.png")


if __name__ == "__main__":
    main()
