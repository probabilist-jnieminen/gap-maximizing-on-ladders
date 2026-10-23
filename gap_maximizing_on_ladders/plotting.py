"""Drawing of a ladder with its edge conductances."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from .ladder import BOTTOM_RAIL, RUNG, TOP_RAIL

EDGE_CMAP = colormaps["Blues"]   # single-hue scale: darker = larger conductance
ZERO_COLOR = "0.7"               # light grey for edges with zero conductance
VERTEX_EDGE = "0.15"
LW_MIN, LW_MAX = 1.0, 6.0        # line widths (points) of the weakest and strongest edge
LABEL_SIZE = 9


def _trim(s: str) -> str:
    """'2.5000' -> '2.5', '3.0000' -> '3'."""

    return s.rstrip("0").rstrip(".") if "." in s else s


def conductance_labels(w, zero_tol: float = 1e-5, decimals: int = 4, unit=None):
    """Express conductances as multiples of a unit (default: the smallest nonzero one).

    An edge counts as zero if w_e <= zero_tol * max(w); this separates true
    zeros from solver noise of order 1e-9. Unless `unit` is given, the unit u
    is the smallest remaining conductance. Returns u, the ratios w_e / u
    (0 for zero edges) and text labels rounded to `decimals` places without
    trailing zeros, e.g. "1", "2.5", "3.1416".
    """

    w = np.asarray(w, dtype=float)
    on = w > zero_tol * w.max()
    if unit is None:
        unit = float(w[on].min())
    unit = float(unit)
    ratios = np.where(on, w / unit, 0.0)
    labels = [_trim(f"{r:.{decimals}f}") for r in ratios]

    return unit, ratios, labels


def draw_ladder(lad, w, ax=None, zero_tol: float = 1e-5, decimals: int = 4, unit=None):
    """Draw the ladder; each edge's width and shade grow with its conductance.

    Every edge is labelled with w_e / u (see conductance_labels); zero edges
    are dashed. The top rail is drawn on top, rungs are vertical. Pass `unit`
    to label several ladders in the same unit. Returns (ax, u).
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(max(5.0, lad.n + 0.5), 2.6))

    n = lad.n
    w = np.asarray(w, dtype=float)
    unit, ratios, labels = conductance_labels(w, zero_tol, decimals, unit)
    scale = w / w.max()   # 1 for the strongest edge

    # Vertex v sits at x = its rail position, y = 1 (top rail) or 0 (bottom rail).
    xy = {v: (v % n, 1.0 if v < n else 0.0) for v in range(2 * n)}

    for (a, b), kind, s, r, label in zip(lad.edges, lad.kind, scale, ratios, labels):
        (x0, y0), (x1, y1) = xy[a], xy[b]
        if r > 0:
            ax.plot([x0, x1], [y0, y1], color=EDGE_CMAP(0.35 + 0.6 * s),
                    lw=LW_MIN + (LW_MAX - LW_MIN) * s, solid_capstyle="round", zorder=1)
        else:
            ax.plot([x0, x1], [y0, y1], color=ZERO_COLOR, lw=LW_MIN,
                    ls=(0, (3, 3)), zorder=1)

        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        if kind == RUNG:   # on the rung itself, with a white backing box
            ax.text(xm, ym, label, ha="center", va="center", fontsize=LABEL_SIZE, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
        elif kind == TOP_RAIL:
            ax.text(xm, ym + 0.12, label, ha="center", va="bottom", fontsize=LABEL_SIZE)
        else:
            ax.text(xm, ym - 0.12, label, ha="center", va="top", fontsize=LABEL_SIZE)

    xs, ys = zip(*xy.values())
    ax.scatter(xs, ys, s=24, c="white", edgecolors=VERTEX_EDGE, linewidths=1.0, zorder=2)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    
    return ax, unit


def draw_comparison(lad, w, w_ref, zero_tol: float = 1e-5, decimals: int = 4):
    """Three panels: ladder with w, ladder with w_ref, relative error per edge.

    w_ref is the reference (e.g. the closed form) and must be positive. Both
    ladders are labelled in multiples of the smallest nonzero reference
    conductance, so the labels can be compared edge by edge. The bottom panel
    shows |w_e - w_ref_e| / w_ref_e against position along the ladder (rungs
    at integer positions, rail edges halfway between), on a log scale with a
    floor at 1e-12. Returns (fig, axes).
    """

    n = lad.n
    w = np.asarray(w, dtype=float)
    w_ref = np.asarray(w_ref, dtype=float)
    unit = conductance_labels(w_ref, zero_tol, decimals)[0]

    fig, axes = plt.subplots(3, 1, figsize=(max(5.0, n + 0.5), 7.5),
                             gridspec_kw=dict(height_ratios=[1, 1, 1.1]))
    draw_ladder(lad, w, ax=axes[0], zero_tol=zero_tol, decimals=decimals, unit=unit)
    draw_ladder(lad, w_ref, ax=axes[1], zero_tol=zero_tol, decimals=decimals, unit=unit)
    axes[0].set_title("solver", loc="left", fontsize=10)
    axes[1].set_title("closed form (conjectured)", loc="left", fontsize=10)

    err = np.abs(w - w_ref) / w_ref
    x = np.where(lad.kind == RUNG, lad.position, lad.position + 0.5)
    ax = axes[2]
    for kind, marker, name in [(RUNG, "o", "rungs"), (TOP_RAIL, "s", "top rail"),
                               (BOTTOM_RAIL, "^", "bottom rail")]:
        sel = lad.kind == kind
        ax.semilogy(x[sel], np.maximum(err[sel], 1e-12), marker, ls="none", ms=4, label=name)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel("position along the ladder")
    ax.set_ylabel("relative error")
    ax.legend(loc="upper center", ncol=3, fontsize=8, frameon=False)

    return fig, axes
