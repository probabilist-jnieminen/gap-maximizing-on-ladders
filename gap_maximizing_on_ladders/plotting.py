"""Drawing of a ladder with its edge conductances."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from .ladder import RUNG, TOP_RAIL

EDGE_CMAP = colormaps["Blues"]   # single-hue scale: darker = larger conductance
ZERO_COLOR = "0.7"               # light grey for edges with zero conductance
VERTEX_EDGE = "0.15"
LW_MIN, LW_MAX = 1.0, 6.0        # line widths (points) of the weakest and strongest edge
LABEL_SIZE = 9


def _trim(s: str) -> str:
    """'2.5000' -> '2.5', '3.0000' -> '3'."""
    return s.rstrip("0").rstrip(".") if "." in s else s


def conductance_labels(w, zero_tol: float = 1e-5, decimals: int = 4):
    """Express conductances as multiples of the smallest nonzero one.

    An edge counts as zero if w_e <= zero_tol * max(w); this separates true
    zeros from solver noise of order 1e-9. The unit u is the smallest
    remaining conductance. Returns u, the ratios w_e / u (0 for zero edges)
    and text labels rounded to `decimals` places without trailing zeros,
    e.g. "1", "2.5", "3.1416".
    """
    w = np.asarray(w, dtype=float)
    on = w > zero_tol * w.max()
    unit = float(w[on].min())
    ratios = np.where(on, w / unit, 0.0)
    labels = [_trim(f"{r:.{decimals}f}") for r in ratios]
    return unit, ratios, labels


def draw_ladder(lad, w, ax=None, zero_tol: float = 1e-5, decimals: int = 4):
    """Draw the ladder; each edge's width and shade grow with its conductance.

    Every edge is labelled with w_e / u (see conductance_labels); zero edges
    are dashed. The top rail is drawn on top, rungs are vertical.
    Returns (ax, u).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(max(5.0, lad.n + 0.5), 2.6))

    n = lad.n
    w = np.asarray(w, dtype=float)
    unit, ratios, labels = conductance_labels(w, zero_tol, decimals)
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
