"""Ladder graph with n rungs: construction, edge ordering, incidence matrix.

Conventions (the rest of the package relies on these):

* Vertex (position i, side s) has index s*n + i, so the top rail is
  0..n-1 and the bottom rail is n..2n-1. This is networkx's own labelling.
* Edges are ordered: the n rungs, then the n-1 top-rail edges, then the
  n-1 bottom-rail edges, each group by increasing position. A conductance
  vector w follows this order, so w[i] belongs to ladder.edges[i].
* Position of a rung is its index i (0..n-1). Position of a rail edge
  joining rail positions i and i+1 is i (0..n-2).
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np

RUNG, TOP_RAIL, BOTTOM_RAIL = 0, 1, 2


@dataclass(frozen=True, eq=False)
class Ladder:
    """A ladder graph together with its labelled, ordered edges."""

    n: int                       # number of rungs (the graph has 2n vertices)
    graph: nx.Graph              # the networkx graph itself
    edges: list                  # ordered list of (u, v) pairs with u < v
    kind: np.ndarray             # per edge: RUNG, TOP_RAIL or BOTTOM_RAIL
    position: np.ndarray         # per edge: position along the ladder
    B: np.ndarray                # incidence matrix, shape (3n-2, 2n)


def make_ladder(n: int) -> Ladder:
    """Build the ladder graph with n rungs (n >= 1)."""
    
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    G = nx.ladder_graph(n)

    # Label every edge by looking at its endpoints, then sort into the
    # agreed order (kind first, then position).
    labelled = []
    for a, b in G.edges:
        u, v = min(a, b), max(a, b)
        if u < n <= v:            # one end on each rail: a rung
            labelled.append((RUNG, u, (u, v)))
        elif v < n:               # both ends on the top rail
            labelled.append((TOP_RAIL, u, (u, v)))
        else:                     # both ends on the bottom rail
            labelled.append((BOTTOM_RAIL, u - n, (u, v)))
    labelled.sort()

    edges = [e for _, _, e in labelled]
    kind = np.array([k for k, _, _ in labelled])
    position = np.array([p for _, p, _ in labelled])

    # networkx returns vertices x edges; we want one row per edge.
    B = nx.incidence_matrix(
        G, nodelist=range(2 * n), edgelist=edges, oriented=True
    ).toarray().T

    return Ladder(n=n, graph=G, edges=edges, kind=kind, position=position, B=B)
