"""Weighted Laplacian of a graph from its incidence matrix."""

import numpy as np


def laplacian(B: np.ndarray, w) -> np.ndarray:
    """Return L(w) = B^T diag(w) B.

    B is the (edges x vertices) incidence matrix and w the edge conductances,
    in the same edge order as the rows of B.
    """
    
    w = np.asarray(w, dtype=float)
    if w.shape != (B.shape[0],):
        raise ValueError(
            f"w must have shape ({B.shape[0]},), got {w.shape}"
        )
    
    return B.T @ np.diag(w) @ B


def algebraic_connectivity(B: np.ndarray, w) -> float:
    """Second-smallest eigenvalue of L(w)."""

    return float(np.linalg.eigvalsh(laplacian(B, w))[1])
