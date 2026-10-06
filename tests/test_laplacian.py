import numpy as np
import pytest

from gap_maximizing_on_ladders import algebraic_connectivity, laplacian, make_ladder

NS = range(1, 16)


def random_weights(m, seed):
    return np.random.default_rng(seed).uniform(0.1, 1.0, size=m)


@pytest.mark.parametrize("n", NS)
def test_symmetric_zero_row_sums_psd(n):
    lad = make_ladder(n)
    L = laplacian(lad.B, random_weights(len(lad.edges), seed=n))
    np.testing.assert_allclose(L, L.T, atol=1e-12)
    np.testing.assert_allclose(L.sum(axis=1), 0, atol=1e-12)
    assert np.linalg.eigvalsh(L)[0] > -1e-12


@pytest.mark.parametrize("n", NS)
def test_matches_degree_minus_adjacency(n):
    """B^T diag(w) B equals D - A built independently from the edge list."""
    lad = make_ladder(n)
    w = random_weights(len(lad.edges), seed=n)
    A = np.zeros((2 * n, 2 * n))
    for (u, v), w_e in zip(lad.edges, w):
        A[u, v] = A[v, u] = w_e
    D = np.diag(A.sum(axis=1))
    np.testing.assert_allclose(laplacian(lad.B, w), D - A, atol=1e-12)


@pytest.mark.parametrize("n", NS)
def test_quadratic_form_and_trace(n):
    lad = make_ladder(n)
    w = random_weights(len(lad.edges), seed=n)
    L = laplacian(lad.B, w)
    x = np.random.default_rng(100 + n).normal(size=2 * n)
    expected = sum(w_e * (x[u] - x[v]) ** 2 for (u, v), w_e in zip(lad.edges, w))
    assert x @ L @ x == pytest.approx(expected)
    assert np.trace(L) == pytest.approx(2 * w.sum())


def test_single_rung():
    lad = make_ladder(1)
    assert algebraic_connectivity(lad.B, [1.0]) == pytest.approx(2.0)


def test_four_cycle_uniform():
    lad = make_ladder(2)
    w = np.full(4, 0.25)
    assert algebraic_connectivity(lad.B, w) == pytest.approx(0.5)


@pytest.mark.parametrize("n", NS)
def test_unit_weight_spectrum_exact(n):
    """Ladder = path x K2, so spectrum = {2-2cos(pi k/n)} and the same + 2."""
    lad = make_ladder(n)
    path = 2 - 2 * np.cos(np.pi * np.arange(n) / n)
    expected = np.sort(np.concatenate([path, path + 2]))
    got = np.linalg.eigvalsh(laplacian(lad.B, np.ones(len(lad.edges))))
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_wrong_weight_shape():
    lad = make_ladder(3)
    with pytest.raises(ValueError):
        laplacian(lad.B, np.ones(5))
