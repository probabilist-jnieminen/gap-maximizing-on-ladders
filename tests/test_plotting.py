import numpy as np
import pytest
from matplotlib.figure import Figure   # draws without opening a window

from gap_maximizing_on_ladders import conductance_labels, draw_ladder, make_ladder


def test_labels_are_multiples_of_smallest():
    unit, ratios, labels = conductance_labels([0.1, 0.25, 0.65])
    assert unit == pytest.approx(0.1)
    np.testing.assert_allclose(ratios, [1.0, 2.5, 6.5])
    assert labels == ["1", "2.5", "6.5"]


def test_zero_edges_and_rounding():
    # 1e-12 is below zero_tol * max(w), so it counts as zero, not as the unit.
    unit, ratios, labels = conductance_labels([1e-12, 0.3, 0.3 * np.pi])
    assert unit == pytest.approx(0.3)
    assert ratios[0] == 0
    assert labels == ["0", "1", "3.1416"]


@pytest.mark.parametrize("n", [1, 2, 5])
def test_draw_ladder_runs(n):
    lad = make_ladder(n)
    m = len(lad.edges)
    ax = Figure().subplots()
    ax_out, unit = draw_ladder(lad, np.full(m, 1 / m), ax=ax)
    assert ax_out is ax
    assert unit == pytest.approx(1 / m)
    assert len(ax.texts) == m   # one label per edge
