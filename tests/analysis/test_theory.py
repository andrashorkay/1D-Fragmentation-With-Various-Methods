# SPDX-License-Identifier: GPL-2.0-only
"""Dimensional and scaling checks for Grady's analytical reference law."""

import numpy as np
import pytest

from src.analysis.calculations.theory import grady_fragment_size


def test_grady_formula_and_rate_scaling():
    rates = np.array([1e4, 1e5, 1e6])
    sizes = grady_fragment_size(rates, rho=3900, gc=50)
    np.testing.assert_allclose(
        sizes,
        (24 * 50 / (3900 * rates**2)) ** (1 / 3),
    )
    assert sizes[1] / sizes[0] == pytest.approx(10 ** (-2 / 3))
    assert sizes[2] / sizes[1] == pytest.approx(10 ** (-2 / 3))


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf])
def test_grady_rejects_invalid_rates(rate):
    with pytest.raises(ValueError):
        grady_fragment_size(rate, rho=3900, gc=50)


def test_grady_rejects_invalid_material_data():
    with pytest.raises(ValueError):
        grady_fragment_size([1e5], rho=0, gc=50)
    with pytest.raises(ValueError):
        grady_fragment_size([1e5], rho=3900, gc=-1)
