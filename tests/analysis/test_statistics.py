# SPDX-License-Identifier: GPL-2.0-only
"""Empirical fragment distributions and normalisation."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.calculations.statistics import (
    distribution_statistics,
    empirical_ccdf,
    empirical_cdf,
    fragment_values,
    histogram_distribution,
)


def test_length_normalisation_has_unit_mean(fragments):
    original = fragments.copy(deep=True)
    values = fragment_values(fragments, normalised=True)
    assert values.mean() == pytest.approx(1.0)
    pd.testing.assert_frame_equal(fragments, original)


def test_distribution_quantiles_and_variation(fragments):
    stats = distribution_statistics(fragment_values(fragments))
    assert stats.count == 3
    assert stats.total == pytest.approx(0.004)
    assert stats.mean == pytest.approx(0.004 / 3)
    assert stats.standard_deviation == pytest.approx(np.std([0.001, 0.002, 0.001]))
    assert stats.q25 <= stats.median <= stats.q75
    assert stats.coefficient_of_variation > 0


def test_empirical_cdf_and_survival():
    x, cdf = empirical_cdf([1, 1, 2, 4])
    survival_x, ccdf = empirical_ccdf([1, 1, 2, 4])
    np.testing.assert_allclose(x, [1, 2, 4])
    np.testing.assert_allclose(cdf, [0.5, 0.75, 1.0])
    np.testing.assert_allclose(survival_x, x)
    np.testing.assert_allclose(ccdf, [0.5, 0.25, 0.0])


def test_histogram_density_integrates_to_one():
    heights, edges = histogram_distribution([1, 2, 3, 4], bins=3, density=True)
    assert np.dot(heights, np.diff(edges)) == pytest.approx(1.0)
    counts, _ = histogram_distribution([1, 2, 3, 4], bins=3, density=False)
    assert sum(counts) == 4


def test_missing_mass_is_error(fragments):
    with pytest.raises(ValueError, match="mass"):
        fragment_values(fragments.drop(columns="mass"), quantity="mass")


def test_zero_mean_cannot_be_normalised():
    with pytest.raises(ValueError, match="zero mean"):
        fragment_values(pd.DataFrame({"mass": [0, 0]}), quantity="mass", normalised=True)
