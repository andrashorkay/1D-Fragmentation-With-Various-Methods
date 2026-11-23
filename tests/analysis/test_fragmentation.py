# SPDX-License-Identifier: GPL-2.0-only
"""Fragment statistics and strain-rate scaling regression tests."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.calculations.fragmentation import (
    final_fragment_statistics,
    fit_strain_rate_scaling,
    fragment_statistics,
)


def test_fragment_statistics_lengths_and_masses(fragments):
    result = fragment_statistics(fragments)
    assert result.count == 3
    assert result.total_size == pytest.approx(0.004)
    assert result.mean_size == pytest.approx(0.004 / 3)
    assert result.median_size == pytest.approx(0.001)
    assert result.min_size == pytest.approx(0.001)
    assert result.max_size == pytest.approx(0.002)
    assert result.total_mass == pytest.approx(0.4)
    assert result.mean_mass == pytest.approx(0.4 / 3)


def test_mass_is_optional_for_pd_raw_fragments(fragments):
    result = fragment_statistics(fragments.drop(columns="mass"))
    assert result.mean_mass is None
    assert result.total_mass is None


def test_only_last_snapshot_is_used(make_case, tmp_path):
    case = make_case("PD", tmp_path)
    old = case.fragments.iloc[[0]].copy()
    old["step"] = 1
    old["time"] = 0.0
    old["size"] = 0.004
    case.fragments = pd.concat([old, case.fragments], ignore_index=True)
    result = final_fragment_statistics(case)
    assert result.count == 3
    assert result.total_size == pytest.approx(0.004)


def test_exact_power_law_recovers_exponent():
    rates = np.array([1e3, 2e3, 1e4, 1e5, 1e6])
    sizes = 0.002 * (rates / 1e3) ** (-2 / 3)
    fit = fit_strain_rate_scaling(rates, sizes)
    assert fit.exponent == pytest.approx(-2 / 3, abs=1e-12)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.n_points == 5
    limited = fit_strain_rate_scaling(rates, sizes, fit_min=1e4)
    assert limited.n_points == 3
    assert limited.exponent == pytest.approx(-2 / 3, abs=1e-12)


@pytest.mark.parametrize("bad", [[0.0, 1.0], [-1.0, 1.0], [1.0, np.nan]])
def test_invalid_fragment_sizes_rejected(bad):
    with pytest.raises(ValueError):
        fragment_statistics(pd.DataFrame({"size": bad}))


def test_rate_fit_requires_two_distinct_rates():
    with pytest.raises(ValueError, match="distinct"):
        fit_strain_rate_scaling([1e5, 1e5], [0.001, 0.002])
