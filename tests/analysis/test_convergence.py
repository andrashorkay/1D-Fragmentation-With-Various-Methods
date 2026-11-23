# SPDX-License-Identifier: GPL-2.0-only
"""Numerical regression against the completed PD timestep study."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.calculations.convergence import (
    build_convergence_table,
    check_convergence,
    relative_change,
    relative_errors_to_reference,
    successive_absolute_changes,
    successive_relative_changes,
)


@pytest.fixture
def pd_dt_results():
    return pd.DataFrame(
        {
            "dt_factor": [0.2, 0.1, 0.05, 0.025],
            "final_fracture_energy_J": [0.006444, 0.006416, 0.006409, 0.006396],
            "peak_average_stress_Pa": [2.292781e8, 2.476734e8, 2.476735e8, 2.517804e8],
            "raw_num_fragments": [97, 96, 96, 99],
        }
    )


def test_current_denominator_matches_original_pd_study():
    expected = abs(0.006396 - 0.006409) / 0.006396
    assert relative_change(0.006409, 0.006396) == pytest.approx(expected)
    assert relative_change(0.006409, 0.006396, denominator="previous") == pytest.approx(
        abs(0.006396 - 0.006409) / 0.006409
    )


def test_dt_table_and_finest_reference(pd_dt_results):
    table = build_convergence_table(
        pd_dt_results.sample(frac=1.0, random_state=1),
        parameter="dt_factor",
        quantities=["final_fracture_energy_J", "peak_average_stress_Pa"],
        sort_ascending=False,
    )
    np.testing.assert_allclose(table.dt_factor.to_numpy(), [0.2, 0.1, 0.05, 0.025])
    assert np.isnan(table["relative_change_final_fracture_energy_J"].iloc[0])
    assert table["relative_change_final_fracture_energy_J"].iloc[-1] == pytest.approx(
        abs(0.006396 - 0.006409) / 0.006396
    )
    assert table["relative_error_final_fracture_energy_J"].iloc[-1] == 0
    assert table["absolute_change_raw_num_fragments"].iloc[-1] == 3
    result = check_convergence(
        table,
        quantities=["final_fracture_energy_J", "peak_average_stress_Pa"],
        tolerance=0.02,
    )
    assert result["passes_tolerance"].all()


def test_zero_reference_and_undefined_relative_change():
    np.testing.assert_allclose(relative_errors_to_reference([3, 0], reference_index=-1)[1:], [0])
    assert np.isnan(relative_errors_to_reference([3, 0])[0])
    assert np.isnan(relative_change(2, 0))
    assert relative_change(0, 0) == 0


def test_absolute_changes_for_discrete_counts():
    values = successive_absolute_changes([1, 3, 3, 7])
    assert np.isnan(values[0])
    np.testing.assert_allclose(values[1:], [2, 0, 4])


def test_invalid_values_and_tolerances(pd_dt_results):
    with pytest.raises(ValueError):
        successive_relative_changes([1, np.nan])
    with pytest.raises(KeyError):
        build_convergence_table(pd_dt_results, parameter="dt_factor", quantities=["nonexistent"])
    table = build_convergence_table(pd_dt_results, parameter="dt_factor", quantities=["final_fracture_energy_J"])
    with pytest.raises(ValueError):
        check_convergence(table, quantities=["final_fracture_energy_J"], tolerance=-0.1)
