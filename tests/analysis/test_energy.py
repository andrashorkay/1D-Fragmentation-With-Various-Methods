# SPDX-License-Identifier: GPL-2.0-only
"""Method-specific energy definitions and conservation invariants."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.calculations.energy import (
    energy_balance_history,
    energy_components,
    energy_partition,
    estimate_initial_energy,
    recoverable_energy,
    summarise_energy,
)


def test_pd_balance_uses_recorded_total_not_displayed_components(energy_history):
    # The mock PD recorded total includes energy absent from plotted components.
    assert energy_history["total_energy"].iloc[-1] != pytest.approx(
        energy_history.loc[2, ["kinetic_energy", "recoverable_energy", "fracture_energy"]].sum()
    )
    balance = energy_balance_history(energy_history)
    assert estimate_initial_energy(energy_history) == pytest.approx(5.0)
    np.testing.assert_allclose(balance["energy_error"], 0, atol=1e-12)
    np.testing.assert_allclose(balance["relative_energy_error"], 0, atol=1e-12)
    result = summarise_energy(energy_history)
    assert result.last_total_energy_J == pytest.approx(5.4)
    assert result.last_recoverable_energy_J == pytest.approx(0.4)
    assert result.max_relative_energy_error == pytest.approx(0.0, abs=1e-12)


def test_czm_recoverable_includes_reversible_cohesive_energy():
    frame = pd.DataFrame(
        {
            "time": [0.0], "kinetic_energy": [1.0],
            "elastic_energy": [2.0], "reversible_cohesive_energy": [0.5],
            "fracture_energy": [1.0], "contact_energy": [0.25],
        }
    )
    assert recoverable_energy(frame).iloc[0] == pytest.approx(2.5)
    components = energy_components(frame)
    assert components["recoverable_energy"].iloc[0] == pytest.approx(2.5)
    fractions = energy_partition(frame)
    assert fractions["recoverable_fraction"].iloc[0] == pytest.approx(2.5 / 4.75)
    assert fractions.filter(like="_fraction").iloc[0].sum() == pytest.approx(1.0)


def test_zero_partition_is_undefined():
    frame = pd.DataFrame(
        {"time": [0.0], "kinetic_energy": [0.0],
         "elastic_energy": [0.0], "fracture_energy": [0.0]}
    )
    part = energy_partition(frame)
    assert np.isnan(part["kinetic_fraction"].iloc[0])


def test_initial_energy_inference_requires_t0_or_error_column(energy_history):
    sample = energy_history.loc[1:].drop(columns="energy_error")
    with pytest.raises(ValueError, match="Cannot infer initial energy"):
        energy_balance_history(sample)
    balance = energy_balance_history(sample, initial_energy=5.0)
    assert balance["energy_error"].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_invalid_time_history_is_rejected(energy_history):
    broken = energy_history.copy()
    broken.loc[2, "time"] = -1
    with pytest.raises(ValueError):
        energy_balance_history(broken)
