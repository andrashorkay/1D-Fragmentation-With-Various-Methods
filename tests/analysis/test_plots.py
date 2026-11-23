# SPDX-License-Identifier: GPL-2.0-only
"""Headless rendering checks (plots do not open windows or write files)."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.analysis.calculations.convergence import build_convergence_table
from src.analysis.calculations.energy import energy_balance_history, energy_partition
from src.analysis.calculations.fracture_locations import fracture_snapshots
from src.analysis.plots.convergence import plot_convergence_quantity
from src.analysis.plots.distributions import plot_fragment_distribution
from src.analysis.plots.energy import (
    plot_energy_balance,
    plot_energy_history,
    plot_energy_partition,
)
from src.analysis.plots.history import plot_history
from src.analysis.plots.spatial import plot_fracture_positions


@pytest.mark.parametrize("kind", ["hist", "cdf", "ccdf"])
def test_distribution_modes_render(kind, fragments):
    fig, ax = plot_fragment_distribution(fragments, kind=kind, label="PD")
    assert fig is ax.figure
    assert ax.get_xlabel()
    plt.close(fig)


def test_convergence_plot_uses_numeric_table():
    raw = pd.DataFrame({"dt_factor": [0.2, 0.1, 0.05], "value": [2.0, 1.0, 0.9]})
    table = build_convergence_table(raw, parameter="dt_factor", quantities=["value"])
    fig, ax = plot_convergence_quantity(
        table, parameter="dt_factor", quantity="value", mode="successive"
    )
    assert len(ax.lines) == 1
    assert ax.lines[0].get_ydata()[-1] == pytest.approx(100 * (0.1 / 0.9))
    plt.close(fig)


def test_energy_plot_modes(energy_history):
    fig, ax = plot_energy_history(energy_history)
    assert len(ax.lines) >= 3
    plt.close(fig)

    fig, ax = plot_energy_balance(energy_balance_history(energy_history))
    assert len(ax.lines) >= 1
    plt.close(fig)

    fig, ax = plot_energy_partition(energy_partition(energy_history), stacked=True)
    assert fig is ax.figure
    plt.close(fig)


def test_history_plot_uses_time_axis(energy_history):
    fig, ax = plot_history(energy_history, quantity="average_stress", time_unit="us")
    assert ax.lines[0].get_xdata()[-1] == pytest.approx(2.0)
    plt.close(fig)


def test_spatial_plot_can_stack_runs(make_case, tmp_path):
    pd_case = make_case("PD", tmp_path / "pd")
    pf_case = make_case("PF", tmp_path / "pf")
    snaps = fracture_snapshots(pd_case) + fracture_snapshots(pf_case)
    fig, ax = plot_fracture_positions(snaps)
    assert len(ax.get_yticks()) == 2
    plt.close(fig)
