# SPDX-License-Identifier: GPL-2.0-only
"""Plot energy histories, energy balance and energy partition.

This module only renders Matplotlib figures. It does not read simulation files,
calculate energy balances or save plots. Numerical inputs come from
``src.analysis.calculations.energy``.

Energies are supplied in joules, time in seconds, and fractions as numbers
between 0 and 1. The functions return ``(figure, axes)`` so that a workflow
can show, save, or overlay figures as needed.
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

TimeUnit = Literal["s", "ms", "us", "ns"]
EnergyUnit = Literal["J", "mJ", "uJ", "kJ"]
EnergyQuantity = Literal[
    "all",
    "kinetic",
    "recoverable",
    "fracture",
    "contact",
    "total",
    "external_work",
]
BalanceQuantity = Literal["relative", "absolute", "totals"]

_TIME_SCALES = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
_ENERGY_SCALES = {"J": 1.0, "mJ": 1e3, "uJ": 1e6, "kJ": 1e-3}
_ENERGY_LABELS = {
    "kinetic_energy": "Kinetic energy",
    "recoverable_energy": "Recoverable energy",
    "fracture_energy": "Fracture / phase-field energy",
    "contact_energy": "Contact energy",
    "total_energy": "Recorded total energy",
    "external_work": "External work",
}
_FRACTION_LABELS = {
    "kinetic_fraction": "Kinetic",
    "recoverable_fraction": "Recoverable",
    "fracture_fraction": "Fracture / phase-field",
    "contact_fraction": "Contact",
}


def _prepare_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    """Create axes if none were provided by the calling workflow."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
    return fig, ax


def _time_axis(data: pd.DataFrame, unit: TimeUnit) -> np.ndarray:
    """Validate and scale a history's time column."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Expected a non-empty Pandas DataFrame")
    if unit not in _TIME_SCALES:
        raise ValueError(f"Unknown time unit: {unit}")
    if "time" not in data:
        raise KeyError("Missing 'time' column")

    time = pd.to_numeric(data["time"], errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(time)) or np.any(time < 0):
        raise ValueError("Time must contain finite, non-negative values")
    if np.any(np.diff(time) < 0):
        raise ValueError("Time must be ordered in ascending order")
    return time * _TIME_SCALES[unit]


def _values(
    data: pd.DataFrame, column: str, *, allow_nan: bool = False
) -> np.ndarray:
    """Read a numerical column, rejecting non-finite values where required."""
    if column not in data:
        raise KeyError(f"Missing '{column}' column")
    values = pd.to_numeric(data[column], errors="raise").to_numpy(dtype=float)
    if np.any(np.isinf(values)) or (not allow_nan and np.any(np.isnan(values))):
        raise ValueError(f"Column '{column}' contains invalid values")
    return values


def _set_time_label(ax: Axes, time_unit: TimeUnit) -> None:
    ax.set_xlabel(f"Time ({'µs' if time_unit == 'us' else time_unit})")


def _set_energy_label(ax: Axes, energy_unit: EnergyUnit) -> None:
    ax.set_ylabel(f"Energy ({'µJ' if energy_unit == 'uJ' else energy_unit})")


def _check_energy_unit(unit: EnergyUnit) -> float:
    if unit not in _ENERGY_SCALES:
        raise ValueError(f"Unknown energy unit: {unit}")
    return _ENERGY_SCALES[unit]


def plot_energy_history(
    energies: pd.DataFrame,
    *,
    quantity: EnergyQuantity = "all",
    method: str | None = None,
    time_unit: TimeUnit = "us",
    energy_unit: EnergyUnit = "mJ",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot energy components from ``energy_components(case.history)``.

    ``quantity='all'`` plots kinetic, recoverable and fracture energy, plus
    contact energy if recorded. ``quantity='total'`` plots the solver-recorded
    total energy; it is never reconstructed by summing displayed components.
    For PD, displayed recoverable energy excludes ghost-region contributions.
    For PF, the 'fracture' component is the recorded phase-field energy.
    """
    available = {
        "kinetic": "kinetic_energy",
        "recoverable": "recoverable_energy",
        "fracture": "fracture_energy",
        "contact": "contact_energy",
        "total": "total_energy",
        "external_work": "external_work",
    }
    if quantity != "all" and quantity not in available:
        raise ValueError(f"Unknown energy quantity: {quantity}")

    times = _time_axis(energies, time_unit)
    scale = _check_energy_unit(energy_unit)
    columns = (
        ["kinetic_energy", "recoverable_energy", "fracture_energy"]
        if quantity == "all"
        else [available[quantity]]
    )
    if quantity == "all" and "contact_energy" in energies:
        columns.append("contact_energy")

    # Validate all data before drawing onto axes.
    series = {column: _values(energies, column) * scale for column in columns}
    fig, ax = _prepare_axes(ax)
    for column, values in series.items():
        label = _ENERGY_LABELS[column]
        if method:
            label = f"{method.upper()} — {label}"
        ax.plot(times, values, label=label)

    _set_time_label(ax, time_unit)
    _set_energy_label(ax, energy_unit)
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig, ax


def plot_energy_balance(
    balance: pd.DataFrame,
    *,
    quantity: BalanceQuantity = "relative",
    time_unit: TimeUnit = "us",
    energy_unit: EnergyUnit = "mJ",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot conservation diagnostics from ``energy_balance_history(history)``.

    ``relative``: absolute relative error expressed in percent.
    ``absolute``: signed balance residual in the chosen energy unit.
    ``totals``: solver-recorded total versus initial energy + external work.

    Balance always uses recorded total energy, not a reconstructed sum.
    """
    if quantity not in ("relative", "absolute", "totals"):
        raise ValueError(f"Unknown energy-balance quantity: {quantity}")

    times = _time_axis(balance, time_unit)
    scale = _check_energy_unit(energy_unit)
    if quantity == "relative":
        values = (_values(balance, "relative_energy_error") * 100.0,)
    elif quantity == "absolute":
        values = (_values(balance, "energy_error") * scale,)
    else:
        values = (
            _values(balance, "total_energy") * scale,
            _values(balance, "expected_total_energy") * scale,
        )

    fig, ax = _prepare_axes(ax)
    if quantity == "relative":
        ax.plot(times, values[0], label="Relative energy-balance error")
        ax.set_ylabel("Relative energy-balance error (%)")
        ax.set_ylim(bottom=0)
    elif quantity == "absolute":
        ax.plot(times, values[0], label="Signed energy-balance residual")
        ax.axhline(0.0, linewidth=1, linestyle=":")
        ax.set_ylabel(
            f"Energy-balance residual ({'µJ' if energy_unit == 'uJ' else energy_unit})"
        )
    else:
        ax.plot(times, values[0], label="Recorded total energy")
        ax.plot(times, values[1], "--", label="Initial energy + external work")
        _set_energy_label(ax, energy_unit)

    _set_time_label(ax, time_unit)
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig, ax


def plot_energy_partition(
    fractions: pd.DataFrame,
    *,
    time_unit: TimeUnit = "us",
    stacked: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot energy fractions from ``energy_partition(case.history)``.

    Each fraction uses the sum of displayed components, *not* total energy.
    The optional stacked area is only valid for non-negative fractions.
    Rows where the tracked energy is zero have undefined (NaN) fractions;
    such rows are left unplotted.
    """
    times = _time_axis(fractions, time_unit)
    columns = [column for column in _FRACTION_LABELS if column in fractions]
    if not columns:
        raise KeyError("No recognised energy-fraction columns")

    series = [
        _values(fractions, column, allow_nan=True) * 100 for column in columns
    ]
    valid = np.logical_and.reduce([np.isfinite(values) for values in series])
    if not np.any(valid):
        raise ValueError("Energy partition has no valid recorded values")

    fig, ax = _prepare_axes(ax)
    labels = [_FRACTION_LABELS[column] for column in columns]
    if stacked:
        stacked_values = np.vstack([values[valid] for values in series])
        if np.any(stacked_values < -1e-9):
            raise ValueError(
                "Stacked partition requires non-negative fractions"
            )
        totals = stacked_values.sum(axis=0)
        if not np.allclose(totals, 100, rtol=1e-5, atol=1e-5):
            raise ValueError("Stacked partition fractions must sum to 100%")
        ax.stackplot(times[valid], *stacked_values, labels=labels, alpha=0.75)
        ax.set_ylim(0, 100)
    else:
        for values, label in zip(series, labels):
            ax.plot(times, values, label=label)

    _set_time_label(ax, time_unit)
    ax.set_ylabel("Fraction of tracked energy (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig, ax


__all__ = [
    "plot_energy_history",
    "plot_energy_balance",
    "plot_energy_partition",
]
