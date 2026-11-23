# SPDX-License-Identifier: GPL-2.0-only
"""Plotting utilities for numerical convergence studies.

This module is deliberately independent of file I/O and numerical post-
processing. It expects tabular convergence summaries produced by
``src.analysis.calculations.convergence`` and returns Matplotlib figures.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_PARAMETER_LABELS = {
    "dt": "Time-step parameter",
    "dt_factor": "Time-step factor",
    "dx": "Mesh spacing",
    "number_elements": "Number of elements",
    "horizon": "Horizon",
    "l0": r"Regularisation length $l_0$",
    "l0_scale": r"Regularisation-length scale factor",
}

_QUANTITY_LABELS = {
    "peak_average_stress_Pa": "Peak average stress [Pa]",
    "final_fracture_energy_J": "Final fracture energy [J]",
    "final_recoverable_energy_J": "Final recoverable energy [J]",
    "final_kinetic_energy_J": "Final kinetic energy [J]",
    "max_relative_energy_error": "Maximum relative energy error [-]",
    "raw_num_fragments": "Raw number of fragments [-]",
    "num_fragments": "Number of fragments [-]",
    "mean_fragment_size_m": "Mean fragment size [m]",
    "simulation_wall_time_s": "Simulation wall time [s]",
}


def _column_name(base: str, mode: str) -> str:
    if mode == "value":
        return base
    if mode == "successive":
        return f"relative_change_{base}"
    if mode == "reference":
        return f"relative_error_{base}"
    raise ValueError(
        f"Unknown mode '{mode}'. Expected 'value', 'successive' or 'reference'."
    )


def _parameter_label(parameter: str) -> str:
    return _PARAMETER_LABELS.get(parameter, parameter)


def _quantity_label(quantity: str, mode: str) -> str:
    base = _QUANTITY_LABELS.get(quantity, quantity)
    if mode == "value":
        return base
    if mode == "successive":
        return f"Successive relative change of {base}"
    if mode == "reference":
        return f"Relative error to reference of {base}"
    return base


def _prepare_xy(
    table: pd.DataFrame,
    parameter: str,
    quantity: str,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    if parameter not in table.columns:
        raise KeyError(f"Parameter column '{parameter}' not found")

    y_column = _column_name(quantity, mode)
    if y_column not in table.columns:
        raise KeyError(f"Convergence column '{y_column}' not found")

    x = pd.to_numeric(table[parameter], errors="coerce")
    y = pd.to_numeric(table[y_column], errors="coerce")
    mask = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())

    return x.to_numpy()[mask], y.to_numpy()[mask], table.loc[mask]


def plot_convergence_quantity(
    table: pd.DataFrame,
    *,
    parameter: str,
    quantity: str,
    mode: str = "value",
    method: str | None = None,
    log_x: bool | None = None,
    log_y: bool = False,
    percent: bool = True,
    invert_x: bool | None = None,
    ax=None,
):
    """Plot one convergence quantity against the varied parameter.

    Parameters
    ----------
    table:
        DataFrame returned by ``build_convergence_table``.
    parameter:
        Name of the varied parameter column.
    quantity:
        Base quantity name, e.g. ``final_fracture_energy_J``.
    mode:
        ``'value'`` plots the recorded quantity itself.
        ``'successive'`` plots successive relative changes.
        ``'reference'`` plots relative errors to the chosen reference case.
    method:
        Optional method label used in the title.
    log_x, log_y:
        Axis scaling controls.
    percent:
        When plotting relative changes or reference errors, multiply by 100.
    invert_x:
        If ``None``, invert the x-axis automatically for parameters where
        "smaller" means "finer" (time-step factor, mesh spacing, l0 scale).
    ax:
        Optional Matplotlib axes.
    """

    x, y, _ = _prepare_xy(table, parameter, quantity, mode)

    if percent and mode in {"successive", "reference"}:
        y = 100.0 * y

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(x, y, marker="o", linestyle="-")

    if log_x is None:
        log_x = parameter in {
            "dt",
            "dt_factor",
            "dx",
            "horizon",
            "l0",
            "l0_scale",
        }
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    if invert_x is None:
        invert_x = parameter in {"dt", "dt_factor", "dx", "l0", "l0_scale"}
    if invert_x:
        ax.invert_xaxis()

    ylabel = _quantity_label(quantity, mode)
    if percent and mode in {"successive", "reference"}:
        ylabel = ylabel + " [%]"

    ax.set_xlabel(_parameter_label(parameter))
    ax.set_ylabel(ylabel)

    if method:
        title = f"{method} convergence"
    else:
        title = "Convergence study"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    return fig, ax


def plot_convergence_comparison(
    table: pd.DataFrame,
    *,
    parameter: str,
    quantities: Sequence[str],
    mode: str = "value",
    method: str | None = None,
    percent: bool = True,
    log_x: bool | None = None,
    log_y: bool = False,
    ax=None,
):
    """Overlay several convergence quantities on one plot."""

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for quantity in quantities:
        x, y, _ = _prepare_xy(table, parameter, quantity, mode)
        if percent and mode in {"successive", "reference"}:
            y = 100.0 * y
        ax.plot(
            x,
            y,
            marker="o",
            linestyle="-",
            label=_QUANTITY_LABELS.get(quantity, quantity),
        )

    if log_x is None:
        log_x = parameter in {
            "dt",
            "dt_factor",
            "dx",
            "horizon",
            "l0",
            "l0_scale",
        }
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    if parameter in {"dt", "dt_factor", "dx", "l0", "l0_scale"}:
        ax.invert_xaxis()

    ylabel = "Value"
    if mode == "successive":
        ylabel = "Successive relative change"
    elif mode == "reference":
        ylabel = "Relative error to reference"
    if percent and mode in {"successive", "reference"}:
        ylabel += " [%]"

    ax.set_xlabel(_parameter_label(parameter))
    ax.set_ylabel(ylabel)
    ax.set_title(f"{method} convergence" if method else "Convergence study")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    return fig, ax


def plot_convergence_tolerance(
    table: pd.DataFrame,
    *,
    parameter: str,
    quantity: str,
    tolerance: float,
    mode: str = "successive",
    method: str | None = None,
    percent: bool = True,
    ax=None,
):
    """Plot one convergence error quantity with a tolerance line."""

    fig, ax = plot_convergence_quantity(
        table,
        parameter=parameter,
        quantity=quantity,
        mode=mode,
        method=method,
        percent=percent,
        ax=ax,
    )

    tol = 100.0 * tolerance if percent else tolerance
    ax.axhline(tol, linestyle="--")

    return fig, ax
