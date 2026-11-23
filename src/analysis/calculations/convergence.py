# SPDX-License-Identifier: GPL-2.0-only
"""Method-independent convergence calculations for CZM, PD and PF.

No file I/O or plotting belongs in this module. Values retain their original
units; relative changes and errors are dimensionless (not percentages).

IMPORTANT: Rows must be ordered from the coarsest/least refined setup to the
finest/reference setup. For an ordinary timestep study, this means decreasing
``dt`` or ``time_step_factor``. Changing the PD horizon or PF ``l0`` is not
necessarily a convergence refinement: the user must choose a meaningful order
and reference configuration for those studies.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def _numeric_vector(values: ArrayLike, name: str) -> np.ndarray:
    """Return a non-empty, one-dimensional array of finite numbers."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numerical values")
    return array


def relative_change(
    previous: float, current: float, *, denominator: str = "current"
) -> float:
    """Return the absolute relative difference between two values.

    By default, use the *current* (finer) value in the denominator, matching
    the convention in the original PD timestep study:

        abs(current - previous) / abs(current)

    ``denominator='previous'`` uses the previous (coarser) value instead.
    A zero denominator gives NaN unless both values are zero (change = 0).
    """
    if denominator not in {"current", "previous"}:
        raise ValueError("denominator must be 'current' or 'previous'")
    if not np.isfinite(previous) or not np.isfinite(current):
        raise ValueError("Comparison values must be finite")

    baseline = current if denominator == "current" else previous
    if baseline == 0:
        return 0.0 if previous == current else float("nan")
    return float(abs(current - previous) / abs(baseline))


def successive_relative_changes(
    values: ArrayLike, *, denominator: str = "current"
) -> np.ndarray:
    """Compare each entry with the preceding entry, in the supplied order.

    The first entry is NaN (no preceding case). Return dimensionless fractions,
    e.g. 0.02 represents a 2% change.
    """
    array = _numeric_vector(values, "values")
    changes = np.full(array.size, np.nan, dtype=float)
    for i in range(1, array.size):
        changes[i] = relative_change(
            array[i - 1], array[i], denominator=denominator
        )
    return changes


def successive_absolute_changes(values: ArrayLike) -> np.ndarray:
    """Absolute differences between successive entries (useful for counts)."""
    array = _numeric_vector(values, "values")
    changes = np.full(array.size, np.nan, dtype=float)
    changes[1:] = np.abs(np.diff(array))
    return changes


def relative_errors_to_reference(
    values: ArrayLike, *, reference_index: int = -1
) -> np.ndarray:
    """Compute abs(value - reference) / abs(reference) for every case.

    The reference is explicitly selected by its row index (last by default).
    If reference == 0, only identical zero values receive an error of zero;
    other entries receive NaN because a relative error is undefined.
    """
    array = _numeric_vector(values, "values")
    try:
        reference = float(array[reference_index])
    except IndexError as exc:
        raise ValueError("reference_index is outside the data range") from exc

    if reference == 0:
        errors = np.full(array.size, np.nan, dtype=float)
        errors[array == 0] = 0.0
        return errors
    return np.abs(array - reference) / abs(reference)


def build_convergence_table(
    data: pd.DataFrame,
    *,
    parameter: str,
    quantities: Sequence[str],
    reference_index: int = -1,
    sort_ascending: bool | None = None,
    relative_denominator: str = "current",
    absolute_quantities: Sequence[str] = ("raw_num_fragments", "num_fragments"),
) -> pd.DataFrame:
    """Add relative changes and reference errors to one convergence study.

    Parameters
    ----------
    data:
        One row per simulation, with the varied ``parameter`` and measured
        ``quantities`` as columns. No file I/O is performed here.
    parameter:
        Column containing the controlled variable (e.g. ``dt_factor``).
    quantities:
        Continuous quantities to compare, such as ``final_fracture_energy_J``
        or ``peak_average_stress_Pa``.
    reference_index:
        Index into the ordered rows for reference errors. Defaults to the last
        case, which should be the finest-resolution case in a convergence study.
    sort_ascending:
        ``None`` preserves the caller's order. ``False`` is useful for timestep
        factors descending from coarse to fine. Do not sort by horizon or l0
        without a justified reference choice.
    relative_denominator:
        Use ``'current'`` (default) or ``'previous'`` for successive changes.
    absolute_quantities:
        Discrete diagnostic counts for which absolute, not relative, changes
        are added if the columns are present. Missing diagnostic columns are
        ignored; missing requested ``quantities`` are errors.

    Returns
    -------
    pd.DataFrame
        Copy of input with ``relative_change_<quantity>`` and
        ``relative_error_<quantity>`` columns. For counts, adds
        ``absolute_change_<quantity>``.
    """
    if data.empty:
        raise ValueError("Convergence table cannot be empty")
    if parameter not in data.columns:
        raise KeyError(f"Varied parameter column '{parameter}' not found")
    if not quantities:
        raise ValueError("Provide at least one convergence quantity")
    if len(set(quantities)) != len(quantities):
        raise ValueError("Convergence quantities must be unique")
    missing = sorted(set(quantities).difference(data.columns))
    if missing:
        raise KeyError(f"Missing convergence quantities: {missing}")
    if relative_denominator not in {"current", "previous"}:
        raise ValueError("relative_denominator must be 'current' or 'previous'")

    result = data.copy()
    if sort_ascending is not None:
        result = result.sort_values(parameter, ascending=sort_ascending)
    result = result.reset_index(drop=True)

    # Check reference_index against the reordered data even before calculating.
    if not (-len(result) <= reference_index < len(result)):
        raise ValueError("reference_index is outside the data range")

    for quantity in quantities:
        values = _numeric_vector(result[quantity].to_numpy(), quantity)
        result[f"relative_change_{quantity}"] = successive_relative_changes(
            values, denominator=relative_denominator
        )
        result[f"relative_error_{quantity}"] = relative_errors_to_reference(
            values, reference_index=reference_index
        )

    for quantity in absolute_quantities:
        if quantity in result.columns:
            result[f"absolute_change_{quantity}"] = successive_absolute_changes(
                result[quantity].to_numpy()
            )

    return result


def check_convergence(
    table: pd.DataFrame,
    *,
    quantities: Sequence[str],
    tolerance: float = 0.02,
) -> pd.DataFrame:
    """Assess changes between the last two cases against a tolerance.

    The input must be the output of :func:`build_convergence_table`, ordered
    coarse-to-fine. This checks successive changes, not proof of asymptotic
    convergence. Discrete fragment counts and the energy-balance residual should
    be treated as diagnostics rather than automatic pass/fail criteria.

    ``tolerance`` is a fraction: 0.02 means 2%. Undefined relative changes
    are reported as not passing, not silently treated as zero.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if len(table) < 2:
        raise ValueError("At least two cases are required to check convergence")
    if not quantities:
        raise ValueError("Provide at least one convergence quantity")

    rows = []
    for quantity in quantities:
        col = f"relative_change_{quantity}"
        if col not in table.columns or quantity not in table.columns:
            raise KeyError(f"Missing convergence columns for '{quantity}'")
        change = float(table[col].iloc[-1])
        rows.append(
            {
                "quantity": quantity,
                "previous_value": float(table[quantity].iloc[-2]),
                "current_value": float(table[quantity].iloc[-1]),
                "relative_change": change,
                "relative_change_percent": 100.0 * change,
                "tolerance_percent": 100.0 * tolerance,
                "passes_tolerance": bool(
                    np.isfinite(change) and change <= tolerance
                ),
            }
        )
    return pd.DataFrame(rows)
