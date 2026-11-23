# SPDX-License-Identifier: GPL-2.0-only
"""Numerical convergence and sensitivity studies."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from ..io import load_case
from .common import (
    _fragment_length, display_or_close, output_path, save_csv, save_figure,
)

_CONVERGENCE_QUANTITIES = {
    "peak_stress": "peak_average_stress_Pa",
    "fracture_energy": "final_fracture_energy_J",
    "recoverable_energy": "final_recoverable_energy_J",
    "kinetic_energy": "final_kinetic_energy_J",
    "fragments": "raw_num_fragments",
    "mean_fragment_size": "mean_fragment_size_m",
    "energy_error": "max_relative_energy_error",
    "wall_time": "simulation_wall_time_s",
}


def _number_or_nan(value) -> float:
    """Return a finite numerical value or NaN for unavailable diagnostics."""
    try:
        result = float(value)
    except (ValueError, TypeError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _convergence_parameter(cases: list, parameter: str) -> str:
    """Select the recorded parameter shared by all cases."""
    candidates = {
        "dt": ("time_step_factor", "dt"),
        "dx": ("dx", "number_elements"),
        "horizon": ("horizon",),
        "l0": ("l0_scale", "l0"),
    }[parameter]
    for name in candidates:
        if all(np.isfinite(_number_or_nan(case.parameters.get(name))) for case in cases):
            return "dt_factor" if name == "time_step_factor" else name
    if parameter in {"dt", "l0"} and all(
        case.path.name.startswith(("factor_", "alpha_")) for case in cases
    ):
        return "dt_factor" if parameter == "dt" else "l0_scale"
    raise ValueError(
        f"Cannot find a common recorded '{parameter}' parameter in all cases"
    )


def _parameter_value(case, axis: str) -> float:
    """Extract the varying setting, using a case-folder fallback if needed."""
    name = "time_step_factor" if axis == "dt_factor" else axis
    value = _number_or_nan(case.parameters.get(name))
    if not np.isfinite(value):
        for prefix in ("factor_", "alpha_"):
            if case.path.name.startswith(prefix):
                value = _number_or_nan(case.path.name[len(prefix):])
                break
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid or missing {axis} for {case.path}")
    return value


def _convergence_record(case, axis: str) -> dict:
    """Extract one row of original-unit convergence diagnostics."""
    history = case.history
    summary = case.summary
    last = history.iloc[-1]

    recoverable = last.get("recoverable_energy")
    if recoverable is None:
        recoverable = _number_or_nan(last.get("elastic_energy")) + _number_or_nan(
            last.get("reversible_cohesive_energy", 0.0)
        )

    raw_count = summary.get("raw_num_fragments")
    if raw_count is None and case.diagnostics is not None:
        if "raw_num_fragments" in case.diagnostics.columns:
            raw_count = case.diagnostics["raw_num_fragments"].iloc[-1]
    if raw_count is None:
        raw_count = summary.get("num_fragments", last.get("num_fragments"))

    balance_error = _number_or_nan(summary.get("max_relative_energy_error"))
    if not np.isfinite(balance_error):
        from ..calculations.energy import energy_balance_history
        try:
            balance = energy_balance_history(history)
            balance_error = float(balance["relative_energy_error"].max())
        except (ValueError, KeyError):
            pass

    wall = summary.get("simulation_wall_time", last.get("simulation_wall_time"))

    try:
        size, _, _, _ = _fragment_length(case)
    except (KeyError, ValueError):
        size = float("nan")

    return {
        "case": case.path.name,
        "source_directory": str(case.path),
        axis: _parameter_value(case, axis),
        "peak_average_stress_Pa": _number_or_nan(history["average_stress"].max())
        if "average_stress" in history else float("nan"),
        "final_fracture_energy_J": _number_or_nan(last.get("fracture_energy")),
        "final_recoverable_energy_J": _number_or_nan(recoverable),
        "final_kinetic_energy_J": _number_or_nan(last.get("kinetic_energy")),
        "raw_num_fragments": _number_or_nan(raw_count),
        "mean_fragment_size_m": _number_or_nan(size),
        "max_relative_energy_error": balance_error,
        "simulation_wall_time_s": _number_or_nan(wall),
        "final_time_s": _number_or_nan(summary.get("physical_time", last.get("time"))),
    }


def run_convergence(args) -> pd.DataFrame:
    """Load completed cases, calculate differences and plot one quantity."""
    from ..calculations.convergence import build_convergence_table
    from ..plots.convergence import plot_convergence_quantity

    method = args.method.upper()
    study_dir = Path(args.input).expanduser().resolve()
    folders = sorted(path for path in study_dir.iterdir() if path.is_dir())
    cases = []
    for folder in folders:
        if not (folder / "summary.out").is_file():
            warnings.warn(f"Skipping unfinished case: {folder}", stacklevel=2)
            continue
        try:
            cases.append(load_case(method, folder))
        except (ValueError, KeyError, FileNotFoundError) as exc:
            warnings.warn(f"Skipping unreadable case {folder.name}: {exc}", stacklevel=2)

    if len(cases) < 2:
        raise ValueError(f"At least two completed cases are required in {study_dir}")

    axis = _convergence_parameter(cases, args.parameter)
    raw = pd.DataFrame(_convergence_record(case, axis) for case in cases)
    if raw[axis].duplicated().any():
        raise ValueError(f"Duplicate values of {axis} in {study_dir}")

    # The last case in the resulting table is the reference. For dt/dx,
    # decreasing values mean refinement; for N, increasing values do.
    # Horizon and l0 studies are sensitivities, not necessarily convergence.
    ascending = axis == "number_elements" or args.parameter in {"horizon", "l0"}
    raw = raw.sort_values(axis, ascending=ascending).reset_index(drop=True)

    quantity = _CONVERGENCE_QUANTITIES[args.quantity]
    if not np.isfinite(raw[quantity].to_numpy(dtype=float)).all():
        raise ValueError(f"The selected quantity '{args.quantity}' is unavailable in some cases")

    # Calculate all metrics available in every simulation; don't fail the
    # study because an optional metric is missing from an output format.
    numeric = [
        q for q in _CONVERGENCE_QUANTITIES.values()
        if q not in {"raw_num_fragments"}
        and np.isfinite(raw[q].to_numpy(dtype=float)).all()
    ]
    if quantity == "raw_num_fragments":
        # Counts are tracked as absolute differences, not relative errors.
        if args.mode != "value":
            raise ValueError("For fragment counts, use --mode value; see absolute_change_raw_num_fragments in the CSV")
    elif quantity not in numeric:
        numeric.append(quantity)

    table = build_convergence_table(
        raw,
        parameter=axis,
        quantities=numeric,
        sort_ascending=None,  # Already ordered coarse-to-fine.
    )

    fig, ax = plot_convergence_quantity(
        table,
        parameter=axis,
        quantity=quantity,
        method=method,
        mode=args.mode,
    )
    if args.tolerance is not None:
        ax.axhline(args.tolerance * 100, linestyle="--", label="Tolerance")
        ax.legend()
    fig.tight_layout()

    print(f"\n{method} {args.parameter} convergence: {study_dir}")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nReference: {table['case'].iloc[-1]} ({axis}={table[axis].iloc[-1]:g})")

    if args.save or args.output is not None:
        figure_path = output_path(
            method, study_dir, args.output,
            f"convergence_{args.parameter}_{args.quantity}_{args.mode}",
        )
        save_figure(fig, figure_path)
        summary_path = save_csv(table, figure_path, suffix="_summary")
        print(f"Figure: {figure_path}\nSummary: {summary_path}")

    display_or_close(fig, no_show=args.no_show)
    return table
