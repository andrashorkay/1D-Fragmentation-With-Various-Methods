# SPDX-License-Identifier: GPL-2.0-only
"""Energy histories, conservation and partition workflow."""

from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd

from ..io import load_case
from .common import (
    display_or_close, output_path, save_csv, save_figure, save_json,
)

def _energy_output_path(
    method: str,
    case_dir: Path,
    explicit: Path | None,
    *,
    view: str,
    quantity: str,
    stacked: bool,
) -> Path:
    """Choose a descriptive output path for a single energy figure."""
    label = (
        f"energy_{quantity}" if view == "history"
        else f"energy_balance_{quantity}" if view == "balance"
        else "energy_partition" + ("_stacked" if stacked else "")
    )
    return output_path(method, case_dir, explicit, label)


def run_energy(args) -> pd.DataFrame:
    """Load one case, analyse its recorded energies and create one figure.

    This also works on *partial* runs where the energy-history file is
    readable, even if there is no final ``summary.out``. A partial run must
    not be interpreted as physically completed.
    """
    from ..calculations.energy import (
        energy_balance_history,
        energy_components,
        energy_partition,
        summarise_energy,
    )
    from ..plots.energy import (
        plot_energy_balance,
        plot_energy_history,
        plot_energy_partition,
    )

    method = args.method.upper()
    case_dir = Path(args.input).expanduser().resolve()
    case = load_case(method, case_dir)

    view = getattr(args, "view", "history")
    quantity = getattr(args, "quantity", "all")
    balance_kind = getattr(args, "balance_kind", "relative")
    stacked = getattr(args, "stacked", False)
    time_unit = getattr(args, "time_unit", "us")
    energy_unit = getattr(args, "energy_unit", "mJ")
    initial_energy = getattr(args, "initial_energy", None)

    # Keep method-specific energy definitions in calculations/energy.py.
    components = energy_components(case.history)
    if view == "history":
        analysed = components
        fig, _ = plot_energy_history(
            analysed,
            method=method,
            quantity=quantity,
            time_unit=time_unit,
            energy_unit=energy_unit,
        )
        file_quantity = quantity
    elif view == "balance":
        analysed = energy_balance_history(
            components, initial_energy=initial_energy
        )
        fig, _ = plot_energy_balance(
            analysed,
            quantity=balance_kind,
            time_unit=time_unit,
            energy_unit=energy_unit,
        )
        file_quantity = balance_kind
    elif view == "partition":
        analysed = energy_partition(components)
        fig, _ = plot_energy_partition(
            analysed,
            time_unit=time_unit,
            stacked=stacked,
        )
        file_quantity = "all"
    else:
        raise ValueError(f"Unknown energy view: {view!r}")

    fig.tight_layout()

    # The last *recorded* row may precede the simulation's final state.
    is_completed = (case_dir / "summary.out").is_file()
    final_row = components.iloc[-1]
    report = {
        "method": method,
        "source_directory": str(case_dir),
        "run_status": "completed" if is_completed else "partial_or_unverified",
        "last_recorded_time_s": float(final_row["time"]),
        "n_samples": len(components),
        "last_kinetic_energy_J": float(final_row["kinetic_energy"]),
        "last_recoverable_energy_J": float(final_row["recoverable_energy"]),
        "last_fracture_energy_J": float(final_row["fracture_energy"]),
    }
    if "contact_energy" in components:
        report["last_contact_energy_J"] = float(final_row["contact_energy"])
    if "total_energy" in components:
        report["last_total_energy_J"] = float(final_row["total_energy"])
    if "external_work" in components:
        report["last_external_work_J"] = float(final_row["external_work"])

    # Balance diagnostics require a recorded total and external work, and
    # either an inferred or user-supplied initial energy. A missing balance
    # must not prevent plotting the individual energy components.
    if {"total_energy", "external_work"}.issubset(components.columns):
        try:
            energy_summary = summarise_energy(
                components, initial_energy=initial_energy
            )
        except (ValueError, KeyError) as exc:
            warnings.warn(f"Energy-balance summary unavailable: {exc}", stacklevel=2)
        else:
            report["initial_total_energy_J"] = energy_summary.initial_total_energy_J
            report["last_energy_error_J"] = energy_summary.last_energy_error_J
            report["max_relative_energy_error"] = (
                energy_summary.max_relative_energy_error
            )

    print(f"\n{method} energy analysis: {case_dir}")
    if not is_completed:
        print("WARNING: No summary.out; this may be an interrupted simulation.")
    print(f"Last recorded time: {report['last_recorded_time_s']:.6g} s")
    print(f"Kinetic energy:     {report['last_kinetic_energy_J']:.6g} J")
    print(f"Recoverable energy: {report['last_recoverable_energy_J']:.6g} J")
    print(f"Fracture energy:    {report['last_fracture_energy_J']:.6g} J")
    if "max_relative_energy_error" in report:
        print(
            "Maximum relative energy error: "
            f"{100 * report['max_relative_energy_error']:.6g}%"
        )

    if args.save or args.output is not None:
        path = _energy_output_path(
            method,
            case_dir,
            args.output,
            view=view,
            quantity=file_quantity,
            stacked=stacked,
        )
        save_figure(fig, path)
        csv_path = save_csv(analysed, path)
        json_path = save_json(report, path, allow_nan=True)
        print(f"Figure: {path}\nData: {csv_path}\nSummary: {json_path}")

    display_or_close(fig, no_show=args.no_show)
    return analysed
