# SPDX-License-Identifier: GPL-2.0-only
"""Fragment-size and fragment-mass distribution workflow."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import load_case
from .common import (
    display_or_close, output_path, save_csv, save_figure, save_json,
)

def _distribution_output_path(
    method: str,
    case_dir: Path,
    explicit: Path | None,
    *,
    quantity: str,
    kind: str,
    normalised: bool,
    raw: bool,
    density: bool,
) -> Path:
    """Select a distinct filename for each distribution configuration."""
    name = f"distribution_{quantity}_{kind}"
    if normalised:
        name += "_normalised"
    if raw:
        name += "_raw"
    if kind == "hist" and not density:
        name += "_counts"
    return output_path(method, case_dir, explicit, name)


def run_distribution(args) -> pd.DataFrame:
    """Analyse one saved fragment snapshot and plot its distribution.

    Unlike the strain-rate workflow, this needs *individual fragment records*.
    The overall mean in the history cannot determine a size distribution.
    Incomplete cases are allowed but explicitly identified in the summary.

    The exported CSV contains plot data in display units; the separate
    samples CSV retains the original per-fragment measurements in SI units.
    """
    from ..calculations.statistics import (
        distribution_statistics,
        empirical_ccdf,
        empirical_cdf,
        fragment_values,
        histogram_distribution,
    )
    from ..plots.distributions import plot_fragment_distribution

    method = args.method.upper()
    case_dir = Path(args.input).expanduser().resolve()
    quantity = args.quantity
    kind = getattr(args, "kind", "hist")
    raw = bool(getattr(args, "raw", False))
    normalised = bool(getattr(args, "normalised", False))
    density = not bool(getattr(args, "counts", False))
    unit = getattr(args, "unit", None)
    log_x = bool(getattr(args, "log_x", False))
    bins = getattr(args, "bins", 30)

    if kind not in {"hist", "cdf", "ccdf"}:
        raise ValueError("Distribution kind must be 'hist', 'cdf' or 'ccdf'")
    if quantity not in {"length", "mass"}:
        raise ValueError("Distribution quantity must be 'length' or 'mass'")
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("--bins must be a positive integer")
    if raw and method != "PD":
        raise ValueError("--raw is available only for PD")
    if raw and quantity == "mass":
        raise ValueError("PD raw fragments have no mass; use filtered fragments")
    if normalised and unit is not None:
        raise ValueError("--unit cannot be specified with --normalised")
    if getattr(args, "counts", False) and kind != "hist":
        raise ValueError("--counts applies only to --kind hist")

    units = {
        "length": {"m": 1.0, "mm": 1e3, "um": 1e6},
        "mass": {"kg": 1.0, "g": 1e3, "mg": 1e6},
    }
    if unit is None:
        unit = "mm" if quantity == "length" else "mg"
    if unit not in units[quantity]:
        raise ValueError(
            f"Invalid unit {unit!r} for {quantity}; "
            f"expected one of {tuple(units[quantity])}"
        )

    case = load_case(method, case_dir)
    # The reader isolates the last snapshot for PD (step) and CZM/PF (time).
    fragments = case.final_fragments(raw=raw)
    values_si = fragment_values(fragments, quantity=quantity)
    stats = distribution_statistics(values_si, quantity=quantity)
    values = fragment_values(
        fragments, quantity=quantity, normalised=normalised
    )
    displayed = values * (1.0 if normalised else units[quantity][unit])

    fig, _ = plot_fragment_distribution(
        fragments,
        quantity=quantity,
        kind=kind,
        bins=bins,
        normalised=normalised,
        density=density,
        unit=unit,
        log_x=log_x,
        label=method + (" (raw)" if raw else ""),
    )

    if kind == "hist":
        heights, edges = histogram_distribution(
            displayed, quantity=quantity, bins=bins, density=density
        )
        analysed = pd.DataFrame(
            {"bin_left": edges[:-1], "bin_right": edges[1:], "height": heights}
        )
    else:
        x, y = (
            empirical_cdf(displayed, quantity=quantity)
            if kind == "cdf"
            else empirical_ccdf(displayed, quantity=quantity)
        )
        analysed = pd.DataFrame({"value": x, "probability": y})

    snapshot_time = (
        float(fragments["time"].iloc[-1])
        if "time" in fragments.columns else None
    )
    snapshot_step = (
        int(fragments["step"].iloc[-1])
        if "step" in fragments.columns else None
    )
    completed = (case_dir / "summary.out").is_file()

    # JSON standard does not permit NaN or Infinity.
    reported_stats = asdict(stats)
    for key, value in reported_stats.items():
        if isinstance(value, float) and not np.isfinite(value):
            reported_stats[key] = None

    report = {
        "method": method,
        "source_directory": str(case_dir),
        "run_status": "completed" if completed else "partial_or_unverified",
        "snapshot_time_s": snapshot_time,
        "snapshot_step": snapshot_step,
        "fragment_set": "raw" if raw else "filtered_or_standard",
        "quantity": quantity,
        "kind": kind,
        "normalised": normalised,
        "display_unit": "dimensionless" if normalised else unit,
        "histogram_density": density if kind == "hist" else None,
        "statistics_SI": reported_stats,
    }

    print(f"\n{method} fragment distribution: {case_dir}")
    if not completed:
        print("WARNING: No summary.out; fragment snapshot may be incomplete.")
    print(f"Fragment set: {'raw' if raw else 'filtered/standard'}")
    print(f"Count: {stats.count}")
    print(f"Mean: {stats.mean:.6g} {'m' if quantity == 'length' else 'kg'}")
    print(f"Median: {stats.median:.6g} {'m' if quantity == 'length' else 'kg'}")
    print(f"Coefficient of variation: {stats.coefficient_of_variation:.6g}")

    if args.save or args.output is not None:
        path = _distribution_output_path(
            method, case_dir, args.output, quantity=quantity, kind=kind,
            normalised=normalised, raw=raw, density=density,
        )
        save_figure(fig, path)
        csv_path = save_csv(analysed, path)
        samples_path = save_csv(
            pd.DataFrame({"value_SI": values_si, "display_value": displayed}),
            path,
            suffix="_samples",
        )
        json_path = save_json(report, path)
        print(
            f"Figure: {path}\n"
            f"Distribution data: {csv_path}\n"
            f"Fragment samples: {samples_path}\n"
            f"Summary: {json_path}"
        )

    display_or_close(fig, no_show=args.no_show)
    return analysed
