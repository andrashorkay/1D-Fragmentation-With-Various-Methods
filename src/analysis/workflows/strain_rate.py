# SPDX-License-Identifier: GPL-2.0-only
"""Strain-rate scaling analysis workflow."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from ..calculations.fragmentation import fit_strain_rate_scaling
from ..io import load_case
from ..io.readers import find_case_directories
from ..plots.strain_rate import plot_strain_rate
from .common import (
    _fragment_length, display_or_close, output_path,
    save_csv, save_figure, save_json,
)

def _strain_rate(case, folder: Path) -> float:
    """Read strain rate from the case parameters, checking the folder name."""
    configured = case.parameters.get("strain_rate")
    folder_rate = None
    try:
        folder_rate = float(folder.name.removeprefix("rate_"))
    except ValueError:
        pass

    if configured is None:
        if folder_rate is None:
            raise ValueError("Strain rate missing in parameters and directory name")
        warnings.warn(
            f"{folder.name}: no recorded strain_rate; using directory name",
            stacklevel=2,
        )
        rate = folder_rate
    else:
        rate = float(configured)
        if folder_rate is not None and not np.isclose(rate, folder_rate, rtol=1e-6):
            raise ValueError(
                f"Strain rate mismatch: parameters={rate}, folder={folder_rate}"
            )

    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"Invalid strain rate: {rate}")
    return rate


def _warn_mixed_setups(cases: list) -> None:
    """Flag numerical changes that could confound a strain-rate comparison."""
    fixed_keys = (
        "dx",
        "horizon",
        "number_elements",
        "time_step_factor",
        "l0",
        "l0_scale",
        "bar_length",
        "length",
    )
    for key in fixed_keys:
        values = [case.parameters[key] for case in cases if key in case.parameters]
        if len(values) < 2:
            continue
        try:
            numeric = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            different = len({str(value) for value in values}) > 1
        else:
            different = not np.allclose(numeric, numeric[0], rtol=1e-9, atol=0)
        if different:
            warnings.warn(
                f"Parameter {key!r} changes between cases; "
                "the scaling may include numerical effects",
                stacklevel=2,
            )


def _material_properties(case) -> tuple[float, float]:
    """Return density (kg/m^3) and toughness (J/m^2) for Grady's formula."""
    parameters = case.parameters

    rho = next(
        (parameters[key] for key in ("rho", "density") if key in parameters),
        None,
    )
    gc = next(
        (parameters[key] for key in ("G_c", "gc", "toughness") if key in parameters),
        None,
    )

    # Heterogeneous PF may contain many damage material zones with a common gc.
    if gc is None and case.method == "PF":
        zones = case.materials.get("phase_field", {}).values()
        values = [float(zone["gc"]) for zone in zones if "gc" in zone]
        if values and np.allclose(values, values[0], rtol=1e-10, atol=0):
            gc = values[0]

    if rho is None or gc is None:
        raise ValueError(
            f"Cannot compute Grady curve for {case.path}: missing density or G_c"
        )

    rho, gc = float(rho), float(gc)
    if not (np.isfinite(rho) and np.isfinite(gc) and rho > 0 and gc > 0):
        raise ValueError(f"Invalid density or G_c in {case.path}")
    return rho, gc


def _nominal_strength(value: float | int | str) -> float:
    """Read nominal strength in Pa from a scalar or an Akantu distribution.

    For example, Akantu may record ``262000000 uniform [0, 1000000]``.
    The leading scalar is the nominal reference strength; the distribution
    expression is retained in the case's original material parameters.
    """
    if isinstance(value, str):
        # Accept a plain scalar or the project's Akantu uniform specification.
        match = re.fullmatch(
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
            r"(?:\s+uniform\s*\[[^\]]+\])?\s*",
            value,
        )
        if match is None:
            raise ValueError(
                f"Cannot extract nominal sigma_c from {value!r}. "
                "Provide --reference-strength in Pa."
            )
        value = match.group(1)

    strength = float(value)
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError(f"Nominal sigma_c must be positive and finite: {value!r}")
    return strength


def _normalisation_scales(
    cases: list, reference_strength: float | None = None
) -> tuple[float, float]:
    """Compute reference scales from material parameters, checking consistency.

    For a heterogeneous material, ``sigma_c`` is the nominal reference
    strength (not the spatially varying individual strengths). If it is not
    recorded in the simulation input, it must be supplied explicitly.
    """
    from ..calculations.theory import (
        characteristic_fragment_size,
        characteristic_strain_rate,
    )

    def get_reference(case) -> tuple[float, float, float, float]:
        params = case.parameters
        E = params.get("E")
        rho, gc = _material_properties(case)
        sigma_c = (
            reference_strength
            if reference_strength is not None
            else next((params[k] for k in ("sigma_c", "strength") if k in params), None)
        )
        if E is None or sigma_c is None:
            missing = "E" if E is None else "sigma_c"
            raise ValueError(
                f"Missing {missing} in {case.path}. "
                "For missing nominal strength use --reference-strength 262e6."
            )
        return (float(E), rho, _nominal_strength(sigma_c), gc)

    if not cases:
        raise ValueError("No cases available for normalisation")
    reference = get_reference(cases[0])
    for case in cases[1:]:
        actual = get_reference(case)
        if not np.allclose(reference, actual, rtol=1e-8, atol=0):
            raise ValueError(
                "Material reference parameters differ between simulations: "
                f"{cases[0].path} has {reference}, but {case.path} has {actual}. "
                "Use a consistent reference across the sweep."
            )

    E, rho, sigma_c, gc = reference
    return (
        characteristic_strain_rate(E, rho, sigma_c, gc),
        characteristic_fragment_size(E, sigma_c, gc),
    )


def _rate_directories(
    low_directory: Path,
    high_directory: Path | None,
    high_from: float | None,
) -> list[Path]:
    """Select rate cases from one sweep or two resolution-specific sweeps.

    In mixed-resolution mode, the low-resolution directory supplies rates
    below ``high_from`` and the high-resolution directory supplies rates
    at or above ``high_from``. Never substitute a low-resolution result when
    its required high-resolution counterpart is missing.
    """

    def index_cases(directory: Path) -> dict[float, Path]:
        indexed: dict[float, Path] = {}
        for case_dir in find_case_directories(directory, "rate_*"):
            try:
                rate = float(case_dir.name.removeprefix("rate_"))
            except ValueError:
                warnings.warn(
                    f"Skipping case with invalid rate directory: {case_dir}",
                    stacklevel=2,
                )
                continue
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError(f"Invalid strain rate in directory {case_dir}")
            if rate in indexed:
                raise ValueError(
                    f"Duplicate strain rate {rate:g} in {directory}: "
                    f"{indexed[rate].name} and {case_dir.name}"
                )
            indexed[rate] = case_dir
        return indexed

    low_cases = index_cases(low_directory)
    if high_directory is None:
        return [low_cases[rate] for rate in sorted(low_cases)]

    if high_from is None or not np.isfinite(high_from) or high_from <= 0:
        raise ValueError(
            "A positive --high-from value is required with --high-rate-input"
        )

    high_cases = index_cases(high_directory)
    selected = {rate: path for rate, path in low_cases.items() if rate < high_from}
    selected.update(
        (rate, path) for rate, path in high_cases.items() if rate >= high_from
    )

    missing_high = sorted(
        rate for rate in low_cases if rate >= high_from and rate not in high_cases
    )
    if missing_high:
        warnings.warn(
            "Omitting rates without a high-resolution case in "
            f"{high_directory}: {missing_high}. "
            "They will not be taken from the lower-resolution sweep.",
            stacklevel=2,
        )

    return [selected[rate] for rate in sorted(selected)]


def run_strain_rate(args) -> pd.DataFrame:
    """Load a rate_* sweep, fit scaling, and plot mean fragment length.

    Every run must contain a final summary.out so that interrupted simulations
    cannot silently enter a published fit.  Each row documents the origin of
    its fragment-size measurement.
    """
    from ..calculations.theory import grady_fragment_size

    method = args.method.upper()
    sweep_dir = Path(args.input).expanduser().resolve()
    high_input = getattr(args, "high_rate_input", None)
    high_from = getattr(args, "high_from", None)
    high_directory = (
        Path(high_input).expanduser().resolve() if high_input is not None else None
    )
    if high_directory is None and high_from is not None:
        raise ValueError("--high-from requires --high-rate-input")

    directories = _rate_directories(sweep_dir, high_directory, high_from)

    records = []
    loaded_cases = []
    grady_sizes = {}
    for directory in directories:
        if not (directory / "summary.out").is_file():
            warnings.warn(
                f"Skipping unfinished run {directory.name}: no summary.out",
                stacklevel=2,
            )
            continue

        try:
            case = load_case(method, directory)
            rate = _strain_rate(case, directory)
            size, count, source, snapshot_time = _fragment_length(case)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            warnings.warn(f"Skipping {directory.name}: {exc}", stacklevel=2)
            continue

        if getattr(args, "theory", "none") == "grady":
            rho, gc = _material_properties(case)
            grady_sizes[rate] = float(grady_fragment_size(rate, rho=rho, gc=gc))

        loaded_cases.append(case)
        records.append(
            {
                "case": directory.name,
                "source_directory": str(directory.parent),
                "number_elements": case.parameters.get("number_elements"),
                "strain_rate": rate,
                "mean_fragment_size": size,  # metres
                "num_fragments": count,
                "size_source": source,
                "fragment_snapshot_time": snapshot_time,  # seconds
                "final_simulation_time": case.summary.get("physical_time"),
                "stopping_reason": case.summary.get("stopping_reason", ""),
            }
        )

    if not records:
        raise ValueError(f"No completed, readable strain-rate cases in {sweep_dir}")

    _warn_mixed_setups(loaded_cases)

    result = pd.DataFrame.from_records(records).sort_values("strain_rate")
    result = result.reset_index(drop=True)

    if result["strain_rate"].duplicated().any():
        duplicates = result.loc[result["strain_rate"].duplicated(), "strain_rate"]
        raise ValueError(
            "Multiple cases at the same strain rate: "
            f"{duplicates.tolist()}. Select one consistent sweep."
        )

    rates = result["strain_rate"].to_numpy(dtype=float)
    sizes = result["mean_fragment_size"].to_numpy(dtype=float)
    fit = None

    # Restrict the power-law fit, without removing any points from the plot.
    # Indexing applies to completed cases after mixed-mesh selection and
    # sorting by physical strain rate (not to integration time steps).
    fit_skip_first = getattr(args, "fit_skip_first", 0)
    if fit_skip_first < 0:
        raise ValueError("--fit-skip-first must be non-negative")
    if fit_skip_first and fit_skip_first > len(rates) - 2:
        raise ValueError(
            f"Cannot skip {fit_skip_first} points: only {len(rates)} "
            "completed rates are available, and the fit needs at least two"
        )

    fit_min = args.fit_min
    if fit_skip_first:
        cutoff = float(rates[fit_skip_first])
        fit_min = max(fit_min, cutoff) if fit_min is not None else cutoff

    if len(result) >= 2:
        fit = fit_strain_rate_scaling(
            rates,
            sizes,
            fit_min=fit_min,
            fit_max=args.fit_max,
        )
    elif fit_skip_first or args.fit_min is not None or args.fit_max is not None:
        raise ValueError("At least two rates are needed for the requested fit")

    # Make the selection explicit in the exported CSV.
    result["used_in_fit"] = (
        (result["strain_rate"] >= fit.min_rate)
        & (result["strain_rate"] <= fit.max_rate)
        if fit is not None
        else False
    )

    theory_sizes = None
    if getattr(args, "theory", "none") == "grady":
        theory_sizes = np.asarray([grady_sizes[rate] for rate in rates])

    characteristic_rate = None
    characteristic_size = None
    if getattr(args, "normalised", False):
        characteristic_rate, characteristic_size = _normalisation_scales(
            loaded_cases, getattr(args, "reference_strength", None)
        )
        result["rate_hat"] = result["strain_rate"] / characteristic_rate
        result["size_hat"] = result["mean_fragment_size"] / characteristic_size
        if theory_sizes is not None:
            result["grady_hat"] = theory_sizes / characteristic_size

    fig, _ = plot_strain_rate(
        rates,
        sizes,
        method=method,
        fit=fit,
        theory_sizes=theory_sizes,
        size_unit="mm",
        characteristic_rate=characteristic_rate,
        characteristic_size=characteristic_size,
    )
    fig.tight_layout()

    print(f"\n{method} strain-rate sweep: {sweep_dir}")
    if characteristic_rate is not None:
        print(
            f"Reference: eps_dot_0 = {characteristic_rate:.6g} s^-1, "
            f"s_0 = {characteristic_size * 1e3:.6g} mm"
        )
    if high_directory is not None:
        print(
            f"High-resolution cases: {high_directory} "
            f"for strain rates >= {high_from:g} s^-1"
        )
    print()
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    if fit is None:
        print("\nNo power-law fit: fewer than two completed cases.")
    else:
        print(
            f"\nScaling exponent m = {fit.exponent:.6g}, "
            f"R² = {fit.r_squared:.6g} "
            f"({fit.n_points} rates: {fit.min_rate:g}–{fit.max_rate:g} s^-1)"
        )

    save = args.save or args.output is not None
    if save:
        figure_path = output_path(method, sweep_dir, args.output, "mean_fragment_size")
        if high_directory is not None and args.output is None:
            figure_path = figure_path.with_name(
                figure_path.stem + "_mixed_mesh" + figure_path.suffix
            )
        if characteristic_rate is not None and args.output is None:
            figure_path = figure_path.with_name(
                figure_path.stem + "_normalised" + figure_path.suffix
            )
        save_figure(fig, figure_path)
        csv_path = save_csv(result, figure_path, suffix="_summary")
        print(f"\nFigure: {figure_path}\nSummary: {csv_path}")

        if fit is not None:
            fit_path = save_json(asdict(fit), figure_path, suffix="_fit")
            print(f"Fit: {fit_path}")

    display_or_close(fig, no_show=args.no_show)

    return result
