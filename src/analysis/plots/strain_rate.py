# SPDX-License-Identifier: GPL-2.0-only
"""Plot mean fragment length versus applied strain rate.

This module only creates figures. Input data, power-law fits, and theoretical
fragment sizes are computed by the IO and calculation layers.
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from ..calculations.fragmentation import ScalingFit


SizeUnit = Literal["m", "mm", "um"]
_SIZE_UNITS: dict[SizeUnit, tuple[float, str]] = {
    "m": (1.0, "m"),
    "mm": (1e3, "mm"),
    "um": (1e6, r"$\mu$m"),
}


def _positive_array(values: ArrayLike, name: str) -> np.ndarray:
    """Convert a non-empty, positive, finite 1D input into an array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must contain only positive, finite values")
    return array


def plot_strain_rate(
    strain_rates: ArrayLike,
    mean_sizes: ArrayLike,
    *,
    method: str,
    fit: ScalingFit | None = None,
    theory_sizes: ArrayLike | None = None,
    reference_exponent: float | None = None,
    size_unit: SizeUnit = "mm",
    characteristic_rate: float | None = None,
    characteristic_size: float | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a strain-rate sweep on logarithmic axes.

    Parameters
    ----------
    strain_rates:
        Applied strain rates in s^-1.
    mean_sizes:
        Mean fragment lengths in metres, one per strain rate.
    method:
        Legend label (for example, ``"PD"``, ``"CZM"`` or ``"PF"``).
    fit:
        An optional fit computed by ``fit_strain_rate_scaling``. It is drawn
        only across the strain rates that were used for fitting.
    theory_sizes:
        Optional *calculated* theoretical fragment lengths in metres, at the
        same strain rates (for example, a Grady prediction). Calculation of
        this theory belongs in ``calculations/theory.py``.
    reference_exponent:
        Optional slope-only guide, such as ``-2/3``. Its amplitude is anchored
        to the geometric mean of the measured points; it is NOT a physical
        prediction of absolute fragment lengths.
    size_unit:
        Display units for the vertical axis: ``"m"``, ``"mm"``, or ``"um"``.
        All numerical inputs remain in SI units (metres).
    characteristic_rate, characteristic_size:
        Optional positive reference scales (s^-1 and m). Supply both to
        display dimensionless rates and lengths; the fit remains in SI units.
    ax:
        Existing Matplotlib axes, if plotting multiple methods together.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes. Showing or saving is left to the caller.
    """
    rates = _positive_array(strain_rates, "Strain rates")
    sizes = _positive_array(mean_sizes, "Mean fragment lengths")
    if rates.shape != sizes.shape:
        raise ValueError("Strain rates and mean fragment lengths must match")

    if size_unit not in _SIZE_UNITS:
        raise ValueError(f"Unknown size unit: {size_unit}")
    scale, unit_label = _SIZE_UNITS[size_unit]
    if (characteristic_rate is None) != (characteristic_size is None):
        raise ValueError(
            "Provide both characteristic_rate and characteristic_size"
        )
    normalised = characteristic_rate is not None
    if normalised:
        if not all(
            np.isfinite(v) and v > 0
            for v in (characteristic_rate, characteristic_size)
        ):
            raise ValueError(
                "Characteristic rate and size must be positive and finite"
            )

    theory = None
    if theory_sizes is not None:
        theory = _positive_array(theory_sizes, "Theoretical fragment lengths")
        if theory.shape != rates.shape:
            raise ValueError("Theoretical lengths must match the strain rates")

    if reference_exponent is not None and not np.isfinite(reference_exponent):
        raise ValueError("Reference exponent must be finite")

    order = np.argsort(rates)
    rates = rates[order]
    sizes = sizes[order]
    if theory is not None:
        theory = theory[order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    plot_rates = rates / characteristic_rate if normalised else rates
    plot_sizes = sizes / characteristic_size if normalised else sizes * scale
    ax.loglog(plot_rates, plot_sizes, "o-", label=method.upper())

    if fit is not None:
        fit_rates = np.geomspace(fit.min_rate, fit.max_rate, 150)
        fit_sizes = fit.prefactor * fit_rates**fit.exponent
        ax.loglog(
            fit_rates / characteristic_rate if normalised else fit_rates,
            fit_sizes / characteristic_size
            if normalised
            else fit_sizes * scale,
            "--",
            label=f"{method.upper()} fit: $m={fit.exponent:.3f}$ ($R^2={fit.r_squared:.3f}$)",
        )

    if theory is not None:
        ax.loglog(
            plot_rates,
            theory / characteristic_size if normalised else theory * scale,
            "-.",
            label="Grady prediction",
        )

    if reference_exponent is not None:
        centre_rate = float(np.exp(np.mean(np.log(rates))))
        centre_size = float(np.exp(np.mean(np.log(sizes))))
        guide = centre_size * (rates / centre_rate) ** reference_exponent
        ax.loglog(
            plot_rates,
            guide / characteristic_size if normalised else guide * scale,
            ":",
            label=f"Slope reference: $m={reference_exponent:.3f}$",
        )

    if normalised:
        ax.set_xlabel(
            r"Normalised strain rate $\dot{\varepsilon}/\dot{\varepsilon}_0$"
        )
        ax.set_ylabel(r"Normalised mean fragment size $\overline{s}/s_0$")
    else:
        ax.set_xlabel(r"Strain rate $\dot{\varepsilon}$ (s$^{-1}$)")
        ax.set_ylabel(
            rf"Mean fragment length $\overline{{\ell}}$ ({unit_label})"
        )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    return fig, ax
