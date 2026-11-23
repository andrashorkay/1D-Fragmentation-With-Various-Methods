# SPDX-License-Identifier: GPL-2.0-only
"""Method-independent fragmentation statistics and strain-rate scaling.

No file reading or plotting belongs in this module. All lengths are in metres,
masses in kilograms, and strain rates in inverse seconds.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


class FragmentCase(Protocol):
    """Minimal interface implemented by the CZM, PD and PF case readers."""

    def final_fragments(self, raw: bool = False) -> pd.DataFrame: ...


@dataclass(frozen=True)
class FragmentStatistics:
    """Statistics for one fragment snapshot (not a time-averaged quantity)."""

    count: int
    mean_size: float
    median_size: float
    std_size: float
    min_size: float
    max_size: float
    total_size: float
    mean_mass: float | None
    total_mass: float | None


@dataclass(frozen=True)
class ScalingFit:
    """Least-squares fit of mean_size = prefactor * strain_rate**exponent."""

    exponent: float
    prefactor: float
    r_squared: float
    n_points: int
    min_rate: float
    max_rate: float


def _positive_finite(values: ArrayLike, name: str) -> np.ndarray:
    """Validate strictly positive, finite numerical measurements."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must contain only positive, finite values")
    return array


def fragment_statistics(fragments: pd.DataFrame) -> FragmentStatistics:
    """Summarise the individual fragments in one snapshot.

    Each row must represent one fragment and contain its length in ``size``.
    If ``mass`` is present, mass statistics are also returned. PD's raw
    fragment table has no mass column; its mass statistics are therefore None.
    """

    if "size" not in fragments:
        raise ValueError("Fragment table must contain a 'size' column")

    sizes = _positive_finite(fragments["size"].to_numpy(), "Fragment sizes")

    mean_mass = None
    total_mass = None
    if "mass" in fragments:
        masses = np.asarray(fragments["mass"], dtype=float)
        if not np.all(np.isfinite(masses)) or np.any(masses < 0):
            raise ValueError("Fragment masses must be finite and non-negative")
        mean_mass = float(np.mean(masses))
        total_mass = float(np.sum(masses))

    return FragmentStatistics(
        count=int(sizes.size),
        mean_size=float(np.mean(sizes)),
        median_size=float(np.median(sizes)),
        std_size=float(np.std(sizes, ddof=0)),
        min_size=float(np.min(sizes)),
        max_size=float(np.max(sizes)),
        total_size=float(np.sum(sizes)),
        mean_mass=mean_mass,
        total_mass=total_mass,
    )


def final_fragment_statistics(
    case: FragmentCase, *, raw: bool = False
) -> FragmentStatistics:
    """Summarise the last *saved* fragment snapshot from a loaded case.

    ``raw=True`` is supported only by PD. This does not necessarily describe
    the final simulated time if the simulation ended between output dumps.
    """

    return fragment_statistics(case.final_fragments(raw=raw))


def fit_strain_rate_scaling(
    strain_rates: ArrayLike,
    mean_sizes: ArrayLike,
    *,
    fit_min: float | None = None,
    fit_max: float | None = None,
) -> ScalingFit:
    """Fit ``mean_size = A * strain_rate**m`` in log-log space.

    The returned exponent ``m`` is dimensionless; ``prefactor`` is ``A``.
    The input must contain at least two distinct positive strain rates within
    the requested fit interval. This is an unweighted least-squares fit.
    """

    rates = _positive_finite(strain_rates, "Strain rates")
    sizes = _positive_finite(mean_sizes, "Mean fragment sizes")

    if rates.shape != sizes.shape:
        raise ValueError("Strain rates and mean sizes must have equal lengths")

    if fit_min is not None and (not np.isfinite(fit_min) or fit_min <= 0):
        raise ValueError("fit_min must be positive and finite")
    if fit_max is not None and (not np.isfinite(fit_max) or fit_max <= 0):
        raise ValueError("fit_max must be positive and finite")
    if fit_min is not None and fit_max is not None and fit_min >= fit_max:
        raise ValueError("fit_min must be smaller than fit_max")

    selected = np.ones(rates.size, dtype=bool)
    if fit_min is not None:
        selected &= rates >= fit_min
    if fit_max is not None:
        selected &= rates <= fit_max

    selected_rates = rates[selected]
    selected_sizes = sizes[selected]
    if selected_rates.size < 2 or np.unique(selected_rates).size < 2:
        raise ValueError("The fit needs at least two distinct strain rates")

    x = np.log(selected_rates)
    y = np.log(selected_sizes)
    exponent, log_prefactor = np.polyfit(x, y, deg=1)
    residual_sum = np.sum((y - (exponent * x + log_prefactor)) ** 2)
    total_sum = np.sum((y - np.mean(y)) ** 2)
    r_squared = float(1 - residual_sum / total_sum) if total_sum > 0 else float("nan")

    return ScalingFit(
        exponent=float(exponent),
        prefactor=float(np.exp(log_prefactor)),
        r_squared=r_squared,
        n_points=int(selected_rates.size),
        min_rate=float(np.min(selected_rates)),
        max_rate=float(np.max(selected_rates)),
    )
