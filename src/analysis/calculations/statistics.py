# SPDX-License-Identifier: GPL-2.0-only
"""Method-independent statistics for fragment length and mass distributions.

This module does not read files or create figures. Fragment lengths are
assumed to be in metres and fragment masses in kilograms.

Use the last recorded fragment snapshot from a case reader; do not combine
records from multiple times into one distribution.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

Quantity = Literal["length", "mass"]


class FragmentCase(Protocol):
    """Minimal interface shared by the three method-specific case readers."""

    def final_fragments(self, raw: bool = False) -> pd.DataFrame: ...


@dataclass(frozen=True)
class DistributionStatistics:
    """Descriptive statistics for a single fragment snapshot."""

    quantity: Quantity
    count: int
    mean: float
    median: float
    standard_deviation: float
    coefficient_of_variation: float
    minimum: float
    maximum: float
    q05: float
    q25: float
    q75: float
    q95: float
    interquartile_range: float
    total: float


def _validate_values(values: ArrayLike, *, quantity: Quantity) -> np.ndarray:
    """Return a finite one-dimensional array of physical measurements."""

    if quantity not in ("length", "mass"):
        raise ValueError("quantity must be 'length' or 'mass'")

    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("Distribution requires a non-empty 1D array")
    if not np.all(np.isfinite(result)):
        raise ValueError("Distribution contains NaN or infinite values")
    if quantity == "length" and np.any(result <= 0):
        raise ValueError("Fragment lengths must be strictly positive")
    if quantity == "mass" and np.any(result < 0):
        raise ValueError("Fragment masses must be non-negative")
    return result


def fragment_values(
    fragments: pd.DataFrame,
    *,
    quantity: Quantity = "length",
    normalised: bool = False,
) -> np.ndarray:
    """Extract fragment lengths or masses from one fragment snapshot.

    ``quantity='length'`` uses the ``size`` column (metres).
    ``quantity='mass'`` uses the ``mass`` column (kilograms).
    If normalised, return each measurement divided by the sample mean.

    The method is not allowed to reconstruct masses from lengths: doing so
    would silently assume a density/area that may not apply to every method.
    """

    column = {"length": "size", "mass": "mass"}.get(quantity)
    if column is None:
        raise ValueError("quantity must be 'length' or 'mass'")
    if column not in fragments.columns:
        raise ValueError(
            f"Fragment table has no '{column}' column. "
            "PD raw-fragment records, for instance, do not contain masses."
        )

    values = _validate_values(fragments[column].to_numpy(), quantity=quantity)
    if normalised:
        mean = float(np.mean(values))
        if mean <= 0:
            raise ValueError("Cannot normalise a distribution with zero mean")
        values = values / mean
    return values


def final_fragment_values(
    case: FragmentCase,
    *,
    quantity: Quantity = "length",
    normalised: bool = False,
    raw: bool = False,
) -> np.ndarray:
    """Extract a distribution from the last *saved* fragment snapshot.

    ``raw=True`` is supported only by PD. A saved snapshot can precede the
    physical final time when an output interval is larger than one step.
    """

    return fragment_values(
        case.final_fragments(raw=raw),
        quantity=quantity,
        normalised=normalised,
    )


def distribution_statistics(
    values: ArrayLike,
    *,
    quantity: Quantity = "length",
) -> DistributionStatistics:
    """Summarise a distribution in original SI units.

    Standard deviation is the population statistic (``ddof=0``), consistent
    with the existing ``fragmentation.fragment_statistics`` implementation.
    """

    data = _validate_values(values, quantity=quantity)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=0))
    q05, q25, q75, q95 = np.percentile(data, [5, 25, 75, 95])
    return DistributionStatistics(
        quantity=quantity,
        count=int(data.size),
        mean=mean,
        median=float(np.median(data)),
        standard_deviation=std,
        coefficient_of_variation=float(std / mean)
        if mean > 0
        else float("nan"),
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
        q05=float(q05),
        q25=float(q25),
        q75=float(q75),
        q95=float(q95),
        interquartile_range=float(q75 - q25),
        total=float(np.sum(data)),
    )


def fragment_distribution_statistics(
    fragments: pd.DataFrame,
    *,
    quantity: Quantity = "length",
) -> DistributionStatistics:
    """Compute summary statistics from a table of individual fragments."""

    return distribution_statistics(
        fragment_values(fragments, quantity=quantity), quantity=quantity
    )


def empirical_cdf(
    values: ArrayLike, *, quantity: Quantity = "length"
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique sorted x values and the empirical CDF P(X <= x).

    The returned CDF is right-continuous; repeated measurements are grouped.
    """

    data = np.sort(_validate_values(values, quantity=quantity))
    x, counts = np.unique(data, return_counts=True)
    return x, np.cumsum(counts, dtype=float) / data.size


def empirical_ccdf(
    values: ArrayLike, *, quantity: Quantity = "length"
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique sorted x values and the empirical survival P(X > x)."""

    x, cdf = empirical_cdf(values, quantity=quantity)
    return x, 1.0 - cdf


def histogram_distribution(
    values: ArrayLike,
    *,
    quantity: Quantity = "length",
    bins: int | ArrayLike = 30,
    density: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return histogram heights and edges (``np.histogram`` convention).

    ``density=True`` produces a probability *density*, whose area integrates
    to one; the individual bin heights do not generally sum to one.
    """

    data = _validate_values(values, quantity=quantity)
    if isinstance(bins, (int, np.integer)) and bins < 1:
        raise ValueError("bins must be a positive integer")
    heights, edges = np.histogram(data, bins=bins, density=density)
    return heights, edges
