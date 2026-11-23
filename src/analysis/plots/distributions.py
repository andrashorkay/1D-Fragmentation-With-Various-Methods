# SPDX-License-Identifier: GPL-2.0-only
"""Matplotlib plots for fragment-length and fragment-mass distributions.

Plots require already-loaded fragment tables. This module does not discover
simulation files, calculate strain-rate fits, save files, or call plt.show().
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..calculations.statistics import (
    empirical_ccdf,
    empirical_cdf,
    fragment_values,
)

Quantity = Literal["length", "mass"]
Kind = Literal["hist", "cdf", "ccdf"]

_UNIT_SCALES = {
    "length": {"m": 1.0, "mm": 1e3, "um": 1e6},
    "mass": {"kg": 1.0, "g": 1e3, "mg": 1e6},
}


def plot_fragment_distribution(
    fragments: pd.DataFrame,
    *,
    quantity: Quantity = "length",
    kind: Kind = "hist",
    bins: int = 30,
    normalised: bool = False,
    density: bool = True,
    unit: str | None = None,
    log_x: bool = False,
    label: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a histogram, empirical CDF, or empirical CCDF.

    Parameters
    ----------
    fragments:
        A *single snapshot* of fragment data, e.g. ``case.final_fragments()``.
    quantity:
        ``'length'`` reads ``size`` (m); ``'mass'`` reads ``mass`` (kg).
    kind:
        ``'hist'`` for a histogram, ``'cdf'`` for P(X <= x), ``'ccdf'`` for
        P(X > x). Both CDFs are empirical; no parametric fit is assumed.
    normalised:
        Plot x / mean(x). The original data are unchanged.
    density:
        For histograms, use a density (area=1), or counts if False.
    unit:
        Display unit if not normalised. Length: m, mm, um; mass: kg, g, mg.
    log_x:
        Show a logarithmic horizontal axis. Requires strictly positive data.
    label:
        Optional legend label, useful for overlaying methods/cases via ax.
    ax:
        Existing axes for overlays; otherwise create a new figure.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes without saving or displaying.
    """

    if kind not in ("hist", "cdf", "ccdf"):
        raise ValueError("kind must be 'hist', 'cdf' or 'ccdf'")
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")

    values = fragment_values(
        fragments, quantity=quantity, normalised=normalised
    )
    unit_options = _UNIT_SCALES[quantity]
    if unit is None:
        unit = "mm" if quantity == "length" else "mg"
    if unit not in unit_options:
        raise ValueError(
            f"Invalid {quantity} unit: {unit}; choose from {tuple(unit_options)}"
        )
    if log_x and np.any(values <= 0):
        raise ValueError("log_x requires strictly positive fragment values")

    factor = 1.0 if normalised else unit_options[unit]
    displayed = values * factor

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    if kind == "hist":
        ax.hist(
            displayed,
            bins=bins,
            density=density,
            histtype="step",
            linewidth=1.7,
            label=label,
        )
        ax.set_ylabel("Probability density" if density else "Fragment count")
    else:
        if kind == "cdf":
            x, y = empirical_cdf(displayed, quantity=quantity)
            ax.set_ylabel(r"Empirical CDF $P(X\leq x)$")
        else:
            x, y = empirical_ccdf(displayed, quantity=quantity)
            ax.set_ylabel(r"Empirical CCDF $P(X>x)$")
        ax.step(x, y, where="post", label=label)
        ax.set_ylim(0, 1.02)

    if log_x:
        ax.set_xscale("log")

    if normalised:
        xlabel = (
            r"Normalised fragment length $\ell/\overline{\ell}$"
            if quantity == "length"
            else r"Normalised fragment mass $m/\overline{m}$"
        )
    else:
        name = "Fragment length" if quantity == "length" else "Fragment mass"
        xlabel = f"{name} ({unit})"

    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.25)
    if label is not None:
        ax.legend()

    fig.tight_layout()
    return fig, ax
