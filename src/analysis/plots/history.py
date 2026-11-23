# SPDX-License-Identifier: GPL-2.0-only
"""Plot scalar simulation diagnostics against time.

This plotting module does not read files, calculate physics, or save figures.
For fracture positions along the bar, see :mod:`.spatial`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


_TIME_SCALES = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
_QUANTITY_UNITS = {
    "average_stress": "Pa",
    "max_stress": "Pa",
    "reaction_left": "N",
    "reaction_right": "N",
    "strain": "–",
    "average_strain": "–",
    "max_strain": "–",
    "displacement": "m",
    "num_fragments": "–",
    "raw_num_fragments": "–",
    "mean_fragment_size": "m",
    "min_fragment_size": "m",
    "max_fragment_size": "m",
    "kinetic_energy": "J",
    "recoverable_energy": "J",
    "elastic_energy": "J",
    "fracture_energy": "J",
    "contact_energy": "J",
    "total_energy": "J",
    "external_work": "J",
    "energy_error": "J",
    "mean_damage": "–",
    "max_damage": "–",
    "fraction_broken": "–",
}


def plot_history(
    history: pd.DataFrame,
    *,
    quantity: str,
    time_unit: str = "us",
    label: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a recorded scalar quantity versus time, with optional overlays."""
    if time_unit not in _TIME_SCALES:
        raise ValueError(f"Unsupported time unit: {time_unit}")
    if "time" not in history:
        raise KeyError("Diagnostic table does not contain a 'time' column")
    if quantity not in history:
        raise KeyError(
            f"Quantity '{quantity}' not found. Columns: {', '.join(history.columns)}"
        )

    x = pd.to_numeric(history["time"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(history[quantity], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        raise ValueError(f"No finite samples for '{quantity}'")
    x, y = x[finite], y[finite]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    ax.plot(x * _TIME_SCALES[time_unit], y, label=label)
    ax.set_xlabel(f"Time [{time_unit}]")
    unit = _QUANTITY_UNITS.get(quantity)
    name = quantity.replace("_", " ").capitalize()
    ax.set_ylabel(f"{name} [{unit}]" if unit else name)
    ax.grid(True, alpha=0.25)
    if label:
        ax.legend()
    return fig, ax
