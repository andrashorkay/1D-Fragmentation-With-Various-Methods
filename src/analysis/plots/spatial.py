# SPDX-License-Identifier: GPL-2.0-only
"""Visualise fracture locations in a one-dimensional reference bar.

This module only draws already-calculated fracture positions. Fracture
extraction lives in ``calculations/fracture_locations.py``; I/O and figure
saving belong to the workflow.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..calculations.fracture_locations import FractureSnapshot

_POSITION_SCALES = {"m": 1.0, "mm": 1e3, "um": 1e6}


def plot_fracture_positions(
    snapshots: Sequence[FractureSnapshot],
    *,
    position_unit: str = "mm",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a horizontal bar and fracture markers for every snapshot.

    Snapshots can come from successive saved times or different simulations.
    The positions are *reference* coordinates, not deformed crack openings.
    A mark repeated in later snapshots indicates a persistent fracture, not
    a new fracture event.
    """
    if not snapshots:
        raise ValueError("At least one snapshot is required")
    if position_unit not in _POSITION_SCALES:
        raise ValueError(f"Unsupported position unit: {position_unit}")

    scale = _POSITION_SCALES[position_unit]
    if ax is None:
        height = max(2.5, min(16.0, 1.4 + 0.47 * len(snapshots)))
        fig, ax = plt.subplots(figsize=(10, height))
    else:
        fig = ax.figure

    for index, snapshot in enumerate(snapshots):
        row = len(snapshots) - 1 - index
        ax.hlines(
            row,
            snapshot.left_m * scale,
            snapshot.right_m * scale,
            linewidth=3,
            color="0.55",
            zorder=1,
        )
        if snapshot.positions_m:
            positions = np.asarray(snapshot.positions_m) * scale
            ax.plot(
                positions,
                np.full(len(positions), row),
                linestyle="None",
                marker="|",
                markersize=18,
                markeredgewidth=2,
                color="tab:red",
                zorder=3,
            )
        ax.annotate(
            f"{snapshot.count} cracks",
            (snapshot.right_m * scale, row),
            xytext=(0, 7),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
        )

    ax.set_yticks(np.arange(len(snapshots)))
    ax.set_yticklabels([snapshot.label for snapshot in reversed(snapshots)])
    ax.set_ylim(-0.65, len(snapshots) - 0.05)
    ax.set_xlabel(f"Reference position [{position_unit}]")
    ax.set_title("Fracture positions along the bar")
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig, ax
