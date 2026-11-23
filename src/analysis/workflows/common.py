# SPDX-License-Identifier: GPL-2.0-only
"""Shared workflow infrastructure; no method-specific scientific calculations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..calculations.fragmentation import final_fragment_statistics
from ..config import DATA_DIR, DEFAULT_FIGURE_DPI, FIGURES_DIR


def output_path(
    method: str,
    source_dir: Path,
    explicit: Path | None,
    filename: str,
) -> Path:
    """Return a validated figure path, preserving method-relative layout.

    Explicit paths are used unchanged (apart from tilde expansion). Otherwise,
    mirror the source directory below figures/METHOD and append .png.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
    else:
        source_dir = Path(source_dir).expanduser().resolve()
        try:
            relative = source_dir.relative_to((DATA_DIR / method).resolve())
        except ValueError:
            relative = Path(source_dir.name)
        path = FIGURES_DIR / method / relative / f"{filename}.png"
    if path.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("--output must end in .png, .pdf or .svg")
    return path


def save_figure(fig, path: Path) -> Path:
    """Save a figure with the project's standard resolution and bounding box."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    return path


def save_csv(table: pd.DataFrame, path: Path, *, suffix: str = "_data") -> Path:
    """Save an analysis table beside the chosen figure."""
    filename = path.with_name(path.stem + suffix + ".csv")
    filename.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(filename, index=False)
    return filename


def save_json(
    report: dict,
    path: Path,
    *,
    suffix: str = "_summary",
    allow_nan: bool = False,
) -> Path:
    """Save JSON metadata beside the chosen figure."""
    filename = path.with_name(path.stem + suffix + ".json")
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(
        json.dumps(report, indent=2, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
    )
    return filename


def display_or_close(fig, *, no_show: bool) -> None:
    """Centralised handling of interactive and headless figures."""
    if no_show:
        plt.close(fig)
    else:
        plt.show()


def save_bundle(
    *,
    fig,
    analysed: pd.DataFrame,
    report: dict,
    path: Path,
) -> None:
    """Save a figure, source CSV and JSON report with consistent names."""
    save_figure(fig, path)
    data_path = save_csv(analysed, path)
    report_path = save_json(report, path)
    print(f"Figure: {path}\nData: {data_path}\nSummary: {report_path}")



def _fragment_length(case) -> tuple[float, int, str, float]:
    """Mean fragment length in metres, preferring measured fragments.

    If no individual fragment records exist, use the last recorded mean in
    the global diagnostic history.  The fallback is labelled in the output.
    """
    if case.fragments is not None and not case.fragments.empty:
        stats = final_fragment_statistics(case)
        snapshot_time = float(case.fragments["time"].max())
        return stats.mean_size, stats.count, "fragments", snapshot_time

    last = case.history.iloc[-1]
    size = float(last["mean_fragment_size"])
    if not np.isfinite(size) or size <= 0:
        raise ValueError("No positive mean fragment size in fragments or history")

    count = int(last["num_fragments"])
    if count < 1:
        raise ValueError("Invalid fragment count in history")
    snapshot_time = float(case.history["time"].iloc[-1])
    return size, count, "history", snapshot_time
