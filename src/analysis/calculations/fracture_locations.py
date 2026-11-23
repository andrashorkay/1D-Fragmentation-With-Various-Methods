# SPDX-License-Identifier: GPL-2.0-only
"""Locate internal fracture boundaries from saved fragment snapshots.

All three solvers use an undeformed bar centred at x = 0. PD stores the
boundaries explicitly; CZM and PF store ordered reference fragment lengths.
The returned positions are reference-coordinate estimates, not deformed gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FractureSnapshot:
    """Internal cuts in one saved snapshot, in reference coordinates (metres)."""

    label: str
    source: str
    method: str
    time_s: float | None
    step: int | None
    left_m: float
    right_m: float
    positions_m: tuple[float, ...]

    @property
    def count(self) -> int:
        return len(self.positions_m)


class FragmentCase(Protocol):
    method: str
    path: object
    fragments: pd.DataFrame | None
    raw_fragments: pd.DataFrame | None
    parameters: dict


def _positions_from_snapshot(
    fragments: pd.DataFrame,
    method: str,
    bar_length: float | None,
) -> tuple[float, float, tuple[float, ...]]:
    if fragments.empty:
        raise ValueError("A fracture snapshot cannot be empty")

    if "x_left" in fragments.columns:
        ordered = fragments.sort_values("x_left", kind="stable")
    elif "centre" in fragments.columns:
        ordered = fragments.sort_values("centre", kind="stable")
    elif "fragment_id" in fragments.columns:
        ordered = fragments.sort_values("fragment_id", kind="stable")
    else:
        raise ValueError(
            "Fragment positions require fragment_id, centre or x_left to "
            "establish the physical ordering"
        )

    if method == "PD" and {"x_left", "x_right"}.issubset(ordered.columns):
        lefts = ordered["x_left"].to_numpy(dtype=float)
        rights = ordered["x_right"].to_numpy(dtype=float)
        if not np.all(np.isfinite(lefts)) or not np.all(np.isfinite(rights)):
            raise ValueError("Non-finite PD fragment boundaries")
        if np.any(rights <= lefts):
            raise ValueError("PD fragment boundaries must be increasing")
        length = float(rights[-1] - lefts[0])
        if len(ordered) > 1 and not np.allclose(
            rights[:-1], lefts[1:], rtol=1e-8, atol=1e-10 * length
        ):
            raise ValueError("PD fragments do not form a contiguous bar")
        return (
            float(lefts[0]),
            float(rights[-1]),
            tuple(map(float, rights[:-1])),
        )

    if "size" not in ordered.columns:
        raise ValueError(
            "Missing size column required to locate fracture boundaries"
        )

    sizes = ordered["size"].to_numpy(dtype=float)
    if not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
        raise ValueError("Fragment sizes must be positive and finite")
    total = float(sizes.sum())
    if bar_length is None:
        length = total
    else:
        length = float(bar_length)
        if not np.isfinite(length) or length <= 0:
            raise ValueError("bar_length must be positive and finite")
        if not np.isclose(total, length, rtol=5e-3, atol=1e-12):
            raise ValueError(
                f"Fragment lengths sum to {total:g} m, but bar length is "
                f"{length:g} m. Cannot reliably locate fractures."
            )

    left = -0.5 * length
    positions = left + np.cumsum(sizes[:-1])
    return left, 0.5 * length, tuple(map(float, positions))


def fracture_snapshots(
    case: FragmentCase,
    *,
    all_snapshots: bool = False,
    max_snapshots: int = 12,
    raw: bool = False,
    label: str | None = None,
) -> list[FractureSnapshot]:
    """Extract final or sampled snapshots for a stacked 1D fracture map.

    The snapshot sampling is evenly spaced through available *saved dumps*;
    it does not interpolate or claim to know exact fracture initiation times.
    """
    method = case.method.upper()
    if raw and method != "PD":
        raise ValueError("Only PD provides raw fragment records")
    if not isinstance(max_snapshots, int) or max_snapshots < 1:
        raise ValueError("max_snapshots must be a positive integer")

    table = case.raw_fragments if raw else case.fragments
    if table is None or table.empty:
        name = "raw_fragments.out" if raw else "the fragment table"
        raise ValueError(f"No {name} in {case.path}")
    group_by = "step" if method == "PD" and "step" in table else "time"
    if group_by not in table:
        raise ValueError(f"Missing '{group_by}' in {case.path} fragment table")

    snapshots = list(table.groupby(group_by, sort=True, dropna=True))
    if not snapshots:
        raise ValueError("No usable fragment snapshots")
    if all_snapshots and len(snapshots) > max_snapshots:
        selection = np.unique(
            np.rint(np.linspace(0, len(snapshots) - 1, max_snapshots)).astype(
                int
            )
        )
        snapshots = [snapshots[i] for i in selection]
    elif not all_snapshots:
        snapshots = [snapshots[-1]]

    base_label = label or f"{method} | {getattr(case.path, 'name', case.path)}"
    parameter_length = case.parameters.get(
        "bar_length", case.parameters.get("length")
    )
    length = float(parameter_length) if parameter_length is not None else None

    result = []
    for _, subset in snapshots:
        left, right, positions = _positions_from_snapshot(
            subset, method, length
        )
        time_s = float(subset["time"].iloc[-1]) if "time" in subset else None
        step = int(subset["step"].iloc[-1]) if "step" in subset else None
        suffix = f" | {time_s * 1e6:.4g} µs" if time_s is not None else ""
        result.append(
            FractureSnapshot(
                label=base_label + suffix,
                source=str(case.path),
                method=method,
                time_s=time_s,
                step=step,
                left_m=left,
                right_m=right,
                positions_m=positions,
            )
        )
    return result
