# SPDX-License-Identifier: GPL-2.0-only
"""Reference-position fracture maps workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..io import load_case
from .common import display_or_close, output_path, save_bundle

def run_spatial(args) -> pd.DataFrame:
    """Draw reference-position fracture marks for one or several 1D bars.

    Every row is a saved fragment snapshot, not an instantaneous crack event.
    """
    from ..calculations.fracture_locations import fracture_snapshots
    from ..plots.spatial import plot_fracture_positions

    method = args.method.upper()
    case_dir = Path(args.input).expanduser().resolve()
    directories = [case_dir, *getattr(args, "stack_input", [])]
    all_snapshots = bool(getattr(args, "all_snapshots", False))
    max_snapshots = int(getattr(args, "max_snapshots", 12))
    raw = bool(getattr(args, "raw", False))
    position_unit = getattr(args, "position_unit", "mm")

    snapshots = []
    for directory in directories:
        case = load_case(method, directory)
        snapshots.extend(
            fracture_snapshots(
                case,
                raw=raw,
                all_snapshots=all_snapshots,
                max_snapshots=max_snapshots,
            )
        )

    fig, _ = plot_fracture_positions(
        snapshots, position_unit=position_unit
    )
    rows = []
    for index, snap in enumerate(snapshots):
        for position in snap.positions_m or (float("nan"),):
            rows.append({
                "snapshot": index,
                "label": snap.label,
                "source": snap.source,
                "time_s": snap.time_s,
                "step": snap.step,
                "fracture_position_m": position,
                "num_fractures": snap.count,
            })
    analysed = pd.DataFrame(rows)
    print(f"{method} spatial fractures: {len(snapshots)} snapshot(s)")
    for snap in snapshots:
        print(f"  {snap.label}: {snap.count} fractures")

    if args.save or args.output is not None:
        filename = "fracture_positions"
        if all_snapshots:
            filename += "_snapshots"
        if raw:
            filename += "_raw"
        if len(directories) > 1:
            filename += "_stacked"
        path = output_path(method, case_dir, args.output, filename)
        save_bundle(
            fig=fig,
            analysed=analysed,
            report={
                "analysis": "spatial",
                "method": method,
                "position_unit": position_unit,
                "raw_fragments": raw,
                "all_snapshots": all_snapshots,
                "cases": [str(d) for d in directories],
                "snapshots": [
                    {
                        "source": snap.source,
                        "label": snap.label,
                        "time_s": snap.time_s,
                        "step": snap.step,
                        "left_m": snap.left_m,
                        "right_m": snap.right_m,
                        "num_fractures": snap.count,
                        "fracture_positions_m": list(snap.positions_m),
                    }
                    for snap in snapshots
                ],
            },
            path=path,
        )

    display_or_close(fig, no_show=args.no_show)
    return analysed
