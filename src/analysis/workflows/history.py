# SPDX-License-Identifier: GPL-2.0-only
"""Scalar simulation time histories workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..io import load_case
from .common import display_or_close, output_path, save_bundle

def run_history(args) -> pd.DataFrame:
    """Plot a diagnostic time series, optionally overlaying other cases."""
    from ..plots.history import plot_history

    method = args.method.upper()
    case_dir = Path(args.input).expanduser().resolve()
    directories = [case_dir, *getattr(args, "stack_input", [])]
    quantity = args.quantity
    source = getattr(args, "source", "history")
    time_unit = getattr(args, "time_unit", "us")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    curves = []
    for directory in directories:
        case = load_case(method, directory)
        if source == "diagnostics":
            table = getattr(case, "diagnostics", None)
        else:
            table = case.history
        if table is None or table.empty:
            raise ValueError(f"No {source} data in {directory}")
        if "time" not in table or quantity not in table:
            raise ValueError(
                f"Missing 'time' or '{quantity}' in {directory}. "
                f"Available: {', '.join(table.columns)}"
            )
        curve = table[["time", quantity]].copy()
        curve["case"] = str(directory)
        curves.append(curve)
        plot_history(
            table,
            quantity=quantity,
            time_unit=time_unit,
            label=f"{method} | {directory.name}",
            ax=ax,
        )

    analysed = pd.concat(curves, ignore_index=True)
    ax.set_title(f"{method} | {quantity.replace('_', ' ')}")
    fig.tight_layout()
    print(f"{method} {quantity}: {len(directories)} case(s)")

    if args.save or args.output is not None:
        filename = f"history_{quantity}" + (
            "_overlay" if len(directories) > 1 else ""
        )
        path = output_path(method, case_dir, args.output, filename)
        save_bundle(
            fig=fig,
            analysed=analysed,
            report={
                "analysis": "history",
                "method": method,
                "quantity": quantity,
                "source": source,
                "time_unit": time_unit,
                "cases": [str(d) for d in directories],
            },
            path=path,
        )

    display_or_close(fig, no_show=args.no_show)
    return analysed
