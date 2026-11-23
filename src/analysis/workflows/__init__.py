# SPDX-License-Identifier: GPL-2.0-only
"""CLI workflows, grouped by analysis type.

The public ``run_analysis(args)`` entry point is compatible with the previous
single-file ``src.analysis.workflows`` module. No CLI changes are required.
"""

from __future__ import annotations

from .convergence import run_convergence
from .distribution import run_distribution
from .energy import run_energy
from .history import run_history
from .spatial import run_spatial
from .strain_rate import run_strain_rate

WORKFLOWS = {
    "strain-rate": run_strain_rate,
    "convergence": run_convergence,
    "energy": run_energy,
    "distribution": run_distribution,
    "history": run_history,
    "spatial": run_spatial,
}

__all__ = [
    "run_analysis", "run_strain_rate", "run_convergence", "run_energy",
    "run_distribution", "run_history", "run_spatial", "WORKFLOWS",
]


def run_analysis(args) -> None:
    """Dispatch an argparse Namespace to the matching analysis workflow."""
    try:
        workflow = WORKFLOWS[args.analysis]
    except KeyError as exc:
        raise NotImplementedError(
            f"The '{args.analysis}' workflow is not yet implemented. "
            "Currently available: strain-rate, convergence, energy, "
            "distribution, history, spatial."
        ) from exc
    workflow(args)
