"""
Author: Andras Horkay
Contact: andras.horkay@gmail.com

Input/output interface for fragmentation simulation results.

Provides a unified interface for loading CZM, PD and PF simulations.
"""

from pathlib import Path

from .czm import CZMCase, load_czm_case
from .pd import PDCase, load_pd_case
from .pf import PFCase, load_pf_case


def load_case(
    method: str,
    directory: str | Path,
) -> CZMCase | PDCase | PFCase:
    """
    Load simulation results for a given numerical method.

    Parameters
    ----------
    method : str
        Numerical method: 'CZM', 'PD' or 'PF'.
    directory : str | Path
        Directory containing the simulation output files.

    Returns
    -------
    CZMCase | PDCase | PFCase
        Loaded simulation data.
    """

    method = method.upper()

    loaders = {
        "CZM": load_czm_case,
        "PD": load_pd_case,
        "PF": load_pf_case,
    }

    if method not in loaders:
        raise ValueError(
            f"Unknown method '{method}'. Supported methods: {', '.join(loaders)}"
        )

    return loaders[method](directory)


__all__ = [
    "CZMCase",
    "PDCase",
    "PFCase",
    "load_czm_case",
    "load_pd_case",
    "load_pf_case",
    "load_case",
]
