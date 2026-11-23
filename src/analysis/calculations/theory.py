# SPDX-License-Identifier: GPL-2.0-only
"""Analytical fragmentation reference laws in SI units."""

import numpy as np
from numpy.typing import ArrayLike


def grady_fragment_size(
    strain_rate: ArrayLike, *, rho: float, gc: float
) -> np.ndarray:
    """Return Grady's characteristic fragment length in metres.

    s = [24 G_c / (rho * strain_rate**2)]**(1/3), with strain_rate in s^-1,
    density rho in kg/m^3, and fracture energy G_c in J/m^2.
    """
    rates = np.asarray(strain_rate, dtype=float)
    if not np.all(np.isfinite(rates)) or np.any(rates <= 0):
        raise ValueError("strain_rate must be positive and finite")
    if not (np.isfinite(rho) and rho > 0 and np.isfinite(gc) and gc > 0):
        raise ValueError("rho and gc must be positive and finite")
    return np.cbrt(24.0 * gc / (rho * rates**2))


def characteristic_fragment_size(E: float, sigma_c: float, gc: float) -> float:
    """Reference fragment length s0 = E * G_c / sigma_c**2, in metres."""
    _positive_material_properties(E=E, sigma_c=sigma_c, gc=gc)
    return float(E * gc / sigma_c**2)


def characteristic_strain_rate(
    E: float, rho: float, sigma_c: float, gc: float
) -> float:
    """Reference strain rate epsdot0 = sigma_c**3 * sqrt(E/rho) / (E**2 * G_c)."""
    _positive_material_properties(E=E, rho=rho, sigma_c=sigma_c, gc=gc)
    return float(sigma_c**3 * np.sqrt(E / rho) / (E**2 * gc))


def _positive_material_properties(**values: float) -> None:
    """Check dimensional material parameters before applying reference laws."""
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
