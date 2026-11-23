# SPDX-License-Identifier: GPL-2.0-only
"""Method-independent energy calculations for CZM, PD and PF.

Functions accept Pandas DataFrames from the ``io`` readers; they never read
files or create plots. All energies and work are in joules, and time is in
seconds. Relative errors and component fractions are dimensionless.

Physical distinctions
---------------------
* CZM: recoverable = bulk elastic + reversible cohesive energy; contact
  energy remains separate; ``fracture_energy`` is dissipated fracture energy.
* PD: the reader's ``recoverable_energy`` refers to *physical* elastic
  energy, whereas the recorded ``total_energy`` includes ghost-region bond
  energy. Never reconstruct PD ``total_energy`` from displayed components.
* PF: ``fracture_energy`` is the recorded phase-field energy, which is not
  necessarily identical to dissipated cohesive/bond-breaking energy.

For every method, energy balance uses the solver-recorded ``total_energy``
and ``external_work``. The sign convention is:

    energy_error = external_work - (total_energy - initial_energy)

An energy history may belong to an interrupted run; "last" means last
*recorded* diagnostic row, not necessarily the completed simulation state.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnergySummary:
    """Energy quantities at the last recorded point of an energy history."""

    n_samples: int
    last_time_s: float
    initial_total_energy_J: float
    last_kinetic_energy_J: float
    last_recoverable_energy_J: float
    last_fracture_energy_J: float
    last_contact_energy_J: float | None
    last_total_energy_J: float
    last_external_work_J: float
    last_energy_error_J: float
    max_relative_energy_error: float


def _validate_history(history: pd.DataFrame, required: tuple[str, ...]) -> None:
    """Check that an energy history is non-empty, ordered and numerical."""

    if not isinstance(history, pd.DataFrame) or history.empty:
        raise ValueError("Energy history must be a non-empty DataFrame")

    missing = sorted(set(required).difference(history.columns))
    if missing:
        raise KeyError(f"Missing energy-history columns: {missing}")

    for column in required:
        try:
            values = pd.to_numeric(history[column], errors="raise").to_numpy(
                dtype=float
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column '{column}' must be numerical") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Column '{column}' contains non-finite values")

    if "time" in required:
        times = history["time"].to_numpy(dtype=float)
        if np.any(times < 0) or np.any(np.diff(times) < 0):
            raise ValueError(
                "Energy-history times must be non-negative and ordered"
            )


def recoverable_energy(history: pd.DataFrame) -> pd.Series:
    """Return recoverable energy, using readers' common alias when available.

    Otherwise derive it from the original recorded columns. In CZM the
    reversible cohesive contribution must be included, even if bulk elastic
    energy is separately reported.
    """

    if "recoverable_energy" in history:
        columns = ("recoverable_energy",)
    elif "reversible_cohesive_energy" in history:
        columns = ("elastic_energy", "reversible_cohesive_energy")
    else:
        columns = ("elastic_energy",)

    _validate_history(history, columns)
    result = sum((history[column] for column in columns), start=0)
    return result.astype(float).rename("recoverable_energy")


def energy_components(history: pd.DataFrame) -> pd.DataFrame:
    """Return recorded energy components with a consistent recoverable alias.

    Original columns are retained. No conservation identity is imposed on
    these components; specifically, PD's physical elastic energy does not
    include all ghost-region terms in its recorded total energy.
    """

    _validate_history(history, ("time", "kinetic_energy", "fracture_energy"))
    result = history.copy()
    result["recoverable_energy"] = recoverable_energy(result)
    if "contact_energy" in result:
        _validate_history(result, ("contact_energy",))
    return result


def estimate_initial_energy(history: pd.DataFrame) -> float:
    """Infer the solver's initial total energy from the recorded balance.

    Uses the identity E0 = total_energy - external_work + energy_error.
    A median across rows reduces sensitivity to text-output rounding. When
    the energy-error column is absent, inference is only possible if the
    first recorded row is at t=0; otherwise the caller must provide E0.
    """

    required = ("time", "total_energy", "external_work")
    if "energy_error" in history:
        required += ("energy_error",)
    _validate_history(history, required)

    if "energy_error" in history:
        estimates = (
            history["total_energy"].to_numpy(dtype=float)
            - history["external_work"].to_numpy(dtype=float)
            + history["energy_error"].to_numpy(dtype=float)
        )
        return float(np.median(estimates))

    if history["time"].iloc[0] == 0:
        return float(
            history["total_energy"].iloc[0] - history["external_work"].iloc[0]
        )

    raise ValueError(
        "Cannot infer initial energy without recorded 'energy_error' "
        "or a t=0 diagnostic row; supply initial_energy explicitly"
    )


def energy_balance_history(
    history: pd.DataFrame, *, initial_energy: float | None = None
) -> pd.DataFrame:
    """Calculate the energy-balance history from recorded totals and work.

    Returns columns ``time``, ``total_energy``, ``external_work``,
    ``initial_energy``, ``expected_total_energy``, ``energy_error`` and
    ``relative_energy_error``. If a solver-recorded ``energy_error`` exists,
    it is also retained as ``reported_energy_error`` for checking.

    Relative error uses the existing project's convention:

      |error| / max(|total_energy|, |E0 + external_work|, machine epsilon)

    Do not substitute the sum of visible components for ``total_energy``.
    """

    required = ("time", "total_energy", "external_work")
    if "energy_error" in history:
        required += ("energy_error",)
    _validate_history(history, required)

    if initial_energy is None:
        initial_energy = estimate_initial_energy(history)
    else:
        initial_energy = float(initial_energy)
        if not np.isfinite(initial_energy):
            raise ValueError("initial_energy must be finite")

    total = history["total_energy"].to_numpy(dtype=float)
    work = history["external_work"].to_numpy(dtype=float)
    expected = initial_energy + work
    residual = work - (total - initial_energy)
    scale = np.maximum.reduce(
        (
            np.abs(total),
            np.abs(expected),
            np.full(len(total), np.finfo(float).eps),
        )
    )

    result = pd.DataFrame(
        {
            "time": history["time"].to_numpy(dtype=float),
            "total_energy": total,
            "external_work": work,
            "initial_energy": np.full(len(total), initial_energy),
            "expected_total_energy": expected,
            "energy_error": residual,
            "relative_energy_error": np.abs(residual) / scale,
        },
        index=history.index,
    )
    if "energy_error" in history:
        result["reported_energy_error"] = history["energy_error"].to_numpy(
            dtype=float
        )
    return result


def relative_energy_error(
    history: pd.DataFrame, *, initial_energy: float | None = None
) -> pd.Series:
    """Return dimensionless absolute energy-balance errors at each time."""

    return energy_balance_history(history, initial_energy=initial_energy)[
        "relative_energy_error"
    ]


def energy_partition(history: pd.DataFrame) -> pd.DataFrame:
    """Calculate fractions of *displayed* kinetic, recoverable and fracture energy.

    Contact energy is included if present. Fractions use the sum of these
    displayed components, NOT the solver-recorded total energy (important
    for PD ghost-region accounting). Where this sum is zero, all fractions
    are NaN rather than manufacturing a partition.
    """

    data = energy_components(history)
    components = ["kinetic_energy", "recoverable_energy", "fracture_energy"]
    if "contact_energy" in data:
        components.append("contact_energy")

    values = data[components].to_numpy(dtype=float)
    tracked = values.sum(axis=1)
    fractions = np.full(values.shape, np.nan, dtype=float)
    nonzero = tracked != 0
    fractions[nonzero, :] = values[nonzero, :] / tracked[nonzero, None]

    result = pd.DataFrame(
        {"time": data["time"].to_numpy(dtype=float), "tracked_energy": tracked},
        index=data.index,
    )
    for index, column in enumerate(components):
        result[column.replace("_energy", "_fraction")] = fractions[:, index]
    return result


def summarise_energy(
    history: pd.DataFrame, *, initial_energy: float | None = None
) -> EnergySummary:
    """Summarise the final *recorded* row and maximum energy-balance error."""

    data = energy_components(history)
    balance = energy_balance_history(data, initial_energy=initial_energy)
    last = data.iloc[-1]
    last_balance = balance.iloc[-1]

    return EnergySummary(
        n_samples=len(data),
        last_time_s=float(last["time"]),
        initial_total_energy_J=float(last_balance["initial_energy"]),
        last_kinetic_energy_J=float(last["kinetic_energy"]),
        last_recoverable_energy_J=float(last["recoverable_energy"]),
        last_fracture_energy_J=float(last["fracture_energy"]),
        last_contact_energy_J=(
            float(last["contact_energy"]) if "contact_energy" in data else None
        ),
        last_total_energy_J=float(last["total_energy"]),
        last_external_work_J=float(last["external_work"]),
        last_energy_error_J=float(last_balance["energy_error"]),
        max_relative_energy_error=float(balance["relative_energy_error"].max()),
    )


__all__ = [
    "EnergySummary",
    "recoverable_energy",
    "energy_components",
    "estimate_initial_energy",
    "energy_balance_history",
    "relative_energy_error",
    "energy_partition",
    "summarise_energy",
]
