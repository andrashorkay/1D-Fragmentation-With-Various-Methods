# SPDX-License-Identifier: GPL-2.0-only
"""Shared synthetic fixtures; no simulation binaries or data downloads needed."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import pandas as pd
import pytest


@pytest.fixture
def fragments() -> pd.DataFrame:
    """One final snapshot, with lengths summing to a 4 mm bar."""
    return pd.DataFrame(
        {
            "time": [1e-6] * 3,
            "fragment_id": [0, 1, 2],
            "size": [0.001, 0.002, 0.001],
            "mass": [0.1, 0.2, 0.1],
            "centre": [-0.0015, 0.0, 0.0015],
        }
    )


@pytest.fixture
def energy_history() -> pd.DataFrame:
    """Perfectly balanced mock case, including ghost-inclusive recorded total."""
    return pd.DataFrame(
        {
            "time": [0.0, 1e-6, 2e-6],
            "kinetic_energy": [0.0, 0.1, 0.2],
            "recoverable_energy": [0.2, 0.3, 0.4],
            "elastic_energy": [0.2, 0.3, 0.4],
            "fracture_energy": [0.0, 0.05, 0.1],
            "total_energy": [5.0, 5.2, 5.4],
            "external_work": [0.0, 0.2, 0.4],
            "energy_error": [0.0, 0.0, 0.0],
            "average_stress": [0.0, 2e8, 1e8],
            "num_fragments": [1, 2, 3],
            "mean_fragment_size": [0.004, 0.002, 0.004 / 3],
        }
    )


@dataclass
class SyntheticCase:
    path: Path
    method: str
    history: pd.DataFrame
    fragments: pd.DataFrame | None = None
    raw_fragments: pd.DataFrame | None = None
    parameters: dict = field(default_factory=lambda: {"bar_length": 0.004})
    summary: dict = field(default_factory=lambda: {"physical_time": 2e-6})
    diagnostics: pd.DataFrame | None = None
    materials: dict = field(default_factory=dict)

    def final_fragments(self, raw: bool = False) -> pd.DataFrame:
        if raw and self.method != "PD":
            raise ValueError("Raw fragments are PD-only")
        table = self.raw_fragments if raw else self.fragments
        if table is None or table.empty:
            raise ValueError("Missing fragment snapshot")
        key = "step" if self.method == "PD" and "step" in table else "time"
        return table.loc[table[key] == table[key].max()].copy()


@pytest.fixture
def make_case(fragments, energy_history):
    def factory(method: str, path: Path, *, strain_rate: float = 1e5):
        table = fragments.copy()
        if method == "PD":
            table["step"] = 100
            table["x_left"] = [-0.002, -0.001, 0.001]
            table["x_right"] = [-0.001, 0.001, 0.002]
            raw = table.drop(columns=["mass"]).copy()
        else:
            raw = None
        params = {
            "bar_length": 0.004,
            "strain_rate": strain_rate,
            "time_step_factor": 0.05,
            "number_elements": 1000,
            "dx": 4e-6,
            "E": 370e9,
            "rho": 3900,
            "sigma_c": 262e6,
            "G_c": 50,
        }
        return SyntheticCase(
            path=path,
            method=method,
            parameters=params,
            history=energy_history.copy(),
            fragments=table,
            raw_fragments=raw,
            summary={"physical_time": 2e-6, "max_relative_energy_error": 0.0},
            diagnostics=energy_history.copy() if method == "PD" else None,
        )

    return factory
