"""
Author: Andras Horkay
Contact: andras.horkay@gmail.com

Read peridynamic (PD) simulation results without analysing or plotting them.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .readers import find_file, read_key_value_file, read_parameters, read_table


@dataclass
class PDCase:
    """Data loaded from one PD simulation directory."""

    path: Path
    parameters: dict[str, float | int | bool | str]
    history: pd.DataFrame
    diagnostics: pd.DataFrame | None
    fragments: pd.DataFrame | None
    raw_fragments: pd.DataFrame | None
    summary: dict
    material_properties: pd.DataFrame | None
    method: str = "PD"

    def final_fragments(self, raw: bool = False) -> pd.DataFrame:
        """Return the last recorded fragment snapshot (raw or filtered)."""

        table = self.raw_fragments if raw else self.fragments
        label = "raw_fragments.out" if raw else "fragments.out"

        if table is None or table.empty:
            raise ValueError(f"No fragment data in {self.path / label}")

        # Use the integer step rather than comparing floating-point times.
        final_step = table["step"].max()
        return table.loc[table["step"] == final_step].copy()


def _read_optional_table(directory: Path, filename: str) -> pd.DataFrame | None:
    """Return a table if it exists, otherwise None."""

    path = directory / filename
    return read_table(path) if path.is_file() else None


def _read_summary(directory: Path) -> dict:
    """Read the one-row final summary, when available."""

    table = _read_optional_table(directory, "summary.out")
    if table is None or table.empty:
        return {}

    if len(table) != 1:
        raise ValueError(f"Expected one row in {directory / 'summary.out'}")

    return table.iloc[0].to_dict()


def _parse_parameter(key: str, value: str) -> float | int | bool | str:
    """Convert numerical PD settings to Python values."""

    if key == "enable_early_stopping":
        return value.lower() in ("true", "1")

    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    for converter in (int, float):
        try:
            return converter(value)
        except ValueError:
            continue

    return value


def load_pd_case(directory: str | Path) -> PDCase:
    """Load one PD simulation without computing derived statistics.

    Parameters
    ----------
    directory : str | Path
        Directory containing results.out (or a configured diagnostics file),
        PD diagnostics, fragments, summary and metadata.

    Returns
    -------
    PDCase
        Original output tables, common column aliases, and run metadata.
    """

    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"PD case directory not found: {directory}")

    # Parameters used by the batch script, if available.
    parameter_file = directory / "parameters_used.dat"
    parameters = read_parameters(parameter_file) if parameter_file.is_file() else {}

    # Metadata is written by the PD executable and contains realised dx, dt,
    # horizon, and other values. Prefer it when both sources define a key.
    metadata_file = directory / "pd_metadata.out"
    if metadata_file.is_file():
        parameters.update(read_key_value_file(metadata_file))

    parameters = {
        key: _parse_parameter(key, value) for key, value in parameters.items()
    }

    # A run may use a non-default diagnostics filename.
    configured_name = parameters.get("diagnostics_file", "results.out")
    history_file = find_file(
        directory,
        configured_name,
        "results.out",
        "diagnostics.out",
        "energies.out",
    )
    history = read_table(history_file)
    if history.empty:
        raise ValueError(f"No data rows in {history_file}")

    required_columns = {
        "step",
        "time",
        "kinetic_energy",
        "elastic_energy",
        "fracture_energy",
        "average_stress",
        "num_fragments",
        "mean_fragment_size",
    }
    missing = required_columns.difference(history.columns)
    if missing:
        raise ValueError(
            f"Missing PD history columns in {history_file}: {sorted(missing)}"
        )

    # Physical elastic energy in results.out excludes full ghost-boundary
    # energy. Retain the original column and provide a common analysis alias.
    history["recoverable_energy"] = history["elastic_energy"]

    return PDCase(
        path=directory,
        parameters=parameters,
        history=history,
        diagnostics=_read_optional_table(directory, "pd_diagnostics.out"),
        fragments=_read_optional_table(directory, "fragments.out"),
        raw_fragments=_read_optional_table(directory, "raw_fragments.out"),
        summary=_read_summary(directory),
        material_properties=_read_optional_table(directory, "material_properties.out"),
    )
