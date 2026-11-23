"""
Author: Andras Horkay
Contact: andras.horkay@gmail.com

Load phase-field (PF) simulation results without analysing or plotting them.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .readers import find_file, read_key_value_file, read_table


ParameterValue = float | int | bool | str


@dataclass
class PFCase:
    """Data loaded from one phase-field simulation directory.

    ``history`` and ``fragments`` retain their original output columns.
    ``materials`` preserves solid and phase-field definitions separately.
    Spatial nodal/material snapshots are read only when requested.
    """

    path: Path
    parameters: dict[str, ParameterValue]
    history: pd.DataFrame
    diagnostics: pd.DataFrame | None
    fragments: pd.DataFrame | None
    raw_fragments: pd.DataFrame | None
    summary: dict[str, ParameterValue]
    material_properties: pd.DataFrame | None
    method: str = "PF"
    materials: dict[str, dict[str, dict[str, ParameterValue]]] = field(
        default_factory=dict
    )

    def final_fragments(self, raw: bool = False) -> pd.DataFrame:
        """Return fragments from the last *saved* snapshot.

        The final saved snapshot may precede ``summary['physical_time']``
        if the last simulation step was not a diagnostic dump.
        """

        if raw:
            raise ValueError("PF does not write a separate raw-fragment table")

        if self.fragments is None or self.fragments.empty:
            raise ValueError(f"No fragment data in {self.path}")

        if "time" not in self.fragments.columns:
            raise ValueError(f"Missing 'time' in {self.path} fragment data")

        last_time = self.fragments["time"].max()
        return self.fragments.loc[self.fragments["time"] == last_time].copy()

    def load_nodal_field(self, step: int | None = None) -> pd.DataFrame:
        """Read a nodal snapshot (latest available, by default).

        Returns the original columns: x, displacement, velocity,
        acceleration and damage. Large snapshots are not cached on the case.
        """

        return _read_snapshot(self.path, "nodal", step)

    def load_material_field(self, step: int | None = None) -> pd.DataFrame:
        """Read a material-point snapshot (latest available, by default)."""

        return _read_snapshot(self.path, "material", step)


def _parse_value(value: str) -> ParameterValue:
    """Convert a scalar Akantu or summary value to a Python type."""

    value = value.strip().strip('"').strip("'")
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    for converter in (int, float):
        try:
            return converter(value)
        except ValueError:
            pass
    return value


def _read_pf_parameters(
    path: Path,
) -> tuple[
    dict[str, ParameterValue],
    dict[str, dict[str, dict[str, ParameterValue]]],
]:
    """Parse the project's nested Akantu PF input file.

    Preserves the distinction between ``material phasefield ...`` in the
    solid-mechanics model and ``phasefield linear ...`` in the damage model.
    Supports one homogeneous phase field or multiple named spatial zones.
    This is not a general-purpose Akantu input parser.
    """

    user: dict[str, ParameterValue] = {}
    materials: dict[str, dict[str, dict[str, ParameterValue]]] = {
        "solid": {},
        "phase_field": {},
    }
    stack: list[tuple[str, dict[str, ParameterValue] | None]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, original in enumerate(file, start=1):
            line = original.split("#", maxsplit=1)[0].strip()
            if not line:
                continue

            if line.endswith("["):
                label = line[:-1].strip()
                block: dict[str, ParameterValue] | None = None
                if label == "user parameters":
                    block = user
                elif label.startswith("material "):
                    block = {"material_type": label[len("material ") :]}
                elif label.startswith("phasefield "):
                    block = {"phasefield_type": label[len("phasefield ") :]}
                stack.append((label, block))
                continue

            if line == "]":
                if not stack:
                    raise ValueError(f"Unexpected ']' in {path}:{line_number}")
                label, block = stack.pop()
                if block is None or label == "user parameters":
                    continue

                group = "solid" if label.startswith("material ") else "phase_field"
                name = str(block.get("name", f"unnamed_{len(materials[group]) + 1}"))
                if name in materials[group]:
                    raise ValueError(f"Duplicate {group} block '{name}' in {path}")
                materials[group][name] = block
                continue

            if "=" not in line:
                raise ValueError(f"Invalid PF input at {path}:{line_number}: {line}")

            key, value = (part.strip() for part in line.split("=", maxsplit=1))
            if not key or not value:
                raise ValueError(f"Invalid assignment at {path}:{line_number}")

            block = next(
                (item for _, item in reversed(stack) if item is not None), None
            )
            if block is None:
                continue  # Model-level settings outside user/material blocks.
            if key in block:
                raise ValueError(f"Duplicate parameter '{key}' in {path}")
            block[key] = _parse_value(value)

    if stack:
        raise ValueError(f"Unclosed material block(s) in {path}")

    parameters = dict(user)

    # Solid properties should agree between the solid model and phase model;
    # choose the solid definition as authoritative for mechanical properties.
    if len(materials["solid"]) == 1:
        solid = next(iter(materials["solid"].values()))
        for key in ("E", "rho", "nu"):
            if key in solid:
                parameters.setdefault(key, solid[key])

    # Only expose a single l0/gc as a global property for a homogeneous PF
    # model. Multi-zone models have locally varying properties instead.
    if len(materials["phase_field"]) == 1:
        phase = next(iter(materials["phase_field"].values()))
        for key in ("l0", "gc"):
            if key in phase:
                parameters.setdefault(key, phase[key])
        if "gc" in phase:
            parameters.setdefault("G_c", phase["gc"])

    return parameters, materials


def _read_summary(directory: Path) -> dict[str, ParameterValue]:
    """Read the PF key-value run summary, if available."""

    path = directory / "summary.out"
    if not path.is_file():
        return {}
    return {
        key: _parse_value(value) for key, value in read_key_value_file(path).items()
    }


def _read_optional_table(directory: Path, filename: str) -> pd.DataFrame | None:
    """Read an optional numerical table."""

    path = directory / filename
    return read_table(path) if path.is_file() else None


def _read_snapshot(directory: Path, kind: str, step: int | None) -> pd.DataFrame:
    """Read a requested PF spatial snapshot from ``fields/``."""

    fields = directory / "fields"
    if step is not None:
        if step < 0:
            raise ValueError("Snapshot step must be non-negative")
        path = fields / f"{kind}_{step}.out"
    else:
        candidates = []
        for path in fields.glob(f"{kind}_*.out"):
            suffix = path.stem.removeprefix(f"{kind}_")
            if suffix.isdecimal():
                candidates.append((int(suffix), path))
        if not candidates:
            raise FileNotFoundError(f"No {kind} snapshots in {fields}")
        path = max(candidates, key=lambda pair: pair[0])[1]

    return read_table(path)


def load_pf_case(directory: str | Path) -> PFCase:
    """Load one PF simulation, retaining recorded outputs without analysis.

    The directory should contain ``energies_pf.out`` (or the configured
    diagnostic filename). Material configuration and other outputs are
    optional. Individual fragment lengths and masses come from the actual
    PF fragment writer, not from a reconstructed fragment count.
    """

    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"PF case directory not found: {directory}")

    parameter_file = next(
        (
            directory / filename
            for filename in (
                "parameters_used.dat",
                "material_generated.dat",
                "material.dat",
            )
            if (directory / filename).is_file()
        ),
        None,
    )
    if parameter_file is not None:
        parameters, materials = _read_pf_parameters(parameter_file)
    else:
        parameters, materials = {}, {"solid": {}, "phase_field": {}}

    # The l0-sensitivity sweep stores its scale in case_info.txt because the
    # actual local l0(x) values are generated at run time from the common field.
    info_file = directory / "case_info.txt"
    if info_file.is_file():
        with info_file.open("r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("l0_scale="):
                    parameters["l0_scale"] = _parse_value(line.split("=", 1)[1])
                    break

    configured = str(parameters.get("diagnostics_file", "energies_pf.out"))
    history_file = find_file(
        directory, configured, "energies_pf.out", "energies.out", "results.out"
    )
    history = read_table(history_file)
    if history.empty:
        raise ValueError(f"No PF diagnostic records in {history_file}")

    required = {
        "step",
        "time",
        "kinetic_energy",
        "elastic_energy",
        "fracture_energy",
        "num_fragments",
        "mean_fragment_size",
        "average_stress",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(
            f"Missing PF diagnostic columns in {history_file}: {sorted(missing)}"
        )

    # PF's bulk potential energy is the recoverable elastic contribution;
    # fracture_energy is the separately recorded phase-field energy.
    history["recoverable_energy"] = history["elastic_energy"]

    fragment_name = str(parameters.get("fragment_file", "fragments_pf.out"))
    fragments = _read_optional_table(directory, fragment_name)
    if fragments is None and fragment_name != "fragments_pf.out":
        fragments = _read_optional_table(directory, "fragments_pf.out")
    if fragments is None:
        fragments = _read_optional_table(directory, "fragments.out")

    return PFCase(
        path=directory,
        parameters=parameters,
        history=history,
        diagnostics=None,  # Global PF diagnostics are stored in history.
        fragments=fragments,
        raw_fragments=None,  # PF has one 1D damage-threshold definition.
        summary=_read_summary(directory),
        material_properties=None,  # Optional spatial fields are loaded lazily.
        materials=materials,
    )
