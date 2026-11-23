"""
Author: Andras Horkay
Contact: andras.horkay@gmail.com

Load cohesive-zone model (CZM) simulation results without analysing them.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .readers import find_file, read_key_value_file, read_table


@dataclass
class CZMCase:
    """Data loaded from one CZM simulation directory."""

    path: Path
    parameters: dict[str, float | int | bool | str]
    history: pd.DataFrame
    diagnostics: pd.DataFrame | None
    fragments: pd.DataFrame | None
    raw_fragments: pd.DataFrame | None
    summary: dict[str, float | int | bool | str]
    material_properties: pd.DataFrame | None
    method: str = "CZM"
    materials: dict[str, dict[str, float | int | bool | str]] = field(
        default_factory=dict
    )

    def final_fragments(self, raw: bool = False) -> pd.DataFrame:
        """Return the fragments from the last recorded time snapshot.

        CZM writes a single fragment definition; it does not distinguish
        between raw and horizon-filtered fragments as PD does.
        """

        if raw:
            raise ValueError(
                "CZM does not provide a separate raw-fragment table"
            )

        if self.fragments is None or self.fragments.empty:
            raise ValueError(f"No fragment data in {self.path}")

        if "time" not in self.fragments.columns:
            raise ValueError(
                f"Missing 'time' column in {self.path / 'fragments.out'}"
            )

        final_time = self.fragments["time"].max()
        return self.fragments.loc[self.fragments["time"] == final_time].copy()


def _parse_value(value: str) -> float | int | bool | str:
    """Convert one Akantu parameter or summary value to a Python type."""

    value = value.strip().strip('"').strip("'")

    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    for converter in (int, float):
        try:
            return converter(value)
        except ValueError:
            pass

    return value


def _read_material_parameters(
    path: Path,
) -> tuple[
    dict[str, float | int | bool | str],
    dict[str, dict[str, float | int | bool | str]],
]:
    """Read user parameters and the two material blocks in CZM material.dat.

    This is a focused reader for the project's Akantu input layout, not a
    general Akantu parser. The nested material sections are kept separate.
    """

    sections: dict[str, dict[str, float | int | bool | str]] = {
        "user": {},
        "elastic": {},
        "cohesive": {},
    }
    stack: list[str | None] = []
    encountered: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, original in enumerate(file, start=1):
            line = original.split("#", maxsplit=1)[0].strip()
            if not line:
                continue

            if line.endswith("["):
                label = line[:-1].strip()
                section = None

                if label == "user parameters":
                    section = "user"
                elif label.startswith("material "):
                    material_type = label[len("material ") :].strip()
                    if material_type.startswith("elastic"):
                        section = "elastic"
                    elif material_type.startswith("cohesive"):
                        section = "cohesive"

                if section is not None:
                    if section in encountered:
                        raise ValueError(
                            f"Multiple '{section}' blocks in {path}; "
                            "this reader expects one of each"
                        )
                    encountered.add(section)
                    if label.startswith("material "):
                        sections[section]["material_type"] = material_type

                stack.append(section)
                continue

            if line == "]":
                if not stack:
                    raise ValueError(
                        f"Unexpected closing bracket at {path}:{line_number}"
                    )
                stack.pop()
                continue

            if "=" not in line:
                raise ValueError(
                    f"Unrecognised line in {path} at {line_number}: {line}"
                )

            key, value = (part.strip() for part in line.split("=", maxsplit=1))
            if not key or not value:
                raise ValueError(f"Invalid assignment at {path}:{line_number}")

            # Ignore non-material model settings. The current input layout
            # defines only user, elastic and cohesive parameters.
            section = next(
                (item for item in reversed(stack) if item is not None), None
            )
            if section is None:
                continue

            if key in sections[section]:
                raise ValueError(f"Duplicate parameter '{key}' in {path}")

            sections[section][key] = _parse_value(value)

    if stack:
        raise ValueError(f"Unclosed bracket(s) in {path}")

    # Put frequently used physical settings at the top level so that the
    # analysis interface mirrors PD's flat case.parameters dictionary.
    parameters = dict(sections["user"])
    for section, keys in (
        ("elastic", ("E", "rho", "nu")),
        ("cohesive", ("sigma_c", "G_c", "delta_c")),
    ):
        for key in keys:
            if key in sections[section]:
                if key in parameters:
                    raise ValueError(f"Ambiguous parameter '{key}' in {path}")
                parameters[key] = sections[section][key]

    materials = {
        "elastic": sections["elastic"],
        "cohesive": sections["cohesive"],
    }
    return parameters, materials


def _read_optional_table(directory: Path, filename: str) -> pd.DataFrame | None:
    """Read an optional whitespace-separated table."""

    path = directory / filename
    return read_table(path) if path.is_file() else None


def _read_summary(directory: Path) -> dict[str, float | int | bool | str]:
    """Read the CZM summary.out key-value pairs, if present."""

    path = directory / "summary.out"
    if not path.is_file():
        return {}

    return {
        key: _parse_value(value)
        for key, value in read_key_value_file(path).items()
    }


def load_czm_case(directory: str | Path) -> CZMCase:
    """Load one CZM simulation, without computing derived statistics.

    The case directory is expected to contain energies.out (or the output
    filename specified in parameters_used.dat). Parameters, fragments and
    summary are optional; large spatial field snapshots are not loaded.
    """

    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"CZM case directory not found: {directory}")

    parameter_file = next(
        (
            directory / filename
            for filename in ("parameters_used.dat", "material.dat")
            if (directory / filename).is_file()
        ),
        None,
    )
    if parameter_file is not None:
        parameters, materials = _read_material_parameters(parameter_file)
    else:
        parameters, materials = {}, {}

    output_name = str(parameters.get("output_file", "energies.out"))
    history_file = find_file(
        directory, output_name, "energies.out", "results.out"
    )
    history = read_table(history_file)

    if history.empty:
        raise ValueError(f"No data rows in {history_file}")

    required = {
        "step",
        "time",
        "kinetic_energy",
        "elastic_energy",
        "reversible_cohesive_energy",
        "fracture_energy",
        "contact_energy",
        "num_fragments",
        "mean_fragment_size",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(
            f"Missing CZM history columns in {history_file}: {sorted(missing)}"
        )

    # Cohesive-zone elastic storage is recoverable. Retain both original
    # energy columns; contact energy remains a distinct contribution.
    history["recoverable_energy"] = (
        history["elastic_energy"] + history["reversible_cohesive_energy"]
    )

    fragment_name = str(parameters.get("fragment_file", "fragments.out"))
    fragments = _read_optional_table(directory, fragment_name)
    if fragments is None and fragment_name != "fragments.out":
        fragments = _read_optional_table(directory, "fragments.out")

    return CZMCase(
        path=directory,
        parameters=parameters,
        history=history,
        diagnostics=None,  # CZM global diagnostics are in history itself.
        fragments=fragments,
        raw_fragments=None,  # No separate raw/filtered definition in CZM.
        summary=_read_summary(directory),
        material_properties=None,  # Spatial fields are loaded on demand.
        materials=materials,
    )
