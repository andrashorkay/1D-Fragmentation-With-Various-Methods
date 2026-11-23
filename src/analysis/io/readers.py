"""
Author: Andras Horkay
Contact: andras.horkay@gmail.com

General-purpose file readers for fragmentation simulation results.

This module handles common file formats independently of the
numerical method. Method-specific interpretation belongs in
io/czm.py, io/pd.py and io/pf.py.
"""

from pathlib import Path

import pandas as pd


# ============================================================================
# Numerical tables


def read_table(path: str | Path) -> pd.DataFrame:
    """
    Read a whitespace-separated table with a commented column header.

    Expected format:
        # step time kinetic_energy fracture_energy
        0 0.0 0.0 0.0
        1 1e-9 0.1 0.01

    Parameters
    ----------
    path : str | Path
        Path to the numerical output file.

    Returns
    -------
    pd.DataFrame
        Table with column names taken from the header.
    """

    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Output file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        header = next(
            (line.strip() for line in file if line.strip()),
            None,
        )

    if header is None or not header.startswith("#"):
        raise ValueError(f"Expected a column header beginning with '#' in {path}")

    columns = header[1:].split()

    if not columns:
        raise ValueError(f"Empty column header in {path}")

    if len(columns) != len(set(columns)):
        raise ValueError(f"Duplicate column names in {path}")

    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=columns,
    )


# ============================================================================
# Flat parameter files


def read_parameters(path: str | Path) -> dict[str, str]:
    """
    Read a flat parameter file containing 'key = value' assignments.

    Expected format:
        E = 370e9
        density = 3900
        strain_rate = 1e5

    Values are returned as strings. Method-specific readers
    convert them to the required Python types.

    This function does not parse nested Akantu material blocks.
    """

    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Parameter file not found: {path}")

    parameters = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.split("#", maxsplit=1)[0].strip()

            if not line:
                continue

            if "=" not in line:
                raise ValueError(
                    f"Invalid parameter line {line_number} in {path}: {line}"
                )

            key, value = line.split("=", maxsplit=1)

            key = key.strip()
            value = value.strip()

            if not key or not value:
                raise ValueError(f"Invalid assignment on line {line_number} in {path}")

            if key in parameters:
                raise ValueError(f"Duplicate parameter '{key}' in {path}")

            parameters[key] = value

    return parameters


# ============================================================================
# Key-value metadata


def read_key_value_file(path: str | Path) -> dict[str, str]:
    """
    Read whitespace-separated key-value metadata.
    Expected format:
        length 0.005
        strain_rate 100000
        num_points 8001

    """

    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    values = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.split("#", maxsplit=1)[0].strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                raise ValueError(f"Invalid metadata line {line_number} in {path}")

            key, value = parts

            if key in values:
                raise ValueError(f"Duplicate metadata key '{key}' in {path}")

            values[key] = value

    return values


# ============================================================================
# File discovery


def find_file(directory: str | Path, *filenames: str) -> Path:
    """
    Return the first existing file among the supplied filenames.

    Candidate filenames are checked in the order provided.
    """

    directory = Path(directory).expanduser()

    if not directory.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")

    for filename in filenames:
        candidate = directory / filename

        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"None of {filenames} found in {directory}")


# ============================================================================
# Simulation directory discovery


def find_case_directories(
    directory: str | Path,
    pattern: str = "rate_*",
) -> list[Path]:
    """
    Discover simulation directories matching a glob pattern.

    Directories are returned in alphabetical order.
    Numerical sorting belongs in the corresponding analysis.
    """

    directory = Path(directory).expanduser()

    if not directory.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")

    cases = sorted(path for path in directory.glob(pattern) if path.is_dir())

    if not cases:
        raise FileNotFoundError(
            f"No case directories matching '{pattern}' in {directory}"
        )

    return cases
