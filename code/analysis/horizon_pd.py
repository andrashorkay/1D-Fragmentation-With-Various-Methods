from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================ #
# Paths and physical parameters.

BASE_PATH = Path("data/PD/horizon_sweep")

E_TARGET = 370e9
L = 0.005
A = 1e-6
STRAIN_RATE = 1e4

# Original/reference horizon.
REFERENCE_HORIZON = 5.015e-5


# ============================================================================ #
# Readers.


def read_results(file_path):
    with open(file_path, "r") as file:
        header = file.readline().strip()

    columns = header[1:].split()

    return pd.read_csv(
        file_path,
        sep=r"\s+",
        comment="#",
        names=columns,
    )


def read_parameter(file_path, key):
    pattern = re.compile(rf"^[\s]*{re.escape(key)}[\s]*=[\s]*(.+?)[\s]*$")

    with open(file_path, "r") as file:
        for line in file:
            match = pattern.match(line)

            if match:
                return float(match.group(1))

    raise ValueError(f"Parameter '{key}' not found in {file_path}")


def read_energy_error(case_path):
    logs = list(case_path.glob("run_*.log"))

    if len(logs) != 1:
        return np.nan

    pattern = re.compile(r"Energy error:\s*([0-9.eE+-]+)")

    with open(logs[0], "r") as file:
        for line in file:
            match = pattern.search(line)

            if match:
                return float(match.group(1))

    return np.nan


# ============================================================================ #
# Effective Young's modulus.


def effective_modulus(data, strain_rate, area, length):
    time = data["time"].to_numpy()
    strain_energy = data["strain_energy"].to_numpy()
    dissipated = data["dissipated"].to_numpy()
    num_fragments = data["num_fragments"].to_numpy()

    strain = strain_rate * time

    elastic = (
        (time > 0.0) & (num_fragments == 1) & np.isclose(dissipated, 0.0, atol=1e-14)
    )

    if not np.any(elastic):
        return np.nan

    E_eff = 2.0 * strain_energy[elastic] / (area * length * strain[elastic] ** 2)

    return np.median(E_eff)


# ============================================================================ #
# Analyse one horizon.


def analyse_case(case_path):
    param_file = case_path / "parameters_used.dat"
    results_file = case_path / "results.out"

    horizon = read_parameter(
        param_file,
        "horizon",
    )

    dx = read_parameter(
        param_file,
        "dx",
    )

    strain_rate = read_parameter(
        param_file,
        "strain_rate",
    )

    data = read_results(results_file)

    num_points = int(round(L / dx))

    final_fragments = int(data["num_fragments"].iloc[-1])

    if final_fragments > 0:
        mean_fragment_size = L / final_fragments
    else:
        mean_fragment_size = np.nan

    fragmented = data["num_fragments"] > 1

    if fragmented.any():
        first_index = np.flatnonzero(fragmented.to_numpy())[0]

        onset_time = data["time"].iloc[first_index]
    else:
        onset_time = np.nan

    final_dissipated = data["dissipated"].iloc[-1]

    final_time = data["time"].iloc[-1]

    energy_error = read_energy_error(case_path)

    E_eff = effective_modulus(
        data,
        strain_rate,
        A,
        L,
    )

    return {
        "horizon": horizon,
        "horizon_um": horizon * 1e6,
        "dx": dx,
        "dx_um": dx * 1e6,
        "horizon_over_dx": horizon / dx,
        "num_points": num_points,
        "num_fragments": final_fragments,
        "mean_fragment_size_mm": mean_fragment_size * 1e3,
        "onset_us": onset_time * 1e6,
        "dissipated": final_dissipated,
        "energy_error": energy_error,
        "E_eff_GPa": E_eff / 1e9,
        "E_error_percent": abs(E_eff - E_TARGET) / E_TARGET * 100.0,
        "final_time_us": final_time * 1e6,
    }


# ============================================================================ #
# Read all horizon cases.

case_paths = [path for path in BASE_PATH.glob("horizon_*") if path.is_dir()]

results = []

for case_path in case_paths:
    try:
        results.append(analyse_case(case_path))

    except Exception as error:
        print(f"Skipping {case_path.name}: {error}")


results = pd.DataFrame(results)

if results.empty:
    raise RuntimeError(f"No valid horizon results found in {BASE_PATH}")

results = results.sort_values("horizon").reset_index(drop=True)


# ============================================================================ #
# Find reference horizon.

reference_index = (results["horizon"] - REFERENCE_HORIZON).abs().idxmin()

reference = results.loc[reference_index]


# ============================================================================ #
# Relative changes from reference horizon.

results["fragment_size_diff_percent"] = (
    abs(results["mean_fragment_size_mm"] - reference["mean_fragment_size_mm"])
    / reference["mean_fragment_size_mm"]
    * 100.0
)


if abs(reference["dissipated"]) > 0.0:
    results["dissipated_diff_percent"] = (
        abs(results["dissipated"] - reference["dissipated"])
        / abs(reference["dissipated"])
        * 100.0
    )
else:
    results["dissipated_diff_percent"] = np.nan


if not np.isnan(reference["onset_us"]):
    results["onset_diff_percent"] = (
        abs(results["onset_us"] - reference["onset_us"])
        / abs(reference["onset_us"])
        * 100.0
    )
else:
    results["onset_diff_percent"] = np.nan


# ============================================================================ #
# Print main results.

columns = [
    "horizon_um",
    "dx_um",
    "horizon_over_dx",
    "num_fragments",
    "mean_fragment_size_mm",
    "onset_us",
    "dissipated",
    "energy_error",
    "E_eff_GPa",
    "E_error_percent",
    "final_time_us",
]

print()
print("PD horizon sensitivity:")
print()

print(
    results[columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)


# ============================================================================ #
# Print comparison with the original horizon.

print()
print("Reference horizon: " f"{reference['horizon_um']:.6g} um")
print()

comparison_columns = [
    "horizon_um",
    "fragment_size_diff_percent",
    "dissipated_diff_percent",
    "onset_diff_percent",
]

print(
    results[comparison_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)


# ============================================================================ #
# Save summary.

summary_file = BASE_PATH / "horizon_summary.csv"

results.to_csv(
    summary_file,
    index=False,
)

print()
print(f"Summary written to {summary_file}")


# ============================================================================ #
# Plots.

x = results["horizon_um"]


# ---------------------------------------------------------------------------- #
# Mean fragment size.

plt.figure()

plt.plot(
    x,
    results["mean_fragment_size_mm"],
    marker="o",
)

plt.axvline(
    REFERENCE_HORIZON * 1e6,
    linestyle="--",
    label="Reference horizon",
)

plt.xlabel(r"Horizon $\delta$ [$\mu$m]")

plt.ylabel("Mean fragment size [mm]")

plt.legend()
plt.tight_layout()


# ---------------------------------------------------------------------------- #
# Number of fragments.

plt.figure()

plt.plot(
    x,
    results["num_fragments"],
    marker="o",
)

plt.axvline(
    REFERENCE_HORIZON * 1e6,
    linestyle="--",
)

plt.xlabel(r"Horizon $\delta$ [$\mu$m]")

plt.ylabel("Final number of fragments")

plt.tight_layout()


# ---------------------------------------------------------------------------- #
# Fragmentation onset.

plt.figure()

plt.plot(
    x,
    results["onset_us"],
    marker="o",
)

plt.axvline(
    REFERENCE_HORIZON * 1e6,
    linestyle="--",
)

plt.xlabel(r"Horizon $\delta$ [$\mu$m]")

plt.ylabel(r"Fragmentation onset [$\mu$s]")

plt.tight_layout()


# ---------------------------------------------------------------------------- #
# Dissipated energy.

plt.figure()

plt.plot(
    x,
    results["dissipated"],
    marker="o",
)

plt.axvline(
    REFERENCE_HORIZON * 1e6,
    linestyle="--",
)

plt.xlabel(r"Horizon $\delta$ [$\mu$m]")

plt.ylabel("Dissipated energy [J]")

plt.tight_layout()


# ---------------------------------------------------------------------------- #
# Effective Young's modulus.

plt.figure()

plt.plot(
    x,
    results["E_eff_GPa"],
    marker="o",
    label=r"$E_{\mathrm{eff}}$",
)

plt.axhline(
    E_TARGET / 1e9,
    linestyle="--",
    label="Target E",
)

plt.axvline(
    REFERENCE_HORIZON * 1e6,
    linestyle=":",
    label="Reference horizon",
)

plt.xlabel(r"Horizon $\delta$ [$\mu$m]")

plt.ylabel(r"$E_{\mathrm{eff}}$ [GPa]")

plt.legend()
plt.tight_layout()


# ---------------------------------------------------------------------------- #
# Energy error.

plt.figure()

plt.plot(
    x,
    results["energy_error"],
    marker="o",
)

plt.axvline(
    REFERENCE_HORIZON * 1e6,
    linestyle="--",
)

plt.xlabel(r"Horizon $\delta$ [$\mu$m]")

plt.ylabel("Energy error")

plt.tight_layout()


# ============================================================================ #

plt.show()
