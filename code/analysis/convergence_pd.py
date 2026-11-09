from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================ #
# Paths and physical parameters.

BASE_PATH = Path("data/PD/discretisation_convergence")

E_TARGET = 370e9
L = 0.005
A = 1e-6
STRAIN_RATE = 1e4
HORIZON = 5.015e-5


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
# Analyse one case.


def analyse_case(case_path):
    param_file = case_path / "parameters_used.dat"
    results_file = case_path / "results.out"

    dx = read_parameter(param_file, "dx")
    horizon = read_parameter(param_file, "horizon")
    strain_rate = read_parameter(
        param_file,
        "strain_rate",
    )

    data = read_results(results_file)

    num_points = int(round(L / dx))

    final_fragments = int(data["num_fragments"].iloc[-1])

    mean_fragment_size = L / final_fragments if final_fragments > 0 else np.nan

    fragmented = data["num_fragments"] > 1

    if fragmented.any():
        first_index = np.flatnonzero(fragmented)[0]
        onset_time = data["time"].iloc[first_index]
    else:
        onset_time = np.nan

    final_dissipated = data["dissipated"].iloc[-1]

    energy_error = read_energy_error(case_path)

    E_eff = effective_modulus(
        data,
        strain_rate,
        A,
        L,
    )

    return {
        "dx": dx,
        "dx_um": dx * 1e6,
        "num_points": num_points,
        "horizon": horizon,
        "horizon_over_dx": horizon / dx,
        "num_fragments": final_fragments,
        "mean_fragment_size_mm": mean_fragment_size * 1e3,
        "onset_us": onset_time * 1e6,
        "dissipated": final_dissipated,
        "energy_error": energy_error,
        "E_eff_GPa": E_eff / 1e9,
        "E_error_percent": abs(E_eff - E_TARGET) / E_TARGET * 100.0,
    }


# ============================================================================ #
# Read all cases.

case_paths = [path for path in BASE_PATH.glob("dx_*") if path.is_dir()]

results = []

for case_path in case_paths:
    try:
        results.append(analyse_case(case_path))
    except Exception as error:
        print(f"Skipping {case_path.name}: {error}")


results = pd.DataFrame(results)

results = results.sort_values("num_points").reset_index(drop=True)


# ============================================================================ #
# Print results.

columns = [
    "num_points",
    "dx_um",
    "horizon_over_dx",
    "num_fragments",
    "mean_fragment_size_mm",
    "onset_us",
    "dissipated",
    "energy_error",
    "E_eff_GPa",
    "E_error_percent",
]

print()
print("PD discretisation convergence:")
print()

print(
    results[columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)


# ============================================================================ #
# Relative differences from finest discretisation.

reference = results.iloc[-1]

results["fragment_size_diff_percent"] = (
    abs(results["mean_fragment_size_mm"] - reference["mean_fragment_size_mm"])
    / reference["mean_fragment_size_mm"]
    * 100.0
)

results["dissipated_diff_percent"] = (
    abs(results["dissipated"] - reference["dissipated"])
    / abs(reference["dissipated"])
    * 100.0
)

results["onset_diff_percent"] = (
    abs(results["onset_us"] - reference["onset_us"])
    / abs(reference["onset_us"])
    * 100.0
)


print()
print("Difference relative to finest discretisation:")
print()

print(
    results[
        [
            "num_points",
            "fragment_size_diff_percent",
            "dissipated_diff_percent",
            "onset_diff_percent",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)


# ============================================================================ #
# Plots.

x = results["num_points"]


plt.figure()

plt.plot(
    x,
    results["mean_fragment_size_mm"],
    marker="o",
)

plt.xlabel("Number of material points")
plt.ylabel("Mean fragment size [mm]")
plt.tight_layout()


plt.figure()

plt.plot(
    x,
    results["num_fragments"],
    marker="o",
)

plt.xlabel("Number of material points")
plt.ylabel("Final number of fragments")
plt.tight_layout()


plt.figure()

plt.plot(
    x,
    results["onset_us"],
    marker="o",
)

plt.xlabel("Number of material points")
plt.ylabel(r"Fragmentation onset [$\mu$s]")
plt.tight_layout()


plt.figure()

plt.plot(
    x,
    results["dissipated"],
    marker="o",
)

plt.xlabel("Number of material points")
plt.ylabel("Dissipated energy [J]")
plt.tight_layout()


plt.figure()

plt.plot(
    x,
    results["energy_error"],
    marker="o",
)

plt.xlabel("Number of material points")
plt.ylabel("Energy error")
plt.tight_layout()


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

plt.xlabel("Number of material points")
plt.ylabel(r"$E_{\mathrm{eff}}$ [GPa]")
plt.legend()
plt.tight_layout()


plt.show()
