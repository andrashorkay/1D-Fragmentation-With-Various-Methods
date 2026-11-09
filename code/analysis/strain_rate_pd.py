from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_PATH = Path("data/PD/strain_rate_sweep")


# ============================================================================ #
# Physical parameters.

RHO = 3900.0
E = 370e9
SIGMA_C = 262e6
GC = 50.0

L = 0.005
A = 1e-6


# ============================================================================ #
# Characteristic quantities.

C = np.sqrt(E / RHO)

STRAIN_RATE_0 = SIGMA_C**3 * C / (E**2 * GC)

S_0 = E * GC / SIGMA_C**2


print()
print(f"Wave speed c       = {C:.6g} m/s")
print(f"Characteristic rate = " f"{STRAIN_RATE_0:.6g} 1/s")
print(f"Characteristic size = " f"{S_0 * 1e3:.6g} mm")


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
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$")

    with open(file_path, "r") as file:
        for line in file:
            match = pattern.match(line)

            if match:
                return float(match.group(1))

    raise ValueError(f"Parameter '{key}' not found in {file_path}")


def energy_balance(data):
    kinetic = data["kinetic"].to_numpy()
    strain = data["strain_energy"].to_numpy()
    dissipated = data["dissipated"].to_numpy()
    external = data["external_work"].to_numpy()

    total = kinetic + strain + dissipated
    total_0 = total[0]

    error = external - (total - total_0)

    max_error = np.max(np.abs(error))

    if abs(total_0) > 0.0:
        relative_error = max_error / abs(total_0) * 100.0
    else:
        relative_error = np.nan

    return max_error, relative_error


# ============================================================================ #
# Analyse one strain rate.


def analyse_case(case_path):
    param_file = case_path / "parameters_used.dat"
    results_file = case_path / "results.out"

    strain_rate = read_parameter(
        param_file,
        "strain_rate",
    )

    dx = read_parameter(
        param_file,
        "dx",
    )

    horizon = read_parameter(
        param_file,
        "horizon",
    )

    scatter = read_parameter(
        param_file,
        "scatter",
    )

    dt_factor = read_parameter(
        param_file,
        "time_step_factor",
    )

    data = read_results(results_file)

    final_fragments = int(data["num_fragments"].iloc[-1])

    mean_size = L / final_fragments if final_fragments > 0 else np.nan

    fragmented = data["num_fragments"] > 1

    if fragmented.any():
        index = np.flatnonzero(fragmented.to_numpy())[0]

        onset = data["time"].iloc[index]
    else:
        onset = np.nan

    final_dissipated = data["dissipated"].iloc[-1]

    max_error, relative_error = energy_balance(data)

    rate_hat = strain_rate / STRAIN_RATE_0

    size_hat = mean_size / S_0

    grady_size = (24.0 * GC / (RHO * strain_rate**2)) ** (1.0 / 3.0)

    grady_hat = grady_size / S_0

    zmr_hat = 4.5 / (1.0 + 4.5 * rate_hat ** (2.0 / 3.0))

    return {
        "strain_rate": strain_rate,
        "rate_hat": rate_hat,
        "dx_um": dx * 1e6,
        "horizon_um": horizon * 1e6,
        "horizon_over_dx": horizon / dx,
        "scatter": scatter,
        "time_step_factor": dt_factor,
        "num_fragments": final_fragments,
        "mean_fragment_size_mm": mean_size * 1e3,
        "size_hat": size_hat,
        "grady_size_mm": grady_size * 1e3,
        "grady_hat": grady_hat,
        "zmr_hat": zmr_hat,
        "onset_us": onset * 1e6,
        "dissipated": final_dissipated,
        "dissipated_per_area": final_dissipated / A,
        "equivalent_Gc_cuts": final_dissipated / (GC * A),
        "max_energy_error": max_error,
        "energy_error_percent": relative_error,
        "final_time_us": data["time"].iloc[-1] * 1e6,
    }


# ============================================================================ #
# Read all rates.

case_paths = [path for path in BASE_PATH.glob("rate_*") if path.is_dir()]

results = []

for case_path in case_paths:
    try:
        results.append(analyse_case(case_path))

    except Exception as error:
        print(f"Skipping {case_path.name}: {error}")


results = pd.DataFrame(results)

if results.empty:
    raise RuntimeError(f"No strain-rate results found in {BASE_PATH}")

results = results.sort_values("strain_rate").reset_index(drop=True)


# ============================================================================ #
# Print table.

columns = [
    "strain_rate",
    "rate_hat",
    "num_fragments",
    "mean_fragment_size_mm",
    "size_hat",
    "grady_hat",
    "zmr_hat",
    "onset_us",
    "dissipated",
    "equivalent_Gc_cuts",
    "energy_error_percent",
]

print()
print("PD strain-rate sweep:")
print()

print(
    results[columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)


# ============================================================================ #
# Save summary.

summary_file = BASE_PATH / "strain_rate_summary.csv"

results.to_csv(
    summary_file,
    index=False,
)

print()
print(f"Summary written to {summary_file}")


# ============================================================================ #
# Fragment size versus strain rate.

plt.figure()

plt.plot(
    results["strain_rate"],
    results["mean_fragment_size_mm"],
    marker="o",
    label="PD",
)

plt.plot(
    results["strain_rate"],
    results["grady_size_mm"],
    linestyle="--",
    label="Grady",
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"Strain rate $\dot{\epsilon}$ [s$^{-1}$]")

plt.ylabel("Mean fragment size [mm]")

plt.legend()
plt.tight_layout()


# ============================================================================ #
# Normalised fragment size.

plt.figure()

plt.plot(
    results["rate_hat"],
    results["size_hat"],
    marker="o",
    label="PD",
)

plt.plot(
    results["rate_hat"],
    results["grady_hat"],
    linestyle="--",
    label="Grady",
)

plt.plot(
    results["rate_hat"],
    results["zmr_hat"],
    linestyle=":",
    label="ZMR",
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$\dot{\epsilon}/\dot{\epsilon}_0$")

plt.ylabel(r"$\bar{s}/s_0$")

plt.legend()
plt.tight_layout()


# ============================================================================ #
# Number of fragments.

plt.figure()

plt.plot(
    results["strain_rate"],
    results["num_fragments"],
    marker="o",
)

plt.xscale("log")

plt.xlabel(r"Strain rate $\dot{\epsilon}$ [s$^{-1}$]")

plt.ylabel("Final number of fragments")

plt.tight_layout()


# ============================================================================ #
# Fragmentation onset.

plt.figure()

plt.plot(
    results["strain_rate"],
    results["onset_us"],
    marker="o",
)

plt.xscale("log")

plt.xlabel(r"Strain rate $\dot{\epsilon}$ [s$^{-1}$]")

plt.ylabel(r"Fragmentation onset [$\mu$s]")

plt.tight_layout()


# ============================================================================ #
# Dissipated energy.

plt.figure()

plt.plot(
    results["strain_rate"],
    results["dissipated"],
    marker="o",
)

plt.xscale("log")

plt.xlabel(r"Strain rate $\dot{\epsilon}$ [s$^{-1}$]")

plt.ylabel("Final dissipated energy [J]")

plt.tight_layout()


# ============================================================================ #
# Energy conservation.

plt.figure()

plt.plot(
    results["strain_rate"],
    results["energy_error_percent"],
    marker="o",
)

plt.xscale("log")

plt.xlabel(r"Strain rate $\dot{\epsilon}$ [s$^{-1}$]")

plt.ylabel("Maximum energy error [%]")

plt.tight_layout()


# ============================================================================ #

plt.show()
