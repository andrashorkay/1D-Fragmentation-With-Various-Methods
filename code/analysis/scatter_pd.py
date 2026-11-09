from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_PATH = Path("data/PD/scatter_sweep")

E_TARGET = 370e9
L = 0.005
A = 1e-6

REFERENCE_SCATTER = 0.01


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


def effective_modulus(data, strain_rate):
    time = data["time"].to_numpy()
    strain = strain_rate * time

    elastic = (
        (time > 0.0)
        & (data["num_fragments"].to_numpy() == 1)
        & np.isclose(
            data["dissipated"].to_numpy(),
            0.0,
            atol=1e-14,
        )
    )

    if not np.any(elastic):
        return np.nan

    E_eff = (
        2.0 * data["strain_energy"].to_numpy()[elastic] / (A * L * strain[elastic] ** 2)
    )

    return np.median(E_eff)


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


def analyse_case(case_path):
    param_file = case_path / "parameters_used.dat"
    results_file = case_path / "results.out"

    scatter = read_parameter(
        param_file,
        "scatter",
    )

    dx = read_parameter(param_file, "dx")
    horizon = read_parameter(param_file, "horizon")

    strain_rate = read_parameter(
        param_file,
        "strain_rate",
    )

    data = read_results(results_file)

    final_fragments = int(data["num_fragments"].iloc[-1])

    mean_size = L / final_fragments

    fragmented = data["num_fragments"] > 1

    if fragmented.any():
        index = np.flatnonzero(fragmented.to_numpy())[0]

        onset = data["time"].iloc[index]
    else:
        onset = np.nan

    max_error, relative_error = energy_balance(data)

    E_eff = effective_modulus(
        data,
        strain_rate,
    )

    return {
        "scatter": scatter,
        "dx_um": dx * 1e6,
        "horizon_um": horizon * 1e6,
        "num_fragments": final_fragments,
        "mean_fragment_size_mm": mean_size * 1e3,
        "onset_us": onset * 1e6,
        "dissipated": data["dissipated"].iloc[-1],
        "max_energy_error": max_error,
        "energy_error_percent": relative_error,
        "E_eff_GPa": E_eff / 1e9,
        "final_time_us": data["time"].iloc[-1] * 1e6,
    }


case_paths = [path for path in BASE_PATH.glob("scatter_*") if path.is_dir()]

results = []

for case_path in case_paths:
    try:
        results.append(analyse_case(case_path))
    except Exception as error:
        print(f"Skipping {case_path.name}: {error}")


results = pd.DataFrame(results)

if results.empty:
    raise RuntimeError(f"No scatter results found in {BASE_PATH}")

results = results.sort_values("scatter").reset_index(drop=True)


reference_index = (results["scatter"] - REFERENCE_SCATTER).abs().idxmin()

reference = results.loc[reference_index]


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


print()
print("PD scatter sensitivity:")
print()

print(results.to_string(index=False,float_format=lambda x: f"{x:.6g}",))


summary_file = BASE_PATH / "scatter_summary.csv"

results.to_csv(summary_file,index=False,)


x = results["scatter"]


plt.figure()

plt.plot(x,results["mean_fragment_size_mm"],marker="o")

plt.axvline(REFERENCE_SCATTER,linestyle="--",label="Reference scatter",)

plt.xlabel("Scatter")
plt.ylabel("Mean fragment size [mm]")
plt.legend()
plt.tight_layout()


plt.figure()

plt.plot(x,results["num_fragments"],marker="o",)

plt.axvline(REFERENCE_SCATTER,linestyle="--",)

plt.xlabel("Scatter")
plt.ylabel("Final number of fragments")
plt.tight_layout()


plt.figure()

plt.plot(x,results["onset_us"],marker="o",)

plt.axvline(REFERENCE_SCATTER,linestyle="--",)

plt.xlabel("Scatter")
plt.ylabel(r"Fragmentation onset [$\mu$s]")
plt.tight_layout()


plt.figure()

plt.plot(x,results["dissipated"],marker="o",)

plt.axvline(REFERENCE_SCATTER,linestyle="--")

plt.xlabel("Scatter")
plt.ylabel("Final dissipated energy [J]")
plt.tight_layout()


plt.figure()

plt.plot(x,results["energy_error_percent"],marker="o",)

plt.xlabel("Scatter")
plt.ylabel("Maximum energy error [%]")
plt.tight_layout()


plt.figure()

plt.plot(x,results["E_eff_GPa"],marker="o",)

plt.axhline(E_TARGET / 1e9,linestyle="--",label="Target E",)

plt.xlabel("Scatter")
plt.ylabel(r"$E_{\mathrm{eff}}$ [GPa]")
plt.legend()
plt.tight_layout()


plt.show()
