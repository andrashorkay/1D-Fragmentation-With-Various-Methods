from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "python"))

from helpers import get_pf_case_path, read_output_file, read_parameters_file


# =============================================================================
# Plot style
# =============================================================================

plt.rcParams.update(
    {
        "figure.figsize": (9, 6),
        "figure.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.fontsize": 13,
        "legend.frameon": True,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,
        "axes.linewidth": 1.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "mathtext.fontset": "stix",
    }
)


# =============================================================================
# Paths and parameters
# =============================================================================

DATA_PATH = (
    ROOT / "data" / "PF" /
    "strain_rate_sweep_increment_displacement_true"
)

FIGURE_PATH = (
    ROOT / "figures" / "PF" /
    "strain_rate_diagnostics_increment_displacement_true"
)

FIGURE_PATH.mkdir(parents=True, exist_ok=True)

N = 10000

LOW_RATES = ["1e3", "2e3", "5e3"]
DAMAGE_RATES = ["1e4", "2e4", "1e5", "2e5", "1e6", "1e7"]

E = 370e9
SIGMA_C = 262e6


# =============================================================================
# Read one PF diagnostics file
# =============================================================================

def read_diagnostics(rate):
    run_path = get_pf_case_path(DATA_PATH, N, "strain_rate", rate)
    params = read_parameters_file(run_path / "parameters_used.dat")

    diagnostics_name = params["user"].get(
        "diagnostics_file", "energies_pf.out"
    )

    data = read_output_file(run_path / diagnostics_name)

    return run_path, data


# =============================================================================
# Find and read the final nodal field
# =============================================================================

def step_number(path):
    match = re.search(r"nodal_(\d+)\.out$", path.name)

    if match is None:
        return -1

    return int(match.group(1))


def final_nodal_file(run_path):
    files = list((run_path / "fields").glob("nodal_*.out"))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No nodal field files found in {run_path}"
        )

    return max(files, key=step_number)


def read_nodal_file(path):
    with open(path, "r") as file:
        header = file.readline().strip().lstrip("#").split()

    data = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=header,
    )

    return data


# =============================================================================
# 1. Average strain history
# =============================================================================

fig, ax = plt.subplots()

for rate_text in LOW_RATES:
    rate = float(rate_text)
    _, data = read_diagnostics(rate_text)

    time_us = data["time"].to_numpy() * 1e6
    average_strain = data["average_strain"].to_numpy()
    expected_strain = rate * data["time"].to_numpy()

    ax.plot(
        time_us,
        average_strain,
        label=f"PF {rate_text} s$^{{-1}}$",
    )

    ax.plot(
        time_us,
        expected_strain,
        "--",
        label=f"Expected {rate_text}",
    )

ax.set_xlabel(r"Time [$\mu$s]")
ax.set_ylabel("Average axial strain")
ax.set_title("PF average strain history")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_PATH / "average_strain_history.png")


# =============================================================================
# 2. Average and maximum stress history
# =============================================================================

fig, ax = plt.subplots()

for rate_text in LOW_RATES:
    rate = float(rate_text)
    _, data = read_diagnostics(rate_text)

    time = data["time"].to_numpy()
    time_us = time * 1e6

    average_stress = data["average_stress"].to_numpy() / 1e6
    max_stress = data["max_stress"].to_numpy() / 1e6

    ax.plot(
        time_us,
        average_stress,
        label=f"Average {rate_text}",
    )

    ax.plot(
        time_us,
        max_stress,
        ":",
        label=f"Maximum {rate_text}",
    )

    critical_time = SIGMA_C / (E * rate)
    elastic_time = np.linspace(
        0.0,
        min(time.max(), critical_time),
        200,
    )

    elastic_stress = E * rate * elastic_time / 1e6

    ax.plot(
        elastic_time * 1e6,
        elastic_stress,
        "--",
        label=f"Elastic {rate_text}",
    )

ax.axhline(
    SIGMA_C / 1e6,
    color="black",
    linestyle="--",
    label=r"$\sigma_c$",
)

ax.set_xlabel(r"Time [$\mu$s]")
ax.set_ylabel("Axial stress [MPa]")
ax.set_title("PF stress history")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_PATH / "stress_history.png")


# =============================================================================
# 3. Print quantitative loading comparison
# =============================================================================

print()
print("Low-rate loading check")
print("======================")

for rate_text in LOW_RATES:
    rate = float(rate_text)
    _, data = read_diagnostics(rate_text)

    time = data["time"].to_numpy()
    strain = data["average_strain"].to_numpy()
    stress = data["average_stress"].to_numpy()

    expected_strain = rate * time
    expected_stress = E * expected_strain

    valid = expected_strain > 0.0

    strain_ratio = np.median(
        strain[valid] / expected_strain[valid]
    )

    elastic = expected_stress <= SIGMA_C
    elastic &= expected_stress > 0.0

    if elastic.any():
        stress_ratio = np.median(
            stress[elastic] / expected_stress[elastic]
        )
    else:
        stress_ratio = np.nan

    expected_failure_time = SIGMA_C / (E * rate) * 1e6

    print()
    print(f"strain rate = {rate_text} s^-1")
    print(
        f"expected failure time = "
        f"{expected_failure_time:.6f} us"
    )
    print(
        f"median strain / expected strain = "
        f"{strain_ratio:.6f}"
    )
    print(
        f"median stress / E*eps before sigma_c = "
        f"{stress_ratio:.6f}"
    )
    print(
        f"maximum average stress = "
        f"{stress.max() / 1e6:.6f} MPa"
    )


# =============================================================================
# 4. Final damage profiles
# =============================================================================

fig, ax = plt.subplots()

print()
print("Final damage profiles")
print("=====================")

for rate_text in DAMAGE_RATES:
    run_path, _ = read_diagnostics(rate_text)

    nodal_file = final_nodal_file(run_path)
    nodal = read_nodal_file(nodal_file)

    profile = nodal.groupby(
        "x",
        as_index=False,
    )["damage"].max()

    profile = profile.sort_values("x")

    x_mm = profile["x"].to_numpy() * 1e3
    damage = profile["damage"].to_numpy()

    ax.plot(
        x_mm,
        damage,
        label=f"{rate_text} s$^{{-1}}$",
    )

    fraction_050 = np.mean(damage >= 0.50)
    fraction_090 = np.mean(damage >= 0.90)
    fraction_095 = np.mean(damage >= 0.95)

    print()
    print(f"strain rate = {rate_text} s^-1")
    print(f"final nodal file = {nodal_file.name}")
    print(f"maximum damage = {damage.max():.6f}")
    print(
        f"fraction d >= 0.50 = "
        f"{100.0 * fraction_050:.3f} %"
    )
    print(
        f"fraction d >= 0.90 = "
        f"{100.0 * fraction_090:.3f} %"
    )
    print(
        f"fraction d >= 0.95 = "
        f"{100.0 * fraction_095:.3f} %"
    )

ax.axhline(
    0.95,
    color="black",
    linestyle="--",
    label="Fragment threshold",
)

ax.set_xlabel("Position [mm]")
ax.set_ylabel("Damage")
ax.set_title("Final phase-field damage profiles")
ax.set_ylim(-0.02, 1.02)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_PATH / "final_damage_profiles.png")


print()
print(f"Figures saved to {FIGURE_PATH}")

plt.show()