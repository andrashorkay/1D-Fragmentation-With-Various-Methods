from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "python"))

from helpers import (
    get_pf_case_path,
    pf_run_values,
    read_output_file,
    read_parameters_file,
)


# ============================================================================
# Plot style
# ============================================================================

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
        "legend.fontsize": 15,
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

COLORS = {
    "blue": "#5F8FAF",
    "orange": "#E6A15C",
    "red": "#D97A7A",
    "green": "#7FB685",
    "purple": "#A58CC2",
}


# ============================================================================
# Paths and simulation parameters
# ============================================================================

DATA_PATH = (
    ROOT
    / "data"
    / "PF"
    / "strain_rate_sweep_increment_displacement_true"
)

FIGURE_PATH = (
    ROOT
    / "figures"
    / "PF"
    / "strain_rate_sweep_increment_displacement_true"
)

SUMMARY_FILE = (
    ROOT
    / "data"
    / "PF"
    / "strain_rate_summary_increment_displacement_true.csv"
)

FIGURE_PATH.mkdir(parents=True, exist_ok=True)

N = 10000

STRAIN_RATES = [
    "1e3",
    "2e3",
    "5e3",
    "1e4",
    "2e4",
    "5e4",
    "1e5",
    "2e5",
    "5e5",
    "1e6",
    "2e6",
    "5e6",
    "1e7",
]


# ============================================================================
# Material parameters
# ============================================================================

RHO = 3900.0
E = 370e9
SIGMA_C = 262e6
GC = 50.0

C = np.sqrt(E / RHO)
RATE_0 = SIGMA_C**3 * C / (E**2 * GC)
SIZE_0 = E * GC / SIGMA_C**2

print(f"Wave speed c          = {C:.6g} m/s")
print(f"Characteristic rate   = {RATE_0:.6g} 1/s")
print(f"Characteristic size   = {SIZE_0 * 1e3:.6g} mm")


# ============================================================================
# Read summary.out
# ============================================================================

def read_run_summary(file_path):
    values = {}

    with open(file_path, "r") as file:
        for line in file:
            parts = line.split(maxsplit=1)

            if len(parts) == 2:
                values[parts[0]] = parts[1].strip()

    return values


# ============================================================================
# Read all strain-rate cases
# ============================================================================

rows = []

for rate in STRAIN_RATES:
    run_path = get_pf_case_path(DATA_PATH, N, "strain_rate", rate)

    if not run_path.exists():
        print(f"Skipping missing case: {run_path}")
        continue

    values = pf_run_values(run_path)

    params_file = run_path / "parameters_used.dat"
    params = read_parameters_file(params_file)

    diagnostics_name = params["user"].get(
        "diagnostics_file",
        "energies_pf.out",
    )

    diagnostics = read_output_file(run_path / diagnostics_name)

    summary = read_run_summary(run_path / "summary.out")

    strain_rate = float(rate)
    bar_length = params["user"]["bar_length"]

    num_fragments = int(summary.get(
        "num_fragments",
        values["num_fragments"],
    ))

    if num_fragments > 0:
        mean_fragment_size = bar_length / num_fragments
    else:
        mean_fragment_size = np.nan

    total_energy_0 = abs(diagnostics["total_energy"].iloc[0])
    max_energy_error = np.abs(diagnostics["energy_error"]).max()

    if total_energy_0 > 0.0:
        energy_error_percent = 100.0 * max_energy_error / total_energy_0
    else:
        energy_error_percent = np.nan

    fragmentation_time = values["fragmentation_time"]

    row = {
        "strain_rate": strain_rate,
        "rate_hat": strain_rate / RATE_0,
        "num_fragments": num_fragments,
        "mean_fragment_size_mm": mean_fragment_size * 1e3,
        "size_hat": mean_fragment_size / SIZE_0,
        "onset_us": fragmentation_time * 1e6,
        "fracture_energy_per_area": values["phase_energy_per_height"],
        "energy_error_percent": energy_error_percent,
        "peak_average_stress_MPa": values["peak_average_stress"] / 1e6,
        "peak_max_stress_MPa": values["peak_max_stress"] / 1e6,
        "final_time_us": values["final_time"] * 1e6,
    }

    rows.append(row)


df = pd.DataFrame(rows)
df = df.sort_values("strain_rate").reset_index(drop=True)


# ============================================================================
# Theory
# ============================================================================

df["grady_hat"] = (24.0 / df["rate_hat"]**2) ** (1.0 / 3.0)

df["zmr_hat"] = 4.5 / (
    1.0 + 4.5 * df["rate_hat"] ** (2.0 / 3.0)
)


# ============================================================================
# Save summary
# ============================================================================

SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(SUMMARY_FILE, index=False)

print("\nPF strain-rate sweep:\n")
print(df.to_string(index=False))
print(f"\nSaved summary to {SUMMARY_FILE}")


# ============================================================================
# Normalised fragment size
# ============================================================================

rate_hat_theory = np.logspace(
    np.log10(df["rate_hat"].min()),
    np.log10(df["rate_hat"].max()),
    400,
)

grady_hat = (24.0 / rate_hat_theory**2) ** (1.0 / 3.0)

zmr_hat = 4.5 / (
    1.0 + 4.5 * rate_hat_theory ** (2.0 / 3.0)
)

plt.figure()

plt.loglog(
    df["rate_hat"],
    df["size_hat"],
    "o-",
    color=COLORS["orange"],
    label="Phase field",
)

plt.loglog(
    rate_hat_theory,
    grady_hat,
    "--",
    color=COLORS["green"],
    label="Grady",
)

plt.loglog(
    rate_hat_theory,
    zmr_hat,
    "--",
    color=COLORS["purple"],
    label="ZMR",
)

plt.xlabel(r"Normalised strain rate $\dot{\varepsilon}/\dot{\varepsilon}_0$")
plt.ylabel(r"Normalised fragment size $\bar{s}/s_0$")
plt.title("Phase-field fragment size")
plt.legend()
plt.tight_layout()

plt.savefig(FIGURE_PATH / "fragment_size_normalised.png")


# ============================================================================
# Number of fragments
# ============================================================================

plt.figure()

plt.semilogx(
    df["strain_rate"],
    df["num_fragments"],
    "o-",
    color=COLORS["orange"],
)

plt.xlabel(r"Strain rate $\dot{\varepsilon}$ [s$^{-1}$]")
plt.ylabel("Number of fragments")
plt.title("Phase-field fragment count")
plt.tight_layout()

plt.savefig(FIGURE_PATH / "fragment_count.png")


# ============================================================================
# Fragmentation onset
# ============================================================================

plt.figure()

plt.semilogx(
    df["strain_rate"],
    df["onset_us"],
    "o-",
    color=COLORS["orange"],
)

plt.xlabel(r"Strain rate $\dot{\varepsilon}$ [s$^{-1}$]")
plt.ylabel(r"Fragmentation onset [$\mu$s]")
plt.title("Phase-field fragmentation onset")
plt.tight_layout()

plt.savefig(FIGURE_PATH / "fragmentation_onset.png")


# ============================================================================
# Phase-field energy
# ============================================================================

plt.figure()

plt.loglog(
    df["strain_rate"],
    df["fracture_energy_per_area"],
    "o-",
    color=COLORS["orange"],
)

plt.xlabel(r"Strain rate $\dot{\varepsilon}$ [s$^{-1}$]")
plt.ylabel(r"Final phase-field energy [J/m$^2$]")
plt.title("Phase-field fracture energy")
plt.tight_layout()

plt.savefig(FIGURE_PATH / "phase_energy.png")


# ============================================================================
# Peak average stress
# ============================================================================

plt.figure()

plt.semilogx(
    df["strain_rate"],
    df["peak_average_stress_MPa"],
    "o-",
    color=COLORS["orange"],
)

plt.xlabel(r"Strain rate $\dot{\varepsilon}$ [s$^{-1}$]")
plt.ylabel("Peak average stress [MPa]")
plt.title("Phase-field peak stress")
plt.tight_layout()

plt.savefig(FIGURE_PATH / "peak_stress.png")


# ============================================================================
# Energy balance error
# ============================================================================

plt.figure()

plt.loglog(
    df["strain_rate"],
    df["energy_error_percent"],
    "o-",
    color=COLORS["orange"],
)

plt.xlabel(r"Strain rate $\dot{\varepsilon}$ [s$^{-1}$]")
plt.ylabel("Maximum energy error [%]")
plt.title("Phase-field energy conservation")
plt.tight_layout()

plt.savefig(FIGURE_PATH / "energy_error.png")


plt.show()