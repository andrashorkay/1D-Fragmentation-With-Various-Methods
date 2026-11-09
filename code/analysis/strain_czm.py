import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from python import helpers


# =============================================================================
# Plot parameters
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


# =============================================================================
# Paths
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR_10000 = (
    PROJECT_DIR / "data" / "CZM" / "strain_rate_sweep" / "N_10000"
)

DATA_DIR_50000 = (
    PROJECT_DIR / "data" / "CZM" / "strain_rate_sweep" / "N_50000"
)

SUMMARY_FILE = (
    PROJECT_DIR / "data" / "CZM" / "strain_rate_sweep"
    / "strain_rate_summary.csv"
)

FIGURE_DIR = PROJECT_DIR / "figures" / "CZM" / "strain_rate_sweep"

SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Material parameters
# =============================================================================

first_run = next(DATA_DIR_10000.glob("rate_*"))
params = helpers.read_parameters_file(first_run / "parameters_used.dat")

E = params["elastic"]["E"]
rho = params["elastic"]["rho"]
sigma_c = params["cohesive"]["sigma_c"]
G_c = params["cohesive"]["G_c"]

eps_dot_0 = helpers.characteristic_strain_rate(E, rho, sigma_c, G_c)
s_0 = helpers.characteristic_fragment_size(E, sigma_c, G_c)


# =============================================================================
# Read one strain-rate sweep
# =============================================================================

def read_strain_rate_sweep(data_dir, number_elements):

    rows = []

    for run_path in data_dir.glob("rate_*"):

        parameter_file = run_path / "parameters_used.dat"
        output_file = run_path / "energies.out"

        if not parameter_file.exists() or not output_file.exists():
            continue

        params = helpers.read_parameters_file(parameter_file)
        data = helpers.read_output_file(output_file)

        strain_rate = params["user"]["strain_rate"]
        mean_fragment_size = data["mean_fragment_size"].iloc[-1]

        if "num_fragments" in data.columns:
            num_fragments = data["num_fragments"].iloc[-1]
        else:
            num_fragments = np.nan

        rows.append(
            {
                "N": number_elements,
                "strain_rate": strain_rate,
                "num_fragments": num_fragments,
                "mean_fragment_size": mean_fragment_size,
                "mean_fragment_size_mm": mean_fragment_size * 1e3,
                "rate_hat": strain_rate / eps_dot_0,
                "size_hat": mean_fragment_size / s_0,
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values("strain_rate").reset_index(drop=True)

    return result


# =============================================================================
# Read N = 10,000 and N = 50,000
# =============================================================================

results_10000 = read_strain_rate_sweep(DATA_DIR_10000, 10000)
results_50000 = read_strain_rate_sweep(DATA_DIR_50000, 50000)


# =============================================================================
# Save summary
# =============================================================================

summary = pd.concat(
    [results_10000, results_50000],
    ignore_index=True,
)

summary = summary.sort_values(["N", "strain_rate"]).reset_index(drop=True)
summary.to_csv(SUMMARY_FILE, index=False)

print("\nCZM strain-rate sweep:\n")
print(summary.to_string(index=False))
print(f"\nSaved summary to {SUMMARY_FILE}")


# =============================================================================
# Extract arrays for plotting
# =============================================================================

strain_rates_10000 = results_10000["strain_rate"].to_numpy()
fragment_sizes_10000 = results_10000["mean_fragment_size"].to_numpy()

strain_rates_50000 = results_50000["strain_rate"].to_numpy()
fragment_sizes_50000 = results_50000["mean_fragment_size"].to_numpy()

eps_norm_10000 = results_10000["rate_hat"].to_numpy()
s_norm_10000 = results_10000["size_hat"].to_numpy()

eps_norm_50000 = results_50000["rate_hat"].to_numpy()
s_norm_50000 = results_50000["size_hat"].to_numpy()


# =============================================================================
# Grady prediction
# =============================================================================

minimum_rate = min(strain_rates_10000.min(), strain_rates_50000.min())
maximum_rate = max(strain_rates_10000.max(), strain_rates_50000.max())

eps_dot_grady = np.logspace(
    np.log10(minimum_rate),
    np.log10(maximum_rate),
    300,
)

s_grady = np.array(
    [
        helpers.grady_fragment_size(rho, G_c, eps_dot)
        for eps_dot in eps_dot_grady
    ]
)

eps_norm_grady = eps_dot_grady / eps_dot_0
s_norm_grady = s_grady / s_0


# =============================================================================
# Plot
# =============================================================================

plt.figure()

plt.loglog(
    eps_norm_grady,
    s_norm_grady,
    label="Grady",
    color=COLORS["green"],
    zorder=2,
    linewidth=2.5,
)

plt.scatter(
    eps_norm_10000,
    s_norm_10000,
    label="CZM, N = 10,000",
    color=COLORS["blue"],
    zorder=3,
)

plt.scatter(
    eps_norm_50000,
    s_norm_50000,
    label="CZM, N = 50,000",
    color=COLORS["orange"],
    zorder=3,
)

plt.xlabel(r"Normalised strain rate $\dot{\varepsilon}/\dot{\varepsilon}_0$")
plt.ylabel(r"Normalised mean fragment size $s/s_0$")
plt.legend()

plt.tight_layout()

figure_file = FIGURE_DIR / "fragment_size_strain_rate.png"
plt.savefig(figure_file)

print(f"Saved figure to {figure_file}")

plt.show()