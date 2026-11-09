import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from python import helpers

# ============================================================================ #
# Plot parameters

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


# ============================================================================ #
# Paths

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "PF" / "mesh_convergence"

N_requested = [1000, 2000, 3000, 5000, 7500, 10000, 20000, 30000]


# ============================================================================ #
# Keep only simulations that have actually finished

Ns = []

for N in N_requested:
    run_path = DATA_DIR / f"N_{N}"
    energy_file_exists = (run_path / "energies_pf.out").exists() or (
        run_path / "energies.out"
    ).exists()

    if (
        (run_path / "parameters_used.dat").exists()
        and (run_path / "summary.out").exists()
        and energy_file_exists
    ):
        Ns.append(N)

print("Completed meshes:", Ns)

if len(Ns) == 0:
    raise RuntimeError("No completed PF mesh-convergence simulations found.")


# ============================================================================ #
# Read convergence values

results = helpers.pf_mesh_convergence_values(DATA_DIR, Ns)
results = results.sort_values("N").reset_index(drop=True)


# ============================================================================ #
# Read common parameters

first_run = DATA_DIR / f"N_{Ns[0]}"
params = helpers.read_parameters_file(first_run / "parameters_used.dat")

L = params["user"]["bar_length"]
l0 = helpers.read_scalar_parameter(first_run / "parameters_used.dat", "l0")


# ============================================================================ #
# Add useful quantities

results["h_um"] = results["mesh_size"] * 1e6
results["h_over_l0"] = results["mesh_size"] / l0
results["fragmentation_time_us"] = results["fragmentation_time"] * 1e6
results["peak_average_stress_MPa"] = results["peak_average_stress"] / 1e6

# Mean fragment size used for the 1D fragmentation comparison.
results["mean_fragment_size_mm"] = L / results["num_fragments"] * 1e3

# Original fragment span obtained from the PhaseFieldFragmentManager.
results["detected_mean_span_mm"] = results["mean_fragment_size"] * 1e3

# Total length represented by the detected connected components.
results["identified_length_mm"] = (
    results["mean_fragment_size"] * results["num_fragments"] * 1e3
)


# ============================================================================ #
# Relative difference from finest completed mesh

reference = results.iloc[-1]

results["fragment_size_difference_percent"] = (
    np.abs(results["mean_fragment_size_mm"] - reference["mean_fragment_size_mm"])
    / reference["mean_fragment_size_mm"]
    * 100
)

results["peak_stress_difference_percent"] = (
    np.abs(results["peak_average_stress"] - reference["peak_average_stress"])
    / np.abs(reference["peak_average_stress"])
    * 100
)

results["phase_energy_difference_percent"] = (
    np.abs(results["phase_energy_per_height"] - reference["phase_energy_per_height"])
    / np.abs(reference["phase_energy_per_height"])
    * 100
)


# ============================================================================ #
# Print main convergence table

columns = [
    "N",
    "h_um",
    "h_over_l0",
    "num_fragments",
    "mean_fragment_size_mm",
    "fragmentation_time_us",
    "peak_average_stress_MPa",
    "phase_energy_per_height",
    "max_energy_error_per_height",
]

print()
print("Main convergence quantities:")
print(results[columns].to_string(index=False))


# ============================================================================ #
# Print fragment-detection diagnostic table

diagnostic_columns = [
    "N",
    "num_fragments",
    "mean_fragment_size_mm",
    "detected_mean_span_mm",
    "identified_length_mm",
]

print()
print("Fragment detection diagnostics:")
print(results[diagnostic_columns].to_string(index=False))


# ============================================================================ #
# Print differences relative to finest completed mesh

difference_columns = [
    "N",
    "fragment_size_difference_percent",
    "peak_stress_difference_percent",
    "phase_energy_difference_percent",
]

print()
print("Difference relative to finest completed mesh [%]:")
print(results[difference_columns].to_string(index=False))


# ============================================================================ #
# Mean fragment size

plt.figure()
plt.plot(results["N"], results["mean_fragment_size_mm"], "o-", color=COLORS["blue"])
plt.xlabel("Number of elements N")
plt.ylabel("Mean fragment size [mm]")
plt.tight_layout()


# ============================================================================ #
# Number of fragments

plt.figure()
plt.plot(results["N"], results["num_fragments"], "o-", color=COLORS["orange"])
plt.xlabel("Number of elements N")
plt.ylabel("Number of fragments")
plt.yticks(
    np.arange(results["num_fragments"].min(), results["num_fragments"].max() + 1)
)
plt.tight_layout()


# ============================================================================ #
# Peak average stress

plt.figure()
plt.plot(results["N"], results["peak_average_stress_MPa"], "o-", color=COLORS["red"])
plt.xlabel("Number of elements N")
plt.ylabel("Peak average stress [MPa]")
plt.tight_layout()


# ============================================================================ #
# Phase-field fracture energy

plt.figure()
plt.plot(results["N"], results["phase_energy_per_height"], "o-", color=COLORS["green"])
plt.xlabel("Number of elements N")
plt.ylabel("Phase-field energy / bar height")
plt.tight_layout()


# ============================================================================ #
# Energy error

plt.figure()
plt.semilogy(
    results["N"], results["max_energy_error_per_height"], "o-", color=COLORS["purple"]
)
plt.xlabel("Number of elements N")
plt.ylabel("Maximum |energy error| / bar height")
plt.tight_layout()


# ============================================================================ #
# Mesh resolution relative to l0

plt.figure()
plt.plot(results["N"], results["h_over_l0"], "o-", color=COLORS["blue"])
plt.xlabel("Number of elements N")
plt.ylabel(r"$h/l_0$")
plt.tight_layout()


# ============================================================================ #
# Identified fragment length

plt.figure()
plt.plot(results["N"], results["identified_length_mm"], "o-", color=COLORS["orange"])
plt.axhline(L * 1e3, linestyle="--", color=COLORS["red"], label="Bar length")
plt.xlabel("Number of elements N")
plt.ylabel("Total detected fragment span [mm]")
plt.legend()
plt.tight_layout()


# ============================================================================ #
# Phase-field energy evolution

plt.figure()

energy_colors = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["red"],
    COLORS["green"],
    COLORS["purple"],
]

for i, N in enumerate(Ns):
    run_path = helpers.get_pf_case_path(DATA_DIR, N)
    time, energies = helpers.pf_energy_evolution(run_path)
    color = energy_colors[i % len(energy_colors)]
    plt.plot(time * 1e6, energies["phase"], label=f"N = {N}", color=color)

plt.xlabel(r"Time [$\mu$s]")
plt.ylabel("Phase-field energy / bar height")
plt.legend()
plt.tight_layout()


# ============================================================================ #
# Final damage profiles for selected meshes

# ============================================================================ #
# Final damage profiles for selected meshes


def final_nodal_file(run_path):
    files = list((run_path / "fields").glob("nodal_*.out"))

    if len(files) == 0:
        raise FileNotFoundError(f"No nodal fields found in {run_path / 'fields'}")

    def step_number(path):
        match = re.search(r"nodal_(\d+)\.out", path.name)
        return int(match.group(1)) if match else -1

    return max(files, key=step_number)


selected_meshes = [10000, 20000, 30000]

plt.figure()

for N in selected_meshes:
    run_path = DATA_DIR / f"N_{N}"
    data = helpers.read_output_file(final_nodal_file(run_path))

    profile = data.groupby("x")["damage"].agg(["min", "max"]).reset_index()

    plt.plot(profile["x"] * 1e3, profile["min"], label=f"N = {N}")

plt.xlabel("x [mm]")
plt.ylabel("Minimum damage through bar height")
plt.legend()
plt.tight_layout()

plt.figure()

for N in selected_meshes:
    run_path = DATA_DIR / f"N_{N}"
    data = helpers.read_output_file(final_nodal_file(run_path))

    profile = data.groupby("x")["damage"].agg(["min", "max"]).reset_index()
    profile["damage_difference"] = profile["max"] - profile["min"]

    plt.plot(profile["x"] * 1e3, profile["damage_difference"], label=f"N = {N}")

plt.xlabel("x [mm]")
plt.ylabel("Max damage - min damage")
plt.legend()
plt.tight_layout()


# ============================================================================ #
# Stopping-time diagnostics

results["final_time_us"] = results["final_time"] * 1e6

print()
print("Final simulation times:")
print(
    results[
        [
            "N",
            "num_fragments",
            "final_time_us",
            "phase_energy_per_height",
        ]
    ].to_string(index=False)
)

# ============================================================================ #
# Independent 1D fragment count

print()
print("Independent 1D fragment count:")

for N in Ns:
    run_path = DATA_DIR / f"N_{N}"
    params_file = run_path / "parameters_used.dat"

    params = helpers.read_parameters_file(params_file)
    damage_limit = params["user"]["damage_limit"]

    data = helpers.read_output_file(final_nodal_file(run_path))
    profile = data.groupby("x", as_index=False)["damage"].mean()
    profile = profile.sort_values("x").reset_index(drop=True)

    broken = profile["damage"].to_numpy() >= damage_limit

    crack_starts = broken & np.concatenate(([True], ~broken[:-1]))
    num_cracks = np.sum(crack_starts)

    num_fragments_1d = num_cracks + 1

    print(
        f"N = {N:5d} | manager = "
        f"{results.loc[results['N'] == N, 'num_fragments'].iloc[0]} | "
        f"1D count = {num_fragments_1d} | cracks = {num_cracks}"
    )


# ============================================================================ #
# Show all figures

plt.show()
