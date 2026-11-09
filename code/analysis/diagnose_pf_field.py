from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "python"))

from helpers import get_pf_case_path, read_parameters_file


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
        "legend.fontsize": 12,
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
    "strain_rate_damage_evolution_increment_displacement_true"
)

FIGURE_PATH.mkdir(parents=True, exist_ok=True)

N = 10000

STRAIN_RATES = [
    "1e3",
    "2e3",
    "5e3",
    "1e4",
    "1e5",
    "1e6",
    "1e7",
]

DAMAGE_ONSET = 0.01
FRAGMENT_THRESHOLD = 0.95


# =============================================================================
# Nodal-field utilities
# =============================================================================

def step_number(path):
    match = re.search(r"nodal_(\d+)\.out$", path.name)

    if match is None:
        return -1

    return int(match.group(1))


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


def damage_profile(path):
    nodal = read_nodal_file(path)

    profile = nodal.groupby(
        "x",
        as_index=False,
    )["damage"].max()

    return profile.sort_values("x")


def get_nodal_files(run_path):
    files = list((run_path / "fields").glob("nodal_*.out"))
    files = sorted(files, key=step_number)

    if len(files) == 0:
        raise FileNotFoundError(
            f"No nodal files found in {run_path / 'fields'}"
        )

    return files


# =============================================================================
# Find first saved field containing damage
# =============================================================================

def find_damage_onset(files):
    for index, path in enumerate(files):
        profile = damage_profile(path)

        if profile["damage"].max() >= DAMAGE_ONSET:
            return index

    return None


# =============================================================================
# Select five states from onset to final state
# =============================================================================

def select_damage_states(files, onset_index):
    final_index = len(files) - 1

    indices = np.linspace(
        onset_index,
        final_index,
        5,
    )

    indices = np.rint(indices).astype(int)

    indices = np.unique(indices)

    return [(index, files[index]) for index in indices]


# =============================================================================
# Damage evolution for each strain rate
# =============================================================================

print()
print("Phase-field damage evolution")
print("============================")

for rate_text in STRAIN_RATES:
    run_path = get_pf_case_path(
        DATA_PATH,
        N,
        "strain_rate",
        rate_text,
    )

    params = read_parameters_file(
        run_path / "parameters_used.dat"
    )

    files = get_nodal_files(run_path)

    onset_index = find_damage_onset(files)

    print()
    print(f"strain rate = {rate_text} s^-1")
    print(f"number of saved fields = {len(files)}")

    if onset_index is None:
        print("No damage detected.")
        continue

    selected = select_damage_states(
        files,
        onset_index,
    )

    fig, ax = plt.subplots()

    labels = [
        "Damage onset",
        "25% after onset",
        "50% after onset",
        "75% after onset",
        "Final",
    ]

    for label, (_, path) in zip(labels, selected):
        profile = damage_profile(path)

        x_mm = profile["x"].to_numpy() * 1e3
        damage = profile["damage"].to_numpy()

        max_damage = damage.max()
        fraction_050 = np.mean(damage >= 0.50)
        fraction_095 = np.mean(damage >= 0.95)

        ax.plot(
            x_mm,
            damage,
            label=label,
        )

        print()
        print(f"{label}:")
        print(f"  file = {path.name}")
        print(f"  maximum damage = {max_damage:.6f}")
        print(
            f"  fraction d >= 0.50 = "
            f"{100.0 * fraction_050:.3f} %"
        )
        print(
            f"  fraction d >= 0.95 = "
            f"{100.0 * fraction_095:.3f} %"
        )

    ax.axhline(
        FRAGMENT_THRESHOLD,
        color="black",
        linestyle="--",
        label="Fragment threshold",
    )

    ax.set_xlabel("Position [mm]")
    ax.set_ylabel("Damage")
    ax.set_title(
        rf"PF damage evolution, "
        rf"$\dot{{\varepsilon}}={rate_text}$ s$^{{-1}}$"
    )

    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()

    figure_name = f"damage_evolution_rate_{rate_text}.png"

    fig.savefig(
        FIGURE_PATH / figure_name
    )

    plt.close(fig)


# =============================================================================
# Compare damaged fraction versus simulation stage
# =============================================================================

fig, ax = plt.subplots()

for rate_text in STRAIN_RATES:
    run_path = get_pf_case_path(
        DATA_PATH,
        N,
        "strain_rate",
        rate_text,
    )

    files = get_nodal_files(run_path)
    onset_index = find_damage_onset(files)

    if onset_index is None:
        continue

    selected = select_damage_states(
        files,
        onset_index,
    )

    stages = []
    fractions = []

    for stage, (_, path) in enumerate(selected):
        profile = damage_profile(path)
        damage = profile["damage"].to_numpy()

        stages.append(stage)
        fractions.append(
            100.0 * np.mean(damage >= FRAGMENT_THRESHOLD)
        )

    ax.plot(
        stages,
        fractions,
        "o-",
        label=rf"$10^{{{int(np.log10(float(rate_text)))}}}$ s$^{{-1}}$",
    )

ax.set_xticks(
    [0, 1, 2, 3, 4],
    [
        "Onset",
        "25%",
        "50%",
        "75%",
        "Final",
    ],
)

ax.set_ylabel(r"Bar with $d\geq0.95$ [\%]")
ax.set_xlabel("Post-damage evolution")
ax.set_title("Growth of highly damaged region")
ax.legend()
fig.tight_layout()

fig.savefig(
    FIGURE_PATH / "damaged_fraction_evolution.png"
)


print()
print(f"Figures saved to {FIGURE_PATH}")

plt.show()