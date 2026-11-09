import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from python import helpers


# ============================================================================ #
# Plot parameters.

plt.rcParams.update(
    {
        # Figure
        "figure.figsize": (9, 6),
        "figure.dpi": 300,

        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
        "font.size": 14,

        # Titles and labels
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,

        # Tick labels
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,

        # Lines and markers
        "lines.linewidth": 2,
        "lines.markersize": 6,

        # Legend
        "legend.fontsize": 15,
        "legend.frameon": True,

        # Grid
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,

        # Axes
        "axes.linewidth": 1.0,

        # Saving figures
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # Mathematical text
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


METHOD_STYLE = {
    "CZM": {"color": COLORS["blue"], "marker": "o"},
    "PF": {"color": COLORS["orange"], "marker": "s"},
    "PD": {"color": COLORS["red"], "marker": "^"},
}


# ============================================================================ #
# Paths.

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"

OUTPUT_DIR = PROJECT_DIR / "figures" / "method_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================ #
# Common physical parameters.

RHO = 3900.0
E = 370e9
SIGMA_C = 262e6
GC = 50.0
L = 0.005

A_PD = 1e-6


# ============================================================================ #
# Characteristic quantities.

C = np.sqrt(E / RHO)

STRAIN_RATE_0 = SIGMA_C**3 * C / (E**2 * GC)
S_0 = E * GC / SIGMA_C**2


print()
print("Common physical parameters")
print("==========================")
print(f"E             = {E / 1e9:.6g} GPa")
print(f"rho           = {RHO:.6g} kg/m^3")
print(f"sigma_c       = {SIGMA_C / 1e6:.6g} MPa")
print(f"Gc            = {GC:.6g} N/m")
print(f"L             = {L * 1e3:.6g} mm")

print()
print("Characteristic quantities")
print("==========================")
print(f"c             = {C:.6g} m/s")
print(f"strain_rate_0 = {STRAIN_RATE_0:.6g} 1/s")
print(f"s_0           = {S_0 * 1e3:.6g} mm")


# ============================================================================ #
# Find summary files.

def find_summary(method):
    base = DATA_DIR / method

    candidates = [
        base / "strain_rate_sweep" / "strain_rate_summary.csv",
        base / "strain_sweep" / "strain_rate_summary.csv",
        base / "strain_rate_summary.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    if not base.exists():
        raise FileNotFoundError(f"Data directory does not exist: {base}")

    matches = list(base.rglob("*summary*.csv"))

    matches = [
        path
        for path in matches
        if "strain" in path.name.lower() or "rate" in path.name.lower()
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No strain-rate summary found for {method} inside {base}"
        )

    print()
    print(f"Several possible summary files found for {method}:")

    for path in matches:
        print(f"  {path}")

    raise RuntimeError(f"Could not uniquely identify the {method} summary.")


# ============================================================================ #
# Column utilities.

def normalise_column_name(name):
    name = str(name).lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)

    return name.strip("_")


def normalise_columns(data):
    return data.rename(
        columns={
            column: normalise_column_name(column)
            for column in data.columns
        }
    )


def get_column(data, aliases):
    for alias in aliases:
        alias = normalise_column_name(alias)

        if alias in data.columns:
            return pd.to_numeric(data[alias], errors="coerce")

    return None


# ============================================================================ #
# Common column aliases.

RATE_ALIASES = [
    "strain_rate",
    "rate",
    "strainrate",
]

FRAGMENT_ALIASES = [
    "num_fragments",
    "final_fragments",
    "fragment_count",
    "number_fragments",
    "n_fragments",
    "nfrag",
    "nf",
]

MEAN_SIZE_ALIASES = [
    "mean_fragment_size_mm",
    "mean_size_mm",
    "fragment_size_mm",
    "mean_fragment_size",
]

ONSET_US_ALIASES = [
    "onset_us",
    "fragmentation_onset_us",
    "first_fragmentation_us",
    "first_fragmentation_time_us",
    "tfrag_us",
]

ONSET_S_ALIASES = [
    "onset",
    "onset_time",
    "fragmentation_onset",
    "first_fragmentation_time",
    "tfrag",
]

ENERGY_ERROR_ALIASES = [
    "energy_error_percent",
    "max_energy_error_percent",
    "max_error_percent",
]


# ============================================================================ #
# Fracture-energy extraction.
#
# Output is always energy per fracture area [J/m^2].

def fracture_energy_per_area(data, method):
    direct_aliases = {
        "CZM": [
            "fracture_energy_per_area",
            "fracture_energy_per_height",
            "fracture_energy_over_h",
            "dissipated_per_area",
            "dissipated_per_height",
            "ef_per_h",
        ],
        "PF": [
            "fracture_energy_per_area",
            "phase_energy_per_area",
            "phase_energy_per_height",
            "phase_energy_over_h",
            "fracture_energy_per_height",
            "fracture_energy_over_h",
        ],
        "PD": [
            "dissipated_per_area",
            "fracture_energy_per_area",
        ],
    }

    direct = get_column(data, direct_aliases[method])

    if direct is not None:
        return direct

    if method == "PD":
        raw = get_column(data, ["dissipated", "final_dissipated"])

        if raw is not None:
            return raw / A_PD

    if method == "CZM":
        raw = get_column(
            data,
            [
                "fracture_energy",
                "final_fracture_energy",
                "dissipated",
            ],
        )

    else:
        raw = get_column(
            data,
            [
                "phase_energy",
                "final_phase_energy",
                "fracture_energy",
            ],
        )

    height = get_column(
        data,
        [
            "h",
            "height",
            "mesh_height",
            "h_m",
        ],
    )

    height_um = get_column(data, ["h_um", "height_um"])

    if height is None and height_um is not None:
        height = height_um * 1e-6

    if raw is not None and height is not None:
        return raw / height

    return None


# ============================================================================ #
# Standardise one method.

def standardise_method(method, file_path):
    raw = pd.read_csv(file_path)
    raw = normalise_columns(raw)

    strain_rate = get_column(raw, RATE_ALIASES)

    if strain_rate is None:
        raise ValueError(
            f"{method}: no strain-rate column found in {file_path}"
        )

    num_fragments = get_column(raw, FRAGMENT_ALIASES)
    mean_size = get_column(raw, MEAN_SIZE_ALIASES)

    if num_fragments is not None:
        calculated_size = L / num_fragments * 1e3

        if mean_size is None:
            mean_size = calculated_size

        else:
            valid = num_fragments.notna() & (num_fragments > 0)

            mean_size = mean_size.copy()
            mean_size.loc[valid] = calculated_size.loc[valid]

    if mean_size is None:
        mean_size = pd.Series(np.nan, index=raw.index)

    if num_fragments is None:
        num_fragments = pd.Series(np.nan, index=raw.index)

    onset_us = get_column(raw, ONSET_US_ALIASES)

    if onset_us is None:
        onset_s = get_column(raw, ONSET_S_ALIASES)

        if onset_s is not None:
            onset_us = onset_s * 1e6

    if onset_us is None:
        onset_us = pd.Series(np.nan, index=raw.index)

    fracture_energy = fracture_energy_per_area(raw, method)

    if fracture_energy is None:
        fracture_energy = pd.Series(np.nan, index=raw.index)

    energy_error = get_column(raw, ENERGY_ERROR_ALIASES)

    if energy_error is None:
        energy_error = pd.Series(np.nan, index=raw.index)

    result = pd.DataFrame(
        {
            "method": method,
            "strain_rate": strain_rate,
            "num_fragments": num_fragments,
            "mean_fragment_size_mm": mean_size,
            "onset_us": onset_us,
            "fracture_energy_per_area": fracture_energy,
            "energy_error_percent": energy_error,
        }
    )

    result["rate_hat"] = result["strain_rate"] / STRAIN_RATE_0

    result["size_hat"] = (
        result["mean_fragment_size_mm"] * 1e-3 / S_0
    )

    result = result.sort_values("strain_rate").reset_index(drop=True)

    return result


# ============================================================================ #
# Read the three methods.

summary_files = {}
method_results = []


for method in ["CZM", "PF", "PD"]:
    try:
        path = find_summary(method)

        summary_files[method] = path

        result = standardise_method(method, path)
        method_results.append(result)

        print()
        print(f"{method} summary: {path.relative_to(PROJECT_DIR)}")

    except FileNotFoundError as error:
        print()
        print(f"{method}: {error}")
        print(f"{method} will be skipped until its summary file exists.")


if len(method_results) == 0:
    raise RuntimeError("No method summaries were found.")


combined = pd.concat(method_results, ignore_index=True)

combined = combined.sort_values(
    ["method", "strain_rate"]
).reset_index(drop=True)


# ============================================================================ #
# Save combined numerical results.

combined_file = DATA_DIR / "combined_strain_rate_summary.csv"

combined.to_csv(combined_file, index=False)


print()
print("Combined numerical results")
print("==========================")
print()

print(
    combined.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print()
print(f"Combined table saved to:\n{combined_file}")


# ============================================================================ #
# Theory curves.

minimum_rate = combined["strain_rate"].min()
maximum_rate = combined["strain_rate"].max()

rate_theory = np.logspace(
    np.log10(minimum_rate),
    np.log10(maximum_rate),
    500,
)

rate_hat_theory = rate_theory / STRAIN_RATE_0


# Grady.

grady_size = (
    24.0 * GC / (RHO * rate_theory**2)
) ** (1.0 / 3.0)

grady_size_mm = grady_size * 1e3
grady_hat = grady_size / S_0


# ZMR.

zmr_hat = 4.5 / (1.0 + 4.5 * rate_hat_theory ** (2.0 / 3.0))

zmr_size_mm = zmr_hat * S_0 * 1e3


# ============================================================================ #
# Plot helper.

def plot_methods(ax, y_column):
    plotted = False

    for method in ["CZM", "PF", "PD"]:
        data = combined[combined["method"] == method].copy()
        data = data[np.isfinite(data[y_column])]

        if data.empty:
            continue

        style = METHOD_STYLE[method]

        ax.plot(
            data["strain_rate"],
            data[y_column],
            color=style["color"],
            marker=style["marker"],
            label=method,
        )

        plotted = True

    return plotted


# ============================================================================ #
# 1. Normalised fragment size versus normalised strain rate.

fig, ax = plt.subplots()


for method in ["CZM", "PF", "PD"]:
    data = combined[combined["method"] == method].copy()
    data = data[np.isfinite(data["size_hat"])]

    if data.empty:
        continue

    style = METHOD_STYLE[method]

    ax.plot(
        data["rate_hat"],
        data["size_hat"],
        color=style["color"],
        marker=style["marker"],
        label=method,
    )


ax.plot(
    rate_hat_theory,
    grady_hat,
    color=COLORS["green"],
    linestyle="--",
    label="Grady",
)

ax.plot(
    rate_hat_theory,
    zmr_hat,
    color=COLORS["purple"],
    linestyle=":",
    label="ZMR",
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(
    r"Normalised strain rate $\dot{\varepsilon}/\dot{\varepsilon}_0$"
)

ax.set_ylabel(r"Normalised mean fragment size $\bar{s}/s_0$")

ax.set_title("Dynamic fragmentation")
ax.legend()

fig.tight_layout()

fig.savefig(OUTPUT_DIR / "fragment_size_normalised.png")


# ============================================================================ #
# 2. Dimensional mean fragment size.

fig, ax = plt.subplots()


for method in ["CZM", "PF", "PD"]:
    data = combined[combined["method"] == method].copy()
    data = data[np.isfinite(data["mean_fragment_size_mm"])]

    if data.empty:
        continue

    style = METHOD_STYLE[method]

    ax.plot(
        data["strain_rate"],
        data["mean_fragment_size_mm"],
        color=style["color"],
        marker=style["marker"],
        label=method,
    )


ax.plot(
    rate_theory,
    grady_size_mm,
    color=COLORS["green"],
    linestyle="--",
    label="Grady",
)

ax.plot(
    rate_theory,
    zmr_size_mm,
    color=COLORS["purple"],
    linestyle=":",
    label="ZMR",
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"Strain rate $\dot{\varepsilon}$ [s$^{-1}$]")
ax.set_ylabel("Mean fragment size [mm]")

ax.set_title("Mean fragment size")
ax.legend()

fig.tight_layout()

fig.savefig(OUTPUT_DIR / "fragment_size_dimensional.png")


# ============================================================================ #
# 3. Fragment count.

fig, ax = plt.subplots()

plotted = plot_methods(ax, "num_fragments")


if plotted:
    ax.set_xscale("log")

    ax.set_xlabel(r"Strain rate $\dot{\varepsilon}$ [s$^{-1}$]")
    ax.set_ylabel("Final number of fragments")

    ax.set_title("Fragment count")
    ax.legend()

    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / "fragment_count.png")

else:
    plt.close(fig)


# ============================================================================ #
# 4. Fragmentation onset.

fig, ax = plt.subplots()

plotted = plot_methods(ax, "onset_us")


if plotted:
    ax.set_xscale("log")

    ax.set_xlabel(r"Strain rate $\dot{\varepsilon}$ [s$^{-1}$]")
    ax.set_ylabel(r"Fragmentation onset [$\mu$s]")

    ax.set_title("Fragmentation onset")
    ax.legend()

    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / "fragmentation_onset.png")

else:
    plt.close(fig)


# ============================================================================ #
# 5. Fracture / dissipated energy per unit fracture area.

fig, ax = plt.subplots()

plotted = plot_methods(ax, "fracture_energy_per_area")


if plotted:
    ax.set_xscale("log")

    ax.set_xlabel(r"Strain rate $\dot{\varepsilon}$ [s$^{-1}$]")
    ax.set_ylabel(r"Fracture energy [J/m$^2$]")

    ax.set_title("Fracture energy")
    ax.legend()

    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / "fracture_energy.png")

else:
    plt.close(fig)


# ============================================================================ #
# 6. Energy-conservation error.

fig, ax = plt.subplots()

plotted = plot_methods(ax, "energy_error_percent")


if plotted:
    ax.set_xscale("log")

    ax.set_xlabel(r"Strain rate $\dot{\varepsilon}$ [s$^{-1}$]")
    ax.set_ylabel("Maximum energy error [%]")

    ax.set_title("Energy conservation")
    ax.legend()

    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / "energy_error.png")

else:
    plt.close(fig)


# ============================================================================ #
# Values at the reference strain rate 1e4.

REFERENCE_RATE = 1e4


print()
print(f"Results at strain rate = {REFERENCE_RATE:.6g} 1/s")
print("==============================")
print()


for method in ["CZM", "PF", "PD"]:
    data = combined[combined["method"] == method]

    if data.empty:
        continue

    index = (data["strain_rate"] - REFERENCE_RATE).abs().idxmin()
    row = data.loc[index]

    print(method)
    print(f"  strain rate         = {row['strain_rate']:.6g} 1/s")
    print(f"  rate/rate_0         = {row['rate_hat']:.6g}")
    print(f"  fragments           = {row['num_fragments']:.6g}")
    print(f"  mean fragment size  = {row['mean_fragment_size_mm']:.6g} mm")
    print(f"  s/s_0               = {row['size_hat']:.6g}")
    print()


# ============================================================================ #
# Theoretical values at 1e4 /s.

reference_rate_hat = REFERENCE_RATE / STRAIN_RATE_0

reference_grady = (
    24.0 * GC / (RHO * REFERENCE_RATE**2)
) ** (1.0 / 3.0)

reference_grady_hat = reference_grady / S_0

reference_zmr_hat = (
    4.5 / (1.0 + 4.5 * reference_rate_hat ** (2.0 / 3.0))
)


print("Theory at 1e4 1/s")
print("=================")
print(f"Grady s/s_0 = {reference_grady_hat:.6g}")
print(f"ZMR s/s_0   = {reference_zmr_hat:.6g}")


# ============================================================================ #

print()
print(f"Figures saved to:\n{OUTPUT_DIR}")

plt.show()