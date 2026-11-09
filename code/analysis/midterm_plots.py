from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path

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


METHOD_STYLE = {
    "CZM": {"color": COLORS["blue"], "marker": "o"},
    "PF": {"color": COLORS["orange"], "marker": "s"},
    "PD": {"color": COLORS["red"], "marker": "^"},
}


# =============================================================================
# Paths
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"

OUTPUT_DIR = PROJECT_DIR / "figures" / "method_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Physical parameters
# =============================================================================

L = 0.005
RHO = 3900.0
E = 370e9
SIGMA_C = 262e6
GC = 50.0

C = np.sqrt(E / RHO)

STRAIN_RATE_0 = SIGMA_C**3 * C / (E**2 * GC)
S_0 = E * GC / SIGMA_C**2


# =============================================================================
# Find strain-rate summary files
# =============================================================================

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
        path for path in matches
        if "strain" in path.name.lower() or "rate" in path.name.lower()
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No strain-rate summary found for {method} inside {base}"
        )

    print(f"\nSeveral possible summary files found for {method}:")

    for path in matches:
        print(f"  {path}")

    raise RuntimeError(f"Could not uniquely identify the {method} summary.")


# =============================================================================
# Column utilities
# =============================================================================

def normalise_column_name(name):

    name = str(name).lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)

    return name.strip("_")


def normalise_columns(data):

    return data.rename(
        columns={column: normalise_column_name(column) for column in data.columns}
    )


def get_column(data, aliases):

    for alias in aliases:

        alias = normalise_column_name(alias)

        if alias in data.columns:
            return pd.to_numeric(data[alias], errors="coerce")

    return None


# =============================================================================
# Possible column names
# =============================================================================

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


N_ALIASES = [
    "n",
    "number_elements",
    "num_elements",
]


# =============================================================================
# Read one method
# =============================================================================

def read_method(method):

    file_path = find_summary(method)

    print(f"{method} summary: {file_path.relative_to(PROJECT_DIR)}")

    data = pd.read_csv(file_path)
    data = normalise_columns(data)

    strain_rate = get_column(data, RATE_ALIASES)
    num_fragments = get_column(data, FRAGMENT_ALIASES)
    mean_size = get_column(data, MEAN_SIZE_ALIASES)
    number_elements = get_column(data, N_ALIASES)

    if strain_rate is None:
        raise ValueError(f"{method}: strain-rate column could not be found.")

    if num_fragments is not None:
        mean_size_mm = L / num_fragments * 1e3

    elif mean_size is not None:

        mean_size_mm = mean_size.copy()

        if mean_size_mm.max() < 0.1:
            mean_size_mm *= 1e3

    else:
        raise ValueError(
            f"{method}: neither fragment count nor mean fragment size was found."
        )

    result = pd.DataFrame(
        {
            "strain_rate": strain_rate,
            "mean_fragment_size_mm": mean_size_mm,
        }
    )

    if number_elements is not None:
        result["N"] = number_elements

    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=["strain_rate", "mean_fragment_size_mm"])
    result = result[result["mean_fragment_size_mm"] > 0]

    if method == "CZM" and "N" in result.columns:
        result = result.sort_values(["strain_rate", "N"])

        result = result.sort_values("strain_rate").reset_index(drop=True)

    return result


# =============================================================================
# Read CZM, phase-field and peridynamics results
# =============================================================================

czm = read_method("CZM")
czm_10000 = czm[czm["N"] == 10000].copy()
czm_50000 = czm[czm["N"] == 50000].copy()
pf_results = read_method("PF")
pd_results = read_method("PD")

print("\nCZM")
print(czm.to_string(index=False))

print("\nPhase field")
print(pf_results.to_string(index=False))

print("\nPeridynamics")
print(pd_results.to_string(index=False))

# =============================================================================
# Normalise numerical results
# =============================================================================

czm["rate_hat"] = czm["strain_rate"] / STRAIN_RATE_0
czm["size_hat"] = czm["mean_fragment_size_mm"] * 1e-3 / S_0

pf_results["rate_hat"] = pf_results["strain_rate"] / STRAIN_RATE_0
pf_results["size_hat"] = pf_results["mean_fragment_size_mm"] * 1e-3 / S_0

pd_results["rate_hat"] = pd_results["strain_rate"] / STRAIN_RATE_0
pd_results["size_hat"] = pd_results["mean_fragment_size_mm"] * 1e-3 / S_0

czm_10000["rate_hat"] = czm_10000["strain_rate"] / STRAIN_RATE_0
czm_10000["size_hat"] = (
    czm_10000["mean_fragment_size_mm"] * 1e-3 / S_0
)

czm_50000["rate_hat"] = czm_50000["strain_rate"] / STRAIN_RATE_0
czm_50000["size_hat"] = (
    czm_50000["mean_fragment_size_mm"] * 1e-3 / S_0
)
# =============================================================================
# Grady prediction
# =============================================================================

minimum_rate = min(czm["strain_rate"].min(), pd_results["strain_rate"].min())
maximum_rate = max(czm["strain_rate"].max(), pd_results["strain_rate"].max())

strain_rate_grady = np.logspace(
    np.log10(minimum_rate),
    np.log10(maximum_rate),
    400,
)

fragment_size_grady = (
    24.0 * GC / (RHO * strain_rate_grady**2)
) ** (1.0 / 3.0)

rate_hat_grady = strain_rate_grady / STRAIN_RATE_0
size_hat_grady = fragment_size_grady / S_0


# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots()

ax.plot(
    czm_10000["rate_hat"],
    czm_10000["size_hat"],
    color=COLORS["blue"],
    marker="o",
    label="CZM, N = 10,000",
)

ax.scatter(
    czm_50000["rate_hat"],
    czm_50000["size_hat"],
    color=COLORS["purple"],
    marker="o",
    s=55,
    label="CZM, N = 50,000",
    zorder=5,
)

ax.plot(
    pd_results["rate_hat"],
    pd_results["size_hat"],
    color=METHOD_STYLE["PD"]["color"],
    marker=METHOD_STYLE["PD"]["marker"],
    label="Peridynamics",
)

ax.plot(
    rate_hat_grady,
    size_hat_grady,
    color=COLORS["green"],
    linestyle="--",
    linewidth=2.5,
    label="Grady",
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(
    r"Normalised strain rate $\dot{\varepsilon}/\dot{\varepsilon}_0$"
)
ax.set_ylabel(r"Normalised mean fragment size $s/s_0$")

ax.legend()

fig.tight_layout()


# =============================================================================
# Save
# =============================================================================

figure_file = OUTPUT_DIR / "fragment_size_czm_pd_grady.png"

fig.savefig(figure_file)

print(f"\nFigure saved to:\n{figure_file}")

# =============================================================================
# Plot: CZM, phase field, peridynamics and Grady
# =============================================================================

fig, ax = plt.subplots()

ax.plot(
    czm_10000["rate_hat"],
    czm_10000["size_hat"],
    color=COLORS["blue"],
    marker="o",
    label="CZM, N = 10,000",
)

ax.scatter(
    czm_50000["rate_hat"],
    czm_50000["size_hat"],
    color=COLORS["purple"],
    marker="o",
    s=55,
    label="CZM, N = 50,000",
    zorder=5,
)

ax.plot(
    pf_results["rate_hat"],
    pf_results["size_hat"],
    color=METHOD_STYLE["PF"]["color"],
    marker=METHOD_STYLE["PF"]["marker"],
    label="Phase field",
)

ax.plot(
    pd_results["rate_hat"],
    pd_results["size_hat"],
    color=METHOD_STYLE["PD"]["color"],
    marker=METHOD_STYLE["PD"]["marker"],
    label="Peridynamics",
)

ax.plot(
    rate_hat_grady,
    size_hat_grady,
    color=COLORS["green"],
    linestyle="--",
    linewidth=2.5,
    label="Grady",
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"Normalised strain rate $\dot{\varepsilon}/\dot{\varepsilon}_0$")
ax.set_ylabel(r"Normalised mean fragment size $s/s_0$")

ax.legend()

fig.tight_layout()

figure_file = OUTPUT_DIR / "fragment_size_czm_pf_pd_grady.png"
fig.savefig(figure_file)

print(f"\nFigure saved to:\n{figure_file}")

# =============================================================================
# Fragment-size distribution
# =============================================================================

DISTRIBUTION_RATE = "1e5"

CZM_SWEEP_DIR = DATA_DIR / "CZM" / "strain_rate_sweep"
PD_SWEEP_DIR = DATA_DIR / "PD" / "strain_rate_sweep"

czm_run_50000 = CZM_SWEEP_DIR / "N_50000" / f"rate_{DISTRIBUTION_RATE}"
czm_run_10000 = CZM_SWEEP_DIR / "N_10000" / f"rate_{DISTRIBUTION_RATE}"

if czm_run_50000.exists():
    czm_run = czm_run_50000
elif czm_run_10000.exists():
    czm_run = czm_run_10000
else:
    raise FileNotFoundError(
        f"No CZM run found for strain rate {DISTRIBUTION_RATE}."
    )

pd_matches = list(PD_SWEEP_DIR.rglob(f"rate_{DISTRIBUTION_RATE}"))
pd_matches = [path for path in pd_matches if path.is_dir()]

if len(pd_matches) == 0:
    raise FileNotFoundError(
        f"No PD run found for strain rate {DISTRIBUTION_RATE}."
    )

if len(pd_matches) > 1:
    print("\nPossible PD runs:")
    for path in pd_matches:
        print(path)
    raise RuntimeError("More than one PD run was found.")

pd_run = pd_matches[0]

print(f"\nCZM distribution run:\n{czm_run}")
print(f"\nPD distribution run:\n{pd_run}")

czm_sizes = helpers.final_fragment_sizes(czm_run)
pd_sizes = helpers.final_fragment_sizes(pd_run)

# =============================================================================
# Find fragment file
# =============================================================================

def find_fragment_file(run_path):

    parameter_file = run_path / "parameters_used.dat"

    if parameter_file.exists():

        params = helpers.read_parameters_file(parameter_file)

        if "fragment_file" in params["user"]:

            fragment_file = run_path / params["user"]["fragment_file"]

            if fragment_file.exists():
                return fragment_file

    candidates = []

    for pattern in ["*fragment*.out", "*fragment*.dat", "*fragment*.txt"]:

        candidates.extend(run_path.glob(pattern))

    candidates = [
        path for path in candidates
        if "summary" not in path.name.lower()
    ]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise FileNotFoundError(f"No fragment file found in {run_path}.")

    print(f"\nSeveral possible fragment files found in {run_path}:")

    for path in candidates:
        print(f"  {path}")

    raise RuntimeError("Could not uniquely identify the fragment file.")


# =============================================================================
# Read fragment sizes
# =============================================================================

def read_fragment_sizes(run_path):

    fragment_file = find_fragment_file(run_path)

    print(f"Reading fragment data from: {fragment_file}")

    try:
        data = helpers.read_output_file(fragment_file)

    except Exception:
        data = None

    if data is not None and len(data) > 0:

        data.columns = [
            normalise_column_name(column) for column in data.columns
        ]

        size_names = [
            "size",
            "fragment_size",
            "length",
            "length_over_l",
        ]

        size_column = None

        for name in size_names:

            if name in data.columns:
                size_column = name
                break

        if size_column is not None:

            if "time" in data.columns:

                final_time = data["time"].max()

                data = data[
                    np.isclose(data["time"], final_time)
                ]

            sizes = pd.to_numeric(
                data[size_column],
                errors="coerce",
            ).to_numpy()

            sizes = sizes[np.isfinite(sizes)]
            sizes = sizes[sizes > 0]

            if len(sizes) > 0:
                return sizes

    with open(fragment_file, "r") as file:
        lines = file.readlines()

    header = None

    for line in lines:

        if line.startswith("#") and "fragment" in line.lower():
            header = line[1:].strip().split()

    numerical_rows = []

    for line in lines:

        if line.startswith("#") or len(line.strip()) == 0:
            continue

        try:
            numerical_rows.append([float(value) for value in line.split()])

        except ValueError:
            continue

    values = np.asarray(numerical_rows)

    if values.size == 0:
        raise ValueError(f"No fragment data could be read from {fragment_file}.")

    if header is not None and len(header) == values.shape[1]:

        columns = [normalise_column_name(name) for name in header]
        data = pd.DataFrame(values, columns=columns)

        if "time" in data.columns:

            final_time = data["time"].max()

            data = data[
                np.isclose(data["time"], final_time)
            ]

        for name in ["size", "fragment_size", "length", "length_over_l"]:

            if name in data.columns:

                sizes = data[name].to_numpy()
                sizes = sizes[np.isfinite(sizes)]
                sizes = sizes[sizes > 0]

                return sizes

    if values.shape[1] == 3:

        sizes = values[:, 2]
        sizes = sizes[np.isfinite(sizes)]
        sizes = sizes[sizes > 0]

        return sizes

    raise ValueError(
        f"Could not identify the fragment-size column in {fragment_file}."
    )


# =============================================================================
# Read CZM and peridynamics fragment sizes
# =============================================================================
czm_sizes = read_fragment_sizes(czm_run)
pd_sizes = read_fragment_sizes(pd_run)


print("\nFragment-size distribution")
print("==========================")

print(f"CZM run:           {czm_run.relative_to(PROJECT_DIR)}")
print(f"Peridynamics run:  {pd_run.relative_to(PROJECT_DIR)}")

print(f"\nCZM fragments:          {len(czm_sizes)}")
print(f"Peridynamics fragments: {len(pd_sizes)}")


# =============================================================================
# Normalise by the mean fragment size
# =============================================================================

czm_sizes_norm = czm_sizes / np.mean(czm_sizes)
pd_sizes_norm = pd_sizes / np.mean(pd_sizes)


# =============================================================================
# Empirical cumulative distributions
# =============================================================================

czm_sizes_sorted = np.sort(czm_sizes_norm)
pd_sizes_sorted = np.sort(pd_sizes_norm)

czm_ecdf = np.arange(1, len(czm_sizes_sorted) + 1) / len(czm_sizes_sorted)
pd_ecdf = np.arange(1, len(pd_sizes_sorted) + 1) / len(pd_sizes_sorted)


# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots()

ax.step(
    czm_sizes_sorted,
    czm_ecdf,
    where="post",
    color=COLORS["blue"],
    linewidth=2.5,
    label="CZM",
)

ax.step(
    pd_sizes_sorted,
    pd_ecdf,
    where="post",
    color=COLORS["red"],
    linewidth=2.5,
    label="Peridynamics",
)

ax.set_xlabel(r"Normalised fragment size $s_i/\bar{s}$")
ax.set_ylabel("Cumulative fraction of fragments")

ax.set_xlim(left=0)
ax.set_ylim(0, 1.02)

ax.legend()

fig.tight_layout()


# =============================================================================
# Save
# =============================================================================

distribution_file = OUTPUT_DIR / "fragment_size_distribution_1e5.png"

fig.savefig(distribution_file)

print(f"\nFigure saved to:\n{distribution_file}")

# =============================================================================
# Computational time comparison
# =============================================================================

CZM_TIME_DIR = DATA_DIR / "CZM" / "strain_rate_sweep"
PF_TIME_DIR = DATA_DIR / "PF" / "strain_rate_sweep"
PD_TIME_DIR = DATA_DIR / "PD" / "strain_rate_sweep"


# =============================================================================
# Read run summary
# =============================================================================
# =============================================================================
# Read wall time
# =============================================================================

def read_run_wall_time(run_path):

    run_path = Path(run_path)

    summary_file = run_path / "summary.out"

    if summary_file.exists():

        with open(summary_file, "r") as file:

            for line in file:

                parts = line.strip().split(maxsplit=1)

                if len(parts) != 2:
                    continue

                key, value = parts

                if key == "wall_time":
                    return float(value)

    log_files = list(run_path.glob("run_*.log"))

    for log_file in log_files:

        with open(log_file, "r") as file:
            text = file.read()

        match = re.search(
            r"Wall time:\s*([0-9.eE+-]+)\s*s",
            text,
        )

        if match is not None:
            return float(match.group(1))

        match = re.search(
            r"Done in\s*([0-9.eE+-]+)\s*s",
            text,
        )

        if match is not None:
            return float(match.group(1))

    raise FileNotFoundError(
        f"No wall-time information found in {run_path}"
    )

# =============================================================================
# Extract strain rate from one run
# =============================================================================

def get_run_strain_rate(run_path):

    parameter_file = run_path / "parameters_used.dat"

    if parameter_file.exists():

        params = helpers.read_parameters_file(parameter_file)

        if "strain_rate" in params["user"]:
            return float(params["user"]["strain_rate"])

    name = run_path.name

    if not name.startswith("rate_"):
        raise ValueError(f"Cannot determine strain rate from {run_path}")

    return float(name.replace("rate_", ""))


# =============================================================================
# Read all timing results for one method
# =============================================================================

def read_timing_sweep(method, sweep_dir):

    rows = []

    rate_dirs = list(sweep_dir.rglob("rate_*"))

    rate_dirs = [path for path in rate_dirs if path.is_dir()]
    if method == "CZM":
        rate_dirs = [path for path in rate_dirs if re.search(r"N_\d+", str(path))]

    if len(rate_dirs) == 0:
        raise FileNotFoundError(
            f"No completed strain-rate runs found in {sweep_dir}"
        )

    for run_path in rate_dirs:

        strain_rate = get_run_strain_rate(run_path)

        try:
            wall_time = read_run_wall_time(run_path)
        except FileNotFoundError:
            print(f"Skipping {run_path}: no wall-time information.")
            continue

        row = {
            "method": method,
            "strain_rate": strain_rate,
            "wall_time": wall_time,
            "run_path": str(run_path),
        }

        if method == "CZM":

            match = re.search(r"N_(\d+)", str(run_path))

            if match is not None:
                row["N"] = int(match.group(1))
            else:
                row["N"] = np.nan

        rows.append(row)

    data = pd.DataFrame(rows)

    if method == "CZM" and "N" in data.columns:
        data = data.sort_values(["strain_rate", "N"])
    else:
        data = data.sort_values("strain_rate")
        data = data.drop_duplicates(
            subset="strain_rate",
            keep="last",
        )

    data = data.reset_index(drop=True)

    return data


# =============================================================================
# Read CZM, phase-field and peridynamics timings
# =============================================================================

czm_time = read_timing_sweep(
    "CZM",
    CZM_TIME_DIR,
)

pf_time = read_timing_sweep(
    "PF",
    PF_TIME_DIR,
)

pd_time = read_timing_sweep(
    "PD",
    PD_TIME_DIR,
)


print("\nCZM computational times")
print(czm_time.to_string(index=False))

print("\nPhase-field computational times")
print(pf_time.to_string(index=False))

print("\nPeridynamics computational times")
print(pd_time.to_string(index=False))


# =============================================================================
# Normalise strain rate
# =============================================================================

czm_time["rate_hat"] = czm_time["strain_rate"] / STRAIN_RATE_0
pf_time["rate_hat"] = pf_time["strain_rate"] / STRAIN_RATE_0
pd_time["rate_hat"] = pd_time["strain_rate"] / STRAIN_RATE_0


# =============================================================================
# Plot
# =============================================================================
czm_time_10000 = czm_time[czm_time["N"] == 10000]
czm_time_50000 = czm_time[czm_time["N"] == 50000]

fig, ax = plt.subplots()

ax.scatter(
    czm_time_10000["rate_hat"],
    czm_time_10000["wall_time"],
    color=COLORS["blue"],
    marker="o",
    s=55,
    label="CZM, N = 10,000",
)

ax.scatter(
    czm_time_50000["rate_hat"],
    czm_time_50000["wall_time"],
    color=COLORS["purple"],
    marker="o",
    s=55,
    label="CZM, N = 50,000",
)

ax.scatter(
    pf_time["rate_hat"],
    pf_time["wall_time"],
    color=COLORS["orange"],
    marker="s",
    label="Phase field",
)

ax.scatter(
    pd_time["rate_hat"],
    pd_time["wall_time"],
    color=COLORS["red"],
    marker="^",
    label="Peridynamics",
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(
    r"Normalised strain rate "
    r"$\dot{\varepsilon}/\dot{\varepsilon}_0$"
)

ax.set_ylabel("Wall time [s]")

ax.legend()

fig.tight_layout()


# =============================================================================
# Save
# =============================================================================

time_figure_file = OUTPUT_DIR / "computational_time_comparison.png"

fig.savefig(time_figure_file)

print(f"\nFigure saved to:\n{time_figure_file}")


plt.show()