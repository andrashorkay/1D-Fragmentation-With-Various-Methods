# SPDX-License-Identifier: GPL-2.0-only
"""Command-line interface for the Fragmentation analysis package."""

import argparse
from pathlib import Path

from .config import SUPPORTED_METHODS


def add_common_arguments(parser):
    """Add arguments shared by all analyses."""
    parser.add_argument(
        "--method",
        type=str.upper,
        choices=SUPPORTED_METHODS,
        required=True,
        help="Numerical method: CZM, PD or PF.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing the simulation results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output figure path.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the figure using the default output location.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the figure interactively.",
    )


def create_parser():
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="fragmentation-analysis",
        description="Post-processing tools for dynamic fragmentation.",
    )
    subparsers = parser.add_subparsers(
        dest="analysis",
        required=True,
        help="Type of analysis to perform.",
    )

    strain_rate = subparsers.add_parser(
        "strain-rate",
        help="Analyse mean fragment size versus strain rate.",
    )
    add_common_arguments(strain_rate)
    strain_rate.add_argument(
        "--fit-skip-first",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Exclude the first N completed strain-rate points from the fit "
            "(after sorting by increasing strain rate). All points remain plotted."
        ),
    )
    strain_rate.add_argument(
        "--fit-min",
        type=float,
        default=None,
        help="Minimum strain rate used for the power-law fit.",
    )
    strain_rate.add_argument(
        "--fit-max",
        type=float,
        default=None,
        help="Maximum strain rate used for the power-law fit.",
    )
    strain_rate.add_argument(
        "--theory",
        choices=("none", "grady"),
        default="none",
        help="Optional theoretical comparison.",
    )
    strain_rate.add_argument(
        "--normalised",
        action="store_true",
        help="Use dimensionless strain rate and fragment length based on E, rho, sigma_c and G_c.",
    )
    strain_rate.add_argument(
        "--reference-strength",
        type=float,
        default=None,
        help="Nominal strength in Pa used for normalisation; required if it cannot be read from the run (e.g. PF).",
    )
    strain_rate.add_argument(
        "--high-rate-input",
        type=Path,
        default=None,
        help="Second sweep directory for cases with finer mesh at high strain rates.",
    )
    strain_rate.add_argument(
        "--high-from",
        type=float,
        default=None,
        help="First strain rate to read from --high-rate-input (inclusive).",
    )

    convergence = subparsers.add_parser(
        "convergence",
        help="Analyse numerical convergence.",
    )
    add_common_arguments(convergence)
    convergence.add_argument(
        "--parameter",
        choices=("dt", "dx", "horizon", "l0"),
        required=True,
        help="Parameter varied in the convergence study.",
    )

    convergence.add_argument(
        "--quantity",
        choices=(
            "peak_stress",
            "fracture_energy",
            "recoverable_energy",
            "kinetic_energy",
            "fragments",
            "mean_fragment_size",
            "energy_error",
            "wall_time",
        ),
        default="fracture_energy",
        help="Quantity to compare (default: fracture_energy).",
    )
    convergence.add_argument(
        "--mode",
        choices=("value", "successive", "reference"),
        default="value",
        help="Plot values, successive changes, or errors to the reference case.",
    )
    convergence.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Optional relative tolerance, e.g. 0.02 for 2%% (error modes only).",
    )

    energy = subparsers.add_parser(
        "energy",
        help="Plot simulation energy histories.",
    )
    add_common_arguments(energy)
    energy.add_argument(
        "--view",
        choices=("history", "balance", "partition"),
        default="history",
        help="Energy plot type (default: history).",
    )
    energy.add_argument(
        "--quantity",
        choices=(
            "all",
            "kinetic",
            "recoverable",
            "fracture",
            "contact",
            "total",
            "external_work",
        ),
        default="all",
        help="Component shown with --view history (default: all).",
    )
    energy.add_argument(
        "--balance-kind",
        choices=("relative", "absolute", "totals"),
        default="relative",
        help="Energy conservation plot style for --view balance.",
    )
    energy.add_argument(
        "--stacked",
        action="store_true",
        help="Show a stacked area plot with --view partition.",
    )
    energy.add_argument(
        "--time-unit",
        choices=("s", "ms", "us", "ns"),
        default="us",
        help="Time-axis unit (default: us).",
    )
    energy.add_argument(
        "--energy-unit",
        choices=("J", "mJ", "uJ", "kJ"),
        default="mJ",
        help="Energy-axis unit (default: mJ).",
    )
    energy.add_argument(
        "--initial-energy",
        type=float,
        default=None,
        help="Optional initial total energy in joules for balance calculations.",
    )

    distribution = subparsers.add_parser(
        "distribution",
        help="Analyse fragment size or mass distributions.",
    )
    add_common_arguments(distribution)
    distribution.add_argument(
        "--quantity",
        choices=("length", "mass"),
        default="length",
        help="Fragment quantity to analyse (default: length).",
    )
    distribution.add_argument(
        "--kind",
        choices=("hist", "cdf", "ccdf"),
        default="hist",
        help="Histogram, cumulative distribution, or complementary CDF.",
    )
    distribution.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins (default: 30).",
    )
    distribution.add_argument(
        "--normalised",
        action="store_true",
        help="Divide individual fragment values by their sample mean.",
    )
    distribution.add_argument(
        "--counts",
        action="store_true",
        help="Plot fragment counts rather than density (histograms only).",
    )
    distribution.add_argument(
        "--raw",
        action="store_true",
        help="Use PD raw fragments rather than filtered fragments (length only).",
    )
    distribution.add_argument(
        "--unit",
        choices=("m", "mm", "um", "kg", "g", "mg"),
        default=None,
        help="Display unit (default: mm for length, mg for mass).",
    )
    distribution.add_argument(
        "--log-x",
        action="store_true",
        help="Use a logarithmic horizontal axis (positive values only).",
    )

    history = subparsers.add_parser(
        "history",
        help="Plot recorded scalar diagnostics against time.",
    )
    add_common_arguments(history)
    history.add_argument(
        "--quantity",
        required=True,
        help="Diagnostic column to plot (e.g. average_stress, num_fragments).",
    )
    history.add_argument(
        "--source",
        choices=("history", "diagnostics"),
        default="history",
        help="Data table containing the quantity (diagnostics is PD-specific).",
    )
    history.add_argument(
        "--time-unit",
        choices=("s", "ms", "us", "ns"),
        default="us",
        help="Time-axis unit (default: us).",
    )
    history.add_argument(
        "--stack-input",
        action="append",
        type=Path,
        default=[],
        metavar="DIRECTORY",
        help="Overlay another simulation history; repeatable.",
    )

    spatial = subparsers.add_parser(
        "spatial",
        help="Show fracture positions along the 1D bar (optionally stacked).",
    )
    add_common_arguments(spatial)
    spatial.add_argument(
        "--position-unit",
        choices=("m", "mm", "um"),
        default="mm",
        help="Horizontal position-axis unit (default: mm).",
    )
    spatial.add_argument(
        "--stack-input",
        action="append",
        type=Path,
        default=[],
        metavar="DIRECTORY",
        help="Add another simulation as a separate horizontal bar; repeatable.",
    )
    spatial.add_argument(
        "--all-snapshots",
        action="store_true",
        help="Stack sampled saved snapshots from each case, not only the last.",
    )
    spatial.add_argument(
        "--max-snapshots",
        type=int,
        default=12,
        metavar="N",
        help="Maximum sampled snapshots per case (default: 12).",
    )
    spatial.add_argument(
        "--raw",
        action="store_true",
        help="Use raw rather than filtered fractures (PD only).",
    )
    return parser


def main():
    """Parse arguments and execute the requested analysis."""
    parser = create_parser()
    args = parser.parse_args()

    args.input = args.input.expanduser().resolve()
    if not args.input.is_dir():
        parser.error(f"Input directory does not exist: {args.input}")

    if args.analysis == "strain-rate":
        if args.fit_skip_first < 0:
            parser.error("--fit-skip-first must be non-negative")
        if args.reference_strength is not None:
            if not 0 < args.reference_strength < float("inf"):
                parser.error("--reference-strength must be positive and finite")
            if not args.normalised:
                parser.error("--reference-strength requires --normalised")
        if (
            args.fit_min is not None
            and args.fit_max is not None
            and args.fit_min >= args.fit_max
        ):
            parser.error("--fit-min must be smaller than --fit-max")

        if (args.high_rate_input is None) != (args.high_from is None):
            parser.error(
                "--high-rate-input and --high-from must be supplied together"
            )

        if args.high_from is not None:
            if not 0 < args.high_from < float("inf"):
                parser.error(
                    "--high-from must be a positive finite strain rate"
                )
            args.high_rate_input = args.high_rate_input.expanduser().resolve()
            if not args.high_rate_input.is_dir():
                parser.error(
                    f"High-rate input directory does not exist: {args.high_rate_input}"
                )
            if args.high_rate_input == args.input:
                parser.error("--high-rate-input must differ from --input")

    if args.analysis == "convergence":
        if args.tolerance is not None:
            if not 0 < args.tolerance < float("inf"):
                parser.error("--tolerance must be positive and finite")
            if args.mode == "value":
                parser.error(
                    "--tolerance requires --mode successive or reference"
                )

    if args.analysis == "energy":
        if args.view != "history" and args.quantity != "all":
            parser.error("--quantity applies only to --view history")
        if args.view != "balance" and args.balance_kind != "relative":
            parser.error("--balance-kind requires --view balance")
        if args.stacked and args.view != "partition":
            parser.error("--stacked requires --view partition")
        if args.initial_energy is not None:
            import math

            if not math.isfinite(args.initial_energy):
                parser.error("--initial-energy must be finite")

    if args.analysis == "distribution":
        if args.bins < 1:
            parser.error("--bins must be positive")
        if args.raw and args.method != "PD":
            parser.error("--raw is supported only for PD")
        if args.raw and args.quantity == "mass":
            parser.error("PD raw fragments do not contain mass")
        if args.counts and args.kind != "hist":
            parser.error("--counts requires --kind hist")
        if args.normalised and args.unit is not None:
            parser.error("--unit cannot be combined with --normalised")
        if args.unit is not None:
            allowed = (
                {"m", "mm", "um"}
                if args.quantity == "length"
                else {"kg", "g", "mg"}
            )
            if args.unit not in allowed:
                parser.error(f"Invalid --unit for {args.quantity}: {args.unit}")

    if args.analysis == "spatial":
        if args.raw and args.method != "PD":
            parser.error("--raw is only available for PD")
        if args.max_snapshots < 1:
            parser.error("--max-snapshots must be positive")

    if args.analysis in {"history", "spatial"}:
        args.stack_input = [p.expanduser().resolve() for p in args.stack_input]
        for directory in args.stack_input:
            if not directory.is_dir():
                parser.error(
                    f"Stacked input directory does not exist: {directory}"
                )
        if args.input in args.stack_input or len(set(args.stack_input)) != len(
            args.stack_input
        ):
            parser.error("Each stacked input must be a different directory")

    args.save = args.save or args.output is not None

    from . import workflows

    workflows.run_analysis(args)
