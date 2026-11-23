"""
Author: Andras Horkay
Contact: andras.horkay@gmail.com

Shared configuration for the Fragmentation analysis package.

Defines project paths, supported numerical methods, and default
output settings. Individual datasets are selected through the CLI.
"""

from pathlib import Path

# ============================================================================
# Project directories

# Absolute path to Fragmentation/project/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"
ANALYSIS_DIR = SRC_DIR / "analysis"

DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"


# ============================================================================
# Supported numerical methods
SUPPORTED_METHODS = ("CZM", "PD", "PF")


# ============================================================================
# Default figure settings
DEFAULT_FIGURE_FORMAT = "png"
DEFAULT_FIGURE_DPI = 300

DEFAULT_SAVE_FIGURES = False


# ============================================================================
# Default analysis settings

DEFAULT_DISTRIBUTION_BINS = 30
DEFAULT_LOG_SCALE = True


# ============================================================================
# Helper functions


def get_method_data_dir(method: str) -> Path:
    """Return the data directory for a numerical method."""

    method = method.upper()

    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Expected one of {SUPPORTED_METHODS}."
        )

    return DATA_DIR / method


def get_method_figures_dir(method: str) -> Path:
    """Return the figure directory for a numerical method."""

    method = method.upper()

    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Expected one of {SUPPORTED_METHODS}."
        )

    return FIGURES_DIR / method
