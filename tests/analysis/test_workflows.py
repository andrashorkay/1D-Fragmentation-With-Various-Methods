# SPDX-License-Identifier: GPL-2.0-only
"""CLI and six-workflow smoke tests with temp directories and synthetic cases."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.analysis.cli import create_parser, main
from src.analysis.workflows import WORKFLOWS, run_analysis


def _args(command: str, method: str, directory: Path, *options: str):
    return create_parser().parse_args(
        [command, "--method", method, "--input", str(directory), "--no-show", *options]
    )


def _patch_loader(monkeypatch, workflow: str, make_case):
    module = importlib.import_module(f"src.analysis.workflows.{workflow}")
    monkeypatch.setattr(module, "load_case", make_case)
    return module


def test_dispatcher_lists_all_six_commands():
    assert set(WORKFLOWS) == {
        "strain-rate", "convergence", "energy", "distribution", "history", "spatial"
    }
    with pytest.raises(NotImplementedError):
        run_analysis(SimpleNamespace(analysis="nonexistent"))


def test_cli_passes_options_to_dispatcher(tmp_path, monkeypatch):
    import src.analysis.workflows as workflows

    seen = []
    monkeypatch.setattr(workflows, "run_analysis", seen.append)
    monkeypatch.setattr(sys, "argv", [
        "fragmentation-analysis", "energy", "--method", "PD",
        "--input", str(tmp_path), "--view", "balance", "--balance-kind", "relative",
        "--no-show",
    ])
    main()
    assert seen[0].analysis == "energy"
    assert seen[0].view == "balance"
    assert seen[0].balance_kind == "relative"


def test_cli_rejects_raw_pd_masses(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "fragmentation-analysis", "distribution", "--method", "PD",
        "--input", str(tmp_path), "--quantity", "mass", "--raw",
    ])
    with pytest.raises(SystemExit, match="2"):
        main()


@pytest.mark.parametrize("view,extra", [
    ("history", ["--quantity", "all"]),
    ("balance", ["--balance-kind", "relative"]),
    ("partition", ["--stacked"]),
])
def test_energy_workflow_exports_without_show(
    view, extra, tmp_path, monkeypatch, make_case
):
    case_dir = tmp_path / "run"
    case_dir.mkdir()
    case = make_case("PD", case_dir)
    module = _patch_loader(monkeypatch, "energy", lambda *_: case)
    fig_path = tmp_path / f"energy_{view}.png"
    args = _args("energy", "PD", case_dir, "--view", view,
                 "--output", str(fig_path), *extra)
    result = module.run_energy(args)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert fig_path.is_file()
    report = json.loads((tmp_path / f"energy_{view}_summary.json").read_text())
    assert report["run_status"] == "partial_or_unverified"
    assert (tmp_path / f"energy_{view}_data.csv").is_file()


@pytest.mark.parametrize("kind", ["hist", "cdf", "ccdf"])
def test_distribution_workflow_exports(kind, tmp_path, monkeypatch, make_case):
    case_dir = tmp_path / "run"
    case_dir.mkdir()
    case = make_case("PD", case_dir)
    module = _patch_loader(monkeypatch, "distribution", lambda *_: case)
    fig_path = tmp_path / f"distribution_{kind}.png"
    args = _args("distribution", "PD", case_dir,
                 "--quantity", "length", "--kind", kind,
                 "--output", str(fig_path))
    result = module.run_distribution(args)
    assert not result.empty
    assert fig_path.is_file()
    assert (tmp_path / f"distribution_{kind}_data.csv").is_file()
    assert (tmp_path / f"distribution_{kind}_summary.json").is_file()
    report = json.loads((tmp_path / f"distribution_{kind}_summary.json").read_text())
    assert report["statistics_SI"]["count"] == 3


def test_history_and_spatial_stacking(tmp_path, monkeypatch, make_case):
    first, second = tmp_path / "run1", tmp_path / "run2"
    first.mkdir()
    second.mkdir()
    def loader(method, path):
        return make_case(method, Path(path))

    for workflow in ("history", "spatial"):
        module = _patch_loader(monkeypatch, workflow, loader)
        fig_path = tmp_path / f"{workflow}.png"
        extra = ["--quantity", "average_stress"] if workflow == "history" else []
        args = _args(workflow, "PD", first, "--stack-input", str(second),
                     "--output", str(fig_path), *extra)
        result = getattr(module, f"run_{workflow}")(args)
        assert not result.empty
        assert fig_path.is_file()
        assert (tmp_path / f"{workflow}_summary.json").is_file()


def test_convergence_workflow_preserves_finest_reference(tmp_path, monkeypatch, make_case):
    for factor in (0.2, 0.1, 0.05):
        folder = tmp_path / f"factor_{factor}"
        folder.mkdir()
        (folder / "summary.out").write_text("completed\n")

    def loader(method, folder):
        case = make_case(method, Path(folder))
        factor = float(Path(folder).name.removeprefix("factor_"))
        case.parameters["time_step_factor"] = factor
        case.history["fracture_energy"] *= 1.0 + factor
        return case

    module = _patch_loader(monkeypatch, "convergence", loader)
    fig_path = tmp_path / "dt.png"
    args = _args("convergence", "PD", tmp_path, "--parameter", "dt",
                 "--quantity", "fracture_energy", "--mode", "successive",
                 "--output", str(fig_path))
    table = module.run_convergence(args)
    assert table["dt_factor"].tolist() == pytest.approx([0.2, 0.1, 0.05])
    assert table["relative_error_final_fracture_energy_J"].iloc[-1] == 0
    assert fig_path.is_file()
    assert (tmp_path / "dt_summary.csv").is_file()


def test_strain_rate_workflow_fit_and_exports(tmp_path, monkeypatch, make_case):
    rates = [1e5, 2e5, 5e5]
    for rate in rates:
        folder = tmp_path / f"rate_{rate:g}"
        folder.mkdir()
        (folder / "summary.out").write_text("completed\n")

    def loader(method, folder):
        case = make_case(method, Path(folder))
        rate = float(Path(folder).name.removeprefix("rate_"))
        count = {1e5: 3, 2e5: 4, 5e5: 5}[rate]
        case.parameters["strain_rate"] = rate
        case.fragments = pd.DataFrame({
            "time": [2e-6] * count,
            "size": [0.004 / count] * count,
            "mass": [0.4 / count] * count,
        })
        return case

    module = _patch_loader(monkeypatch, "strain_rate", loader)
    fig_path = tmp_path / "rates.png"
    args = _args("strain-rate", "PD", tmp_path, "--output", str(fig_path),
                 "--fit-skip-first", "1")
    table = module.run_strain_rate(args)
    assert len(table) == 3
    assert table["used_in_fit"].tolist() == [False, True, True]
    assert fig_path.is_file()
    assert (tmp_path / "rates_summary.csv").is_file()
    assert (tmp_path / "rates_fit.json").is_file()
