# SPDX-License-Identifier: GPL-2.0-only
"""Case-reader contracts with small mock solver output tables.

The actual method-specific loaders are exercised; only the low-level text
parser is replaced so tests are independent of header whitespace conventions.
"""

import pandas as pd
import pytest

from src.analysis.io import load_case
from src.analysis.io import czm, pd as pd_reader, pf


def _history(method: str) -> pd.DataFrame:
    columns = {
        "step": [0, 1],
        "time": [0.0, 1e-6],
        "kinetic_energy": [0.0, 0.2],
        "elastic_energy": [1.0, 0.8],
        "fracture_energy": [0.0, 0.1],
        "average_stress": [0.0, 2e8],
        "num_fragments": [1, 2],
        "mean_fragment_size": [0.004, 0.002],
        "total_energy": [1.0, 1.2],
        "external_work": [0.0, 0.2],
        "energy_error": [0.0, 0.0],
    }
    if method == "CZM":
        columns.update(
            reversible_cohesive_energy=[0.0, 0.05],
            contact_energy=[0.0, 0.01],
        )
    return pd.DataFrame(columns)


def _fragment_table(method: str):
    base = {"time": [0.0, 1e-6, 1e-6], "fragment_id": [0, 0, 1], "size": [0.004, 0.002, 0.002]}
    if method == "PD":
        base["step"] = [0, 1, 1]
        base["x_left"] = [-0.002, -0.002, 0.0]
        base["x_right"] = [0.002, 0.0, 0.002]
    return pd.DataFrame(base)


@pytest.mark.parametrize(
    ("method", "module", "filename", "fragment_name"),
    [
        ("PD", pd_reader, "results.out", "fragments.out"),
        ("CZM", czm, "energies.out", "fragments.out"),
        ("PF", pf, "energies_pf.out", "fragments_pf.out"),
    ],
)
def test_reader_aliases_and_last_saved_snapshot(
    method, module, filename, fragment_name, tmp_path, monkeypatch
):
    (tmp_path / filename).write_text("test data\n", encoding="utf-8")
    (tmp_path / fragment_name).write_text("test data\n", encoding="utf-8")
    history = _history(method)
    fragments = _fragment_table(method)

    def table_for(path):
        if path.name == filename:
            return history.copy()
        if path.name == fragment_name:
            return fragments.copy()
        raise AssertionError(f"Unexpected file read: {path}")

    monkeypatch.setattr(module, "read_table", table_for)
    case = load_case(method, tmp_path)
    assert case.method == method
    assert len(case.history) == 2
    assert len(case.final_fragments()) == 2
    assert case.history["recoverable_energy"].iloc[-1] == pytest.approx(
        0.85 if method == "CZM" else 0.8
    )
    with pytest.raises(ValueError):
        case.final_fragments(raw=True)  # None stored for PD; unsupported for CZM/PF.


def test_pd_raw_snapshot_separate_from_filtered(tmp_path):
    filtered = _fragment_table("PD")
    raw = filtered.drop(columns=["size"]).copy()
    case = pd_reader.PDCase(
        path=tmp_path,
        parameters={},
        history=_history("PD"),
        diagnostics=None,
        fragments=filtered,
        raw_fragments=raw,
        summary={},
        material_properties=None,
    )
    assert len(case.final_fragments()) == 2
    assert "size" not in case.final_fragments(raw=True)


def test_no_fragments_reports_missing_data(tmp_path):
    case = pf.PFCase(
        path=tmp_path, parameters={}, history=_history("PF"),
        diagnostics=None, fragments=None, raw_fragments=None,
        summary={}, material_properties=None,
    )
    with pytest.raises(ValueError, match="No fragment"):
        case.final_fragments()
