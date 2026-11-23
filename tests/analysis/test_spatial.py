# SPDX-License-Identifier: GPL-2.0-only
"""Fracture position reconstruction in the undeformed reference bar."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.calculations.fracture_locations import (
    fracture_snapshots,
)


@pytest.mark.parametrize("method", ["CZM", "PF"])
def test_ordered_lengths_reconstruct_cuts(method, make_case, tmp_path):
    case = make_case(method, tmp_path)
    case.fragments = case.fragments.iloc[[2, 0, 1]].copy()  # Unsorted rows.
    snapshot = fracture_snapshots(case)[0]
    assert snapshot.count == 2
    assert snapshot.left_m == pytest.approx(-0.002)
    assert snapshot.right_m == pytest.approx(0.002)
    np.testing.assert_allclose(snapshot.positions_m, [-0.001, 0.001])


def test_pd_uses_recorded_boundaries(make_case, tmp_path):
    case = make_case("PD", tmp_path)
    snapshot = fracture_snapshots(case)[0]
    np.testing.assert_allclose(snapshot.positions_m, [-0.001, 0.001])
    assert snapshot.count == 2
    assert fracture_snapshots(case, raw=True)[0].count == 2


def test_stacked_time_snapshots_do_not_invent_fractures(make_case, tmp_path):
    case = make_case("PD", tmp_path)
    unbroken = pd.DataFrame(
        {
            "time": [0.0], "step": [0], "fragment_id": [0],
            "size": [0.004], "x_left": [-0.002], "x_right": [0.002],
        }
    )
    case.fragments = pd.concat([unbroken, case.fragments], ignore_index=True)
    snapshots = fracture_snapshots(case, all_snapshots=True)
    assert [snapshot.count for snapshot in snapshots] == [0, 2]
    assert [snapshot.step for snapshot in snapshots] == [0, 100]
    assert len(fracture_snapshots(case)) == 1


def test_discontinuous_pd_boundaries_are_rejected(make_case, tmp_path):
    case = make_case("PD", tmp_path)
    case.fragments.loc[1, "x_left"] = -0.0009
    with pytest.raises(ValueError, match="contiguous"):
        fracture_snapshots(case)


def test_length_mismatch_and_invalid_raw_request(make_case, tmp_path):
    case = make_case("PF", tmp_path)
    case.fragments.loc[2, "size"] = 0.0005
    with pytest.raises(ValueError, match="Fragment lengths sum"):
        fracture_snapshots(case)
    with pytest.raises(ValueError, match="Only PD"):
        fracture_snapshots(case, raw=True)
