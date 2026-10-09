"""Scientific boundary tests for the experimental missingness audit."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.methods_briefing_2026.missingness_mechanism_audit import (  # noqa: E402
    compare_missingness_sensitivity,
    profile_missingness_mechanism,
)


def _data():
    return pd.DataFrame({
        "participant_id": ["p1"] * 5 + ["p2"] * 3,
        "trial_id": ["t1"] * 8,
        "time_s": [0, 1, 2, 3, 4, 0, 1, 2],
        "observed": [1, 0, 0, 1, 1, 1, 0, 1],
        "outcome": [10.0, None, None, 14.0, 16.0, 8.0, None, 12.0],
        "condition": ["active"] * 5 + ["passive"] * 3,
    })


def test_profiles_recorded_dropout_runs_without_inferring_mnar():
    result = profile_missingness_mechanism(
        _data(), observed_col="observed", time_col="time_s"
    )
    assert result["summary"]["n_missing"].tolist() == [2, 1]
    assert result["gap_runs"]["n_recorded_missing_rows"].tolist() == [2, 1]
    assert set(result["summary"]["mechanism"]) == {"not_identified"}


def test_pattern_mixture_delta_is_on_outcome_scale():
    result = compare_missingness_sensitivity(
        _data(), outcome_col="outcome", observed_col="observed",
        deltas=[-3.0, 0.0, 3.0],
    )
    active = result[result["condition"] == "active"].reset_index(drop=True)
    assert active["observed_mean"].tolist() == [pytest.approx(40 / 3)] * 3
    assert active.loc[2, "scenario_mean"] == pytest.approx(40 / 3 + 2 / 5 * 3)
    assert (result["inference_status"] == "descriptive_sensitivity_only").all()


def test_rejects_ambiguous_indicator_and_nonmonotone_timestamp():
    data = _data()
    data.loc[0, "observed"] = 2
    with pytest.raises(ValueError, match="indicator"):
        profile_missingness_mechanism(data, observed_col="observed")
    data = _data()
    data.loc[2, "time_s"] = 1
    with pytest.raises(ValueError, match="strictly increase"):
        profile_missingness_mechanism(data, observed_col="observed", time_col="time_s")


def test_group_without_observed_outcomes_is_not_silently_dropped():
    data = _data()
    data.loc[data["condition"] == "passive", "observed"] = 0
    with pytest.raises(ValueError, match="at least one"):
        compare_missingness_sensitivity(data, outcome_col="outcome", observed_col="observed")


def test_same_recorded_rows_do_not_identify_missing_outcome_mechanism():
    """Two hidden potential-outcome worlds have an identical observed export."""
    recorded = pd.DataFrame({
        "condition": ["active"] * 6,
        "observed": [1, 1, 0, 0, 1, 0],
        "outcome": [10.0, 12.0, None, None, 11.0, None],
    })
    # The complete-data truths below are deliberately not inputs to the API.
    full_low = [10.0, 12.0, 2.0, 2.0, 11.0, 2.0]
    full_high = [10.0, 12.0, 20.0, 20.0, 11.0, 20.0]
    assert sum(full_low) / 6 != sum(full_high) / 6
    report = compare_missingness_sensitivity(
        recorded, outcome_col="outcome", observed_col="observed",
        deltas=[-9.0, 0.0, 9.0],
    )
    assert report["observed_mean"].tolist() == [11.0] * 3
    assert report["scenario_mean"].tolist() == [6.5, 11.0, 15.5]
    assert set(report["inference_status"]) == {"descriptive_sensitivity_only"}
