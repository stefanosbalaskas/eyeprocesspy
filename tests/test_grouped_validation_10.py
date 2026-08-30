from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "grouped_folds",
    "grouped_cv",
    "crossed_grouped_folds",
    "crossed_grouped_cv",
    "quantify_process_leakage",
]


def _data():
    rng = np.random.default_rng(41)
    participants = [f"P{i:02d}" for i in range(12)]
    items = [f"I{i:02d}" for i in range(8)]
    rows = []
    for participant_index, participant in enumerate(participants):
        person_effect = rng.normal(0, 0.7)
        for item_index, item in enumerate(items):
            x = rng.normal()
            eta = -0.25 + 0.8 * x + person_effect + 0.08 * item_index
            probability = 1 / (1 + np.exp(-eta))
            y = int(rng.random() < probability)
            rows.append(
                {
                    "participant_id": participant,
                    "item_id": item,
                    "x": x,
                    "y": y,
                    "participant_index": participant_index,
                }
            )
    return pd.DataFrame(rows)


def test_public_grouped_validation_exports():
    assert len(TARGETS) == 5
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_grouped_folds_never_split_declared_groups():
    data = _data()
    folds = ep.grouped_folds(data, group="participant_id", v=4, seed=7)
    assert folds.eyeprocess_class == "eye_grouped_folds"
    assert len(folds["folds"]) == 4

    for fold in folds["folds"]:
        train = set(data.iloc[fold["analysis"]].participant_id)
        test = set(data.iloc[fold["assessment"]].participant_id)
        assert train.isdisjoint(test)
        assert len(fold["assessment"]) > 0


def test_grouped_folds_combined_key_blocks_repeated_crossed_cells():
    data = _data()
    folds = ep.grouped_folds(
        data,
        group=["participant_id", "item_id"],
        v=4,
        seed=9,
    )
    for fold in folds["folds"]:
        train = {tuple(row) for row in data.iloc[fold["analysis"]][["participant_id", "item_id"]].to_numpy()}
        test = {tuple(row) for row in data.iloc[fold["assessment"]][["participant_id", "item_id"]].to_numpy()}
        assert train.isdisjoint(test)


@pytest.mark.parametrize("metric", ["log_loss", "brier", "accuracy"])
def test_grouped_cv_returns_finite_fold_scores(metric):
    result = ep.grouped_cv(
        _data(),
        "y ~ x",
        group="participant_id",
        v=4,
        metric=metric,
        seed=11,
    )
    assert result.eyeprocess_class == "eye_grouped_cv"
    scores = result["results"]
    assert len(scores) == 4
    assert scores.error.isna().all()
    assert np.isfinite(scores.score).all()


def test_crossed_folds_buffer_mixed_held_out_rows():
    data = _data()
    folds = ep.crossed_grouped_folds(
        data,
        groups=["participant_id", "item_id"],
        v=4,
        seed=3,
    )
    assert folds.eyeprocess_class == "eye_crossed_grouped_folds"

    for fold in folds["folds"]:
        analysis = set(fold["analysis"])
        assessment = set(fold["assessment"])
        buffer = set(fold["buffer"])
        assert analysis.isdisjoint(assessment)
        assert analysis.isdisjoint(buffer)
        assert assessment.isdisjoint(buffer)
        assert analysis | assessment | buffer == set(range(len(data)))

        if assessment:
            held_participants = set(data.iloc[list(assessment)].participant_id.astype(str))
            held_items = set(data.iloc[list(assessment)].item_id.astype(str))
            assert not set(data.iloc[list(analysis)].participant_id.astype(str)) & held_participants
            assert not set(data.iloc[list(analysis)].item_id.astype(str)) & held_items


def test_crossed_grouped_cv_reports_buffer_and_scores():
    result = ep.crossed_grouped_cv(
        _data(),
        "y ~ x",
        groups=["participant_id", "item_id"],
        v=4,
        metric="log_loss",
        seed=4,
    )
    assert result.eyeprocess_class == "eye_crossed_grouped_cv"
    scores = result["results"]
    assert len(scores) == 4
    assert (scores.n_buffer > 0).all()
    assert scores.error.isna().all()
    assert np.isfinite(scores.score).all()


def test_crossed_folds_reject_missing_group_levels():
    data = _data()
    data.loc[0, "item_id"] = pd.NA
    with pytest.raises(
        ep.EyeProcessValidationError,
        match="cannot contain missing or empty",
    ):
        ep.crossed_grouped_folds(data, v=4)


def test_quantify_process_leakage_runs_all_frozen_schemes():
    result = ep.quantify_process_leakage(
        _data(),
        "y ~ x",
        group=["participant_id", "item_id"],
        v=4,
        seed=5,
    )
    expected = {
        "row_wise",
        "combined_group",
        "held_participant_id",
        "held_item_id",
        "cross_classified",
    }
    assert set(result.scheme) == expected
    assert (result.folds == 4).all()
    assert (result.successful_folds > 0).all()
    assert np.isfinite(result.mean_log_loss).all()

    row_loss = result.loc[result.scheme.eq("row_wise"), "mean_log_loss"].iloc[0]
    np.testing.assert_allclose(
        result.optimistic_difference,
        result.mean_log_loss - row_loss,
    )


def test_invalid_fold_requests_are_rejected():
    data = _data()
    with pytest.raises(ep.EyeProcessValidationError):
        ep.grouped_folds(data, v=1)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.grouped_folds(data, group="participant_id", v=20)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.crossed_grouped_folds(data, groups=["participant_id"], v=4)
