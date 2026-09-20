from __future__ import annotations

import pandas as pd

import eyeprocesspy.detector_multiverse as dm
import eyeprocesspy.legacy_models as legacy


def _ivt_spec(detector_id: str = "ivt_final"):
    return dm.define_event_detector_spec(
        detector_id,
        "ivt",
        velocity_threshold=30,
        minimum_duration_ms=60,
        maximum_gap_ms=75,
        sampling_rate=60,
        coordinate_unit="degrees",
    )


# ---------------------------------------------------------------------
# import_external_detector_events:
# event has no matching trial interval -> leave trial_id missing
# ---------------------------------------------------------------------


def test_external_event_outside_trials_remains_unassigned():
    data = dm.simulate_detector_multiverse_data(
        n_participants=4,
        seed=20260920,
    )

    intervals = data["intervals"]
    first = intervals.iloc[0]
    recording_id = first["recording_id"]

    last_end = float(
        pd.to_numeric(
            intervals.loc[
                intervals["recording_id"].eq(recording_id),
                "end_time",
            ],
            errors="coerce",
        ).max()
    )

    external = pd.DataFrame(
        {
            "recording_id": [recording_id],
            "episode_type": ["fixation"],
            "start_time": [last_end + 10.0],
            "end_time": [last_end + 10.1],
        }
    )

    spec = dm.define_event_detector_spec(
        "external_final",
        "external",
        callback=lambda **kwargs: pd.DataFrame(),
        implementation="coverage_test",
    )

    out = dm.import_external_detector_events(
        external,
        spec,
        dataset=data,
    )

    assert pd.isna(out.loc[0, "trial_id"])


# ---------------------------------------------------------------------
# match_detected_events:
# trial identity is genuinely optional
# ---------------------------------------------------------------------


def test_event_matching_without_trial_columns():
    reference = pd.DataFrame(
        {
            "recording_id": ["R1"],
            "episode_type": ["fixation"],
            "start_time": [0.10],
            "end_time": [0.30],
        }
    )

    candidate = pd.DataFrame(
        {
            "recording_id": ["R1"],
            "episode_type": ["fixation"],
            "start_time": [0.11],
            "end_time": [0.31],
        }
    )

    out = dm.match_detected_events(
        reference,
        candidate,
    )

    assert len(out) == 1
    assert pd.isna(out.loc[0, "trial_id"])


# ---------------------------------------------------------------------
# AOI assignment:
# - AOI definition without stimulus restriction
# - duplicate geometry row for same AOI must be de-duplicated
# ---------------------------------------------------------------------


def test_detector_aoi_global_definition_and_duplicate_geometry():
    data = dm.simulate_detector_multiverse_data(
        n_participants=4,
        seed=20260921,
    )

    spec = _ivt_spec("ivt_global_aoi")
    result = dm.run_detector_multiverse(
        data,
        [spec],
    )

    branch = result.branches[spec.detector_id].copy()

    fixation_mask = branch["episodes"]["episode_type"].eq("fixation")

    assert fixation_mask.any()

    event_index = branch["episodes"].index[fixation_mask][0]

    definitions = branch["aoi_definitions"].copy()

    definition = definitions.iloc[0]
    aoi_id = str(definition["aoi_id"])

    geometry_rows = branch["aoi_geometry"].loc[
        branch["aoi_geometry"]["aoi_id"].astype(str).eq(aoi_id)
    ]

    assert not geometry_rows.empty

    geometry = geometry_rows.iloc[0]

    # A missing stimulus_id means this AOI is globally applicable.
    definitions.loc[
        definitions["aoi_id"].astype(str).eq(aoi_id),
        "stimulus_id",
    ] = pd.NA

    branch["aoi_definitions"] = definitions

    # Duplicate the same AOI geometry deliberately. The assignment
    # must retain one AOI identity rather than producing ambiguity.
    branch["aoi_geometry"] = pd.concat(
        [
            branch["aoi_geometry"],
            geometry_rows.iloc[[0]].copy(),
        ],
        ignore_index=True,
        sort=False,
    )

    episodes = branch["episodes"].copy()

    episodes.loc[
        event_index,
        "centroid_x",
    ] = float(geometry["x"]) + float(geometry["width"]) / 2

    episodes.loc[
        event_index,
        "centroid_y",
    ] = float(geometry["y"]) + float(geometry["height"]) / 2

    episodes.loc[
        event_index,
        "coordinate_space_id",
    ] = geometry["coordinate_space_id"]

    branch["episodes"] = episodes

    assigned = dm._assign_episode_aois_explicit(
        branch,
        overlap="first",
    )

    assert (
        str(
            assigned["episodes"].loc[
                event_index,
                "aoi_id",
            ]
        )
        == aoi_id
    )


# ---------------------------------------------------------------------
# Detector result with episode catalogue lacking detector_id.
#
# This is a legitimate public-container state: propagation must not
# assume an optional provenance column exists.
# ---------------------------------------------------------------------


def test_detector_propagation_without_detector_id_column():
    data = dm.simulate_detector_multiverse_data(
        n_participants=4,
        seed=20260922,
    )

    spec = _ivt_spec("ivt_no_detector_column")

    result = dm.run_detector_multiverse(
        data,
        [spec],
    )

    branch = result.branches[spec.detector_id].copy()

    branch["episodes"] = branch["episodes"].drop(
        columns=["detector_id"],
        errors="ignore",
    )

    modified = dm.DetectorMultiverseResult(
        multiverse=result.multiverse,
        branches={
            spec.detector_id: branch,
        },
        events=result.events.drop(
            columns=["detector_id"],
            errors="ignore",
        ),
        status=result.status.copy(),
        failures=result.failures.copy(),
        warnings=result.warnings.copy(),
        features=result.features.copy(),
        source_fingerprint=result.source_fingerprint,
    )

    propagated = dm.propagate_detector_to_aoi(
        modified,
        overlap="first",
        continue_on_error=False,
    )

    assert "detector_id" not in propagated.events.columns

    features = dm._derive_branch_features(
        propagated.branches[spec.detector_id],
        spec,
    )

    assert not features.empty


# ---------------------------------------------------------------------
# Strategy-mixture append=False branch.
#
# Extraction is isolated with deterministic synthetic feature data;
# the actual KMeans/scaling implementation is still exercised.
# ---------------------------------------------------------------------


def test_strategy_mixture_append_false_preserves_dataset(
    monkeypatch,
):
    wide = pd.DataFrame(
        {
            "recording_id": [
                "R1",
                "R2",
                "R3",
                "R4",
                "R5",
                "R6",
            ],
            "participant_id": [
                "P1",
                "P2",
                "P3",
                "P4",
                "P5",
                "P6",
            ],
            "trial_id": [
                "T1",
                "T2",
                "T3",
                "T4",
                "T5",
                "T6",
            ],
            "item_id": [
                "I1",
                "I1",
                "I1",
                "I2",
                "I2",
                "I2",
            ],
            "f1": [
                0.0,
                0.1,
                0.2,
                5.0,
                5.1,
                5.2,
            ],
            "f2": [
                0.0,
                0.2,
                0.1,
                5.0,
                5.2,
                5.1,
            ],
        }
    )

    source = {
        "features": pd.DataFrame(
            {
                "sentinel": [1],
            }
        )
    }

    monkeypatch.setattr(
        legacy,
        "_require_dataset",
        lambda x: x,
    )

    monkeypatch.setattr(
        legacy,
        "_features_wide",
        lambda x: wide.copy(),
    )

    out = legacy.fit_strategy_mixture(
        source,
        features=["f1", "f2"],
        centers=2,
        seed=20260920,
        append=False,
    )

    pd.testing.assert_frame_equal(
        out.data["features"],
        source["features"],
    )

    assert out.model.experimental is True
