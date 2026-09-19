from __future__ import annotations

import sys
import types
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.detector_multiverse as dm
import eyeprocesspy.survival as sv


def _ivt(detector_id: str = "ivt30", sampling_rate: float = 60.0):
    return ep.define_event_detector_spec(
        detector_id,
        "ivt",
        velocity_threshold=30,
        minimum_duration_ms=60,
        maximum_gap_ms=75,
        sampling_rate=sampling_rate,
        coordinate_unit="degrees",
    )


def _detector_data(seed: int = 901):
    return ep.simulate_detector_multiverse_data(
        n_participants=4,
        seed=seed,
    )


def _detector_features(seed: int = 902):
    data = _detector_data(seed)
    result = ep.run_detector_multiverse(
        data,
        [_ivt()],
    )
    result = ep.propagate_detector_to_aoi(
        result,
        overlap="first",
        continue_on_error=False,
    )
    return ep.propagate_detector_to_features(
        result,
        continue_on_error=False,
    )


def _survival_data(seed: int = 903):
    return sv.simulate_gaze_survival_example(
        seed=seed,
        n_participants=12,
        trials_per_participant=2,
    )


def test_detector_scalar_numeric_json_and_parameter_guards():
    assert dm._finite_positive(None, "x") is None
    assert dm._finite_positive(2, "x") == 2.0

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="non-empty scalar",
    ):
        dm._scalar_string("   ", "x")

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="required",
    ):
        dm._finite_positive(
            None,
            "x",
            allow_none=False,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="numeric",
    ):
        dm._finite_positive(
            object(),
            "x",
            allow_none=False,
        )

    for value in (0, -1, np.inf, -np.inf):
        with pytest.raises(
            ep.EyeProcessValidationError,
            match="finite and > 0",
        ):
            dm._finite_positive(
                value,
                "x",
                allow_none=False,
            )

    assert dm._normalise_parameters(None) == ()

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="mapping",
    ):
        dm._normalise_parameters([("x", 1)])

    assert dm._normalise_parameters({"z": 2, "a": 1}) == (
        ("a", 1),
        ("z", 2),
    )

    assert dm._jsonable(np.nan) is None
    assert dm._jsonable(np.inf) == "inf"
    assert dm._jsonable(np.int64(3)) == 3
    assert dm._jsonable({"b": np.int64(2)}) == {"b": 2}
    assert dm._jsonable((1, np.nan)) == [1, None]
    assert isinstance(dm._jsonable(object()), str)


def test_detector_spec_aliases_and_validation_guards():
    ivt = ep.define_event_detector_spec(
        "alias_ivt",
        "i-vt",
        velocity_threshold=30,
        minimum_duration_ms=60,
        sampling_rate=60,
    )

    assert ivt.algorithm == "ivt"

    idt = ep.define_event_detector_spec(
        "alias_idt",
        "i-dt",
        dispersion_threshold=1.2,
        minimum_duration_ms=60,
        sampling_rate=60,
    )

    assert idt.algorithm == "idt"

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="algorithm",
    ):
        ep.define_event_detector_spec(
            "bad",
            "unknown",
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="coordinate_unit",
    ):
        ep.define_event_detector_spec(
            "bad",
            "ivt",
            velocity_threshold=30,
            minimum_duration_ms=60,
            sampling_rate=60,
            coordinate_unit="metres",
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="maximum_gap_ms",
    ):
        ep.define_event_detector_spec(
            "bad",
            "ivt",
            velocity_threshold=30,
            minimum_duration_ms=60,
            maximum_gap_ms=0,
            sampling_rate=60,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="noise_factor",
    ):
        ep.define_event_detector_spec(
            "bad",
            "adaptive_velocity",
            minimum_duration_ms=60,
            sampling_rate=60,
            parameters={
                "minimum_velocity_threshold": 20,
            },
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="px2deg",
    ):
        ep.define_event_detector_spec(
            "bad",
            "remodnav",
            minimum_duration_ms=60,
            sampling_rate=60,
            coordinate_unit="pixels",
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="do not use",
    ):
        ep.define_event_detector_spec(
            "bad",
            "vendor",
            callback=lambda **kwargs: None,
        )


def test_detector_multiverse_grid_guards_and_parameter_dimensions():
    base = _ivt("base")

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="base_spec",
    ):
        ep.create_detector_multiverse(
            parameter_grid={
                "velocity_threshold": [20, 30],
            }
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="non-empty mapping",
    ):
        ep.create_detector_multiverse(
            base_spec=base,
            parameter_grid={},
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="at least one value",
    ):
        ep.create_detector_multiverse(
            base_spec=base,
            parameter_grid={
                "velocity_threshold": [],
            },
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="Unknown detector-grid field",
    ):
        ep.create_detector_multiverse(
            base_spec=base,
            parameter_grid={
                "mystery": [1],
            },
        )

    grid = ep.create_detector_multiverse(
        base_spec=base,
        parameter_grid={
            "velocity_threshold": [20, 40],
            "minimum_duration_ms": [50],
        },
    )

    assert len(grid.specs) == 2
    assert {spec.velocity_threshold for spec in grid.specs} == {20, 40}

    adaptive = ep.define_event_detector_spec(
        "adaptive",
        "adaptive_velocity",
        minimum_duration_ms=60,
        sampling_rate=60,
        parameters={
            "noise_factor": 4,
            "minimum_velocity_threshold": 20,
        },
    )

    parameter_grid = ep.create_detector_multiverse(
        base_spec=adaptive,
        parameter_grid={
            "noise_factor": [3, 5],
            "parameters.minimum_velocity_threshold": [15],
        },
    )

    assert {spec.parameter_dict["noise_factor"] for spec in parameter_grid.specs} == {3, 5}

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="unique",
    ):
        ep.create_detector_multiverse([base, base])

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="at least one",
    ):
        ep.create_detector_multiverse([])


def test_detector_fingerprint_sampling_and_clean_branch_paths():
    data = _detector_data(904)

    assert dm._frame_hash(None) is None
    assert dm._frame_hash(pd.DataFrame()) is None

    fingerprint = dm._dataset_fingerprint(data)
    assert fingerprint == dm._dataset_fingerprint(data)

    empty = ep.new_eye_dataset(validate=False)
    assert (
        dm._check_sampling_rate(
            empty,
            _ivt(),
        )
        == []
    )

    flat = data.copy()
    flat["gaze_samples"] = flat["gaze_samples"].copy()
    flat["gaze_samples"]["timestamp_seconds"] = 1.0

    message = dm._check_sampling_rate(
        flat,
        _ivt(),
    )

    assert message
    assert "could not be verified" in message[0]

    mismatch = dm._check_sampling_rate(
        data,
        _ivt(
            "wrong_rate",
            sampling_rate=120,
        ),
    )

    assert mismatch
    assert "differs" in mismatch[0]

    detected = dm.detect_events_with_spec(
        data,
        _ivt("clean"),
    )

    assert not detected["episodes"].empty

    cleaned = dm._clean_branch_dataset(detected)

    assert (
        not cleaned["episodes"]["episode_type"]
        .isin(["fixation", "saccade", "pursuit", "pso"])
        .any()
    )


def test_fake_remodnav_backend_and_explicit_bridge_guards(
    monkeypatch,
):
    class FakeClassifier:
        def __init__(
            self,
            px2deg,
            sampling_rate,
            min_fixation_duration,
        ):
            self.px2deg = px2deg
            self.sampling_rate = sampling_rate
            self.min_fixation_duration = min_fixation_duration

        def preproc(
            self,
            data,
        ):
            return data

        def __call__(
            self,
            data,
            classify_isp=True,
            sort_events=True,
        ):
            return [
                {
                    "label": "SACC",
                    "start_time": 0.01,
                    "end_time": 0.02,
                },
                {
                    "label": "BAD",
                    "start_time": 0.03,
                    "end_time": 0.04,
                },
                {
                    "label": "FIXA",
                    "start_time": 0.05,
                    "end_time": 0.15,
                    "start_x": 4.0,
                    "start_y": 2.0,
                    "end_x": 4.2,
                    "end_y": 2.1,
                    "amp": 0.2,
                    "peak_vel": 10.0,
                },
            ]

    fake_module = types.ModuleType("remodnav")
    fake_module.EyegazeClassifier = FakeClassifier

    monkeypatch.setitem(
        sys.modules,
        "remodnav",
        fake_module,
    )

    data = _detector_data(905)

    spec = ep.define_event_detector_spec(
        "fake_remodnav",
        "remodnav",
        minimum_duration_ms=60,
        sampling_rate=60,
        coordinate_unit="degrees",
        parameters={
            "include_labels": [
                "FIXA",
                "BAD",
            ]
        },
    )

    out = dm.detect_events_with_spec(
        data,
        spec,
    )

    events = out["episodes"]
    events = events[events["detector_id"].eq("fake_remodnav")]

    assert not events.empty
    assert set(events["episode_type"]) == {"fixation"}

    unknown = ep.define_event_detector_spec(
        "bad_parameter",
        "remodnav",
        minimum_duration_ms=60,
        sampling_rate=60,
        parameters={
            "mystery_parameter": 1,
        },
    )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="Unknown REMoDNaV",
    ):
        dm._run_remodnav(
            data,
            unknown,
        )

    normalized = ep.define_event_detector_spec(
        "normalized",
        "remodnav",
        minimum_duration_ms=60,
        sampling_rate=60,
        coordinate_unit="normalized",
        parameters={
            "px2deg": 1.0,
        },
    )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="normalized",
    ):
        dm._run_remodnav(
            data,
            normalized,
        )

    pixel = dm.EventDetectorSpec(
        detector_id="pixel",
        algorithm="remodnav",
        minimum_duration_ms=60,
        sampling_rate=60,
        coordinate_unit="pixels",
    )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="px2deg",
    ):
        dm._run_remodnav(
            data,
            pixel,
        )

    monkeypatch.setattr(
        dm.importlib.metadata,
        "version",
        lambda name: "9.9.9",
    )

    assert dm._remodnav_version() == "9.9.9"


def test_external_detector_normalization_and_trial_inference():
    data = _detector_data(906)

    spec = ep.define_event_detector_spec(
        "external",
        "external",
        callback=lambda **kwargs: pd.DataFrame(),
        implementation="unit-test",
    )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="DataFrame",
    ):
        dm.import_external_detector_events(
            [],
            spec,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="missing",
    ):
        dm.import_external_detector_events(
            pd.DataFrame(
                {
                    "recording_id": ["R1"],
                }
            ),
            spec,
        )

    trial = data["intervals"].loc[data["intervals"]["interval_type"].eq("trial")].iloc[0]

    raw = pd.DataFrame(
        {
            "recording_id": [trial.recording_id],
            "label": ["FIXA"],
            "onset": [float(trial.start_time) + 0.10],
            "duration": [0.10],
        }
    )

    normalized = dm.import_external_detector_events(
        raw,
        spec,
        dataset=data,
    )

    assert normalized.iloc[0].episode_type == "fixation"
    assert normalized.iloc[0].trial_id == trial.trial_id
    assert normalized.iloc[0].duration_ms == pytest.approx(100)
    assert normalized.iloc[0].eye == "combined"
    assert normalized.iloc[0].derived_by == "external"

    bad_time = raw.copy()
    bad_time["duration"] = -0.2

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="end_time >= start_time",
    ):
        dm.import_external_detector_events(
            bad_time,
            spec,
        )

    ambiguous_data = data.copy()
    intervals = ambiguous_data["intervals"].copy()

    duplicate = trial.copy()
    duplicate["interval_id"] = str(trial.interval_id) + "_duplicate"
    duplicate["trial_id"] = str(trial.trial_id) + "_duplicate"

    ambiguous_data["intervals"] = pd.concat(
        [
            intervals,
            pd.DataFrame([duplicate]),
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="multiple trial intervals",
    ):
        dm.import_external_detector_events(
            raw,
            spec,
            dataset=ambiguous_data,
        )


def test_external_eyedataset_vendor_and_failure_paths():
    data = _detector_data(907)

    def eye_dataset_callback(
        data,
        spec,
    ):
        return dm.detect_events_with_spec(
            data,
            _ivt("nested"),
        )

    external = ep.define_event_detector_spec(
        "external_dataset",
        "external",
        callback=eye_dataset_callback,
        implementation="unit-test",
    )

    external_branch = dm.detect_events_with_spec(
        data,
        external,
    )

    assert external_branch["episodes"]["detector_id"].eq("external_dataset").any()

    detected = dm.detect_events_with_spec(
        data,
        _ivt("vendor_seed"),
    )

    vendor_input = data.copy()
    vendor_events = detected["episodes"].copy()

    vendor_events["derived_by"] = "vendor"

    vendor_input["episodes"] = vendor_events

    vendor_spec = ep.define_event_detector_spec(
        "vendor_events",
        "vendor",
        parameters={
            "derived_by": "vendor",
        },
    )

    vendor_branch = dm.detect_events_with_spec(
        vendor_input,
        vendor_spec,
    )

    assert vendor_branch["episodes"]["detector_id"].eq("vendor_events").any()

    def broken(
        data,
        spec,
    ):
        raise RuntimeError("planned detector failure")

    broken_spec = ep.define_event_detector_spec(
        "broken",
        "external",
        callback=broken,
    )

    recorded = ep.run_detector_multiverse(
        data,
        [broken_spec],
    )

    assert recorded.status.iloc[0].status == "failed"

    with pytest.raises(
        RuntimeError,
        match="planned detector failure",
    ):
        ep.run_detector_multiverse(
            data,
            [broken_spec],
            continue_on_error=False,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="EyeDataset",
    ):
        ep.run_detector_multiverse(
            pd.DataFrame(),
            [_ivt()],
        )


def test_event_matching_empty_validation_and_mismatch_paths():
    columns = [
        "recording_id",
        "trial_id",
        "episode_type",
        "start_time",
        "end_time",
    ]

    empty = pd.DataFrame(columns=columns)

    assert dm.match_detected_events(
        empty,
        empty,
    ).empty

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="tolerance",
    ):
        dm.match_detected_events(
            empty,
            empty,
            onset_tolerance_ms=-1,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="DataFrame",
    ):
        dm.match_detected_events(
            [],
            empty,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="missing",
    ):
        dm.match_detected_events(
            pd.DataFrame(
                {
                    "recording_id": ["R1"],
                }
            ),
            empty,
        )

    reference = pd.DataFrame(
        {
            "recording_id": ["R1"],
            "trial_id": ["T1"],
            "episode_type": ["fixation"],
            "start_time": [0.0],
            "end_time": [0.1],
        }
    )

    candidate = reference.copy()
    candidate["recording_id"] = "R2"

    assert dm.match_detected_events(
        reference,
        candidate,
    ).empty

    candidate = reference.copy()
    candidate["trial_id"] = "T2"

    assert dm.match_detected_events(
        reference,
        candidate,
    ).empty

    zero = dm.compare_event_catalogues(
        empty,
        empty,
    ).iloc[0]

    assert np.isnan(zero.matched_event_precision)
    assert np.isnan(zero.matched_event_recall)
    assert np.isnan(zero.f1)


def test_detector_aoi_and_feature_failure_contracts():
    data = _detector_data(908)
    result = ep.run_detector_multiverse(
        data,
        [_ivt()],
    )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="DetectorMultiverseResult",
    ):
        ep.propagate_detector_to_aoi(data)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="overlap",
    ):
        dm._assign_episode_aois_explicit(
            result.branches["ivt30"],
            "mystery",
        )

    no_aoi = result.branches["ivt30"].copy()

    no_aoi["aoi_definitions"] = no_aoi["aoi_definitions"].iloc[0:0].copy()

    no_aoi["aoi_geometry"] = no_aoi["aoi_geometry"].iloc[0:0].copy()

    broken = replace(
        result,
        branches={
            "ivt30": no_aoi,
        },
    )

    failed = ep.propagate_detector_to_aoi(
        broken,
        continue_on_error=True,
    )

    assert failed.failures["stage"].eq("aoi_assignment").any()

    assert failed.status.iloc[0].status == "failed_aoi"

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="No AOIs",
    ):
        ep.propagate_detector_to_aoi(
            broken,
            continue_on_error=False,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="DetectorMultiverseResult",
    ):
        ep.propagate_detector_to_features(data)

    feature_failed = ep.propagate_detector_to_features(
        broken,
        continue_on_error=True,
    )

    assert feature_failed.failures["stage"].eq("feature_propagation").any()


def test_trial_feature_helper_guards():
    data = _detector_data(909)
    branch = dm.detect_events_with_spec(
        data,
        _ivt(),
    )

    no_trials = branch.copy()
    no_trials["intervals"] = no_trials["intervals"].iloc[0:0].copy()

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="trial intervals",
    ):
        dm._trial_rows(no_trials)

    missing_id = branch.copy()
    missing_id["intervals"] = missing_id["intervals"].copy()

    trial_mask = missing_id["intervals"]["interval_type"].eq("trial")

    first_index = missing_id["intervals"].index[trial_mask][0]

    missing_id["intervals"].loc[
        first_index,
        "trial_id",
    ] = pd.NA

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="non-missing trial_id",
    ):
        dm._trial_rows(missing_id)

    duplicate = branch.copy()
    intervals = duplicate["intervals"].copy()

    trial = intervals[intervals["interval_type"].eq("trial")].iloc[0]

    repeated = pd.concat(
        [
            intervals,
            pd.DataFrame([trial]),
        ],
        ignore_index=True,
    )

    duplicate["intervals"] = repeated

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="unique",
    ):
        dm._trial_rows(duplicate)

    assert (
        dm._trial_valid_fraction(
            branch,
            "not-a-recording",
            "not-a-trial",
        )[0]
        == 0
    )

    assert np.isnan(
        dm._pupil_within_fixations(
            branch,
            pd.DataFrame(),
            "x",
            "y",
        )
    )


def test_detector_inference_validation_and_callback_contracts():
    result = _detector_features(910)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="DetectorMultiverseResult",
    ):
        ep.run_detector_inference_multiverse(
            pd.DataFrame(),
            {},
        )

    empty_features = replace(
        result,
        features=pd.DataFrame(),
    )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="propagate_detector_to_features",
    ):
        ep.run_detector_inference_multiverse(
            empty_features,
            {},
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="mapping",
    ):
        ep.run_detector_inference_multiverse(
            result,
            [],
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="explicit model engine",
    ):
        ep.run_detector_inference_multiverse(
            result,
            {
                "engine": "mystery",
                "outcome": "dwell_time_ms",
            },
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="callable",
    ):
        ep.run_detector_inference_multiverse(
            result,
            {
                "engine": "callback",
                "outcome": "dwell_time_ms",
            },
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="outcome",
    ):
        ep.run_detector_inference_multiverse(
            result,
            {
                "engine": "statsmodels_ols",
                "formula": "missing ~ condition_id",
            },
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="minimum_valid_fraction",
    ):
        ep.run_detector_inference_multiverse(
            result,
            {
                "engine": "statsmodels_ols",
                "formula": "dwell_time_ms ~ C(condition_id)",
                "outcome": "dwell_time_ms",
            },
            minimum_valid_fraction=2,
        )

    def not_a_frame(
        data,
        spec,
    ):
        return []

    bad = ep.run_detector_inference_multiverse(
        result,
        {
            "engine": "callback",
            "outcome": "dwell_time_ms",
        },
        model_callback=not_a_frame,
    )

    assert bad.coefficients.empty
    assert bad.failures["error"].str.contains("DataFrame").any()

    def missing_columns(
        data,
        spec,
    ):
        return pd.DataFrame(
            {
                "term": ["x"],
            }
        )

    missing = ep.run_detector_inference_multiverse(
        result,
        {
            "engine": "callback",
            "outcome": "dwell_time_ms",
        },
        model_callback=missing_columns,
    )

    assert missing.coefficients.empty
    assert missing.failures["error"].str.contains("missing").any()


def test_statsmodels_helper_and_tidy_failure_paths():
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x": [0.0, 1.0, 2.0],
        }
    )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="formula",
    ):
        dm._fit_statsmodels(
            data,
            {
                "engine": "statsmodels_ols",
                "formula": "y",
            },
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="Unknown",
    ):
        dm._fit_statsmodels(
            data,
            {
                "engine": "unknown",
                "formula": "y ~ x",
            },
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="groups",
    ):
        dm._fit_statsmodels(
            data,
            {
                "engine": "statsmodels_mixedlm",
                "formula": "y ~ x",
                "groups": "participant",
            },
        )

    class Fit:
        params = pd.Series(
            [1.0],
            index=["x"],
        )
        bse = pd.Series(
            [0.2],
            index=["x"],
        )
        pvalues = pd.Series(
            [0.04],
            index=["x"],
        )

        def conf_int(self):
            raise RuntimeError("planned CI failure")

    tidy = dm._tidy_statsmodels(
        Fit(),
        True,
        3,
    )

    assert tidy.loc[0, "term"] == "x"
    assert np.isnan(tidy.loc[0, "CI_lower"])


def test_detector_inference_stability_edge_paths():
    multiverse = ep.create_detector_multiverse(
        [
            _ivt("a"),
            _ivt("b"),
        ]
    )

    empty = dm.DetectorInferenceResult(
        multiverse=multiverse,
        coefficients=pd.DataFrame(
            columns=[
                "term",
                "detector_id",
                "estimate",
                "CI_lower",
                "CI_upper",
                "converged",
            ]
        ),
        failures=pd.DataFrame(),
        warnings=pd.DataFrame(),
        model_spec={},
    )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="DetectorInferenceResult",
    ):
        ep.assess_detector_inference_stability(
            pd.DataFrame(),
            term="x",
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="direction",
    ):
        ep.assess_detector_inference_stability(
            empty,
            term="x",
            direction="sideways",
        )

    no_term = ep.assess_detector_inference_stability(
        empty,
        term="x",
    )

    assert no_term.iloc[0].convergence_rate == 0

    coefficients = pd.DataFrame(
        {
            "term": ["x", "x"],
            "detector_id": ["a", "b"],
            "estimate": [0.0, 0.0],
            "CI_lower": [np.nan, np.nan],
            "CI_upper": [np.nan, np.nan],
            "converged": [True, True],
        }
    )

    zero = replace(
        empty,
        coefficients=coefficients,
    )

    summary = ep.assess_detector_inference_stability(
        zero,
        term="x",
        substantive_threshold=0.5,
        direction="absolute",
    )

    assert np.isnan(summary.iloc[0].same_sign_proportion)

    assert pd.isna(summary.iloc[0].ci_overlap)

    below = coefficients.copy()
    below["estimate"] = [-2.0, -1.0]
    below["CI_lower"] = [-3.0, -2.0]
    below["CI_upper"] = [-1.0, 0.0]

    below_result = replace(
        empty,
        coefficients=below,
    )

    summary = ep.assess_detector_inference_stability(
        below_result,
        term="x",
        substantive_threshold=-0.5,
        direction="below",
    )

    assert summary.iloc[0].same_sign_proportion == 1


def test_detector_summary_plot_and_simulation_guards():
    assert dm.summarise_detector_events(pd.DataFrame()).empty

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="detector_id",
    ):
        dm.summarise_detector_events(pd.DataFrame({"episode_type": ["fixation"]}))

    result = ep.run_detector_multiverse(
        _detector_data(911),
        [_ivt()],
    )

    assert dm.estimate_detector_agreement(result).empty

    assert dm.summarise_detector_disagreement(result).empty

    result = ep.propagate_detector_to_aoi(
        result,
        overlap="first",
    )

    result = ep.propagate_detector_to_features(
        result,
    )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="term",
    ):
        dm.summarise_detector_robustness(
            result,
            inference=dm.DetectorInferenceResult(
                multiverse=result.multiverse,
                coefficients=pd.DataFrame(),
                failures=pd.DataFrame(),
                warnings=pd.DataFrame(),
                model_spec={},
            ),
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="feature",
    ):
        ep.plot_detector_feature_distributions(
            result,
            feature="not_a_feature",
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="agreement metric",
    ):
        ep.plot_detector_agreement(
            ep.run_detector_multiverse(
                _detector_data(912),
                [
                    _ivt("a"),
                    _ivt("b"),
                ],
            ),
            metric="not_a_metric",
        )

    assert dm._markdown_table(pd.DataFrame()) == ""

    escaped = dm._markdown_table(
        pd.DataFrame(
            {
                "x": [
                    "a|b\nc",
                    pd.NA,
                ]
            }
        )
    )

    assert "a\\|b c" in escaped

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="at least four",
    ):
        ep.simulate_detector_multiverse_data(n_participants=3)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="sampling_rate",
    ):
        ep.simulate_detector_multiverse_data(sampling_rate=0)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="trial_duration_s",
    ):
        ep.simulate_detector_multiverse_data(trial_duration_s=0)


# =================================================================
# SURVIVAL COVERAGE
# =================================================================


def test_survival_low_level_helper_guards(
    monkeypatch,
):
    monkeypatch.setattr(
        sv.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(sv.metadata.PackageNotFoundError),
    )

    assert sv._package_version("not-installed") is None

    with pytest.raises(
        ImportError,
        match="Optional dependency",
    ):
        sv._require_optional(
            "not-installed",
            "for testing",
        )

    with pytest.raises(
        TypeError,
        match="DataFrame",
    ):
        sv._as_dataframe(
            [],
            "data",
        )

    empty = pd.DataFrame(
        columns=[
            "start_time",
            "aoi_id",
            "episode_type",
        ]
    )

    assert (
        sv._event_time_for_trial(
            empty,
            "target",
            "first_fixation",
            "start_time",
            "aoi_id",
            "episode_type",
        )
        is None
    )

    events = pd.DataFrame(
        {
            "start_time": [
                0.1,
                0.2,
                0.3,
            ],
            "aoi_id": [
                "target",
                "target",
                "body",
            ],
            "episode_type": [
                "fixation",
                "fixation",
                "fixation",
            ],
        }
    )

    assert (
        sv._event_time_for_trial(
            events,
            "target",
            "first_revisit",
            "start_time",
            "aoi_id",
            "episode_type",
        )
        is None
    )

    assert (
        sv._event_time_for_trial(
            events.iloc[:2],
            "target",
            "first_transition_into_target",
            "start_time",
            "aoi_id",
            "episode_type",
        )
        is None
    )

    assert (
        sv._event_time_for_trial(
            events.iloc[:2],
            "target",
            "disengagement",
            "start_time",
            "aoi_id",
            "episode_type",
        )
        is None
    )

    with pytest.raises(
        ValueError,
        match="Unsupported event_type",
    ):
        sv._event_time_for_trial(
            events,
            "target",
            "mystery",
            "start_time",
            "aoi_id",
            "episode_type",
        )


def test_prepare_survival_input_validation_branches():
    with pytest.raises(
        ValueError,
        match="must contain",
    ):
        sv.prepare_gaze_survival_data(
            pd.DataFrame(
                {
                    "participant_id": ["P1"],
                }
            )
        )

    base = pd.DataFrame(
        {
            "participant_id": ["P1"],
            "trial_id": ["T1"],
            "start_time": [0.0],
            "end_time": [5.0],
            "event_observed": [0],
        }
    )

    with pytest.raises(
        ValueError,
        match="time_unit",
    ):
        sv.prepare_gaze_survival_data(
            base,
            time_unit="minutes",
        )

    with pytest.raises(
        ValueError,
        match="min_valid_fraction",
    ):
        sv.prepare_gaze_survival_data(
            base,
            min_valid_fraction=1.5,
        )

    with pytest.raises(
        ValueError,
        match="observation window",
    ):
        sv.prepare_gaze_survival_data(
            base.drop(
                columns=[
                    "start_time",
                    "end_time",
                ]
            )
        )

    with pytest.raises(
        ValueError,
        match="time_origin_col",
    ):
        sv.prepare_gaze_survival_data(
            base,
            time_origin="stimulus",
            time_origin_col="missing",
        )

    with pytest.raises(
        ValueError,
        match="non-trial-start",
    ):
        sv.prepare_gaze_survival_data(
            base,
            time_origin="stimulus",
        )

    with pytest.raises(
        ValueError,
        match="observation_end_reason_col",
    ):
        sv.prepare_gaze_survival_data(
            base,
            observation_end_reason_col="missing",
        )

    with pytest.raises(
        ValueError,
        match="valid_observation_col",
    ):
        sv.prepare_gaze_survival_data(
            base,
            valid_observation_col="missing",
        )

    invalid_event = base.copy()
    invalid_event["event_observed"] = 2

    with pytest.raises(
        ValueError,
        match="0/1",
    ):
        sv.prepare_gaze_survival_data(
            invalid_event,
        )

    with pytest.raises(
        ValueError,
        match="target_aoi",
    ):
        sv.prepare_gaze_survival_data(
            base.drop(columns="event_observed"),
            pd.DataFrame(
                {
                    "trial_id": ["T1"],
                    "start_time": [1.0],
                    "aoi_id": ["target"],
                }
            ),
        )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        sv.prepare_gaze_survival_data(
            base.drop(columns="event_observed"),
            pd.DataFrame(
                {
                    "trial_id": ["T1"],
                }
            ),
            target_aoi="target",
        )


def test_prepare_survival_explicit_censor_and_trial_only_join():
    supplied = pd.DataFrame(
        {
            "participant_id": [
                "P1",
                "P2",
            ],
            "trial_id": [
                "A",
                "B",
            ],
            "censor_time": [
                4.0,
                5.0,
            ],
            "trial_duration": [
                4.0,
                5.0,
            ],
            "event_observed": [
                0,
                1,
            ],
            "event_time": [
                np.nan,
                2.0,
            ],
            "target_aoi": [
                "target",
                "target",
            ],
            "condition_id": [
                "c1",
                "c2",
            ],
            "observation_reason": [
                "timeout",
                "complete",
            ],
            "usable": [
                True,
                False,
            ],
        }
    )

    with pytest.warns(RuntimeWarning):
        out = sv.prepare_gaze_survival_data(
            supplied,
            observation_end_reason_col=("observation_reason"),
            valid_observation_col="usable",
            time_origin="custom_origin",
            source_data="synthetic",
            preprocessing_specification="none",
            event_detector="supplied_events",
            quality_rules={
                "declared": True,
            },
        )

    assert (
        out.loc[
            0,
            "observation_end_reason",
        ]
        == "timeout"
    )

    assert (
        out.loc[
            1,
            "censor_reason",
        ]
        == "invalid_observation_flag"
    )

    assert bool(
        out.loc[
            1,
            "review_required",
        ]
    )

    assert not bool(
        out.loc[
            1,
            "analysis_eligible",
        ]
    )

    assert pd.isna(
        out.loc[
            1,
            "event_observed",
        ]
    )

    assert (
        out.loc[
            0,
            "condition",
        ]
        == "c1"
    )

    trials = pd.DataFrame(
        {
            "participant_id": [
                "P1",
                "P2",
            ],
            "trial_id": [
                "unique_a",
                "unique_b",
            ],
            "start_time": [0.0, 0.0],
            "end_time": [5.0, 5.0],
        }
    )

    events = pd.DataFrame(
        {
            "trial_id": ["unique_a"],
            "start_time": [1.0],
            "aoi_id": ["target"],
            "episode_type": ["fixation"],
        }
    )

    with pytest.warns(RuntimeWarning):
        joined = sv.prepare_gaze_survival_data(
            trials,
            events,
            target_aoi="target",
        )

    assert set(joined["event_join_key"]) == {"trial_only"}


def test_validate_survival_all_major_issue_codes():
    base = _survival_data(913)

    duplicate = pd.concat(
        [
            base,
            base.iloc[[0]],
        ],
        ignore_index=True,
    )

    issues = sv.validate_gaze_survival_data(
        duplicate,
        raise_on_error=False,
    )

    assert "duplicated_trials" in set(issues.code)

    cases = []

    invalid_indicator = base.copy()
    invalid_indicator.loc[
        0,
        "event_observed",
    ] = 2
    cases.append(
        (
            invalid_indicator,
            "invalid_event_indicator",
        )
    )

    negative = base.copy()
    negative.loc[
        0,
        "analysis_time",
    ] = -1
    cases.append(
        (
            negative,
            "negative_time",
        )
    )

    missing_window = base.copy()
    missing_window.loc[
        0,
        "censor_time",
    ] = np.nan
    cases.append(
        (
            missing_window,
            "missing_observation_window",
        )
    )

    missing_event = base.copy()
    missing_event.loc[
        0,
        "event_observed",
    ] = 1
    missing_event.loc[
        0,
        "event_time",
    ] = np.nan
    cases.append(
        (
            missing_event,
            "observed_missing_event_time",
        )
    )

    event_mismatch = base.copy()
    event_index = event_mismatch.index[event_mismatch["event_observed"].eq(1)][0]

    event_mismatch.loc[
        event_index,
        "analysis_time",
    ] += 0.25

    cases.append(
        (
            event_mismatch,
            "analysis_time_event_mismatch",
        )
    )

    censor_mismatch = base.copy()
    censor_index = censor_mismatch.index[censor_mismatch["event_observed"].eq(0)][0]

    censor_mismatch.loc[
        censor_index,
        "analysis_time",
    ] -= 0.25

    cases.append(
        (
            censor_mismatch,
            "analysis_time_censor_mismatch",
        )
    )

    zero_follow = base.copy()
    zero_follow.loc[
        censor_index,
        [
            "censor_time",
            "analysis_time",
        ],
    ] = 0.0

    issues = sv.validate_gaze_survival_data(
        zero_follow,
        raise_on_error=False,
    )

    assert "zero_followup_censoring" in set(issues.code)

    invalid_fraction = base.copy()
    invalid_fraction.loc[
        0,
        "valid_data_fraction",
    ] = 1.5

    cases.append(
        (
            invalid_fraction,
            "invalid_valid_fraction",
        )
    )

    for frame, expected in cases:
        issues = sv.validate_gaze_survival_data(
            frame,
            raise_on_error=False,
        )

        assert expected in set(issues.code)

    with pytest.raises(ValueError):
        sv.validate_gaze_survival_data(invalid_fraction)

    missing_columns = sv.validate_gaze_survival_data(
        pd.DataFrame(),
        raise_on_error=False,
    )

    assert "missing_columns" in set(missing_columns.code)


def test_survival_analysis_and_descriptive_guards():
    data = _survival_data(914)

    review = data.copy()
    review.loc[
        0,
        "event_observed",
    ] = np.nan
    review.loc[
        0,
        "analysis_time",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="Non-analyzable",
    ):
        sv._analysis_rows(review)

    ineligible = data.copy()
    ineligible["analysis_eligible"] = True
    ineligible.loc[
        0,
        "analysis_eligible",
    ] = False

    with pytest.raises(
        ValueError,
        match="analysis_eligible",
    ):
        sv._analysis_rows(ineligible)

    with pytest.raises(
        ValueError,
        match="Grouping column",
    ):
        sv.summarise_gaze_censoring(
            data,
            by="missing",
        )

    grouped = sv.summarise_gaze_censoring(
        data,
        by="condition",
    )

    assert len(grouped) >= 2

    with pytest.raises(
        ValueError,
        match="conf_level",
    ):
        sv.estimate_gaze_survival(
            data,
            conf_level=1,
        )

    with pytest.raises(
        ValueError,
        match="Group column",
    ):
        sv.estimate_gaze_survival(
            data,
            group="missing",
        )

    curve = sv.estimate_gaze_survival(
        data,
        group=None,
    )

    assert set(curve["group"]) == {"all"}


def test_survival_design_and_model_selection_guards():
    data = _survival_data(915)

    with pytest.raises(
        ValueError,
        match="predictor",
    ):
        sv._design_matrix(
            "",
            data,
        )

    with pytest.raises(
        ValueError,
        match="estimable covariate",
    ):
        sv._design_matrix(
            "1",
            data,
        )

    with pytest.raises(
        ValueError,
        match="ties",
    ):
        sv.fit_gaze_cox_model(
            data,
            "C(condition)",
            ties="mystery",
        )

    with pytest.raises(
        ValueError,
        match="Cluster column",
    ):
        sv.fit_gaze_cox_model(
            data,
            "C(condition)",
            cluster="missing",
        )

    with pytest.raises(
        ValueError,
        match="structure",
    ):
        sv.fit_gaze_mixed_cox_model(
            data,
            "C(condition)",
            structure="unknown",
        )

    with pytest.raises(
        ValueError,
        match="distribution",
    ):
        sv.fit_gaze_aft_model(
            data,
            "C(condition)",
            distribution="gamma",
        )

    with pytest.raises(
        ValueError,
        match="maxiter",
    ):
        sv.fit_gaze_aft_model(
            data,
            "C(condition)",
            distribution="weibull",
            maxiter=0,
        )

    with pytest.raises(
        ValueError,
        match="predictor",
    ):
        sv.fit_gaze_aft_model(
            data,
            "",
            distribution="weibull",
        )


def test_survival_tidy_ph_and_comparison_guards():
    data = _survival_data(916)

    fake = sv.GazeSurvivalFit(
        model_family="unsupported",
        backend="fake",
        result=SimpleNamespace(),
        data=data,
    )

    with pytest.raises(
        ValueError,
        match="conf_level",
    ):
        sv.tidy_gaze_survival_model(
            fake,
            conf_level=0,
        )

    with pytest.raises(
        TypeError,
        match="Unsupported",
    ):
        sv.tidy_gaze_survival_model(fake)

    with pytest.raises(
        TypeError,
        match="Cox",
    ):
        sv.check_gaze_proportional_hazards(fake)

    cox_data = data.iloc[:3].copy()

    cox_data["event_observed"] = [1, 1, 0]

    cox_data["analysis_time"] = [1.0, 2.0, 3.0]

    cox_data["event_time"] = [1.0, 2.0, np.nan]

    cox_data["censor_time"] = [4.0, 4.0, 3.0]

    fake_cox = sv.GazeSurvivalFit(
        model_family="cox",
        backend="fake",
        result=SimpleNamespace(
            schoenfeld_residuals=np.array(
                [
                    [0.1],
                    [0.2],
                    [0.3],
                ]
            )
        ),
        data=cox_data,
        covariate_names=["x"],
    )

    with pytest.raises(
        ValueError,
        match="alpha",
    ):
        sv.check_gaze_proportional_hazards(
            fake_cox,
            alpha=2,
        )

    diagnostic = sv.check_gaze_proportional_hazards(fake_cox)

    assert np.isnan(
        diagnostic.loc[
            0,
            "rho",
        ]
    )

    with pytest.raises(
        ValueError,
        match="one or more",
    ):
        sv.compare_gaze_survival_models()

    with pytest.raises(
        TypeError,
        match="GazeSurvivalFit",
    ):
        sv.compare_gaze_survival_models(object())


def test_survival_prediction_and_quantile_validation():
    data = _survival_data(917)

    cox = sv.fit_gaze_cox_model(
        data,
        "C(condition)",
    )

    prediction = sv.predict_gaze_survival(
        cox,
        data.iloc[[0]],
        times=[
            0.0,
            0.5,
            3.0,
        ],
    )

    assert len(prediction) == 3
    assert prediction.survival.between(
        0,
        1,
    ).all()

    cox_quantiles = sv.estimate_gaze_latency_quantiles(
        cox,
        probs=[
            0.25,
            0.5,
        ],
    )

    assert len(cox_quantiles) == 2

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        sv.predict_gaze_survival(
            cox,
            data.iloc[[0]],
            times=[],
        )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        sv.predict_gaze_survival(
            cox,
            data.iloc[[0]],
            times=[-1],
        )

    unsupported = sv.GazeSurvivalFit(
        model_family="other",
        backend="fake",
        result=SimpleNamespace(),
        data=data,
    )

    with pytest.raises(
        TypeError,
        match="Unsupported",
    ):
        sv.predict_gaze_survival(
            unsupported,
            data.iloc[[0]],
            times=[1],
        )

    with pytest.raises(
        ValueError,
        match="probs",
    ):
        sv.estimate_gaze_latency_quantiles(
            data,
            probs=[],
        )

    with pytest.raises(
        ValueError,
        match="strictly between",
    ):
        sv.estimate_gaze_latency_quantiles(
            data,
            probs=[0],
        )

    with pytest.raises(
        TypeError,
        match="DataFrame or GazeSurvivalFit",
    ):
        sv.estimate_gaze_latency_quantiles(object())


def test_aft_prediction_quantiles_and_contract_failure():
    data = _survival_data(918)

    aft = sv.fit_gaze_aft_model(
        data,
        "C(condition)",
        distribution="weibull",
    )

    prediction = sv.predict_gaze_survival(
        aft,
        data.iloc[[0]],
        times=[0.5, 1.0],
    )

    assert len(prediction) == 2

    quantiles = sv.estimate_gaze_latency_quantiles(
        aft,
        probs=[0.5],
        newdata=data.iloc[[0]],
    )

    assert len(quantiles) == 1

    class BadAFT:
        def predict_survival_function(
            self,
            new,
            times,
        ):
            return pd.DataFrame(
                np.ones(
                    (
                        len(times),
                        len(new) + 1,
                    )
                )
            )

        def predict_percentile(
            self,
            new,
            p,
        ):
            return np.ones(len(new) + 1)

    bad = sv.GazeSurvivalFit(
        model_family="aft_weibull",
        backend="fake",
        result=BadAFT(),
        data=data,
    )

    with pytest.raises(
        RuntimeError,
        match="prediction contract",
    ):
        sv.predict_gaze_survival(
            bad,
            data.iloc[[0]],
            times=[1],
        )

    with pytest.raises(
        RuntimeError,
        match="quantile contract",
    ):
        sv.estimate_gaze_latency_quantiles(
            bad,
            probs=[0.5],
            newdata=data.iloc[[0]],
        )


def test_survival_plot_and_specification_guards():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _survival_data(919)

    with pytest.raises(
        ValueError,
        match="Group column",
    ):
        sv.plot_gaze_hazard(
            data,
            group="missing",
        )

    non_cox = sv.GazeSurvivalFit(
        model_family="aft_weibull",
        backend="fake",
        result=SimpleNamespace(),
        data=data,
    )

    with pytest.raises(
        TypeError,
        match="Cox",
    ):
        sv.plot_gaze_cox_diagnostics(non_cox)

    zero_residual = sv.GazeSurvivalFit(
        model_family="cox",
        backend="fake",
        result=SimpleNamespace(
            schoenfeld_residuals=np.empty(
                (
                    len(data),
                    0,
                )
            )
        ),
        data=data,
        covariate_names=[],
    )

    ax = sv.plot_gaze_cox_diagnostics(zero_residual)

    assert ax is not None

    plt.close("all")

    with pytest.raises(
        ValueError,
        match="non-empty mapping",
    ):
        sv.compare_gaze_survival_specifications(
            {},
            "C(condition)",
            model_families=["cox"],
        )

    with pytest.raises(
        ValueError,
        match="model_families",
    ):
        sv.compare_gaze_survival_specifications(
            {
                "x": data,
            },
            "C(condition)",
            model_families=[],
        )

    with pytest.raises(
        ValueError,
        match="model_families",
    ):
        sv.compare_gaze_survival_specifications(
            {
                "x": data,
            },
            "C(condition)",
            model_families=[
                "automatic",
            ],
        )


def test_survival_simulation_guards():
    with pytest.raises(
        ValueError,
        match="kind",
    ):
        sv.simulate_gaze_survival_inputs("unknown")

    with pytest.raises(
        ValueError,
        match="positive integers",
    ):
        sv.simulate_gaze_survival_inputs(n_participants=0)

    with pytest.raises(
        ValueError,
        match="positive integers",
    ):
        sv.simulate_gaze_survival_inputs(trials_per_participant=0)
