from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy.validation_stress_freeze_09 as vs
from eyeprocesspy.exceptions import EyeProcessValidationError


class _NonIterable:
    def __iter__(self):
        raise TypeError("not iterable")


class _NamedScalar:
    name = "metric"

    def __array__(self, dtype=None, copy=None):
        del copy
        return np.asarray(1.5, dtype=dtype)


class _Exotic:
    def __repr__(self):
        return "<exotic>"


def _small_plan(seed=7):
    return vs.eyeprocess_stress_evidence_plan(
        missing_gaze=[0.0],
        pupil_dropout=[0.0],
        calibration_offset=[0.0],
        sampling_jitter=[0.0],
        aoi_label_noise=[0.0],
        device_shift=[0.0],
        trial_imbalance=[0.0],
        seed=seed,
    )


def _identity_corruptors():
    def identity(data, severity, seed):
        del severity, seed
        return data

    return {name: identity for name in vs._STRESS_FIELDS}


def _full_freeze():
    claims = vs.eyeprocess_validation_claim_matrix(
        "C1", "claim", "E1", "test", status="supported"
    )
    return vs.freeze_eyeprocess_validation_evidence(
        design={"id": 1},
        recovery={"rmse": 0.1},
        stress={"stable": True},
        reliability={"icc": 0.9},
        negative_controls={"pass": True},
        claims=claims,
        provenance={"commit": "abc"},
        source_commit="abc",
    )


def test_private_collection_numeric_and_scalar_helpers(tmp_path):
    assert vs._as_list(None) == []
    assert vs._as_list("x") == ["x"]
    assert vs._as_list(tmp_path / "x") == [tmp_path / "x"]
    assert vs._as_list(pd.Series([1, 2])) == [1, 2]
    assert vs._as_list(3.0) == [3.0]
    assert vs._as_list((1, 2)) == [1, 2]
    wrapped = vs._as_list(_NonIterable())
    assert len(wrapped) == 1 and isinstance(wrapped[0], _NonIterable)

    assert vs._unique([1, 1, 2, 1]) == [1, 2]
    assert vs._numeric_vector([0, 0, 1], name="x") == [0.0, 1.0]
    for bad in ([], [np.nan], [np.inf], [-0.1], ["bad"]):
        with pytest.raises(EyeProcessValidationError, match="finite non-negative"):
            vs._numeric_vector(bad, name="x")

    assert isinstance(vs._now_utc_string(), str)
    assert vs._clean_scalar(np.int64(2)) == 2
    assert vs._clean_scalar(Path("a/b")) == "a/b"
    assert vs._clean_scalar(date(2026, 1, 2)) == "2026-01-02"
    assert vs._clean_scalar(datetime(2026, 1, 2, tzinfo=timezone.utc)).startswith(
        "2026-01-02"
    )
    assert vs._clean_scalar(pd.NA) is None
    assert vs._clean_scalar(np.nan) is None
    assert vs._clean_scalar(np.inf) is None
    assert vs._clean_scalar("x") == "x"


def test_jsonify_restore_all_supported_and_fallback_types(tmp_path):
    frame = pd.DataFrame({"a": [1, np.nan], "b": ["x", "y"]}, index=["r1", "r2"])
    frame.attrs["source"] = Path("evidence/file.csv")
    series = pd.Series([1, 2], index=["a", "b"], name="s")
    array = np.array([[1, 2], [3, 4]])

    payload = {
        "frame": frame,
        "series": series,
        "array": array,
        "tuple": (Path("x"), pd.NA, np.float64(2.5)),
        "mapping": {"when": date(2026, 2, 3)},
        "exotic": _Exotic(),
    }
    encoded = vs._jsonify(payload)
    assert encoded["frame"]["__eyeprocess_type__"] == "dataframe"
    assert encoded["series"]["__eyeprocess_type__"] == "series"
    assert encoded["array"]["__eyeprocess_type__"] == "ndarray"
    assert encoded["exotic"]["repr"] == "<exotic>"

    restored = vs._restore_json(encoded)
    assert isinstance(restored["frame"], pd.DataFrame)
    assert restored["frame"].index.tolist() == ["r1", "r2"]
    assert restored["frame"].attrs["source"] == "evidence/file.csv"
    assert isinstance(restored["series"], pd.Series)
    np.testing.assert_array_equal(restored["array"], array)
    assert restored["tuple"][0] == "x"
    assert restored["tuple"][1] is None

    assert vs._restore_json([{"x": 1}]) == [{"x": 1}]
    assert vs._restore_json(3) == 3
    assert vs._restore_json({"unknown": {"x": 1}}) == {"unknown": {"x": 1}}

    malformed = {
        "__eyeprocess_type__": "ndarray",
        "data": [1, 2, 3],
        "shape": [2, 2],
    }
    malformed_restored = vs._restore_json(malformed)
    assert malformed_restored.shape == (3,)


def test_stress_plan_seed_and_expansion_validation_edges():
    for bad_seed in ("bad", None, 0, -1):
        with pytest.raises(EyeProcessValidationError, match="seed"):
            vs.eyeprocess_stress_evidence_plan(seed=bad_seed)

    with pytest.raises(EyeProcessValidationError, match="missing_gaze"):
        vs.eyeprocess_stress_evidence_plan(missing_gaze=[])
    with pytest.raises(EyeProcessValidationError, match="pupil_dropout"):
        vs.eyeprocess_stress_evidence_plan(pupil_dropout=[np.inf])
    with pytest.raises(EyeProcessValidationError, match=r"\[0,1\)"):
        vs.eyeprocess_stress_evidence_plan(trial_imbalance=[1.0])

    with pytest.raises(EyeProcessValidationError, match="eye_stress_evidence_plan"):
        vs.expand_eyeprocess_stress_evidence_plan({})

    plan = _small_plan(seed=2)
    grid = vs.expand_eyeprocess_stress_evidence_plan(plan)
    assert len(grid) == 7
    assert grid["scenario_id"].iloc[-1] == "STRESS007"


def test_reliability_and_negative_control_plan_validation_edges():
    with pytest.raises(EyeProcessValidationError, match="unsupported reliability"):
        vs.eyeprocess_reliability_evidence_plan(metrics=[])
    with pytest.raises(EyeProcessValidationError, match="invalid bootstrap/seed"):
        vs.eyeprocess_reliability_evidence_plan(bootstrap="bad")
    with pytest.raises(EyeProcessValidationError, match="invalid bootstrap/seed"):
        vs.eyeprocess_reliability_evidence_plan(seed="bad")
    with pytest.raises(EyeProcessValidationError, match="invalid bootstrap/seed"):
        vs.eyeprocess_reliability_evidence_plan(bootstrap=-1)
    with pytest.raises(EyeProcessValidationError, match="invalid bootstrap/seed"):
        vs.eyeprocess_reliability_evidence_plan(seed=0)

    rel = vs.eyeprocess_reliability_evidence_plan(
        metrics=pd.Series(["icc", "icc", "split_half"]), bootstrap=0, seed=1
    )
    assert rel["metrics"] == ["icc", "split_half"]

    with pytest.raises(EyeProcessValidationError, match="unsupported negative"):
        vs.eyeprocess_negative_control_evidence_plan(controls=[])
    with pytest.raises(EyeProcessValidationError, match="positive scalar integers"):
        vs.eyeprocess_negative_control_evidence_plan(replications="bad")
    with pytest.raises(EyeProcessValidationError, match="positive scalar integers"):
        vs.eyeprocess_negative_control_evidence_plan(seed=None)
    with pytest.raises(EyeProcessValidationError, match="positive scalar integers"):
        vs.eyeprocess_negative_control_evidence_plan(replications=0)
    with pytest.raises(EyeProcessValidationError, match="positive scalar integers"):
        vs.eyeprocess_negative_control_evidence_plan(seed=0)

    neg = vs.eyeprocess_negative_control_evidence_plan(
        controls=["permutation", "permutation", "temporal_shift"],
        replications=1,
        seed=1,
    )
    assert neg["controls"] == ["permutation", "temporal_shift"]


def test_recycle_character_and_claim_matrix_missing_text_guards():
    assert vs._recycle(["a", "b"], 5) == ["a", "b", "a", "b", "a"]
    values = vs._as_character_values(
        [None, pd.NA, np.nan, 2],
        4,
    )
    assert values == [None, None, None, "2"]

    with pytest.raises(EyeProcessValidationError, match="non-empty"):
        vs.eyeprocess_validation_claim_matrix([], "claim", "E1", "test")

    with pytest.raises(EyeProcessValidationError, match="claim_id"):
        vs.eyeprocess_validation_claim_matrix(
            [None, "C2"], ["a", "b"], ["E1", "E2"], "test"
        )
    with pytest.raises(EyeProcessValidationError, match="claim_id"):
        vs.eyeprocess_validation_claim_matrix(
            ["", "C2"], ["a", "b"], ["E1", "E2"], "test"
        )

    for kwargs in (
        {"claim": None},
        {"evidence_id": ""},
        {"evidence_type": np.nan},
    ):
        base = {
            "claim_id": "C1",
            "claim": "claim",
            "evidence_id": "E1",
            "evidence_type": "test",
        }
        base.update(kwargs)
        with pytest.raises(EyeProcessValidationError, match="non-empty"):
            vs.eyeprocess_validation_claim_matrix(**base)

    bounded = vs.eyeprocess_validation_claim_matrix(
        ["C1", "C2"],
        "claim",
        ["E1", "E2"],
        "test",
        boundary=["b1", "b2"],
    )
    assert bounded["boundary"].tolist() == ["b1", "b2"]


def test_manifest_sequence_objects_empty_forms_and_source_commit(tmp_path):
    first = tmp_path / "one.txt"
    first.write_text("one", encoding="utf-8")

    manifest = vs.eyeprocess_validation_evidence_manifest(
        files=first,
        objects=[pd.DataFrame({"x": [1]}), {"y": 2}],
        source_commit=None,
        label=123,
    )
    assert manifest["label"] == "123"
    assert manifest["source_commit"] is None
    assert manifest["objects"]["name"].tolist() == ["object_1", "object_2"]
    assert len(manifest["files"]) == 1

    empty = vs.eyeprocess_validation_evidence_manifest()
    assert empty["files"].empty
    assert empty["objects"].empty


def test_freeze_verify_write_read_invalid_class_and_read_failures(tmp_path):
    with pytest.raises(EyeProcessValidationError, match="frozen validation evidence"):
        vs.verify_eyeprocess_validation_evidence({})
    with pytest.raises(EyeProcessValidationError, match="frozen validation evidence"):
        vs.write_eyeprocess_validation_evidence({}, tmp_path / "x.json")

    frozen = vs.freeze_eyeprocess_validation_evidence(design={"id": 1})
    assert frozen["source_commit"] is None

    path = tmp_path / "nested" / "freeze.json"
    vs.write_eyeprocess_validation_evidence(frozen, path)
    assert path.exists()

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    with pytest.raises(EyeProcessValidationError, match="Could not read"):
        vs.read_eyeprocess_validation_evidence(bad_json)

    missing = tmp_path / "missing.json"
    with pytest.raises(EyeProcessValidationError, match="Could not read"):
        vs.read_eyeprocess_validation_evidence(missing)

    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"eyeprocess_class": "wrong"}), encoding="utf-8")
    with pytest.raises(EyeProcessValidationError, match="not an eyeprocess"):
        vs.read_eyeprocess_validation_evidence(wrong)

    encoded = vs._jsonify(frozen)
    encoded["hash"] = "broken"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(encoded), encoding="utf-8")

    with pytest.raises(EyeProcessValidationError, match="hash verification failed"):
        vs.read_eyeprocess_validation_evidence(tampered, verify=True)

    unchecked = vs.read_eyeprocess_validation_evidence(tampered, verify=False)
    assert unchecked["hash"] == "broken"
    assert not vs.verify_eyeprocess_validation_evidence(unchecked)


def test_readiness_and_release_gate_invalid_inputs_hash_and_acceptance_edges():
    with pytest.raises(EyeProcessValidationError, match="frozen validation evidence"):
        vs.eyeprocess_validation_readiness({})

    frozen = _full_freeze()
    readiness = vs.eyeprocess_validation_readiness(
        frozen,
        required=pd.Series(["design", "design", "claims"]),
    )
    assert readiness["ready"] is True
    assert readiness["table"]["requirement"].tolist() == ["design", "claims"]

    with pytest.raises(EyeProcessValidationError, match="readiness must"):
        vs.eyeprocess_validation_release_gate({})

    with pytest.raises(EyeProcessValidationError, match="data.frame"):
        vs.eyeprocess_validation_release_gate(readiness, acceptance=True)
    with pytest.raises(EyeProcessValidationError, match="missing required"):
        vs.eyeprocess_validation_release_gate(
            readiness, acceptance=pd.DataFrame({"status": [True]})
        )

    for acceptance in (
        pd.DataFrame({"pass": []}),
        pd.DataFrame({"pass": [pd.NA]}),
        pd.DataFrame({"pass": [True, False]}),
    ):
        gate = vs.eyeprocess_validation_release_gate(readiness, acceptance=acceptance)
        assert gate["pass"] is False
        assert gate["acceptance"] is False

    tampered = _full_freeze()
    tampered["hash"] = "bad"
    tampered_readiness = vs.eyeprocess_validation_readiness(tampered)
    assert tampered_readiness["ready"] is True
    assert tampered_readiness["hash_valid"] is False
    assert vs.eyeprocess_validation_release_gate(tampered_readiness)["pass"] is False
    assert (
        vs.eyeprocess_validation_release_gate(
            tampered_readiness,
            acceptance=pd.DataFrame({"pass": [True]}),
            require_hash=False,
        )["pass"]
        is True
    )


def test_metric_output_coercion_series_mapping_named_scalar_and_errors():
    assert vs._coerce_metric_output(pd.Series([1, 2], index=["a", "b"])) == {
        "a": 1.0,
        "b": 2.0,
    }
    assert vs._coerce_metric_output({"a": np.nan})["a"] != vs._coerce_metric_output(
        {"a": np.nan}
    )["a"]
    assert vs._coerce_metric_output(_NamedScalar()) == {"metric": 1.5}

    for bad in (
        {},
        {"": 1},
        pd.Series([1, 2], index=["a", "a"]),
        np.array([1.0]),
    ):
        with pytest.raises(EyeProcessValidationError, match="uniquely named"):
            vs._coerce_metric_output(bad)

    with pytest.raises(EyeProcessValidationError, match="numeric"):
        vs._coerce_metric_output({"a": "bad"})
    with pytest.raises(EyeProcessValidationError, match="infinite"):
        vs._coerce_metric_output({"a": np.inf})


def test_stress_executor_argument_validation_and_metric_failure_paths():
    plan = _small_plan()

    with pytest.raises(EyeProcessValidationError, match="eye_stress_evidence_plan"):
        vs.run_eyeprocess_stress_evidence({}, {}, {}, lambda x: {"m": 1})

    for corruptors in (
        {},
        {"missing_gaze": 1},
    ):
        with pytest.raises(EyeProcessValidationError, match="corruptors"):
            vs.run_eyeprocess_stress_evidence({}, plan, corruptors, lambda x: {"m": 1})

    with pytest.raises(EyeProcessValidationError, match="metric_fun must be a function"):
        vs.run_eyeprocess_stress_evidence({}, plan, _identity_corruptors(), 1)

    def corrupt(data, severity, seed):
        del severity, seed
        data = dict(data)
        if data.get("mode") == "baseline":
            data["stage"] = "after"
        return data

    controls = _identity_corruptors()
    controls["missing_gaze"] = lambda data, severity, seed: {
        **dict(data),
        "failure": True,
    }
    controls["pupil_dropout"] = lambda data, severity, seed: {
        **dict(data),
        "changed": True,
    }

    def metric(data):
        if data.get("failure"):
            raise RuntimeError("metric failure")
        if data.get("changed"):
            return {"other": 2.0}
        return {"m": 1.0}

    result = vs.run_eyeprocess_stress_evidence(
        {"mode": "baseline"},
        plan,
        controls,
        metric,
    )
    errors = result["failures"]["error"].tolist()
    assert "metric failure" in errors
    assert "metric_fun names changed after corruption" in errors


def test_stress_executor_zero_nonfinite_and_summary_empty_delta_paths():
    plan = _small_plan()
    result = vs.run_eyeprocess_stress_evidence(
        {"x": 1},
        plan,
        _identity_corruptors(),
        lambda data: {"zero": 0.0, "nan": np.nan},
    )
    zero = result["results"].query("metric == 'zero'")
    assert zero["relative_change"].isna().all()
    nan_rows = result["results"].query("metric == 'nan'")
    assert nan_rows["delta"].isna().all()

    with pytest.raises(EyeProcessValidationError, match="eye_stress_evidence_result"):
        vs.summarise_eyeprocess_stress_evidence({})

    empty = vs._tag(
        {"results": pd.DataFrame()},
        vs._STRESS_RESULT_CLASS,
    )
    assert vs.summarise_eyeprocess_stress_evidence(empty).empty

    all_nan = vs._tag(
        {
            "results": pd.DataFrame(
                {
                    "corruption": ["missing_gaze"],
                    "metric": ["m"],
                    "severity": [0.1],
                    "delta": [np.nan],
                }
            )
        },
        vs._STRESS_RESULT_CLASS,
    )
    summary = vs.summarise_eyeprocess_stress_evidence(all_nan)
    assert np.isnan(summary.loc[0, "mean_delta"])
    assert np.isnan(summary.loc[0, "max_abs_delta"])
