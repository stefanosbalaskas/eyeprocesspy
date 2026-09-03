from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.engine_adapters as ea
from eyeprocesspy.exceptions import EyeProcessBackendError, EyeProcessValidationError
from eyeprocesspy.irt import EyeResult


def _adapter_result(**overrides):
    values = {
        "status": "not_available",
        "engine": "mirt",
        "purpose": "test",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "fit": None,
    }
    values.update(overrides)
    return EyeResult(values, eyeprocess_class="eye_engine_adapter_result")


def _sequence_dataset():
    x = ep.simulate_eye_dataset(
        n_person=2,
        n_item=2,
        sampling_rate=10,
        trial_duration=0.3,
        seed=911,
    )
    return x


def test_private_text_tag_api_version_and_schema_inference_edges():
    with pytest.raises(EyeProcessValidationError, match="engine must be"):
        ea._text(None, "engine")
    with pytest.raises(EyeProcessValidationError, match="scientific `purpose`"):
        ea._text("   ", "purpose")
    assert ea._text("  mirt  ", "engine") == "mirt"

    frame = pd.DataFrame({"x": [1]})
    tagged = ea._tag(frame, "demo")
    assert tagged is not frame
    assert tagged.attrs["eyeprocess_class"] == "demo"

    version = ea.eyeprocess_api_version()
    assert str(version) == "0.5.0"
    assert version.storage_schema == "2.0.0"
    assert version.model_contract == "1.0.0"

    x = _sequence_dataset()
    assert ea.object_schema(x)["class"] == "eye_dataset"

    dynamic = EyeResult(
        {"spec": {"engine": "stan"}, "fit": {}},
        eyeprocess_class="eye_dynamic_irtree",
    )
    assert ea.object_schema(dynamic)["class"] == "eyeprocess_model"
    assert ea.object_schema({"fit": {}})["class"] == "eyeprocess_model"
    assert ea.object_schema({"model": {}})["class"] == "eyeprocess_model"

    assert ea.object_schema("validation_plan")["class"] == "eye_validation_job_plan"
    assert ea.object_schema("validation_collection")["class"] == "eye_validation_collection"
    assert ea.object_schema("vendor_corpus")["class"] == "eye_vendor_corpus"
    assert ea.object_schema("eye_storage")["class"] == "eye_partitioned_storage"

    with pytest.raises(EyeProcessValidationError, match="infer"):
        ea.object_schema(object())
    with pytest.raises(EyeProcessValidationError, match="Unknown object schema"):
        ea.object_schema("not-a-schema")


def test_validate_model_object_nonmapping_recognized_and_strict_paths():
    invalid = ea.validate_model_object(42)
    assert invalid["valid"] is False
    assert invalid["findings"].loc[0, "check"] == "list_like"
    with pytest.raises(EyeProcessValidationError, match="list-like"):
        ea.validate_model_object(42, strict=True)

    unrecognized = EyeResult({}, eyeprocess_class="mystery")
    checked = ea.validate_model_object(unrecognized)
    assert checked["valid"] is False
    assert checked["model_family"] is None

    complete = EyeResult(
        {
            "spec": {"engine": "reference", "interpretation": "validation only"},
            "model": {"diagnostics": {"ok": True}},
        },
        eyeprocess_class="eyeprocess_model",
    )
    valid = ea.validate_model_object(complete)
    assert valid["valid"] is True
    assert valid["model_family"] == "generic"
    assert valid["findings"]["passed"].all()

    warning_only = EyeResult(
        {"specification": {"engine": "reference"}, "fit": {"coef": 1}},
        eyeprocess_class="eyeprocess_model",
    )
    assert ea.validate_model_object(warning_only)["valid"] is True
    with pytest.raises(EyeProcessValidationError, match="Interpretive safeguard"):
        ea.validate_model_object(warning_only, strict=True)


def test_upgrade_model_validation_target_engine_diagnostics_and_provenance_paths():
    with pytest.raises(EyeProcessValidationError, match="list-like"):
        ea.upgrade_eyeprocess_model(1)
    with pytest.raises(EyeProcessValidationError, match="only to contract"):
        ea.upgrade_eyeprocess_model({}, target_version="2.0.0")

    legacy = EyeResult(
        {
            "spec": {"engine": "legacy-engine"},
            "model": {"diagnostics": {"rhat": 1.0}},
        },
        eyeprocess_class="legacy_model",
    )
    upgraded = ea.upgrade_eyeprocess_model(legacy)
    assert upgraded["specification"] == legacy["spec"]
    assert upgraded["fit"] == legacy["model"]
    assert upgraded["engine"] == "legacy-engine"
    assert upgraded["diagnostics"] == {"rhat": 1.0}
    assert upgraded["provenance"]["source_class"] == "legacy_model"

    specification_only = ea.upgrade_eyeprocess_model(
        {"specification": {"engine": "declared"}, "fit": {"coef": 1}, "provenance": {}}
    )
    assert specification_only["engine"] == "declared"
    assert specification_only["provenance"] == {}

    unknown_engine = ea.upgrade_eyeprocess_model({"fit": {"coef": 1}})
    assert unknown_engine["engine"] == "unknown"

    dep = ea.eyeprocess_deprecation("old", "new", "0.1", "1.0", reason="superseded")
    assert dep.loc[0, "replacement"] == "new"
    assert dep.loc[0, "reason"] == "superseded"


def test_engine_registry_status_not_available_and_defensive_available_path(monkeypatch):
    registry = ea.external_model_engines()
    assert registry.attrs["eyeprocess_class"] == "eye_external_engine_registry"
    assert not registry["available"].any()

    with pytest.raises(EyeProcessValidationError, match="engine must be"):
        ea.engine_adapter_status(3)
    with pytest.raises(EyeProcessValidationError, match="Unknown engine"):
        ea.engine_adapter_status("unknown")

    custom = ea._not_available("mirt", "purpose", message="custom message")
    assert custom["message"] == "custom message"

    data = pd.DataFrame({"a": [1, 2]})
    unavailable = ea.fit_external_engine(
        "mirt",
        data,
        specification={"model": 1},
        purpose="  exact comparison  ",
        technical=True,
    )
    assert unavailable["status"] == "not_available"
    assert unavailable["purpose"] == "exact comparison"
    assert unavailable["specification"] == {"model": 1}
    assert unavailable["arguments"] == {"technical": True}

    with pytest.raises(EyeProcessValidationError, match="purpose"):
        ea.fit_external_engine("mirt", data)

    monkeypatch.setitem(ea._EXACT_ENGINE_AVAILABLE, "mirt", True)
    failed = ea.fit_external_engine(
        "mirt", data, specification={"model": 1}, purpose="defensive branch"
    )
    assert failed["status"] == "failed"
    assert failed["result_class"] == "eye_engine_adapter_failure"
    assert "unavailable" in failed["error"]


def test_adapter_validation_and_comparison_invalid_and_scalar_text_paths():
    with pytest.raises(EyeProcessValidationError, match="engine-adapter result"):
        ea.validate_engine_adapter({})

    malformed = _adapter_result(
        status="weird",
        engine="",
        purpose=None,
        timestamp_utc="",
    )
    validation = ea.validate_engine_adapter(malformed, require_fit=True)
    assert validation["valid"] is False
    assert not validation["findings"]["passed"].any()

    fitted = _adapter_result(status="fitted", fit={"coef": 1})
    assert ea.validate_engine_adapter(fitted, require_fit=True)["valid"] is True

    with pytest.raises(EyeProcessValidationError, match="At least one"):
        ea.compare_engine_adapters()
    with pytest.raises(EyeProcessValidationError, match="All results"):
        ea.compare_engine_adapters([_adapter_result(), {}])

    one = _adapter_result()
    two = _adapter_result(engine="TAM", status="failed")
    as_args = ea.compare_engine_adapters(one, two)
    assert len(as_args) == 2
    assert as_args.attrs["eyeprocess_class"] == "eye_engine_adapter_comparison"
    assert "error" in as_args.columns


def test_all_adapter_wrappers_and_gdina_q_dimension_guard():
    d = pd.DataFrame({"I1": [1, 0], "I2": [0, 1]})
    wrappers = [
        ea.fit_mirt_adapter(d, purpose="x"),
        ea.fit_tam_adapter(d, purpose="x"),
        ea.fit_brms_adapter("y ~ x", pd.DataFrame({"y": [1], "x": [1]}), purpose="x"),
        ea.fit_lnirt_adapter(d, purpose="x"),
        ea.fit_traminer_adapter(d, purpose="x"),
        ea.fit_seqhmm_adapter(d, purpose="x"),
        ea.fit_openmx_adapter({"model": "x"}, purpose="x"),
        ea.fit_diffirt_engine_adapter(d, purpose="x"),
        ea.fit_eyetrackingr_adapter(d, purpose="x"),
        ea.fit_pupillometryr_adapter(d, purpose="x"),
    ]
    assert all(x["status"] == "not_available" for x in wrappers)

    with pytest.raises(EyeProcessValidationError, match="two-dimensional"):
        ea.fit_gdina_adapter(d, Q=np.array([1, 0]))


def test_scanpath_sequence_validation_empty_missing_columns_and_collapse_paths():
    with pytest.raises(EyeProcessValidationError, match="eye_dataset"):
        ea._scanpath_sequence({})

    x = _sequence_dataset()
    with pytest.raises(EyeProcessValidationError, match="source must be"):
        ea._scanpath_sequence(x, source="bad")

    samples = x.copy()
    samples["gaze_samples"] = samples["gaze_samples"].drop(columns=["aoi_id"], errors="ignore")
    with pytest.raises(EyeProcessValidationError, match="AOIs have not been assigned"):
        ea._scanpath_sequence(samples, source="samples")

    missing_cols = x.copy()
    missing_cols["episodes"] = pd.DataFrame({"episode_type": ["aoi_visit"]})
    empty = ea._scanpath_sequence(missing_cols, source="visits")
    assert empty.empty

    visits = x.copy()
    visits["episodes"] = pd.DataFrame(
        {
            "recording_id": ["r", "r", "r", "r"],
            "trial_id": ["t", "t", "t", "t"],
            "start_time": [0.0, 0.1, 0.2, 0.3],
            "aoi_id": ["A", "A", "B", pd.NA],
            "episode_type": ["aoi_visit"] * 4,
        }
    )
    collapsed = ea._scanpath_sequence(visits, source="visits", collapse_consecutive=True)
    assert collapsed.loc[0, "sequence"] == "A > B"
    assert collapsed.loc[0, "length"] == 2

    uncollapsed = ea._scanpath_sequence(visits, source="visits", collapse_consecutive=False)
    assert uncollapsed.loc[0, "sequence"] == "A > A > B"
    assert uncollapsed.loc[0, "length"] == 3

    fixation = visits.copy()
    fixation["episodes"] = fixation["episodes"].assign(episode_type="fixation")
    assert not ea._scanpath_sequence(fixation, source="fixations").empty


def test_sequence_conversion_empty_nonempty_and_native_backend_gate():
    x = _sequence_dataset()
    x["episodes"] = pd.DataFrame(
        columns=["recording_id", "trial_id", "start_time", "aoi_id", "episode_type"]
    )

    proc = ea.as_procdata_sequence(x, source="visits")
    assert proc.empty
    assert list(proc.columns) == [
        "recording_id", "trial_id", "action_index", "action", "timestamp_order"
    ]

    wide = ea.as_traminer_sequence(x, source="visits")
    assert wide.empty
    assert list(wide.columns) == ["recording_id", "trial_id"]
    with pytest.raises(EyeProcessBackendError, match="TraMineR"):
        ea.as_traminer_sequence(x, source="visits", create_object=True)

    hmm = ea.as_seqhmm_data(x, source="visits")
    assert hmm["sequences"] == []
    assert hmm["alphabet"] == []
    assert hmm["index"].empty

    x2 = _sequence_dataset()
    x2["episodes"] = pd.DataFrame(
        {
            "recording_id": ["r", "r"],
            "trial_id": ["t", "t"],
            "start_time": [0.0, 0.1],
            "aoi_id": ["A", "B"],
            "episode_type": ["aoi_visit", "aoi_visit"],
        }
    )
    proc2 = ea.as_procdata_sequence(x2)
    assert proc2["action"].tolist() == ["A", "B"]
    wide2 = ea.as_traminer_sequence(x2)
    assert wide2.loc[0, "state_1"] == "A"
    assert wide2.loc[0, "state_2"] == "B"
    hmm2 = ea.as_seqhmm_data(x2)
    assert hmm2["alphabet"] == ["A", "B"]


def test_strict_legacy_adapters_reject_wrong_input_before_backend_gate():
    with pytest.raises(EyeProcessValidationError, match="eye_dataset"):
        ea.fit_diffirt_adapter({}, model="D")
    with pytest.raises(EyeProcessValidationError, match="eye_dataset"):
        ea.fit_openmx_process_model({}, model_builder=lambda x: x)

    x = _sequence_dataset()
    with pytest.raises(EyeProcessValidationError, match="model must"):
        ea.fit_diffirt_adapter(x, model="X")
    with pytest.raises(EyeProcessBackendError, match="diffIRT"):
        ea.fit_diffirt_adapter(x, model="Q")

    with pytest.raises(EyeProcessValidationError, match="must be a function"):
        ea.fit_openmx_process_model(x, model_builder="not callable")
    with pytest.raises(EyeProcessBackendError, match="OpenMx"):
        ea.fit_openmx_process_model(x, model_builder=lambda d: d, include_features=False)


def test_compare_model_engines_argument_validation_paths():
    data = pd.DataFrame({"x": [1.0, 2.0]})

    for engines in ({}, [], {"a": 1}, {1: lambda d: d}):
        with pytest.raises(EyeProcessValidationError, match="engines"):
            ea.compare_model_engines(data, engines, lambda x: x)

    engines = {"a": lambda d: d, "b": lambda d: d}
    with pytest.raises(EyeProcessValidationError, match="extractor"):
        ea.compare_model_engines(data, engines, {"a": lambda x: x})
    with pytest.raises(EyeProcessValidationError, match="reference"):
        ea.compare_model_engines(data, engines, lambda x: {"p": 1.0}, reference="z")
    with pytest.raises(EyeProcessValidationError, match="finite non-negative"):
        ea.compare_model_engines(data, engines, lambda x: {"p": 1.0}, tolerance=np.nan)
    with pytest.raises(EyeProcessValidationError, match="finite non-negative"):
        ea.compare_model_engines(data, engines, lambda x: {"p": 1.0}, tolerance=-1)


def test_compare_model_engines_all_extractor_forms_and_failure_paths():
    data = pd.DataFrame({"x": [1.0, 2.0]})

    def fit_ok(d):
        return d

    def fit_bad(d):
        raise RuntimeError("fit exploded")

    engines = {
        "mapping": fit_ok,
        "series": fit_ok,
        "frame": fit_ok,
        "array_bad": fit_ok,
        "frame_bad": fit_ok,
        "fit_bad": fit_bad,
    }
    extractors = {
        "mapping": lambda fit: {"beta": 1.0},
        "series": lambda fit: pd.Series([1.02], index=["beta"]),
        "frame": lambda fit: pd.DataFrame({"parameter": ["beta"], "estimate": [0.99]}),
        "array_bad": lambda fit: np.array([1.0]),
        "frame_bad": lambda fit: pd.DataFrame({"estimate": [1.0]}),
        "fit_bad": lambda fit: {"beta": 1.0},
    }
    result = ea.compare_model_engines(
        data,
        engines,
        extractors,
        reference="mapping",
        tolerance=0.05,
    )
    assert result["reference"] == "mapping"
    estimates = result["estimates"]
    assert set(estimates.loc[estimates["parameter"].notna(), "engine"]) == {
        "mapping", "series", "frame"
    }
    assert estimates.loc[estimates["engine"].eq("mapping"), "equivalent"].iloc[0]
    errors = estimates["error"].fillna("").str.cat(sep=" ")
    assert "fit exploded" in errors
    assert "named estimates" in errors
    assert "parameter and estimate" in errors

    default_reference = ea.compare_model_engines(
        data,
        {"first": fit_ok},
        lambda fit: pd.Series([1.0], index=["beta"]),
    )
    assert default_reference["reference"] == "first"


def test_plot_engine_comparison_validation_parameter_and_reference_paths():
    with pytest.raises(EyeProcessValidationError, match="eye_engine_comparison"):
        ea.plot_eye_engine_comparison({})

    no_parameters = EyeResult(
        {
            "estimates": pd.DataFrame(
                {
                    "engine": ["a"],
                    "parameter": [pd.NA],
                    "estimate": [np.nan],
                    "reference_estimate": [np.nan],
                }
            )
        },
        eyeprocess_class="eye_engine_comparison",
    )
    with pytest.raises(EyeProcessValidationError, match="No parameter estimates"):
        ea.plot_eye_engine_comparison(no_parameters)

    data = pd.DataFrame({"x": [1.0, 2.0]})
    result = ea.compare_model_engines(
        data,
        {"a": lambda d: d, "b": lambda d: d},
        {"a": lambda f: {"beta": 1.0}, "b": lambda f: {"beta": 1.1}},
        reference="a",
    )
    with pytest.raises(EyeProcessValidationError, match="was not found"):
        ea.plot_eye_engine_comparison(result, parameter="missing")

    fig, ax = plt.subplots()
    returned = ea.plot_eye_engine_comparison(result, parameter="beta", ax=ax)
    assert returned is ax
    assert hasattr(returned, "eyeprocess_plot_data")
    plt.close(fig)

    no_reference = EyeResult(
        {
            "estimates": pd.DataFrame(
                {
                    "engine": ["a"],
                    "parameter": ["beta"],
                    "estimate": [1.0],
                    "reference_estimate": [np.nan],
                }
            )
        },
        eyeprocess_class="eye_engine_comparison",
    )
    ax2 = ea.plot_eye_engine_comparison(no_reference)
    assert len(ax2.lines) == 0
    plt.close(ax2.figure)
