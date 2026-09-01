from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy.validation_orchestration_10 as vo
from eyeprocesspy.exceptions import EyeProcessValidationError


def test_result_containers_attribute_access_and_missing_attribute():
    for cls in (vo.EyeValidationJobPlan, vo.EyeValidationRun, vo.EyeValidationCollection):
        obj = cls(answer=42)
        assert obj.answer == 42
        with pytest.raises(AttributeError, match="missing"):
            _ = obj.missing


@pytest.mark.parametrize("value", [True, "bad", 1.5, -1])
def test_scalar_int_rejects_non_integer_and_below_minimum(value):
    with pytest.raises(EyeProcessValidationError, match="single integer"):
        vo._scalar_int(value, "value", 0)


def test_scalar_int_accepts_integer_like_values():
    assert vo._scalar_int("3", "value", 0) == 3
    assert vo._scalar_int(np.int64(4), "value", 1) == 4


@pytest.mark.parametrize("value", ["bad", np.nan, np.inf, -1.0, 2.0])
def test_scalar_num_range_and_finiteness_guards(value):
    with pytest.raises(EyeProcessValidationError, match="allowed range"):
        vo._scalar_num(value, "value", minimum=0.0, maximum=1.0)


def test_scalar_num_can_allow_infinity_explicitly():
    assert vo._scalar_num(np.inf, "value", finite=False) == np.inf


def test_safe_name_and_missing_detection_contracts():
    assert vo._safe_name(None, "fallback") == "fallback"
    assert vo._safe_name("  A / B  ") == "A-B"
    assert vo._safe_name("---", "fallback") == "fallback"
    assert vo._safe_name(float("nan"), "fallback") == "fallback"
    assert vo._is_missing(None)
    assert vo._is_missing(pd.NA)
    assert vo._is_missing(np.nan)
    assert not vo._is_missing("value")
    assert not vo._is_missing([1, 2])


def test_canonical_scalar_covers_sequences_missing_booleans_and_nonfinite_numbers():
    assert vo._canonical_scalar(None) == "<NULL>"
    assert vo._canonical_scalar(pd.NA) == "<NA>"
    assert vo._canonical_scalar(True) == "TRUE"
    assert vo._canonical_scalar(False) == "FALSE"
    assert vo._canonical_scalar([1, 2]).count(",") == 1
    assert vo._canonical_scalar(np.inf) == "inf"
    assert vo._canonical_scalar(-np.inf) == "-inf"
    assert vo._canonical_scalar("x") == "x"


def test_canonical_row_guards_and_order_invariance():
    left = vo._canonical_row({"b": 2, "a": 1})
    right = vo._canonical_row(pd.Series({"a": 1, "b": 2}))
    frame = vo._canonical_row(pd.DataFrame([{"a": 1, "b": 2}]))
    assert left == right == frame
    with pytest.raises(EyeProcessValidationError, match="one-row"):
        vo._canonical_row(pd.DataFrame([{"a": 1}, {"a": 2}]))
    with pytest.raises(EyeProcessValidationError, match="list or one-row"):
        vo._canonical_row(123)


def test_hash_int_empty_and_sequence_paths_are_deterministic():
    assert vo._hash_int("") == 1
    assert vo._hash_int(["a", "b"]) == vo._hash_int(("a", "b"))
    assert vo._hash_int("abc") == vo._hash_int("abc")


def test_validation_seed_argument_guards():
    with pytest.raises(EyeProcessValidationError, match="replication"):
        vo.validation_seed({"n": 1}, 0)
    with pytest.raises(EyeProcessValidationError, match="base_seed"):
        vo.validation_seed({"n": 1}, 1, base_seed=-1)
    with pytest.raises(EyeProcessValidationError, match="stream"):
        vo.validation_seed({"n": 1}, 1, stream=0)


def test_as_levels_handles_scalar_and_arraylike_inputs():
    assert vo._as_levels("x") == ["x"]
    assert vo._as_levels(pd.Series([1, 2])) == [1, 2]
    assert vo._as_levels(np.array([1, 2])) == [1, 2]
    assert vo._as_levels((1, 2)) == [1, 2]
    assert vo._as_levels(3) == [3]


def test_expand_grid_null_dataframe_mapping_and_guard_paths():
    null_grid = vo._expand_grid(None)
    assert null_grid[".scenario"].tolist() == [1]
    frame = pd.DataFrame({"n": [10, 20]})
    out = vo._expand_grid(frame)
    assert out.equals(frame)
    out.loc[0, "n"] = 999
    assert frame.loc[0, "n"] == 10

    grid = vo._expand_grid({"a": [1, 2], "b": ["x", "y"]})
    assert grid.to_dict("records") == [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "y"},
    ]
    for bad in (pd.DataFrame(), {}, {"a": []}, 42):
        with pytest.raises(EyeProcessValidationError, match="grid"):
            vo._expand_grid(bad)


def test_validation_job_plan_guards_metadata_and_safe_identifiers():
    with pytest.raises(EyeProcessValidationError, match="metadata"):
        vo.validation_job_plan({"n": [1]}, metadata=[])
    plan = vo.validation_job_plan(
        {"n": [1]},
        replications=1,
        model_family=" Demo Family ",
        plan_id=" Plan / One ",
        metadata={"source": "test"},
    )
    assert plan.model_family == "Demo-Family"
    assert plan.plan_id == "Plan-One"
    assert plan.metadata == {"source": "test"}


def test_serialize_deserialize_roundtrip_rich_python_payload():
    stamp = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    payload = {
        "missing": pd.NA,
        "frame": pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        "series": pd.Series([1.0, 2.0], name="s"),
        "array": np.array([[1, 2], [3, 4]]),
        "path": Path("example/file.txt"),
        "tuple": (1, 2),
        "nan": np.nan,
        "posinf": np.inf,
        "neginf": -np.inf,
        "stamp": stamp,
    }
    restored = vo._deserialize(vo._serialize(payload))
    assert restored["missing"] is pd.NA
    pd.testing.assert_frame_equal(restored["frame"], payload["frame"])
    pd.testing.assert_series_equal(restored["series"], payload["series"])
    np.testing.assert_array_equal(restored["array"], payload["array"])
    assert restored["path"] == payload["path"]
    assert restored["tuple"] == [1, 2]
    assert np.isnan(restored["nan"])
    assert restored["posinf"] == np.inf
    assert restored["neginf"] == -np.inf
    assert restored["stamp"] == stamp


def test_serialize_unknown_object_uses_explicit_repr_boundary():
    class Example:
        def __repr__(self):
            return "Example(value=3)"

    encoded = vo._serialize(Example())
    assert encoded["__eye_type__"] == "python_repr"
    assert vo._deserialize(encoded) == "Example(value=3)"


def test_stable_payload_and_object_fingerprint_are_key_order_invariant():
    a = {"b": 2, "a": 1}
    b = {"a": 1, "b": 2}
    assert vo._stable_payload(a) == vo._stable_payload(b)
    assert vo._object_fingerprint(a, "x") == vo._object_fingerprint(b, "x")


def test_function_fingerprint_and_supported_call_paths():
    def f(a, named=0):
        return a + named

    def accepts_kwargs(a, **kwargs):
        return a + kwargs.get("extra", 0)

    assert vo._function_fingerprint(None) is None
    assert vo._function_fingerprint(f).startswith("fn-")
    assert vo._call_supported(f, (2,), {"named": 3, "ignored": 99}) == 5
    assert vo._call_supported(accepts_kwargs, (2,), {"extra": 4}) == 6


def test_capture_call_records_warnings_without_messages():
    def warns(value):
        import warnings

        warnings.warn("expected warning", RuntimeWarning)
        return value * 2

    value, warning_messages, messages = vo._capture_call(warns, (3,))
    assert value == 6
    assert warning_messages == ["expected warning"]
    assert messages == []


def test_deep_size_handles_recursive_and_array_objects():
    recursive = []
    recursive.append(recursive)
    assert vo._deep_size(recursive) > 0
    assert vo._deep_size(pd.DataFrame({"a": [1, 2]})) > 0
    assert vo._deep_size(pd.Series([1, 2])) > 0
    assert vo._deep_size(np.array([1, 2])) > 0
    assert vo._deep_size({"x": [1, 2]}) > 0


def test_truth_mapping_supports_series_mapping_vector_and_invalid_values():
    assert vo._truth_mapping(pd.Series({"a": 1}), ["a"]) == {"a": 1.0}
    mapped = vo._truth_mapping({"a": "bad", "b": 2}, ["a", "b"])
    assert np.isnan(mapped["a"])
    assert mapped["b"] == 2.0
    assert vo._truth_mapping(np.array([1.0, 2.0]), ["a", "b"]) == {"a": 1.0, "b": 2.0}
    assert vo._truth_mapping(np.array([1.0]), ["a", "b"]) == {}


def test_standardize_estimates_guards_mapping_sd_and_interval_aliases():
    with pytest.raises(EyeProcessValidationError, match="data frame"):
        vo._standardize_estimates([1, 2])
    with pytest.raises(EyeProcessValidationError, match="parameter"):
        vo._standardize_estimates(pd.DataFrame({"estimate": [1.0]}))

    mapped = vo._standardize_estimates({"mu": 2.0}, truth={"mu": 1.5})
    assert mapped.loc[0, "parameter"] == "mu"
    assert np.isnan(mapped.loc[0, "std_error"])

    aliases = vo._standardize_estimates(
        pd.DataFrame(
            {
                "parameter": ["mu"],
                "estimate": [2.0],
                "sd": [0.5],
                "q025": [1.1],
                "q975": [2.9],
            }
        ),
        truth={"mu": 1.5},
    )
    assert aliases.loc[0, "std_error"] == pytest.approx(0.5)
    assert aliases.loc[0, "lower"] == pytest.approx(1.1)
    assert aliases.loc[0, "upper"] == pytest.approx(2.9)
