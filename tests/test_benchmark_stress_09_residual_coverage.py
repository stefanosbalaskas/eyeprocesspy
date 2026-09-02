from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.benchmark_stress_09 as bs


class _Unpickleable:
    def __reduce_ex__(self, protocol):
        raise RuntimeError("no pickle")

    def __repr__(self):
        return "_Unpickleable()"


def _stress_result(results: pd.DataFrame) -> dict[str, object]:
    return {
        "results": results,
        "eyeprocess_class": "eye_process_stress_test",
    }


def _benchmark_result(results: pd.DataFrame) -> dict[str, object]:
    return {
        "results": results,
        "eyeprocess_class": "eye_benchmark_result",
    }


def test_private_coercion_and_validation_residual_paths(monkeypatch):
    with pytest.raises(ep.EyeProcessValidationError, match="coercible"):
        bs._as_frame(object(), name="broken")

    with pytest.raises(ep.EyeProcessValidationError, match="missing required"):
        bs._require_columns(pd.DataFrame({"a": [1]}), ["a", "b"], name="tiny")

    assert bs._as_list(None) == []
    assert bs._as_list("x") == ["x"]
    assert bs._as_list(pd.Series([1, 2])) == [1, 2]
    assert bs._as_list(7) == [7]

    np.testing.assert_allclose(bs._numeric(pd.Series(["1", "bad"])), [1.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(bs._numeric(np.asarray([1, 2])), [1.0, 2.0])
    np.testing.assert_allclose(bs._numeric("3"), [3.0])
    np.testing.assert_allclose(bs._numeric(4), [4.0])

    real_asarray = np.asarray
    calls = 0

    def flaky_asarray(value, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("array conversion failed")
        return real_asarray(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(bs.np, "asarray", flaky_asarray)
        assert np.isnan(bs._numeric(object())[0])

    with pytest.raises(ep.EyeProcessValidationError, match="positive integer"):
        bs._positive_integer(np.nan, name="n")
    with pytest.raises(ep.EyeProcessValidationError, match=r"\[0, 1\)"):
        bs._proportion(np.nan)


def test_recursive_size_and_hash_fallback_paths():
    recursive: list[object] = []
    recursive.append(recursive)
    assert bs._deep_size(recursive) > 0
    assert bs._deep_size(pd.Series(["a", "bb"])) > 0
    assert bs._deep_size({"a": (1, 2)}) > 0

    digest = bs._stable_hash(_Unpickleable())
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)


def test_default_benchmark_generator_operation_and_default_run(monkeypatch, capsys):
    tiny = ep.eye_benchmark_design([3], repetitions=1, label="default")
    monkeypatch.setattr(bs, "eye_benchmark_design", lambda: tiny)

    default_result = bs.run_eye_benchmark(None, gc_before=False)
    row = default_result["results"].iloc[0]
    assert row["status"] == "success"
    assert row["n_obs"] == 3
    assert row["input_bytes"] > 0
    assert row["output_bytes"] > 0

    bs.run_eye_benchmark(
        tiny,
        generator=lambda n, row: pd.DataFrame({"pupil": np.ones(n), "valid": np.ones(n)}),
        operation=lambda data, row: {"ok": len(data)},
        gc_before=False,
        progress=True,
    )
    assert "benchmark B00001 n=3" in capsys.readouterr().out

    with pytest.raises(ep.EyeProcessValidationError, match="generator"):
        bs.run_eye_benchmark(tiny, generator=1, operation=lambda data, row: data)
    with pytest.raises(ep.EyeProcessValidationError, match="operation"):
        bs.run_eye_benchmark(tiny, generator=lambda n, row: [n], operation=1)


def test_default_sensitivity_extract_all_contract_forms():
    scalar = bs._default_sensitivity_extract(2.5)
    assert scalar.loc[0, "effect"] == pytest.approx(2.5)

    frame = pd.DataFrame({"effect": [1.0]})
    copied = bs._default_sensitivity_extract(frame)
    pd.testing.assert_frame_equal(copied, frame)
    assert copied is not frame

    mapped = bs._default_sensitivity_extract({"effect": np.float64(1.2), "label": "x", "drop": [1, 2]})
    assert mapped.loc[0, "effect"] == pytest.approx(1.2)
    assert mapped.loc[0, "label"] == "x"
    assert "drop" not in mapped.columns

    with pytest.raises(ep.EyeProcessValidationError, match="Provide metric_fun"):
        bs._default_sensitivity_extract({"drop": [1, 2]})
    with pytest.raises(ep.EyeProcessValidationError, match="Provide metric_fun"):
        bs._default_sensitivity_extract("unsupported")


def test_design_summary_and_scaling_guard_paths():
    with pytest.raises(ep.EyeProcessValidationError, match="n_obs"):
        ep.eye_benchmark_design([], repetitions=1)
    with pytest.raises(ep.EyeProcessValidationError, match="n_obs"):
        ep.eye_benchmark_design([np.nan], repetitions=1)

    empty_success = _benchmark_result(
        pd.DataFrame(
            {
                "status": ["generator_error"],
                "n_obs": [10],
                "elapsed_sec": [np.nan],
                "input_bytes": [np.nan],
                "output_bytes": [np.nan],
            }
        )
    )
    assert ep.summarise_eye_benchmark(empty_success).empty
    empty_scaling = ep.benchmark_scaling_curve(empty_success)
    assert empty_scaling.loc[0, "n_sizes"] == 0
    assert np.isnan(empty_scaling.loc[0, "exponent"])

    one_size = _benchmark_result(
        pd.DataFrame(
            {
                "status": ["success"],
                "n_obs": [10],
                "elapsed_sec": [0.5],
                "input_bytes": [100.0],
                "output_bytes": [10.0],
            }
        )
    )
    one_scaling = ep.benchmark_scaling_curve(one_size)
    assert one_scaling.loc[0, "n_sizes"] == 1
    assert np.isnan(one_scaling.loc[0, "exponent"])

    with pytest.raises(ep.EyeProcessValidationError, match="eye_benchmark_result"):
        ep.summarise_eye_benchmark({})


def test_corruption_plan_and_operator_validation_residuals():
    with pytest.raises(ep.EyeProcessValidationError, match="offsets"):
        ep.synthetic_corruption_plan(gaze_offset_x=np.inf)
    with pytest.raises(ep.EyeProcessValidationError, match="seed"):
        ep.synthetic_corruption_plan(seed=[1, 2])

    data = pd.DataFrame(
        {
            "gaze_x": [0.1, 0.2, 0.3],
            "gaze_y": [0.2, 0.3, 0.4],
            "pupil": [3.0, 3.1, 3.2],
            "timestamp_ms": [1.0, 2.0, 3.0],
            "aoi": ["A", "A", "A"],
            "device": [10.0, 20.0, 30.0],
        }
    )

    with pytest.raises(ep.EyeProcessValidationError, match="proportion"):
        ep.inject_pupil_dropout(data, proportion=None)
    with pytest.raises(ep.EyeProcessValidationError, match="offset_x"):
        ep.inject_calibration_offset(data, offset_x=np.inf)
    with pytest.raises(ep.EyeProcessValidationError, match="sd"):
        ep.inject_sampling_jitter(data, sd=None)
    with pytest.raises(ep.EyeProcessValidationError, match="shift"):
        ep.inject_device_shift(data, "device", np.inf)

    unchanged = ep.inject_aoi_label_noise(data, proportion=0.5, seed=2)
    pd.testing.assert_frame_equal(unchanged, data)

    shifted_all = ep.inject_device_shift(data, "device", 2.0, rows=None)
    np.testing.assert_allclose(shifted_all["device"], [12.0, 22.0, 32.0])
    shifted_indices = ep.inject_device_shift(data, "device", -1.0, rows=[0, 2])
    np.testing.assert_allclose(shifted_indices["device"], [9.0, 20.0, 29.0])


def test_apply_synthetic_corruption_exercises_every_optional_operator():
    data = pd.DataFrame(
        {
            "gaze_x": np.linspace(0.1, 0.9, 20),
            "gaze_y": np.linspace(0.2, 0.8, 20),
            "pupil": np.linspace(3.0, 4.0, 20),
            "timestamp_ms": np.arange(20, dtype=float),
            "aoi": np.tile(["A", "B"], 10),
            "device": np.arange(20, dtype=float),
        }
    )
    plan = ep.synthetic_corruption_plan(
        missingness=0.2,
        pupil_dropout=0.2,
        gaze_offset_x=0.1,
        gaze_offset_y=-0.1,
        sampling_jitter_sd=0.25,
        aoi_label_noise=0.5,
        device_shift=3.0,
        trial_drop=0.2,
        seed=11,
    )
    out = ep.apply_synthetic_corruption(data, plan, aoi="aoi", device_column="device")
    assert len(out) < len(data)
    assert out.attrs["eyeprocess_corruption_plan"]["seed"] == 11
    assert not np.array_equal(out["timestamp_ms"].to_numpy(), data.loc[out.index, "timestamp_ms"].to_numpy())
    assert np.allclose(
        out["device"].to_numpy(),
        data.loc[out.index, "device"].to_numpy() + 3.0,
        equal_nan=True,
    )

    with pytest.raises(ep.EyeProcessValidationError, match="plan"):
        ep.apply_synthetic_corruption(data, {})


def test_stress_pipeline_validation_corruption_and_empty_metric_paths():
    data = pd.DataFrame({"pupil": [3.0, 3.2]})
    plan = ep.synthetic_corruption_plan(seed=1)

    with pytest.raises(ep.EyeProcessValidationError, match="analysis_fun"):
        ep.stress_test_process_pipeline(data, [plan], analysis_fun=1)
    with pytest.raises(ep.EyeProcessValidationError, match="metric_fun"):
        ep.stress_test_process_pipeline(data, [plan], lambda frame, spec: 1.0, metric_fun=1)

    corruption_error = ep.stress_test_process_pipeline(
        data,
        [{}],
        lambda frame, spec: 1.0,
    )
    assert corruption_error["results"].loc[0, "status"] == "corruption_error"

    default_metric_error = ep.stress_test_process_pipeline(
        data,
        [plan],
        lambda frame, spec: object(),
    )
    assert default_metric_error["results"].loc[0, "status"] == "metric_error"

    empty_metric = ep.stress_test_process_pipeline(
        data,
        [plan],
        lambda frame, spec: 1.0,
        metric_fun=lambda fit, spec: pd.DataFrame(),
    )
    assert empty_metric["results"].loc[0, "status"] == "empty_metric"

    with pytest.raises(ep.EyeProcessValidationError, match="metric_fun output"):
        ep.stress_test_process_pipeline(
            data,
            [plan],
            lambda frame, spec: 1.0,
            metric_fun=lambda fit, spec: object(),
        )


def test_stress_summary_invalid_and_all_nonfinite_metrics():
    with pytest.raises(ep.EyeProcessValidationError, match="eye_process_stress_test"):
        ep.stress_test_summary({})

    result = _stress_result(
        pd.DataFrame(
            {
                "status": ["success", "metric_error"],
                "effect": [np.nan, np.inf],
            }
        )
    )
    summary = ep.stress_test_summary(result)
    assert summary.loc[0, "plans"] == 2
    assert summary.loc[0, "successful"] == 1
    assert np.isnan(summary.loc[0, "median"])
    assert np.isnan(summary.loc[0, "min"])
    assert np.isnan(summary.loc[0, "max"])


def test_stress_frontier_none_nan_invalid_and_empty_sides():
    result = _stress_result(
        pd.DataFrame(
            {
                "severity": [0.0, 0.1, 0.2, 0.3],
                "effect": [1.0, 2.0, 3.0, 4.0],
            }
        )
    )

    outcomes = iter([None, float("nan"), True, False])
    frontier = ep.stress_tolerance_frontier(
        result,
        "severity",
        "effect",
        acceptable=lambda value: next(outcomes),
    )
    assert frontier.loc[0, "max_acceptable_severity"] == pytest.approx(0.2)
    assert frontier.loc[0, "first_unacceptable_severity"] == pytest.approx(0.3)

    all_unknown = ep.stress_tolerance_frontier(
        result,
        "severity",
        "effect",
        acceptable=lambda value: None,
    )
    assert np.isnan(all_unknown.loc[0, "max_acceptable_severity"])
    assert np.isnan(all_unknown.loc[0, "first_unacceptable_severity"])

    with pytest.raises(ep.EyeProcessValidationError, match="TRUE/FALSE"):
        ep.stress_tolerance_frontier(
            result,
            "severity",
            "effect",
            acceptable=lambda value: 1,
        )


def test_frontier_missing_columns_and_nonfinite_severity_are_safe():
    with pytest.raises(ep.EyeProcessValidationError, match="missing required"):
        ep.stress_tolerance_frontier(
            _stress_result(pd.DataFrame({"severity": [0.1]})),
            "severity",
            "effect",
            acceptable=lambda value: True,
        )

    result = _stress_result(pd.DataFrame({"severity": [np.nan], "effect": [1.0]}))
    frontier = ep.stress_tolerance_frontier(
        result,
        "severity",
        "effect",
        acceptable=lambda value: True,
    )
    assert math.isnan(frontier.loc[0, "max_acceptable_severity"])
