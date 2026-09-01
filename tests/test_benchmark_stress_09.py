from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "apply_synthetic_corruption",
    "benchmark_memory_estimate",
    "benchmark_scaling_curve",
    "eye_benchmark_design",
    "inject_aoi_label_noise",
    "inject_calibration_offset",
    "inject_device_shift",
    "inject_eye_missingness",
    "inject_pupil_dropout",
    "inject_sampling_jitter",
    "inject_trial_imbalance",
    "run_eye_benchmark",
    "stress_test_process_pipeline",
    "stress_test_summary",
    "stress_tolerance_frontier",
    "summarise_eye_benchmark",
    "synthetic_corruption_plan",
]


def test_public_r075_exports_are_callable():
    assert len(TARGETS) == 17
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_benchmark_design_preserves_frozen_expand_grid_order():
    design = ep.eye_benchmark_design(
        n_obs=[10, 20],
        repetitions=2,
        label="tiny",
    )
    assert design["benchmark_id"].tolist() == [
        "B00001",
        "B00002",
        "B00003",
        "B00004",
    ]
    assert design["n_obs"].tolist() == [10, 20, 10, 20]
    assert design["repetition"].tolist() == [1, 1, 2, 2]
    assert design["label"].tolist() == ["tiny"] * 4
    assert design.attrs["eyeprocess_class"] == "eye_benchmark_design"


def test_small_benchmark_run_summary_scaling_and_memory():
    design = ep.eye_benchmark_design(
        n_obs=[8, 16],
        repetitions=2,
    )

    def generator(n, row):
        return pd.DataFrame(
            {
                "value": np.arange(n, dtype=float),
            }
        )

    def operation(data, row):
        return {"effect": float(data["value"].mean())}

    result = ep.run_eye_benchmark(
        design,
        generator=generator,
        operation=operation,
        gc_before=False,
    )
    assert result["eyeprocess_class"] == "eye_benchmark_result"
    assert (result["results"]["status"] == "success").all()
    assert len(result["results"]) == 4
    assert (result["results"]["elapsed_sec"] >= 0).all()
    assert (result["results"]["input_bytes"] > 0).all()

    summary = ep.summarise_eye_benchmark(result)
    assert summary["n_obs"].tolist() == [8, 16]
    assert summary["runs"].tolist() == [2, 2]
    assert (summary["median_elapsed_sec"] >= 0).all()
    assert (summary["median_input_mb"] > 0).all()

    scaling = ep.benchmark_scaling_curve(result)
    assert scaling.loc[0, "n_sizes"] == 2
    assert np.isfinite(scaling.loc[0, "exponent"])
    assert np.isfinite(scaling.loc[0, "intercept"])

    memory = ep.benchmark_memory_estimate(
        10,
        generator=lambda n: np.zeros(n, dtype=float),
    )
    assert memory.loc[0, "bytes"] >= 80
    assert memory.loc[0, "kb"] == pytest.approx(memory.loc[0, "bytes"] / 1024)


def test_benchmark_failures_are_retained_explicitly():
    design = ep.eye_benchmark_design([5], repetitions=1)

    generated = ep.run_eye_benchmark(
        design,
        generator=lambda n, row: (_ for _ in ()).throw(RuntimeError("generator boom")),
        operation=lambda data, row: data,
    )
    assert generated["results"].loc[0, "status"] == "generator_error"
    assert "generator boom" in generated["results"].loc[0, "error"]

    operated = ep.run_eye_benchmark(
        design,
        generator=lambda n, row: list(range(n)),
        operation=lambda data, row: (_ for _ in ()).throw(RuntimeError("operation boom")),
    )
    assert operated["results"].loc[0, "status"] == "operation_error"
    assert "operation boom" in operated["results"].loc[0, "error"]


def test_corruption_plan_is_explicit_and_reproducible():
    data = pd.DataFrame(
        {
            "gaze_x": np.arange(1, 101) / 100,
            "gaze_y": np.arange(1, 101) / 100,
            "pupil": np.repeat(3.0, 100),
            "timestamp_ms": np.arange(1, 101),
            "aoi": np.tile(["A", "B"], 50),
        }
    )
    plan = ep.synthetic_corruption_plan(
        missingness=0.1,
        pupil_dropout=0.1,
        seed=9,
    )
    first = ep.apply_synthetic_corruption(
        data,
        plan,
        aoi="aoi",
    )
    second = ep.apply_synthetic_corruption(
        data,
        plan,
        aoi="aoi",
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["gaze_x"].isna().sum() > 0
    assert first["pupil"].isna().sum() > 0
    assert first.attrs["eyeprocess_corruption_plan"]["seed"] == 9


def test_each_corruption_operator_preserves_source_semantics():
    data = pd.DataFrame(
        {
            "gaze_x": [0.1, 0.2, 0.3, 0.4],
            "gaze_y": [0.2, 0.3, 0.4, 0.5],
            "pupil": [3.0, 3.1, 3.2, 3.3],
            "timestamp_ms": [1.0, 2.0, 3.0, 4.0],
            "aoi": ["A", "A", "B", "B"],
            "device": [10.0, 20.0, 30.0, 40.0],
        }
    )

    offset = ep.inject_calibration_offset(
        data,
        offset_x=0.5,
        offset_y=-0.25,
    )
    np.testing.assert_allclose(
        offset["gaze_x"],
        [0.6, 0.7, 0.8, 0.9],
    )
    np.testing.assert_allclose(
        offset["gaze_y"],
        [-0.05, 0.05, 0.15, 0.25],
    )

    shifted = ep.inject_device_shift(
        data,
        "device",
        5,
        rows=[True, False, True, False],
    )
    np.testing.assert_allclose(
        shifted["device"],
        [15.0, 20.0, 35.0, 40.0],
    )

    missing_a = ep.inject_eye_missingness(
        data,
        ["gaze_x", "gaze_y"],
        0.5,
        seed=7,
    )
    missing_b = ep.inject_eye_missingness(
        data,
        ["gaze_x", "gaze_y"],
        0.5,
        seed=7,
    )
    pd.testing.assert_frame_equal(missing_a, missing_b)

    dropout_a = ep.inject_pupil_dropout(
        data,
        proportion=0.5,
        seed=8,
    )
    dropout_b = ep.inject_pupil_dropout(
        data,
        proportion=0.5,
        seed=8,
    )
    pd.testing.assert_frame_equal(dropout_a, dropout_b)

    jitter_a = ep.inject_sampling_jitter(
        data,
        sd=0.5,
        seed=11,
    )
    jitter_b = ep.inject_sampling_jitter(
        data,
        sd=0.5,
        seed=11,
    )
    pd.testing.assert_frame_equal(jitter_a, jitter_b)

    noisy_a = ep.inject_aoi_label_noise(
        data,
        proportion=0.75,
        seed=4,
    )
    noisy_b = ep.inject_aoi_label_noise(
        data,
        proportion=0.75,
        seed=4,
    )
    pd.testing.assert_frame_equal(noisy_a, noisy_b)

    imbalance_a = ep.inject_trial_imbalance(
        data,
        proportion=0.25,
        seed=3,
    )
    imbalance_b = ep.inject_trial_imbalance(
        data,
        proportion=0.25,
        seed=3,
    )
    pd.testing.assert_frame_equal(imbalance_a, imbalance_b)


def test_stress_pipeline_summary_and_frontier():
    data = pd.DataFrame(
        {
            "gaze_x": np.linspace(0.1, 0.9, 40),
            "gaze_y": np.linspace(0.2, 0.8, 40),
            "pupil": np.linspace(3.0, 4.0, 40),
            "timestamp_ms": np.arange(40, dtype=float),
        }
    )
    plans = [
        ep.synthetic_corruption_plan(
            missingness=0.0,
            seed=1,
        ),
        ep.synthetic_corruption_plan(
            missingness=0.2,
            seed=2,
        ),
        ep.synthetic_corruption_plan(
            missingness=0.4,
            seed=3,
        ),
    ]

    def analysis(frame, plan):
        return {
            "effect": float(frame["pupil"].mean()),
        }

    result = ep.stress_test_process_pipeline(
        data,
        plans,
        analysis,
    )
    assert result["eyeprocess_class"] == "eye_process_stress_test"
    assert len(result["results"]) == 3
    assert (result["results"]["status"] == "success").all()
    assert result["results"]["missingness"].tolist() == [
        0.0,
        0.2,
        0.4,
    ]

    summary = ep.stress_test_summary(result)
    assert summary.loc[0, "plans"] == 3
    assert summary.loc[0, "successful"] == 3
    assert np.isfinite(summary.loc[0, "median"])

    frontier = ep.stress_tolerance_frontier(
        result,
        severity="missingness",
        metric="effect",
        acceptable=lambda value: bool(value >= 3.4),
    )
    assert np.isfinite(frontier.loc[0, "max_acceptable_severity"])


def test_stress_pipeline_failure_taxonomy_is_preserved():
    data = pd.DataFrame({"pupil": [3.0, 3.2]})
    plan = ep.synthetic_corruption_plan(seed=1)

    analysis_error = ep.stress_test_process_pipeline(
        data,
        [plan],
        lambda frame, spec: (_ for _ in ()).throw(RuntimeError("analysis failed")),
    )
    assert analysis_error["results"].loc[0, "status"] == "analysis_error"

    metric_error = ep.stress_test_process_pipeline(
        data,
        [plan],
        lambda frame, spec: 1.0,
        metric_fun=lambda fit, spec: (_ for _ in ()).throw(RuntimeError("metric failed")),
    )
    assert metric_error["results"].loc[0, "status"] == "metric_error"


def test_validation_boundaries_fail_safely():
    with pytest.raises(
        ep.EyeProcessValidationError,
        match="n_obs",
    ):
        ep.eye_benchmark_design([0], repetitions=1)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="repetitions",
    ):
        ep.eye_benchmark_design([10], repetitions=0)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match=r"\[0, 1\)",
    ):
        ep.synthetic_corruption_plan(missingness=1.0)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="sampling_jitter_sd",
    ):
        ep.synthetic_corruption_plan(sampling_jitter_sd=-1)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="seed",
    ):
        ep.inject_eye_missingness(
            pd.DataFrame({"x": [1, 2]}),
            ["x"],
            0.2,
            seed=-1,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="Logical rows",
    ):
        ep.inject_device_shift(
            pd.DataFrame({"x": [1.0, 2.0]}),
            "x",
            1,
            rows=[True],
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="plans",
    ):
        ep.stress_test_process_pipeline(
            pd.DataFrame({"x": [1]}),
            [],
            lambda frame, plan: 1,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="acceptable",
    ):
        ep.stress_tolerance_frontier(
            {
                "results": pd.DataFrame({"severity": [0.1], "effect": [1.0]}),
                "eyeprocess_class": "eye_process_stress_test",
            },
            "severity",
            "effect",
            acceptable=1,
        )
