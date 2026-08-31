from __future__ import annotations

import pytest

import eyeprocesspy as ep

TARGETS = [
    "plot_aoi_dwell",
    "plot_biometrics",
    "plot_clock_alignment",
    "plot_coordinate_spaces",
    "plot_eye_overview",
    "plot_eye_trace",
    "plot_feature_correlation",
    "plot_feature_distribution",
    "plot_fixations",
    "plot_gaze_heatmap",
    "plot_item_difficulty",
    "plot_missingness",
    "plot_model_diagnostics",
    "plot_pupil_timeseries",
    "plot_sampling_rate",
    "plot_scanpath",
    "plot_signal_quality",
    "plot_transition_matrix",
    "plot_trial_timeline",
]


@pytest.fixture()
def dataset():
    return ep.simulate_eye_dataset(
        n_person=3,
        n_item=3,
        samples_per_trial=20,
        include_pupil=True,
        include_biometrics=True,
        seed=3,
    )


def _close(axis):
    import matplotlib.pyplot as plt

    if axis is not None:
        plt.close(axis.figure)


def test_public_r012_exports_are_callable():
    assert len(TARGETS) == 19
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_frozen_core_plot_smoke_contract(dataset):
    trial_id = (
        dataset["intervals"]
        .loc[
            dataset["intervals"]["interval_type"].eq("trial"),
            "trial_id",
        ]
        .iloc[0]
    )

    axes = [
        ep.plot_eye_overview(dataset),
        ep.plot_eye_trace(dataset, trial_id=trial_id),
        ep.plot_pupil_timeseries(dataset, trial_id=trial_id),
    ]
    for axis in axes:
        assert hasattr(axis, "eyeprocess_plot_data")
        _close(axis)


def test_gaze_fixation_scanpath_and_heatmap_plots(dataset):
    axes = [
        ep.plot_fixations(dataset),
        ep.plot_scanpath(dataset),
        ep.plot_gaze_heatmap(dataset, bins=(8, 6)),
    ]
    for axis in axes:
        assert hasattr(axis, "eyeprocess_plot_data")
        _close(axis)


def test_aoi_and_transition_plots_execute(dataset):
    axes = [
        ep.plot_aoi_dwell(dataset),
        ep.plot_transition_matrix(dataset),
    ]
    for axis in axes:
        assert hasattr(axis, "eyeprocess_plot_data")
        _close(axis)


def test_pupil_biometrics_and_quality_plots(dataset):
    axes = [
        ep.plot_biometrics(dataset),
        ep.plot_signal_quality(dataset),
        ep.plot_sampling_rate(dataset, expected_hz=60),
        ep.plot_missingness(dataset, component="gaze_samples"),
        ep.plot_clock_alignment(dataset),
    ]
    for axis in axes:
        assert hasattr(axis, "eyeprocess_plot_data")
        _close(axis)


def test_feature_and_coordinate_plots(dataset):
    axes = [
        ep.plot_feature_distribution(
            dataset,
            feature_name="not_present",
        ),
        ep.plot_feature_correlation(dataset),
        ep.plot_coordinate_spaces(dataset),
        ep.plot_trial_timeline(dataset),
    ]
    for axis in axes:
        assert hasattr(axis, "eyeprocess_plot_data")
        _close(axis)


def test_item_difficulty_and_model_diagnostics(dataset):
    model = ep.fit_irt(dataset, engine="rasch_glm")
    difficulty = ep.plot_item_difficulty(model)
    diagnostics = ep.plot_model_diagnostics(model)
    assert hasattr(difficulty, "eyeprocess_plot_data")
    assert hasattr(diagnostics, "eyeprocess_plot_data")
    _close(difficulty)
    _close(diagnostics)


def test_selection_and_validation_contracts(dataset):
    with pytest.raises(ep.EyeProcessValidationError):
        ep.plot_gaze_heatmap(dataset, bins=(0, 5))
    with pytest.raises(ep.EyeProcessValidationError):
        ep.plot_transition_matrix(dataset, normalize="bad")
    with pytest.raises(ep.EyeProcessValidationError):
        ep.plot_missingness(dataset, component="events")
