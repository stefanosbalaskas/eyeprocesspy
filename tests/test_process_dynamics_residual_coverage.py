from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.process_dynamics as dyn


def _close(ax):
    if ax is not None and hasattr(ax, "figure"):
        plt.close(ax.figure)


def _fixations(n=8):
    return pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, n),
            "y": np.linspace(1.0, 0.0, n),
            "duration": np.linspace(100.0, 200.0, n),
            "pupil": np.linspace(3.0, 3.7, n),
            "cov": [np.nan] * n,
        }
    )


def test_dataframe_runlength_and_recurrence_private_residual_paths():
    coerced = dyn._df([[1, 2], [3, 4]], "x")
    assert coerced.shape == (2, 2)
    with pytest.raises(ep.EyeProcessValidationError, match="coercible"):
        dyn._df(object(), "x")

    assert dyn._run_lengths([False, True, True, False, True]) == [2, 1]
    matrix = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=int)
    assert dyn._diag_lengths(matrix).ndim == 1
    assert dyn._diag_lengths(matrix, vertical=True).ndim == 1

    aoi = dyn._rec_matrix(["A", "B"], representation="aoi")
    assert aoi.tolist() == [[1, 0], [0, 1]]
    same = dyn._rec_matrix([1.0, 1.0], representation="coordinates")
    assert same.shape == (2, 2) and same.sum() == 4
    supplied = dyn._rec_matrix([0.0, 1.0], [0.0, 2.0], radius=0.25)
    assert supplied.shape == (2, 2)

    zero = dyn.recurrence_features(np.zeros((2, 2), dtype=int))
    assert zero.loc[0, "determinism"] == 0
    recurrent = dyn.recurrence_features(np.eye(4, dtype=int), minimum_line=2)
    assert np.isfinite(recurrent.loc[0, "recurrence_rate"])


def test_gaze_cross_and_windowed_recurrence_alternate_paths_and_plots():
    df = pd.DataFrame(
        {
            "aoi": ["A", "A", "B", "B", "A", "C"],
            "x": [0, 0, 1, 1, 0, 2],
            "y": [0, 1, 1, 2, 0, 2],
        }
    )
    aoi = ep.gaze_recurrence(df, representation="aoi")
    velocity = ep.gaze_recurrence(df[["x", "y"]].to_numpy(float), representation="velocity")
    assert aoi.representation == "aoi"
    assert velocity.representation == "velocity"

    with pytest.raises(ep.EyeProcessValidationError, match="must contain observations"):
        ep.cross_recurrence(np.empty((0, 1)), np.empty((0, 1)))
    cross = ep.cross_recurrence(np.ones(5), np.arange(5.0))
    assert cross.matrix.shape == (5, 5)

    with pytest.raises(ep.EyeProcessValidationError, match="Invalid window"):
        ep.windowed_recurrence(np.array([1.0]), window=2, step=1)
    win_aoi = ep.windowed_recurrence(np.array(["A", "B", "A", "C"], dtype=object), 3, 1)
    win_eye = ep.windowed_recurrence(aoi, 3, 2)
    assert len(win_aoi.summary) >= 1 and len(win_eye.summary) >= 1

    axes = [
        ep.plot_recurrence_matrix(aoi),
        ep.plot_windowed_recurrence(win_eye),
        ep.plot_diagonal_recurrence_profile(aoi),
        ep.plot_crossmodal_recurrence(cross),
        ep.plot_recurrence_network(aoi),
        ep.plot_eye_recurrence(aoi, type="network"),
        ep.plot_eye_cross_recurrence(cross),
        ep.plot_eye_windowed_recurrence(win_eye, metric="missing_metric"),
    ]
    for ax in axes:
        assert hasattr(ax, "eyeprocess_plot_data")
        _close(ax)


def test_point_process_guards_covariates_marked_prediction_and_diagnostics():
    assert dyn._expand_range(np.array([2.0, 2.0]))[0] < 2.0
    assert dyn._expand_range(np.array([1.0, 2.0])) == (1.0, 2.0)

    with pytest.raises(ep.EyeProcessValidationError, match="At least five fixation"):
        ep.fit_fixation_point_process(pd.DataFrame({"x": [1], "y": [1]}))

    incomplete = _fixations()
    incomplete.loc[:4, "x"] = np.nan
    with pytest.raises(ep.EyeProcessValidationError, match="complete fixation"):
        ep.fit_fixation_point_process(incomplete)

    data = _fixations()
    data["x"] = [0, 0, 0, 1, 1, 1, 2, 2]
    data["y"] = [0, 0, 0, 1, 1, 1, 2, 2]
    fit = ep.fit_fixation_point_process(
        data,
        spatial_covariates=["cov", "cov", "missing"],
        interaction="self_exciting",
        grid_size=4,
    )
    assert fit.interaction == "self_exciting"
    assert fit.covariate_map.source.tolist() == ["cov"]
    assert fit.grid["history"].notna().all()

    with pytest.raises(ep.EyeProcessValidationError, match="No requested mark"):
        ep.fit_marked_gaze_process(data.drop(columns=["duration", "pupil"]))
    marked = ep.fit_marked_gaze_process(data, marks=("duration", "pupil"))
    assert set(marked.models) == {"duration", "pupil"}

    with pytest.raises(ep.EyeProcessValidationError, match="eye_fixation_point_process"):
        ep.predict_fixation_intensity({})

    p_default = ep.predict_fixation_intensity(
        fit, pd.DataFrame({"x": [0.2, 0.8], "y": [0.8, 0.2]})
    )
    p_source = ep.predict_fixation_intensity(
        fit,
        pd.DataFrame(
            {"gaze_x": [0.2, 0.8], "gaze_y": [0.8, 0.2], "cov": [1.0, np.nan]}
        ),
    )
    assert p_default["predicted_intensity"].notna().all()
    assert p_source["predicted_intensity"].notna().all()

    one = dyn._result(
        "eye_fixation_point_process",
        grid=pd.DataFrame(
            {
                "count": [1.0],
                "expected": [1.0],
                "x_center": [0.0],
                "y_center": [0.0],
                "residual": [0.0],
            }
        ),
        model={"coef": np.array([0.0])},
    )
    diag = ep.diagnose_gaze_point_process(one)
    assert math.isnan(diag.summary.loc[0, "sd_pearson"])
    assert math.isnan(diag.summary.loc[0, "correlation_observed_expected"])

    diag_full = ep.diagnose_gaze_point_process(fit)
    axes = [
        ep.plot_fixation_intensity(fit),
        ep.plot_spatial_residuals(fit),
        ep.plot_temporal_excitation_kernel(fit),
        ep.plot_covariate_effect_surface(fit),
        ep.plot_observed_expected_fixations(fit),
        ep.plot_eye_fixation_point_process(fit, type="observed_expected"),
        ep.plot_eye_gaze_point_process_diagnostics(diag_full, type="observed_expected"),
        ep.plot_eye_gaze_point_process_diagnostics(diag_full, type="residual"),
    ]
    for ax in axes:
        assert hasattr(ax, "eyeprocess_plot_data")
        _close(ax)


def test_scanpath_path_coercion_distance_and_consensus_residual_paths():
    seqs = [["A", "B", "C"], ["A", "C"], ["A", "B", "B", "C"]]
    assert dyn._paths(seqs) == seqs
    mapped = {"p1": seqs[0], "p2": seqs[1]}
    assert dyn._paths(mapped) == list(mapped.values())

    df = pd.DataFrame(
        {
            "person_id": ["p1"] * 3 + ["p2"] * 3,
            "aoi": ["A", "B", "C", "A", "C", "C"],
            "x": [0, 1, 2, 0, 1, 2],
            "y": [0, 1, 0, 1, 0, 1],
        }
    )
    p_aoi = dyn._paths(df)
    p_xy = dyn._paths(df, aoi_col="missing")
    assert len(p_aoi) == 2 and len(p_xy) == 2
    assert dyn._dist(["A", "B"], ["A", "C"], "edit") == 1
    assert dyn._dist(p_xy[0], p_xy[1], "transport") >= 0
    assert dyn._dist(p_xy[0], p_xy[1], "multimatch") >= 0

    med = ep.representative_scanpath(mapped, method="medoid", distance="edit")
    consensus_seq = ep.representative_scanpath(mapped, method="consensus", distance="edit")
    consensus_xy = ep.representative_scanpath(
        df, method="consensus", aoi_col="missing", distance="multimatch"
    )
    assert med.sequence is True
    assert consensus_seq.sequence is True
    assert consensus_xy.sequence is False
    assert np.asarray(consensus_xy.representative).shape[1] == 2

    direct_disp = ep.scanpath_dispersion(med)
    raw_disp = ep.scanpath_dispersion(mapped)
    assert len(direct_disp) == len(raw_disp) == 2


def test_scanpath_comparison_bootstrap_and_all_plot_dispatch_branches():
    mapped = {
        "p1": ["A", "B", "C"],
        "p2": ["A", "C", "C"],
        "p3": ["B", "C", "A"],
    }
    rep = ep.representative_scanpath(mapped, distance="edit")
    comp = ep.compare_scanpath_distributions(
        mapped, group=["g1", "g1", "g2"], distance="edit", permutations=2
    )
    same = ep.compare_scanpath_distributions(
        mapped, group=["g1", "g1", "g1"], distance="edit", permutations=0
    )
    assert same.observed == 0
    boot = ep.bootstrap_representative_scanpath(rep, draws=3, seed=2)
    assert int(boot.summary["count"].sum()) == 3

    coord = ep.representative_scanpath(
        {
            "a": np.array([[0.0, 0.0], [1.0, 1.0]]),
            "b": np.array([[0.0, 1.0], [1.0, 0.0]]),
        },
        distance="multimatch",
    )

    axes = [
        ep.plot_scanpath_atlas(rep),
        ep.plot_representative_scanpath(coord),
        ep.plot_scanpath_dispersion(rep),
        ep.plot_group_scanpath_transport(comp),
        ep.plot_scanpath_similarity_matrix(rep),
        ep.plot_eye_scanpath_representative(rep, type="representative"),
        ep.plot_eye_scanpath_comparison(comp),
        ep.plot_eye_scanpath_bootstrap(boot),
    ]
    for ax in axes:
        assert hasattr(ax, "eyeprocess_plot_data")
        _close(ax)


def _episode_signal(n=30):
    return pd.DataFrame(
        {
            "time": np.arange(n, dtype=float),
            "signal": np.r_[np.zeros(n // 2), np.ones(n - n // 2) * 5],
        }
    )


def test_changepoint_guards_detection_segmentation_and_time_path():
    with pytest.raises(ep.EyeProcessValidationError, match="At least"):
        ep.detect_process_changepoints(pd.DataFrame({"signal": [1.0] * 5}), channels=["signal"], window=2)

    text = pd.DataFrame({"aoi": ["A"] * 25})
    with pytest.raises(ep.EyeProcessValidationError, match="numeric process channel"):
        ep.detect_process_changepoints(text, channels=["aoi"], window=3)

    data = _episode_signal()
    cp = ep.detect_process_changepoints(
        data, channels=["signal"], time_col="time", window=3, min_segment=2
    )
    assert cp.channels == ["signal"]
    assert len(cp.score) == len(data)

    seg_from_cp = ep.segment_process_episodes(cp)
    seg_from_data = ep.segment_process_episodes(
        data, channels=["signal"], time_col="time", window=3, min_segment=2
    )
    assert len(seg_from_cp.data) == len(data)
    assert len(seg_from_data.data) == len(data)


def test_episode_label_compare_and_plot_residual_paths():
    cp = dyn._result(
        "eye_process_changepoints",
        data=pd.DataFrame({"signal": np.arange(9.0)}),
        channels=["signal"],
        score=np.arange(9.0),
        threshold=3.0,
        changepoints=np.array([3, 6]),
        summary=pd.DataFrame({"index": [4, 7], "time": [4, 7], "score": [4, 7]}),
    )
    seg = ep.segment_process_episodes(cp)

    with pytest.raises(ep.EyeProcessValidationError, match="eye_process_episodes"):
        ep.label_process_episodes({})
    modelled = ep.label_process_episodes(
        seg, model=lambda summary, data: ["m1", "m2", "m3"]
    )
    ruled = ep.label_process_episodes(
        seg,
        rules={
            "first": lambda summary, data: summary.episode_id.eq(1),
            "later": lambda summary, data: summary.episode_id.gt(1),
        },
    )
    default = ep.label_process_episodes(seg)
    assert modelled.summary.episode_label.tolist() == ["m1", "m2", "m3"]
    assert set(ruled.summary.episode_label) == {"first", "later"}
    assert default.summary.episode_label.iloc[-1] == "commitment"

    with pytest.raises(ep.EyeProcessValidationError, match="align"):
        ep.compare_episode_structure(default, ["g"])
    with pytest.raises(ep.EyeProcessValidationError, match="episode object"):
        ep.compare_episode_structure({}, ["g"])
    comparison = ep.compare_episode_structure(default, ["A"] * 3 + ["B"] * 6)
    unlabeled = ep.compare_episode_structure(seg, ["A"] * 3 + ["B"] * 6)
    assert not comparison.table.empty and not unlabeled.table.empty

    axes = [
        ep.plot_process_episodes(default),
        ep.plot_changepoint_ribbons(default),
        ep.plot_episode_waterfall(default),
        ep.plot_episode_transition_graph(default),
        ep.plot_episode_duration_distribution(default),
        ep.plot_eye_process_changepoints(cp),
        ep.plot_eye_process_episodes(default, type="duration"),
        ep.plot_eye_episode_comparison(comparison),
    ]
    for ax in axes:
        assert hasattr(ax, "eyeprocess_plot_data")
        _close(ax)
