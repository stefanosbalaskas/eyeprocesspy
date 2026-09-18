import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eyeprocesspy.aoi_perturbation import (
    OUTSIDE,
    aoi_perturbation_spec,
    apply_aoi_perturbation_grid,
    assess_aoi_inference_stability,
    compare_aoi_assignments,
    convert_aoi_margin_to_degrees,
    convert_aoi_margin_to_pixels,
    create_aoi_perturbation_grid,
    dilate_aoi,
    erode_aoi,
    estimate_aoi_assignment_stability,
    estimate_fixation_assignment_probability,
    jitter_aoi,
    perturb_aoi_geometry,
    plot_aoi_assignment_stability,
    plot_aoi_coefficient_stability,
    plot_aoi_perturbations,
    plot_aoi_robustness_surface,
    recompute_aoi_features,
    report_aoi_sensitivity,
    run_aoi_sensitivity_analysis,
    translate_aoi,
    validate_aoi_geometry,
)
from eyeprocesspy.exceptions import EyeProcessValidationError


def rects():
    return pd.DataFrame(
        {
            "aoi_id": ["headline", "cta"],
            "shape_type": ["rectangle", "rectangle"],
            "xmin": [100.0, 600.0],
            "xmax": [400.0, 900.0],
            "ymin": [80.0, 500.0],
            "ymax": [180.0, 620.0],
        }
    )


def convex_polygon():
    return pd.DataFrame(
        {
            "aoi_id": ["diamond"],
            "shape_type": ["polygon"],
            "polygon": [
                np.array(
                    [[500.0, 200.0], [560.0, 260.0], [500.0, 320.0], [440.0, 260.0]]
                )
            ],
        }
    )


def test_rectangle_dilation_and_erosion():
    d = dilate_aoi(rects(), 10)
    assert d.loc[0, "xmin"] == 90
    assert d.loc[0, "xmax"] == 410
    e = erode_aoi(rects(), 10)
    assert e.loc[0, "xmin"] == 110
    assert e.loc[0, "xmax"] == 390


def test_polygon_dilation_is_true_convex_offset():
    p0 = convex_polygon().iloc[0]["polygon"]
    d = dilate_aoi(convex_polygon(), 10)
    p1 = d.iloc[0]["polygon"]
    assert abs(np.ptp(p1[:, 0]) - np.ptp(p0[:, 0])) > 10
    assert abs(np.ptp(p1[:, 1]) - np.ptp(p0[:, 1])) > 10
    cross = (
        (p1[1, 0] - p1[0, 0]) * (p1[2, 1] - p1[1, 1])
        - (p1[1, 1] - p1[0, 1]) * (p1[2, 0] - p1[1, 0])
    )
    assert abs(cross) > 0


def test_concave_polygon_dilation_is_not_silently_approximated():
    concave = pd.DataFrame(
        {
            "aoi_id": ["concave"],
            "shape_type": ["polygon"],
            "polygon": [
                np.array([[0, 0], [4, 0], [2, 1], [4, 4], [0, 4]], dtype=float)
            ],
        }
    )
    with pytest.raises(EyeProcessValidationError, match="convex polygons only"):
        dilate_aoi(concave, 0.2)


def test_erosion_beyond_valid_geometry_fails():
    with pytest.raises(EyeProcessValidationError, match="collapsed"):
        erode_aoi(rects().iloc[[0]].copy(), 200)


def test_screen_edge_warn_error_clip_are_explicit():
    edge = rects().iloc[[0]].copy()
    edge.loc[:, ["xmin", "xmax"]] = [5, 35]
    with pytest.warns(RuntimeWarning, match="extend beyond"):
        translate_aoi(
            edge,
            x=-10,
            screen_width_px=100,
            screen_height_px=1000,
            boundary_policy="warn",
        )
    with pytest.raises(EyeProcessValidationError, match="extend beyond"):
        translate_aoi(
            edge,
            x=-10,
            screen_width_px=100,
            screen_height_px=1000,
            boundary_policy="error",
        )
    with pytest.warns(RuntimeWarning, match="clipping"):
        clipped = translate_aoi(
            edge,
            x=-10,
            screen_width_px=100,
            screen_height_px=1000,
            boundary_policy="clip",
        )
    assert clipped.iloc[0]["xmin"] == 0


def test_overlap_is_audited_not_resolved():
    a = pd.DataFrame(
        {
            "aoi_id": ["a", "b"],
            "xmin": [0, 5],
            "xmax": [10, 15],
            "ymin": [0, 0],
            "ymax": [10, 10],
        }
    )
    result = validate_aoi_geometry(a)
    assert result["overlap_present"]


def test_zero_area_and_invalid_polygon_fail():
    zero = pd.DataFrame(
        {"aoi_id": ["a"], "xmin": [0], "xmax": [0], "ymin": [0], "ymax": [1]}
    )
    with pytest.raises(EyeProcessValidationError, match="zero or negative"):
        validate_aoi_geometry(zero)
    bow = pd.DataFrame(
        {
            "aoi_id": ["bow"],
            "shape_type": ["polygon"],
            "polygon": [
                np.array([[0, 0], [2, 2], [0, 2], [2, 0]], dtype=float)
            ],
        }
    )
    with pytest.raises(EyeProcessValidationError, match="self-intersects"):
        validate_aoi_geometry(bow)


def test_pixel_degree_roundtrip():
    kwargs = {
        "screen_width_px": 1920,
        "screen_height_px": 1080,
        "viewing_distance": 60,
        "physical_screen_size": (53.1, 29.9),
    }
    deg = convert_aoi_margin_to_degrees((20, 30), **kwargs)
    px = convert_aoi_margin_to_pixels(deg, **kwargs)
    assert np.allclose(px, [20, 30], atol=1e-10)
    assert deg[0] != deg[1] / 1.5


def test_degree_spec_requires_geometry():
    with pytest.raises(EyeProcessValidationError, match="require"):
        aoi_perturbation_spec("x", "dilation", margin_x=0.25, unit="deg")


def test_jitter_is_reproducible():
    a = jitter_aoi(rects(), 5, seed=4)
    b = jitter_aoi(rects(), 5, seed=4)
    c = jitter_aoi(rects(), 5, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert not a[["xmin", "ymin"]].equals(c[["xmin", "ymin"]])


def test_exact_baseline_is_unchanged():
    spec = aoi_perturbation_spec("baseline", "baseline")
    out = perturb_aoi_geometry(rects(), spec)
    pd.testing.assert_frame_equal(
        out["nominal_geometry"], out["perturbed_geometry"]
    )


def test_assignment_comparison_matrix_and_missingness():
    cmp = compare_aoi_assignments(
        ["a", OUTSIDE, "b", None], ["a", "a", OUTSIDE, "b"]
    )
    s = cmp["summary"].iloc[0]
    assert s["n_comparable"] == 3
    assert math.isclose(s["proportion_unchanged"], 1 / 3)
    assert math.isclose(s["proportion_newly_assigned"], 1 / 3)
    assert math.isclose(s["proportion_lost"], 1 / 3)
    assert cmp["reassignment_matrix"].loc["a", "a"] == 1


def test_group_level_assignment_stability():
    a = compare_aoi_assignments(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    b = compare_aoi_assignments(["a", "a", "b", "b"], ["a", "a", "a", "b"])
    meta = pd.DataFrame(
        {
            "observation_id": [1, 2, 3, 4],
            "participant": ["p1", "p1", "p2", "p2"],
            "trial": [1, 2, 1, 2],
        }
    )
    out = estimate_aoi_assignment_stability(
        {"x": a, "y": b}, metadata=meta, group_cols=["participant", "trial"]
    )
    assert set(out["group_summaries"]) == {"participant", "trial"}
    assert len(out["aoi_level"]) > 0


def test_assignment_frequency_is_descriptive():
    out = estimate_fixation_assignment_probability(
        {"baseline": ["a", "b"], "p1": ["a", "a"], "p2": ["b", "b"]}
    )
    one = out[(out.observation_id == 1) & (out.aoi == "a")].iloc[0]
    assert math.isclose(one.assignment_frequency, 2 / 3)
    assert "not a posterior probability" in out.attrs["caveat"]


def test_recompute_features_does_not_infer_dwell_or_zero_missing():
    d = pd.DataFrame(
        {
            "participant": [1, 1, 1],
            "trial": [1, 1, 1],
            "time": [0.1, 0.2, 0.3],
            "duration": [0.1, np.nan, 0.1],
        }
    )
    with pytest.warns(RuntimeWarning, match="dwell is returned as NA"):
        f = recompute_aoi_features(
            d,
            ["a", "a", OUTSIDE],
            participant_col="participant",
            trial_col="trial",
            time_col="time",
        )
    assert np.isnan(f.iloc[0]["dwell"])
    f2 = recompute_aoi_features(
        d,
        ["a", "a", OUTSIDE],
        participant_col="participant",
        trial_col="trial",
        time_col="time",
        duration_col="duration",
    )
    assert math.isclose(f2.iloc[0]["dwell"], 0.1)


def synthetic_data():
    rng = np.random.default_rng(12)
    rows = []
    for participant in range(1, 9):
        cond = participant % 2
        for trial in range(1, 4):
            centers = [
                (250, 130, "headline"),
                (500, 270, "image"),
                (500, 430, "claim"),
                (500, 560, "disclosure"),
                (800, 650, "cta"),
            ]
            for idx, (cx, cy, aoi) in enumerate(centers):
                n = 5 + (3 if aoi == "disclosure" and cond else 0)
                for j in range(n):
                    rows.append(
                        {
                            "obs": len(rows) + 1,
                            "participant": f"p{participant}",
                            "trial": trial,
                            "condition": cond,
                            "x": rng.normal(cx, 35),
                            "y": rng.normal(cy, 22),
                            "duration": rng.uniform(0.06, 0.18),
                            "time": idx + j / 20,
                        }
                    )
    return pd.DataFrame(rows)


def synthetic_aois():
    return pd.DataFrame(
        {
            "aoi_id": ["headline", "image", "claim", "disclosure", "cta"],
            "xmin": [100, 300, 300, 300, 680],
            "xmax": [400, 700, 700, 700, 920],
            "ymin": [80, 190, 360, 500, 600],
            "ymax": [180, 340, 480, 590, 710],
        }
    )


def model_callback(features, assigned, spec):
    target = features[features.aoi.eq("disclosure")].copy()
    if target.empty:
        raise RuntimeError("no disclosure observations")
    condition_map = assigned[["participant", "condition"]].drop_duplicates()
    target = target.merge(
        condition_map, on="participant", how="left", validate="many_to_one"
    )
    y = target["dwell"].to_numpy(float)
    X = np.column_stack(
        [np.ones(len(target)), target["condition"].to_numpy(float)]
    )
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    df = len(y) - X.shape[1]
    if df <= 0:
        return pd.DataFrame(
            [
                {
                    "term": "condition",
                    "estimate": np.nan,
                    "SE": np.nan,
                    "CI_low": np.nan,
                    "CI_high": np.nan,
                    "p_value": np.nan,
                    "model_converged": False,
                    "N": len(y),
                }
            ]
        )
    s2 = np.sum(resid**2) / df
    cov = s2 * np.linalg.inv(X.T @ X)
    se = math.sqrt(cov[1, 1])
    est = beta[1]
    return pd.DataFrame(
        [
            {
                "term": "condition",
                "estimate": est,
                "SE": se,
                "CI_low": est - 1.96 * se,
                "CI_high": est + 1.96 * se,
                "p_value": np.nan,
                "model_converged": True,
                "N": len(y),
            }
        ]
    )


def sensitivity_result():
    kwargs = {
        "screen_width_px": 1024,
        "screen_height_px": 768,
        "viewing_distance": 60,
        "physical_screen_size": (53.1, 29.9),
        "boundary_policy": "allow",
    }
    grid = create_aoi_perturbation_grid(
        dilations=[0.25, 0.5, 1.0],
        erosions=[0.25],
        translations_x=[0.5],
        translations_y=[0.5],
        unit="deg",
        include_baseline=True,
        **kwargs,
    )
    return run_aoi_sensitivity_analysis(
        synthetic_data(),
        synthetic_aois(),
        grid,
        x_col="x",
        y_col="y",
        observation_id_col="obs",
        participant_col="participant",
        trial_col="trial",
        duration_col="duration",
        time_col="time",
        model_callback=model_callback,
        preprocessing_specification={"duration_unit": "seconds"},
        event_detector="synthetic_fixations",
        quality_rules={"missing": "preserve"},
        model_specification={"family": "OLS", "outcome": "disclosure_dwell"},
    )


def test_full_sensitivity_pipeline_and_model_provenance():
    result = sensitivity_result()
    assert result.eyeprocess_class == "eye_aoi_sensitivity"
    assert len(result["grid_result"]["audit"]) == 7
    assert result["grid_result"]["audit"]["status"].eq("completed").all()
    expected = {
        "term",
        "estimate",
        "SE",
        "CI_low",
        "CI_high",
        "p_value",
        "model_converged",
        "N",
        "direction",
    }
    assert expected.issubset(result["models"].columns)
    assert result["provenance"]["event_detector"] == "synthetic_fixations"
    assert result["provenance"]["quality_rules"]["missing"] == "preserve"


def test_observation_ids_must_be_present_unique_and_nonmissing():
    data = synthetic_data().head(20).copy()
    grid = create_aoi_perturbation_grid(dilations=[2])
    data.loc[data.index[0], "obs"] = np.nan
    with pytest.raises(EyeProcessValidationError, match="unique and non-missing"):
        run_aoi_sensitivity_analysis(
            data,
            synthetic_aois(),
            grid,
            x_col="x",
            y_col="y",
            observation_id_col="obs",
            duration_col="duration",
        )
    data = synthetic_data().head(20).copy()
    with pytest.raises(EyeProcessValidationError, match="absent"):
        run_aoi_sensitivity_analysis(
            data,
            synthetic_aois(),
            grid,
            x_col="x",
            y_col="y",
            observation_id_col="missing_id",
            duration_col="duration",
        )


def test_model_callback_failures_and_partial_nonconvergence_are_preserved():
    data = synthetic_data().head(40).copy()
    grid = create_aoi_perturbation_grid(dilations=[5], translations_x=[5])

    def callback(features, assigned, spec):
        pid = spec["perturbation_id"]
        if pid == "dilate_5_px":
            raise RuntimeError("planned failure")
        return pd.DataFrame(
            [
                {
                    "term": "x",
                    "estimate": 1.0,
                    "SE": 0.2,
                    "CI_low": 0.6,
                    "CI_high": 1.4,
                    "p_value": 0.03,
                    "model_converged": pid != "shift_x_5_px",
                    "N": 10,
                }
            ]
        )

    result = run_aoi_sensitivity_analysis(
        data,
        synthetic_aois(),
        grid,
        x_col="x",
        y_col="y",
        duration_col="duration",
        model_callback=callback,
    )
    assert (
        (result["failures"].stage == "model")
        & result["failures"].message.str.contains("planned failure")
    ).any()
    row = result["models"].loc[
        result["models"].perturbation_id.eq("shift_x_5_px")
    ].iloc[0]
    assert not bool(row.model_converged)
    inf = assess_aoi_inference_stability(result, term="x")
    assert inf.iloc[0].n_converged == 1


def test_reports_and_plots():
    result = sensitivity_result()
    text = report_aoi_sensitivity(result)
    assert "not probabilities" in text
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = [
        plot_aoi_perturbations(
            result,
            perturbation_id="dilate_0.5_deg",
            data=synthetic_data(),
            x_col="x",
            y_col="y",
        ),
        plot_aoi_assignment_stability(result),
        plot_aoi_coefficient_stability(result, term="condition"),
        plot_aoi_robustness_surface(result),
    ]
    assert all(ax is not None for ax in axes)
    plt.close("all")


def test_translate_polygon_and_grid_failure_audit():
    p = translate_aoi(convex_polygon(), 3, -4)
    assert np.allclose(p.iloc[0]["polygon"][0], [503, 196])
    grid = create_aoi_perturbation_grid(erosions=[1000])
    applied = apply_aoi_perturbation_grid(rects(), grid)
    assert set(applied["audit"].status) == {"completed", "failed"}


def test_cross_language_parity_fixture_contract():
    fixture = pd.read_csv(
        Path(__file__).parent / "fixtures" / "aoi_perturbation_parity.csv",
        skipinitialspace=True,
    )
    aois = pd.DataFrame(
        {
            "aoi_id": ["a", "b"],
            "xmin": [0.0, 12.0],
            "xmax": [10.0, 22.0],
            "ymin": [0.0, 0.0],
            "ymax": [10.0, 10.0],
        }
    )
    data = fixture[["x", "y"]].copy()
    data["duration"] = 1.0
    grid = create_aoi_perturbation_grid(dilations=[1.0])
    result = run_aoi_sensitivity_analysis(
        data, aois, grid, x_col="x", y_col="y", duration_col="duration"
    )
    for branch in ("baseline", "dilate_1_px"):
        observed = result["assignments"][branch].tolist()
        expected = [
            None if pd.isna(value) else str(value).strip() for value in fixture[branch]
        ]
        assert observed == expected
    comparison = result["comparisons"]["dilate_1_px"]["summary"].iloc[0]
    assert comparison["n_comparable"] == 4
    assert comparison["proportion_unchanged"] == pytest.approx(0.5)
    assert comparison["proportion_newly_assigned"] == pytest.approx(0.5)


def test_public_package_front_door_exports_aoi_sensitivity_api():
    import eyeprocesspy as ep

    expected = {
        "validate_aoi_geometry",
        "create_aoi_perturbation_grid",
        "run_aoi_sensitivity_analysis",
        "assess_aoi_inference_stability",
        "report_aoi_sensitivity",
        "plot_aoi_robustness_surface",
    }
    assert expected.issubset(set(dir(ep)))
