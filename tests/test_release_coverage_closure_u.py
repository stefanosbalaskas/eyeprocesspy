from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy._aoi_assignment_core as ac
import eyeprocesspy._aoi_geometry_primitives as gp
import eyeprocesspy._aoi_geometry_units as gu
import eyeprocesspy._aoi_geometry_validation as gv
import eyeprocesspy._aoi_perturb_ops as po
import eyeprocesspy.measurement_accountability_11 as ma
import eyeprocesspy.plots_aoi_perturbation as pap
from eyeprocesspy.exceptions import EyeProcessValidationError


def _rectangle(
    aoi_id: str = "rect",
    *,
    xmin: float = 0.0,
    xmax: float = 10.0,
    ymin: float = 0.0,
    ymax: float = 10.0,
):
    return pd.DataFrame(
        {
            "aoi_id": [aoi_id],
            "shape_type": ["rectangle"],
            "xmin": [xmin],
            "xmax": [xmax],
            "ymin": [ymin],
            "ymax": [ymax],
        }
    )


def _square(
    aoi_id: str = "poly",
    *,
    clockwise: bool = False,
):
    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ]
    )

    if clockwise:
        points = points[::-1].copy()

    return pd.DataFrame(
        {
            "aoi_id": [aoi_id],
            "shape_type": ["polygon"],
            "polygon": [points],
        }
    )


# =================================================================
# AOI ASSIGNMENT RESIDUAL
# =================================================================


def test_recompute_single_group_scalar_key_branch():
    data = pd.DataFrame(
        {
            "participant": [
                "P1",
                "P1",
                "P2",
            ],
            "duration": [
                0.10,
                0.20,
                0.30,
            ],
            "time": [
                0.20,
                0.40,
                0.10,
            ],
        }
    )

    out = ac.recompute_aoi_features(
        data,
        [
            "a",
            "b",
            "a",
        ],
        participant_col="participant",
        duration_col="duration",
        time_col="time",
        aoi_levels=[
            "a",
            "b",
        ],
    )

    assert set(out["participant"]) == {
        "P1",
        "P2",
    }

    assert len(out) == 4

    p2_b = out[out["participant"].eq("P2") & out["aoi"].eq("b")].iloc[0]

    assert p2_b["fixation_count"] == 0
    assert p2_b["dwell"] == 0


# =================================================================
# GEOMETRY PRIMITIVE RESIDUALS
# =================================================================


def test_segment_intersection_each_collinear_endpoint_branch():
    a = np.array([0.0, 0.0])
    b = np.array([4.0, 0.0])

    # d lies on a-b: second collinear special case.
    assert gp._segments_intersect(
        a,
        b,
        np.array([2.0, 2.0]),
        np.array([2.0, 0.0]),
    )

    # a lies on c-d: third collinear special case.
    assert gp._segments_intersect(
        a,
        np.array([0.0, 2.0]),
        np.array([-1.0, 0.0]),
        np.array([1.0, 0.0]),
    )

    # b lies on c-d: fourth collinear special case.
    assert gp._segments_intersect(
        np.array([0.0, 2.0]),
        np.array([0.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([1.0, 0.0]),
    )


def test_non_self_intersecting_polygon_exercises_skip_paths():
    square = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )

    assert not gp._polygon_self_intersects(square)


def test_convexity_accepts_collinear_boundary_vertex():
    polygon = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )

    assert gp._polygon_is_convex(polygon)


def test_offset_zero_area_guard_after_convexity_override(
    monkeypatch,
):
    monkeypatch.setattr(
        gp,
        "_polygon_is_convex",
        lambda poly: True,
    )

    line = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="area",
    ):
        gp._offset_convex_polygon(
            line,
            0.1,
        )


def test_offset_zero_length_edge_guard(
    monkeypatch,
):
    monkeypatch.setattr(
        gp,
        "_polygon_is_convex",
        lambda poly: True,
    )

    polygon = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="zero-length edge",
    ):
        gp._offset_convex_polygon(
            polygon,
            0.1,
        )


def test_clockwise_polygon_offset_branch():
    polygon = np.array(
        [
            [0.0, 10.0],
            [10.0, 10.0],
            [10.0, 0.0],
            [0.0, 0.0],
        ]
    )

    out = gp._offset_convex_polygon(
        polygon,
        0.5,
    )

    assert out.shape == (4, 2)

    assert abs(gp._signed_polygon_area(out)) > abs(gp._signed_polygon_area(polygon))


def test_offset_invalidated_polygon_guard(
    monkeypatch,
):
    polygon = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )

    monkeypatch.setattr(
        gp,
        "_polygon_self_intersects",
        lambda poly: True,
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="collapsed or invalidated",
    ):
        gp._offset_convex_polygon(
            polygon,
            0.1,
        )


def test_offset_inversion_guard(
    monkeypatch,
):
    polygon = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )

    real_area = gp._signed_polygon_area

    calls = {"n": 0}

    def fake_area(poly):
        calls["n"] += 1

        if calls["n"] == 1:
            return real_area(poly)

        if calls["n"] == 2:
            return abs(real_area(poly))

        return -abs(real_area(poly))

    monkeypatch.setattr(
        gp,
        "_signed_polygon_area",
        fake_area,
    )

    monkeypatch.setattr(
        gp,
        "_polygon_self_intersects",
        lambda poly: False,
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="inverted geometry",
    ):
        gp._offset_convex_polygon(
            polygon,
            0.1,
        )


# =================================================================
# GEOMETRY VALIDATION RESIDUALS
# =================================================================


def test_shape_inference_from_polygon_without_shape_type():
    frame = pd.DataFrame(
        {
            "aoi_id": ["poly"],
            "polygon": [
                np.array(
                    [
                        [0.0, 0.0],
                        [2.0, 0.0],
                        [2.0, 2.0],
                        [0.0, 2.0],
                    ]
                )
            ],
        }
    )

    out = gv.validate_aoi_geometry(frame)

    assert out["geometry"].iloc[0]["shape_type"] == "polygon"


def test_zero_area_polygon_validation_guard():
    frame = pd.DataFrame(
        {
            "aoi_id": ["flat"],
            "shape_type": ["polygon"],
            "polygon": [
                np.array(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [2.0, 0.0],
                    ]
                )
            ],
        }
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="zero area",
    ):
        gv.validate_aoi_geometry(frame)


def test_polygon_containment_both_argument_orders():
    outer = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ]
    )

    inner = np.array(
        [
            [2.0, 2.0],
            [4.0, 2.0],
            [4.0, 4.0],
        ]
    )

    assert gv._polygons_overlap_area(
        inner,
        outer,
    )

    assert gv._polygons_overlap_area(
        outer,
        inner,
    )


def test_polygon_unequal_length_disjoint_fallthrough():
    triangle = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    square = np.array(
        [
            [10.0, 10.0],
            [12.0, 10.0],
            [12.0, 12.0],
            [10.0, 12.0],
        ]
    )

    assert not gv._polygons_overlap_area(
        triangle,
        square,
    )


# =================================================================
# PERTURBATION OPERATION RESIDUALS
# =================================================================


def test_polygon_clip_success_path():
    polygon = pd.DataFrame(
        {
            "aoi_id": ["poly"],
            "shape_type": ["polygon"],
            "polygon": [
                np.array(
                    [
                        [-1.0, 1.0],
                        [5.0, 1.0],
                        [5.0, 5.0],
                        [-1.0, 5.0],
                    ]
                )
            ],
        }
    )

    clipped = po._clip_geometry(
        polygon,
        10,
        10,
    )

    points = clipped.iloc[0]["polygon"]

    assert points[:, 0].min() == 0
    assert points[:, 0].max() == 5


def test_boundary_clip_policy_happy_path():
    geometry = gv.validate_aoi_geometry(
        _rectangle(
            xmin=-2.0,
            xmax=5.0,
            ymin=1.0,
            ymax=5.0,
        )
    )["geometry"]

    with pytest.warns(
        RuntimeWarning,
        match="clipping",
    ):
        out = po._apply_boundary_policy(
            geometry,
            {
                "screen_width_px": 100,
                "screen_height_px": 100,
                "boundary_policy": "clip",
            },
        )

    assert (
        out.loc[
            0,
            "xmin",
        ]
        == 0
    )


def test_rectangle_anisotropic_expansion_success():
    spec = gu.aoi_perturbation_spec(
        "rect_expand",
        "anisotropic_expansion",
        margin_x=2,
        margin_y=1,
        boundary_policy="allow",
    )

    out = po._transform_geometry(
        _rectangle(),
        spec,
    )

    assert out.loc[
        0,
        "xmin",
    ] == pytest.approx(-2)

    assert out.loc[
        0,
        "ymax",
    ] == pytest.approx(11)


def test_degree_polygon_missing_dpp_defensive_guard(
    monkeypatch,
):
    polygon = _square()

    raw_spec = {
        "perturbation_id": "raw_deg",
        "operation": "dilation",
        "unit": "deg",
        "margin_x": 1.0,
        "margin_y": 1.0,
        "translation_x": 0.0,
        "translation_y": 0.0,
        "degrees_per_pixel": None,
    }

    monkeypatch.setattr(
        po,
        "_spec_to_px",
        lambda spec: {
            "perturbation_id": "raw_deg",
            "operation": "dilation",
            "unit": "px",
            "margin_x": 1.0,
            "margin_y": 1.0,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "seed": None,
            "screen_width_px": None,
            "screen_height_px": None,
            "boundary_policy": "allow",
        },
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="lacks degrees-per-pixel",
    ):
        po._transform_geometry(
            polygon,
            raw_spec,
        )


def test_degree_polygon_erosion_sign_branch():
    spec = gu.aoi_perturbation_spec(
        "deg_erode",
        "erosion",
        margin_x=0.1,
        unit="deg",
        degrees_per_pixel=(
            0.1,
            0.1,
        ),
        boundary_policy="allow",
    )

    out = po._transform_geometry(
        _square(),
        spec,
    )

    points = out.iloc[0]["polygon"]

    assert np.ptp(points[:, 0]) < 10


def test_transformed_polygon_invalidity_guard(
    monkeypatch,
):
    spec = gu.aoi_perturbation_spec(
        "translate",
        "translate",
        translation_x=1,
        translation_y=1,
        boundary_policy="allow",
    )

    monkeypatch.setattr(
        po,
        "_polygon_self_intersects",
        lambda poly: True,
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="invalidated polygon",
    ):
        po._transform_geometry(
            _square(),
            spec,
        )


def test_grid_pair_success_variants():
    dataframe_grid = po.create_aoi_perturbation_grid(
        translations_xy=pd.DataFrame(
            {
                "x": [
                    1,
                    2,
                ],
                "y": [
                    3,
                    4,
                ],
            }
        ),
        include_baseline=False,
    )

    assert len(dataframe_grid["specifications"]) == 2

    vector_grid = po.create_aoi_perturbation_grid(
        translations_xy=np.array(
            [
                1.0,
                2.0,
            ]
        ),
        include_baseline=False,
    )

    assert len(vector_grid["specifications"]) == 1

    sequence_grid = po.create_aoi_perturbation_grid(
        translations_xy=[
            [
                1,
                2,
            ],
            [
                3,
                4,
            ],
        ],
        include_baseline=False,
    )

    assert len(sequence_grid["specifications"]) == 2

    generator_grid = po.create_aoi_perturbation_grid(
        translations_xy=(
            pair
            for pair in (
                (
                    1,
                    2,
                ),
                (
                    3,
                    4,
                ),
            )
        ),
        include_baseline=False,
    )

    assert len(generator_grid["specifications"]) == 2

    jitter_grid = po.create_aoi_perturbation_grid(
        jitters=[
            1,
            2,
        ],
        include_baseline=False,
        seed=10,
    )

    assert len(jitter_grid["specifications"]) == 2


# =================================================================
# MEASUREMENT ACCOUNTABILITY RESIDUALS
# =================================================================


def _latency_fixture():
    time = np.arange(
        -0.5,
        2.01,
        0.02,
    )

    pupil = []

    for index, value in enumerate(time):
        if value < 0:
            pupil.append(4.0 + 0.01 * math.sin(index))
        elif value < 0.30:
            pupil.append(4.0)
        else:
            pupil.append(4.0 + 0.8 * (1.0 - math.exp(-(value - 0.30) / 0.20)))

    return (
        list(time),
        pupil,
    )


def test_latency_filter_skips_nonconvertible_pair():
    time, pupil = _latency_fixture()

    time.insert(
        4,
        "not-a-time",
    )

    pupil.insert(
        4,
        "not-a-pupil",
    )

    out = ma.pupil_latency_sensitivity(
        time,
        pupil,
        direction="increase",
        simulations=0,
        threshold_sigma=2,
    )

    assert out["sampling_hz"] > 0


def test_latency_nonzero_noise_path():
    time, pupil = _latency_fixture()

    out = ma.pupil_latency_sensitivity(
        time,
        pupil,
        direction="increase",
        simulations=0,
        threshold_sigma=2,
    )

    assert out["noise_mad_sigma"] > 0
    assert out["sampling_hz"] == pytest.approx(
        50,
        rel=0.05,
    )


def test_event_marker_direction_with_no_finite_effects():
    out = ma.event_marker_qc(
        [
            0.01,
            0.015,
        ],
        tolerance=0.05,
        expected_direction="negative",
        effects=[
            "bad",
            np.nan,
        ],
        min_corroborating=2,
    )

    assert out["direction_consistent"]

    assert out["status"] == "confirmed"


# =================================================================
# AOI PLOTTING RESIDUALS
# =================================================================


def _plot_geometry():
    rect = gv.validate_aoi_geometry(_rectangle("rect"))["geometry"]

    poly = gv.validate_aoi_geometry(_square("poly"))["geometry"]

    return pd.concat(
        [
            rect,
            poly,
        ],
        ignore_index=True,
        sort=False,
    )


def test_plot_aoi_perturbations_grid_assignment_path():
    nominal = _plot_geometry()

    perturbed = nominal.copy()

    rect_mask = perturbed["shape_type"].eq("rectangle")

    perturbed.loc[
        rect_mask,
        [
            "xmin",
            "xmax",
        ],
    ] = (
        perturbed.loc[
            rect_mask,
            [
                "xmin",
                "xmax",
            ],
        ]
        + 1
    )

    poly_index = perturbed.index[perturbed["shape_type"].eq("polygon")][0]

    moved_polygon = perturbed.at[
        poly_index,
        "polygon",
    ].copy()

    moved_polygon[
        :,
        0,
    ] += 1

    perturbed.at[
        poly_index,
        "polygon",
    ] = moved_polygon

    payload = {
        "nominal_geometry": nominal,
        "grid_result": {
            "geometries": {
                "baseline": nominal,
                "shift": perturbed,
            }
        },
        "assignments": {
            "baseline": [
                "rect",
                "poly",
                "rect",
            ],
            "shift": [
                "rect",
                "rect",
                "poly",
            ],
        },
    }

    observations = pd.DataFrame(
        {
            "x": [
                1.0,
                2.0,
                3.0,
            ],
            "y": [
                1.0,
                2.0,
                3.0,
            ],
        }
    )

    ax = pap.plot_aoi_perturbations(
        payload,
        data=observations,
        x_col="x",
        y_col="y",
    )

    assert ax.get_title() == "AOI perturbation: shift"

    plt.close("all")


def test_plot_aoi_perturbations_requires_xy_for_data():
    geometry = _plot_geometry()

    payload = {
        "nominal_geometry": geometry,
        "perturbed_geometry": geometry,
        "perturbation_id": "same",
    }

    with pytest.raises(
        EyeProcessValidationError,
        match="x_col",
    ):
        pap.plot_aoi_perturbations(
            payload,
            data=pd.DataFrame({"x": [1.0]}),
        )

    plt.close("all")


def test_plot_assignment_stability_happy_path():
    payload = {
        "overall": pd.DataFrame(
            {
                "perturbation_id": [
                    "baseline",
                    "shift",
                ],
                "proportion_unchanged": [
                    1.0,
                    0.75,
                ],
            }
        )
    }

    ax = pap.plot_aoi_assignment_stability(payload)

    assert ax.get_ylabel() == "Proportion unchanged"

    plt.close("all")


def test_plot_coefficient_stability_nonconvergence_and_missing_term():
    payload = {
        "models": pd.DataFrame(
            {
                "term": [
                    "condition",
                    "condition",
                ],
                "estimate": [
                    0.2,
                    0.1,
                ],
                "CI_low": [
                    0.1,
                    -0.1,
                ],
                "CI_high": [
                    0.3,
                    0.3,
                ],
                "model_converged": [
                    True,
                    False,
                ],
                "perturbation_id": [
                    "baseline",
                    "shift",
                ],
            }
        )
    }

    ax = pap.plot_aoi_coefficient_stability(
        payload,
        term="condition",
    )

    assert ax.get_title() == "Coefficient stability: condition"

    with pytest.raises(
        EyeProcessValidationError,
        match="No coefficient rows",
    ):
        pap.plot_aoi_coefficient_stability(
            payload,
            term="missing",
        )

    plt.close("all")


def test_plot_robustness_surface_happy_and_failure_paths():
    payload = {
        "stability": {
            "overall": pd.DataFrame(
                {
                    "perturbation_id": [
                        "p1",
                        "p2",
                        "p3",
                        "p4",
                    ],
                    "proportion_unchanged": [
                        0.9,
                        0.8,
                        0.7,
                        0.6,
                    ],
                }
            )
        },
        "grid": {
            "table": pd.DataFrame(
                {
                    "perturbation_id": [
                        "p1",
                        "p2",
                        "p3",
                        "p4",
                    ],
                    "margin_x": [
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                    ],
                    "margin_y": [
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                }
            )
        },
    }

    ax = pap.plot_aoi_robustness_surface(payload)

    assert ax.get_title() == "AOI robustness surface"

    with pytest.raises(
        EyeProcessValidationError,
        match="columns are unavailable",
    ):
        pap.plot_aoi_robustness_surface(
            payload,
            x_col="missing",
        )

    empty = {
        "stability": {
            "overall": pd.DataFrame(
                {
                    "perturbation_id": ["p1"],
                    "proportion_unchanged": [np.nan],
                }
            )
        },
        "grid": {
            "table": pd.DataFrame(
                {
                    "perturbation_id": ["p1"],
                    "margin_x": [np.nan],
                    "margin_y": [np.nan],
                }
            )
        },
    }

    with pytest.raises(
        EyeProcessValidationError,
        match="finite",
    ):
        pap.plot_aoi_robustness_surface(empty)

    plt.close("all")
