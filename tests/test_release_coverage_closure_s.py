from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy._aoi_assignment_core as ac
import eyeprocesspy._aoi_geometry_primitives as gp
import eyeprocesspy._aoi_geometry_units as gu
import eyeprocesspy._aoi_geometry_validation as gv
import eyeprocesspy._aoi_perturb_ops as po
import eyeprocesspy.measurement_accountability_11 as ma
from eyeprocesspy.exceptions import EyeProcessValidationError


def rectangles():
    return pd.DataFrame(
        {
            "aoi_id": ["a", "b"],
            "shape_type": ["rectangle", "rectangle"],
            "xmin": [0.0, 20.0],
            "xmax": [10.0, 30.0],
            "ymin": [0.0, 0.0],
            "ymax": [10.0, 10.0],
        }
    )


def overlapping_rectangles():
    return pd.DataFrame(
        {
            "aoi_id": ["a", "b"],
            "shape_type": ["rectangle", "rectangle"],
            "xmin": [0.0, 5.0],
            "xmax": [10.0, 15.0],
            "ymin": [0.0, 0.0],
            "ymax": [10.0, 10.0],
        }
    )


def square(aoi_id: str = "poly"):
    return pd.DataFrame(
        {
            "aoi_id": [aoi_id],
            "shape_type": ["polygon"],
            "polygon": [
                np.array(
                    [
                        [0.0, 0.0],
                        [10.0, 0.0],
                        [10.0, 10.0],
                        [0.0, 10.0],
                    ]
                )
            ],
        }
    )


def valid_model_row(**updates):
    row = {
        "term": "x",
        "estimate": 1.0,
        "SE": 0.2,
        "CI_low": 0.6,
        "CI_high": 1.4,
        "p_value": 0.05,
        "model_converged": True,
        "N": 10,
    }
    row.update(updates)
    return pd.DataFrame([row])


# =================================================================
# GEOMETRY PRIMITIVES
# =================================================================


def test_geometry_primitive_frame_scalar_and_pair_guards():
    frame = pd.DataFrame({"x": [1]})
    copied = gp._frame(frame, "x")

    assert copied.equals(frame)
    assert copied is not frame

    with pytest.raises(
        EyeProcessValidationError,
        match="DataFrame",
    ):
        gp._frame([], "x")

    with pytest.raises(
        EyeProcessValidationError,
        match="finite numeric",
    ):
        gp._finite_scalar("bad", "x")

    with pytest.raises(
        EyeProcessValidationError,
        match="finite numeric",
    ):
        gp._finite_scalar(np.inf, "x")

    with pytest.raises(
        EyeProcessValidationError,
        match="non-negative",
    ):
        gp._finite_scalar(
            -1,
            "x",
            nonnegative=True,
        )

    assert gp._normalise_pair(3, "x") == (3.0, 3.0)
    assert gp._normalise_pair([1, 2], "x") == (1.0, 2.0)

    with pytest.raises(
        EyeProcessValidationError,
        match="numeric",
    ):
        gp._normalise_pair("12", "x")

    with pytest.raises(
        EyeProcessValidationError,
        match="length-2",
    ):
        gp._normalise_pair([1, 2, 3], "x")


def test_stable_hash_normalizes_complex_values():
    frame = pd.DataFrame(
        {
            "array": [np.array([1.0, 2.0])],
            "sequence": [(np.int64(2), np.nan)],
            "mapping": [{"z": np.float64(2.5), "a": 1}],
            "missing": [pd.NA],
        }
    )

    one = gp._stable_frame_hash(frame)
    two = gp._stable_frame_hash(frame.copy())

    assert isinstance(one, str)
    assert len(one) == 64
    assert one == two


def test_polygon_array_validation_and_closed_ring():
    closed = gp._polygon_array(
        [
            [0, 0],
            [2, 0],
            [2, 2],
            [0, 2],
            [0, 0],
        ]
    )

    assert closed.shape == (4, 2)

    with pytest.raises(
        EyeProcessValidationError,
        match="coercible",
    ):
        gp._polygon_array([["x", "y"]] * 3)

    with pytest.raises(
        EyeProcessValidationError,
        match="at least three",
    ):
        gp._polygon_array([[0, 0], [1, 1]])

    with pytest.raises(
        EyeProcessValidationError,
        match="finite",
    ):
        gp._polygon_array(
            [
                [0, 0],
                [1, np.nan],
                [2, 0],
            ]
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="unique",
    ):
        gp._polygon_array(
            [
                [0, 0],
                [1, 0],
                [0, 0],
            ]
        )


def test_segment_intersection_variants():
    a = np.array([0.0, 0.0])
    b = np.array([2.0, 2.0])
    c = np.array([0.0, 2.0])
    d = np.array([2.0, 0.0])

    assert gp._segments_intersect(a, b, c, d)

    assert gp._segments_intersect(
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([3.0, 0.0]),
    )

    assert not gp._segments_intersect(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([3.0, 0.0]),
    )

    assert gp._on_segment(
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([1.0, 0.0]),
    )

    assert not gp._on_segment(
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([3.0, 0.0]),
    )


def test_polygon_convexity_self_intersection_and_parallel_guard():
    convex = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )

    concave = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )

    bow = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [2.0, 0.0],
        ]
    )

    assert gp._polygon_is_convex(convex)
    assert not gp._polygon_is_convex(concave)
    assert gp._polygon_self_intersects(bow)

    with pytest.raises(
        EyeProcessValidationError,
        match="convex polygons",
    ):
        gp._offset_convex_polygon(
            concave,
            0.5,
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="parallel adjacent edges",
    ):
        gp._line_intersection(
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([2.0, 0.0]),
        )


# =================================================================
# UNITS AND PERTURBATION SPECIFICATION
# =================================================================


def test_screen_geometry_guards():
    kwargs = {
        "screen_width_px": 1920,
        "screen_height_px": 1080,
        "viewing_distance": 60,
        "physical_screen_size": (53.0, 30.0),
    }

    out = gu._screen_geometry(**kwargs)
    assert out[0] == 1920
    assert out[2] > 0

    with pytest.raises(
        EyeProcessValidationError,
        match="pixel dimensions",
    ):
        gu._screen_geometry(
            **{
                **kwargs,
                "screen_width_px": 0,
            }
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="viewing_distance",
    ):
        gu._screen_geometry(
            **{
                **kwargs,
                "viewing_distance": 0,
            }
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="Physical screen",
    ):
        gu._screen_geometry(
            **{
                **kwargs,
                "physical_screen_size": (0, 30),
            }
        )


def test_perturbation_spec_contract_guards():
    with pytest.raises(
        EyeProcessValidationError,
        match="non-empty string",
    ):
        gu.aoi_perturbation_spec("", "baseline")

    with pytest.raises(
        EyeProcessValidationError,
        match="Unsupported",
    ):
        gu.aoi_perturbation_spec("x", "mystery")

    with pytest.raises(
        EyeProcessValidationError,
        match="unit",
    ):
        gu.aoi_perturbation_spec(
            "x",
            "baseline",
            unit="cm",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="boundary_policy",
    ):
        gu.aoi_perturbation_spec(
            "x",
            boundary_policy="guess",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="Dilation",
    ):
        gu.aoi_perturbation_spec(
            "x",
            "dilation",
            margin_x=-1,
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="Erosion",
    ):
        gu.aoi_perturbation_spec(
            "x",
            "erosion",
            margin_x=-1,
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="uniform margin",
    ):
        gu.aoi_perturbation_spec(
            "x",
            "dilation",
            margin_x=1,
            margin_y=2,
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="degrees_per_pixel",
    ):
        gu.aoi_perturbation_spec(
            "x",
            unit="deg",
            degrees_per_pixel=(0, 0.1),
        )

    for seed in (True, -1, 1.5):
        with pytest.raises(
            EyeProcessValidationError,
            match="seed",
        ):
            gu.aoi_perturbation_spec(
                "x",
                seed=seed,
            )


def test_degree_spec_resolves_from_screen_geometry():
    spec = gu.aoi_perturbation_spec(
        "degrees",
        "translate",
        translation_x=0.5,
        unit="deg",
        screen_width_px=1920,
        screen_height_px=1080,
        viewing_distance=60,
        physical_screen_size=(53.0, 30.0),
    )

    assert spec["degrees_per_pixel"] is not None

    px = gu._spec_to_px(spec)

    assert px["translation_x"] > 0
    assert px["source_unit"] == "deg"


def test_spec_value_and_missing_degree_provenance():
    assert (
        gu._spec_value(
            {"x": 2},
            "x",
        )
        == 2
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="mapping-like",
    ):
        gu._spec_value(
            object(),
            "x",
        )

    raw = {
        "unit": "deg",
        "margin_x": 1.0,
        "margin_y": 1.0,
        "translation_x": 0.0,
        "translation_y": 0.0,
        "degrees_per_pixel": None,
        "perturbation_id": "x",
        "operation": "baseline",
        "seed": None,
        "screen_width_px": None,
        "screen_height_px": None,
        "boundary_policy": "allow",
    }

    with pytest.raises(
        EyeProcessValidationError,
        match="lacks",
    ):
        gu._spec_to_px(raw)


# =================================================================
# AOI GEOMETRY VALIDATION
# =================================================================


def test_geometry_validation_identifier_and_shape_guards():
    with pytest.raises(
        EyeProcessValidationError,
        match="at least one",
    ):
        gv.validate_aoi_geometry(pd.DataFrame())

    with pytest.raises(
        EyeProcessValidationError,
        match="identifier",
    ):
        gv.validate_aoi_geometry(
            pd.DataFrame(
                {
                    "xmin": [0],
                    "xmax": [1],
                    "ymin": [0],
                    "ymax": [1],
                }
            )
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="non-missing",
    ):
        gv.validate_aoi_geometry(
            pd.DataFrame(
                {
                    "aoi_id": [pd.NA],
                    "xmin": [0],
                    "xmax": [1],
                    "ymin": [0],
                    "ymax": [1],
                }
            )
        )

    duplicate = rectangles()
    duplicate["aoi_id"] = ["a", "a"]

    with pytest.raises(
        EyeProcessValidationError,
        match="unique",
    ):
        gv.validate_aoi_geometry(duplicate)

    unsupported = rectangles().iloc[[0]].copy()
    unsupported["shape_type"] = "ellipse"

    with pytest.raises(
        EyeProcessValidationError,
        match="shape_type",
    ):
        gv.validate_aoi_geometry(unsupported)


def test_geometry_validation_missing_rectangle_fields():
    with pytest.raises(
        EyeProcessValidationError,
        match="require xmin",
    ):
        gv.validate_aoi_geometry(
            pd.DataFrame(
                {
                    "aoi_id": ["x"],
                    "shape_type": ["rectangle"],
                    "xmin": [0.0],
                }
            )
        )

    bad = rectangles().iloc[[0]].copy()
    bad.loc[:, "xmin"] = np.nan

    with pytest.raises(
        EyeProcessValidationError,
        match="finite numeric",
    ):
        gv.validate_aoi_geometry(bad)


def test_polygon_overlap_area_semantics():
    poly = gp._polygon_array(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
        ]
    )

    identical = poly.copy()

    assert gv._polygons_overlap_area(
        poly,
        identical,
    )

    inner = gp._polygon_array(
        [
            [2, 2],
            [4, 2],
            [4, 4],
            [2, 4],
        ]
    )

    assert gv._polygons_overlap_area(
        poly,
        inner,
    )

    disjoint = gp._polygon_array(
        [
            [20, 20],
            [30, 20],
            [30, 30],
            [20, 30],
        ]
    )

    assert not gv._polygons_overlap_area(
        poly,
        disjoint,
    )

    assert gv._point_strictly_inside_polygon(
        np.array([5.0, 5.0]),
        poly,
    )

    assert not gv._point_strictly_inside_polygon(
        np.array([0.0, 5.0]),
        poly,
    )


def test_identical_polygon_validation_reports_overlap():
    p = square("one")
    q = square("two")

    geometries = pd.concat(
        [p, q],
        ignore_index=True,
    )

    result = gv.validate_aoi_geometry(geometries)

    assert result["overlap_present"]

    with pytest.raises(
        EyeProcessValidationError,
        match="allow_overlap=False",
    ):
        gv.validate_aoi_geometry(
            geometries,
            allow_overlap=False,
        )


def test_touching_rectangles_do_not_count_as_area_overlap():
    touching = pd.DataFrame(
        {
            "aoi_id": ["left", "right"],
            "xmin": [0.0, 10.0],
            "xmax": [10.0, 20.0],
            "ymin": [0.0, 0.0],
            "ymax": [10.0, 10.0],
        }
    )

    result = gv.validate_aoi_geometry(touching)

    assert not result["overlap_present"]


# =================================================================
# ASSIGNMENT CORE
# =================================================================


def test_assign_points_missing_overlap_and_all_policy():
    points = pd.DataFrame(
        {
            "x": [
                2.0,
                7.0,
                12.0,
                30.0,
                np.nan,
            ],
            "y": [
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
            ],
        }
    )

    geometry = overlapping_rectangles()

    assigned = ac._assign_points(
        points,
        geometry,
        x_col="x",
        y_col="y",
    )

    assert assigned.tolist() == [
        "a",
        gp.AMBIGUOUS,
        "b",
        gp.OUTSIDE,
        None,
    ]

    all_memberships = ac._assign_points(
        points,
        geometry,
        x_col="x",
        y_col="y",
        overlap_policy="all",
    )

    assert all_memberships.iloc[1] == "a|b"

    with pytest.raises(
        EyeProcessValidationError,
        match="ambiguous overlapping",
    ):
        ac._assign_points(
            points,
            geometry,
            x_col="x",
            y_col="y",
            overlap_policy="error",
        )


def test_assign_points_input_guards():
    with pytest.raises(
        EyeProcessValidationError,
        match="must contain",
    ):
        ac._assign_points(
            pd.DataFrame({"x": [1]}),
            rectangles(),
            x_col="x",
            y_col="y",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="overlap_policy",
    ):
        ac._assign_points(
            pd.DataFrame(
                {
                    "x": [1],
                    "y": [1],
                }
            ),
            rectangles(),
            x_col="x",
            y_col="y",
            overlap_policy="first",
        )


def test_compare_assignments_empty_and_length_guards():
    with pytest.raises(
        EyeProcessValidationError,
        match="equal length",
    ):
        ac.compare_aoi_assignments(
            ["a"],
            ["a", "b"],
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="same length",
    ):
        ac.compare_aoi_assignments(
            ["a"],
            ["a"],
            ids=[1, 2],
        )

    empty = ac.compare_aoi_assignments(
        [],
        [],
    )

    row = empty["summary"].iloc[0]

    assert row["n_total"] == 0
    assert np.isnan(row["proportion_unchanged"])
    assert np.isnan(row["proportion_missing_comparison"])


def test_assignment_stability_dataframe_and_metadata_guards():
    with pytest.raises(
        EyeProcessValidationError,
        match="must contain",
    ):
        ac.estimate_aoi_assignment_stability(pd.DataFrame({"perturbation_id": ["p"]}))

    with pytest.raises(
        EyeProcessValidationError,
        match="No assignment",
    ):
        ac.estimate_aoi_assignment_stability({})

    missing_id = pd.DataFrame(
        {
            "perturbation_id": ["p"],
            "observation_id": [pd.NA],
            "baseline_aoi": ["a"],
            "perturbed_aoi": ["a"],
        }
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="non-missing",
    ):
        ac.estimate_aoi_assignment_stability(missing_id)

    source = pd.DataFrame(
        {
            "perturbation_id": [
                "p",
                "p",
            ],
            "baseline_aoi": [
                "a",
                None,
            ],
            "perturbed_aoi": [
                "a",
                None,
            ],
        }
    )

    result = ac.estimate_aoi_assignment_stability(source)

    assert result["overall"].iloc[0]["n_comparable"] == 1

    with pytest.raises(
        EyeProcessValidationError,
        match="one row per observation",
    ):
        ac.estimate_aoi_assignment_stability(
            source,
            metadata=pd.DataFrame({"participant": ["x"]}),
        )

    metadata_duplicate = pd.DataFrame(
        {
            "observation_id": [1, 1],
            "participant": ["a", "b"],
        }
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="unique and non-missing",
    ):
        ac.estimate_aoi_assignment_stability(
            source,
            metadata=metadata_duplicate,
        )

    metadata_wrong = pd.DataFrame(
        {
            "observation_id": [1, 9],
            "participant": ["a", "b"],
        }
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="match .* exactly",
    ):
        ac.estimate_aoi_assignment_stability(
            source,
            metadata=metadata_wrong,
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="Grouping column",
    ):
        ac.estimate_aoi_assignment_stability(
            source,
            group_cols=["missing"],
        )


def test_assignment_probability_guards_and_empty_paths():
    with pytest.raises(
        EyeProcessValidationError,
        match="Assignment table",
    ):
        ac.estimate_fixation_assignment_probability(pd.DataFrame({"perturbation_id": ["x"]}))

    with pytest.raises(
        EyeProcessValidationError,
        match="same length",
    ):
        ac.estimate_fixation_assignment_probability(
            {
                "a": ["x"],
                "b": ["x", "y"],
            }
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="ids",
    ):
        ac.estimate_fixation_assignment_probability(
            {
                "a": ["x", "y"],
            },
            ids=[1],
        )

    excluded = ac.estimate_fixation_assignment_probability(
        {
            "baseline": ["a"],
        },
        include_baseline=False,
    )

    assert excluded.empty

    all_missing = ac.estimate_fixation_assignment_probability(
        pd.DataFrame(
            {
                "perturbation_id": [
                    "a",
                    "b",
                ],
                "observation_id": [1, 1],
                "aoi_assignment": [
                    None,
                    None,
                ],
            }
        )
    )

    assert all_missing.empty


def test_recompute_feature_validation_paths():
    data = pd.DataFrame(
        {
            "participant": ["p1", "p1"],
            "trial": [1, 1],
            "duration": [0.1, 0.2],
            "time": [0.4, 0.2],
        }
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="observation_level",
    ):
        ac.recompute_aoi_features(
            data,
            ["a", "b"],
            observation_level="visit",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="one value per",
    ):
        ac.recompute_aoi_features(
            data,
            ["a"],
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="Missing grouping",
    ):
        ac.recompute_aoi_features(
            data,
            ["a", "b"],
            participant_col="missing",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="duration_col",
    ):
        ac.recompute_aoi_features(
            data,
            ["a", "b"],
            duration_col="missing",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="time_col",
    ):
        ac.recompute_aoi_features(
            data,
            ["a", "b"],
            time_col="missing",
        )

    for levels in (
        ["a", "a"],
        ["a", ""],
    ):
        with pytest.raises(
            EyeProcessValidationError,
            match="unique non-empty",
        ):
            ac.recompute_aoi_features(
                data,
                ["a", "b"],
                aoi_levels=levels,
            )


def test_recompute_feature_empty_inference_and_sample_semantics():
    data = pd.DataFrame(
        {
            "duration": [
                0.1,
                0.2,
            ],
            "time": [
                0.4,
                0.2,
            ],
        }
    )

    empty = ac.recompute_aoi_features(
        data,
        [
            gp.OUTSIDE,
            gp.AMBIGUOUS,
        ],
        duration_col="duration",
        time_col="time",
    )

    assert empty.empty

    sample = ac.recompute_aoi_features(
        data,
        ["a", "b"],
        duration_col="duration",
        time_col="time",
        aoi_levels=["a", "b", "c"],
        observation_level="sample",
        perturbation_id="p",
    )

    a = sample[sample["aoi"].eq("a")].iloc[0]

    c = sample[sample["aoi"].eq("c")].iloc[0]

    assert a["sample_count"] == 1
    assert pd.isna(a["fixation_count"])
    assert np.isnan(a["first_fixation"])
    assert a["first_observation"] == pytest.approx(0.4)

    assert c["sample_count"] == 0
    assert c["dwell"] == 0
    assert bool(c["duration_complete"])
    assert bool(c["time_complete"])
    assert not bool(c["inspected"])


def test_validate_model_table_contract_failures():
    with pytest.raises(
        EyeProcessValidationError,
        match="missing required",
    ):
        ac._validate_model_table(
            pd.DataFrame({"term": ["x"]}),
            "p",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="non-missing and non-empty",
    ):
        ac._validate_model_table(
            valid_model_row(term=""),
            "p",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="booleans",
    ):
        ac._validate_model_table(
            valid_model_row(model_converged="yes"),
            "p",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="numeric",
    ):
        ac._validate_model_table(
            valid_model_row(estimate="not-a-number"),
            "p",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="non-negative",
    ):
        ac._validate_model_table(
            valid_model_row(
                model_converged=False,
                N=-1,
            ),
            "p",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="integer-valued",
    ):
        ac._validate_model_table(
            valid_model_row(
                model_converged=False,
                N=1.5,
            ),
            "p",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="finite estimate",
    ):
        ac._validate_model_table(
            valid_model_row(estimate=np.nan),
            "p",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="N > 0",
    ):
        ac._validate_model_table(
            valid_model_row(N=0),
            "p",
        )


def test_validate_model_table_direction_outputs():
    frame = pd.concat(
        [
            valid_model_row(
                term="positive",
                estimate=1,
            ),
            valid_model_row(
                term="negative",
                estimate=-1,
            ),
            valid_model_row(
                term="zero",
                estimate=0,
            ),
            valid_model_row(
                term="missing",
                estimate=np.nan,
                SE=np.nan,
                CI_low=np.nan,
                CI_high=np.nan,
                model_converged=False,
                N=np.nan,
            ),
        ],
        ignore_index=True,
    )

    out = ac._validate_model_table(
        frame,
        "perturbation",
    )

    assert out["direction"].tolist() == [
        "positive",
        "negative",
        "zero",
        "missing",
    ]

    assert set(out["perturbation_id"]) == {"perturbation"}


# =================================================================
# PERTURBATION OPERATIONS
# =================================================================


def test_geometry_bounds_rectangle_and_polygon():
    rect = gv.validate_aoi_geometry(rectangles().iloc[[0]])["geometry"].iloc[0]

    assert po._geometry_bounds(rect) == (0.0, 10.0, 0.0, 10.0)

    poly = gv.validate_aoi_geometry(square())["geometry"].iloc[0]

    assert po._geometry_bounds(poly) == (0.0, 10.0, 0.0, 10.0)


def test_clip_geometry_collapse_guards():
    outside_rectangle = pd.DataFrame(
        {
            "aoi_id": ["r"],
            "shape_type": ["rectangle"],
            "xmin": [-10.0],
            "xmax": [-5.0],
            "ymin": [1.0],
            "ymax": [5.0],
            "polygon": [None],
        }
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="clipping collapsed",
    ):
        po._clip_geometry(
            outside_rectangle,
            100,
            100,
        )

    outside_polygon = pd.DataFrame(
        {
            "aoi_id": ["p"],
            "shape_type": ["polygon"],
            "polygon": [
                np.array(
                    [
                        [-10.0, 0.0],
                        [-5.0, 5.0],
                        [-10.0, 10.0],
                    ]
                )
            ],
        }
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="invalidated polygon",
    ):
        po._clip_geometry(
            outside_polygon,
            100,
            100,
        )


def test_boundary_policy_internal_paths():
    geometry = gv.validate_aoi_geometry(rectangles().iloc[[0]])["geometry"]

    no_screen = {
        "screen_width_px": None,
        "screen_height_px": None,
        "boundary_policy": "error",
    }

    pd.testing.assert_frame_equal(
        po._apply_boundary_policy(
            geometry,
            no_screen,
        ),
        geometry,
    )

    inside = {
        "screen_width_px": 100,
        "screen_height_px": 100,
        "boundary_policy": "error",
    }

    pd.testing.assert_frame_equal(
        po._apply_boundary_policy(
            geometry,
            inside,
        ),
        geometry,
    )

    shifted = geometry.copy()
    shifted.loc[:, "xmin"] = -5.0

    with pytest.raises(
        EyeProcessValidationError,
        match="extend beyond",
    ):
        po._apply_boundary_policy(
            shifted,
            {
                **inside,
                "boundary_policy": "error",
            },
        )

    with pytest.warns(
        RuntimeWarning,
        match="extend beyond",
    ):
        warned = po._apply_boundary_policy(
            shifted,
            {
                **inside,
                "boundary_policy": "warn",
            },
        )

    assert warned.loc[0, "xmin"] == -5

    allowed = po._apply_boundary_policy(
        shifted,
        {
            **inside,
            "boundary_policy": "allow",
        },
    )

    assert allowed.loc[0, "xmin"] == -5


def test_polygon_jitter_and_anisotropic_paths():
    polygon = square()

    jittered = po.jitter_aoi(
        polygon,
        1.0,
        seed=11,
        boundary_policy="allow",
    )

    original = polygon.iloc[0]["polygon"]

    moved = jittered.iloc[0]["polygon"]

    assert not np.allclose(
        original,
        moved,
    )

    expanded_spec = gu.aoi_perturbation_spec(
        "expand",
        "anisotropic_expansion",
        margin_x=2,
        margin_y=1,
        boundary_policy="allow",
    )

    expanded = po._transform_geometry(
        polygon,
        expanded_spec,
    )

    assert (
        np.ptp(
            expanded.iloc[0]["polygon"][
                :,
                0,
            ]
        )
        > 10
    )

    collapse_spec = gu.aoi_perturbation_spec(
        "collapse",
        "anisotropic_expansion",
        margin_x=-6,
        margin_y=0,
        boundary_policy="allow",
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="collapsed polygon",
    ):
        po._transform_geometry(
            polygon,
            collapse_spec,
        )


def test_grid_pair_input_guards():
    with pytest.raises(
        EyeProcessValidationError,
        match="exactly two columns",
    ):
        po.create_aoi_perturbation_grid(
            translations_xy=pd.DataFrame(
                {
                    "x": [1],
                    "y": [2],
                    "z": [3],
                }
            )
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="length two",
    ):
        po.create_aoi_perturbation_grid(translations_xy=np.array([1, 2, 3]))

    with pytest.raises(
        EyeProcessValidationError,
        match="exactly two columns",
    ):
        po.create_aoi_perturbation_grid(
            anisotropic=np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            )
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="grid is empty",
    ):
        po.create_aoi_perturbation_grid(include_baseline=False)

    with pytest.raises(
        EyeProcessValidationError,
        match="not unique",
    ):
        po.create_aoi_perturbation_grid(
            dilations=[1, 1],
            include_baseline=False,
        )


def test_grid_apply_invalid_object_and_partial_failure():
    with pytest.raises(
        EyeProcessValidationError,
        match="create_aoi_perturbation_grid",
    ):
        po.apply_aoi_perturbation_grid(
            rectangles(),
            object(),
        )

    good = gu.aoi_perturbation_spec(
        "good",
        "baseline",
        boundary_policy="allow",
    )

    bad = gu.aoi_perturbation_spec(
        "bad",
        "erosion",
        margin_x=100,
        boundary_policy="allow",
    )

    result = po.apply_aoi_perturbation_grid(
        rectangles(),
        {
            "specifications": [
                good,
                bad,
            ]
        },
    )

    audit = result["audit"].set_index("perturbation_id")

    assert (
        audit.loc[
            "good",
            "status",
        ]
        == "completed"
    )

    assert (
        audit.loc[
            "bad",
            "status",
        ]
        == "failed"
    )

    assert "good" in result["geometries"]
    assert "bad" not in result["geometries"]


# =================================================================
# MEASUREMENT ACCOUNTABILITY
# =================================================================


def test_accountability_numeric_helpers():
    assert ma._finite(
        [
            1,
            "2",
            "bad",
            np.inf,
            None,
        ]
    ) == [1.0, 2.0]

    assert math.isnan(ma._median([]))

    assert ma._median([1, 4, 2, 3]) == pytest.approx(2.5)

    assert math.isnan(ma._mad([]))

    assert ma._mad(
        [1, 2, 3],
        center=2,
    ) == pytest.approx(1)

    assert (
        ma._first_sustained(
            [False, True, True],
            2,
        )
        == 1
    )

    assert (
        ma._first_sustained(
            [False, False],
            0,
        )
        is None
    )


def test_pupil_latency_validation_paths():
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        ma.pupil_latency_sensitivity(
            list(range(8)),
            list(range(7)),
        )

    with pytest.raises(
        ValueError,
        match="fewer than 8 finite",
    ):
        ma.pupil_latency_sensitivity(
            list(range(8)),
            [
                1,
                1,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        )

    with pytest.raises(
        ValueError,
        match="increasing samples",
    ):
        ma.pupil_latency_sensitivity(
            [0.0] * 8,
            [1.0] * 8,
            baseline_window=(-1, 1),
            search_window=(-1, 1),
        )

    with pytest.raises(
        ValueError,
        match="too few samples",
    ):
        ma.pupil_latency_sensitivity(
            [
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
            ],
            [1.0] * 8,
        )


def test_pupil_latency_positive_direction_without_simulation():
    time = [-0.5 + i / 100 for i in range(201)]

    pupil = [(4.0 if t < 0.25 else 4.0 + 0.8 * (1.0 - math.exp(-(t - 0.25) / 0.2))) for t in time]

    out = ma.pupil_latency_sensitivity(
        time,
        pupil,
        direction="increase",
        simulations=0,
        threshold_sigma=2,
    )

    assert out["sampling_hz"] == pytest.approx(
        100,
        rel=1e-6,
    )

    assert out["simulation"]["n"] == 0
    assert out["simulation"]["successful"] == 0

    assert out["provenance"]["direction"] == "increase"


def test_event_marker_qc_all_status_paths():
    with pytest.raises(
        ValueError,
        match="tolerance",
    ):
        ma.event_marker_qc(
            [0],
            tolerance=0,
        )

    missing = ma.event_marker_qc(
        [
            "bad",
            np.nan,
        ],
        tolerance=0.05,
    )

    assert missing["status"] == "implausible"
    assert missing["n"] == 0

    plausible = ma.event_marker_qc(
        [0.04],
        tolerance=0.05,
        min_corroborating=2,
    )

    assert plausible["status"] == "plausible"

    ambiguous_direction = ma.event_marker_qc(
        [0.01, 0.015],
        tolerance=0.05,
        expected_direction="negative",
        effects=[1.0, 2.0],
        min_corroborating=2,
    )

    assert ambiguous_direction["status"] == "ambiguous"

    assert not ambiguous_direction["direction_consistent"]

    implausible = ma.event_marker_qc(
        [-1.0, 1.0],
        tolerance=0.05,
        min_corroborating=2,
    )

    assert implausible["status"] == "implausible"


def test_validation_ladder_aliases_and_statuses():
    supported = ma.validation_ladder(
        {
            "acquisition_qc": "ok",
            "analytical_qc": "passed",
            "construct_check": True,
            "within_person": True,
            "held_out_person": True,
        }
    )

    assert supported["claim_status"] == "supported"
    assert supported["first_blocker"] is None

    qualified = ma.validation_ladder(
        {
            "acquisition_qc": "warning",
            "analytical_qc": True,
            "construct_check": True,
            "within_person": True,
            "held_out_person": True,
        }
    )

    assert qualified["claim_status"] == "qualified"

    failed = ma.validation_ladder(
        {
            "acquisition_qc": False,
            "analytical_qc": True,
            "construct_check": True,
            "within_person": True,
            "held_out_person": True,
        }
    )

    assert failed["claim_status"] == "not_supported"
    assert failed["first_blocker"] == "acquisition_qc"

    missing = ma.validation_ladder(
        {
            "acquisition_qc": True,
            "analytical_qc": True,
            "construct_check": True,
            "within_person": True,
            "held_out_person": "missing",
        }
    )

    assert missing["claim_status"] == "qualified"

    with pytest.raises(
        ValueError,
        match="invalid stage status",
    ):
        ma.validation_ladder({"acquisition_qc": "mystery"})


def test_validation_ladder_generalization_aliases():
    for claim in (
        "generalizable",
        "generalization",
        "out-of-person",
        "population",
    ):
        result = ma.validation_ladder(
            {
                "acquisition_qc": True,
                "analytical_qc": True,
                "construct_check": True,
                "within_person": True,
                "held_out_person": None,
            },
            claim=claim,
        )

        assert result["claim_status"] == "not_supported"
