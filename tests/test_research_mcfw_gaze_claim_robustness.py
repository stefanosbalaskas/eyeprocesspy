from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research" / "gaze_claim_robustness"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mcfw_adapter = _load_module(
    "mcfw_adapter_research_test",
    RESEARCH / "mcfw_adapter.py",
)
mcfw_coordinates = _load_module(
    "mcfw_coordinates_research_test",
    RESEARCH / "mcfw_coordinates.py",
)
mcfw_validation = _load_module(
    "mcfw_validation_research_test",
    RESEARCH / "run_mcfw_detector_validation.py",
)
mcfw_summary = _load_module(
    "mcfw_summary_research_test",
    RESEARCH / "summarise_mcfw_validation.py",
)


def _write_mcfw_trial(
    tmp_path: Path,
    *,
    nonincreasing: bool = False,
) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    participant = root / "data" / "participant_001"
    participant.mkdir(parents=True)
    path = participant / "image_01_1.tsv"

    n = 48
    device = np.arange(n, dtype=np.int64) * 8333 + 1_000_000
    if nonincreasing:
        device[20] = device[19]

    x = np.full(n, 0.5)
    y = np.full(n, 0.5)
    valid = np.ones(n, dtype=bool)
    valid[20] = False

    frame = pd.DataFrame(
        {
            "device_time_stamp": device,
            "system_time_stamp": device + 10_000,
            "left_gaze_point_on_display_area_x": x,
            "left_gaze_point_on_display_area_y": y,
            "left_gaze_point_valid": valid,
            "left_gaze_point_available": True,
            "right_gaze_point_on_display_area_x": x,
            "right_gaze_point_on_display_area_y": y,
            "right_gaze_point_valid": valid,
            "right_gaze_point_available": True,
        }
    )
    frame.to_csv(path, sep="\t", index=False)
    return root, path


def _manifest_row(root: Path, path: Path) -> dict[str, object]:
    return {
        "participant": "participant_001",
        "trial": path.stem,
        "trial_family": "natural_image",
        "path": path.relative_to(root).as_posix(),
        "n_samples": 48,
        "left_usable_fraction": 47 / 48,
        "right_usable_fraction": 47 / 48,
        "either_eye_usable_fraction": 47 / 48,
        "both_eyes_usable_fraction": 47 / 48,
    }


def test_mcfw_timestamp_scale_and_nominal_geometry_are_explicit(tmp_path: Path):
    _root, path = _write_mcfw_trial(tmp_path)
    dataset = mcfw_adapter.load_mcfw_trial(
        path,
        eye="left",
        timestamp_scale_seconds=1e-6,
        nominal_sampling_rate=120.0,
    )
    samples = dataset["gaze_samples"]
    assert samples["timestamp_seconds"].iloc[1] == pytest.approx(0.008333)
    assert samples["valid"].sum() == 47

    angular = mcfw_coordinates.convert_mcfw_normalized_to_degrees(dataset)
    converted = angular["gaze_samples"]
    assert converted["gaze_x"].dropna().eq(0.0).all()
    assert converted["gaze_y"].dropna().eq(0.0).all()
    assert converted["source_gaze_x_normalized"].eq(0.5).all()
    assert converted["valid"].tolist() == samples["valid"].tolist()

    geometry = mcfw_coordinates.geometry_manifest().iloc[0]
    assert geometry["screen_width_px"] == 1920
    assert geometry["screen_height_px"] == 1080
    assert geometry["screen_width_mm"] == pytest.approx(310.0)
    assert geometry["screen_height_mm"] == pytest.approx(175.0)
    assert geometry["viewing_distance_mm"] == pytest.approx(650.0)
    assert geometry["primary_distance_branches"] == 1


def test_mcfw_detector_input_breaks_at_source_unusable_samples(tmp_path: Path):
    _root, path = _write_mcfw_trial(tmp_path)
    dataset = mcfw_adapter.load_mcfw_trial(
        path,
        eye="left",
        timestamp_scale_seconds=1e-6,
    )
    angular = mcfw_coordinates.convert_mcfw_normalized_to_degrees(dataset)
    detector_input = mcfw_validation.prepare_detector_input(angular)
    samples = detector_input["gaze_samples"]

    assert len(samples) == 47
    assert samples["valid"].all()
    assert samples["detector_input_run"].nunique() == 2
    assert samples["trial_id"].str.contains("__usable_run_").all()
    assert detector_input.vendor_metadata["detector_input_policy"][
        "split_at_unusable_samples"
    ]


def test_mcfw_fixation_state_maps_without_inventing_samples():
    source = pd.DataFrame(
        {
            "timestamp_seconds": [0.0, 0.1, 0.2, 0.3, 0.4],
            "valid": [True, True, True, True, True],
        }
    )
    episodes = pd.DataFrame(
        {
            "episode_type": ["fixation", "fixation"],
            "start_time": [0.1, 0.3],
            "end_time": [0.2, 0.3],
        }
    )
    state = mcfw_validation.fixation_state_on_source_timeline(source, episodes)
    assert state.tolist() == [False, True, True, True, False]


def test_mcfw_pair_metrics_have_frozen_edge_semantics():
    assert np.isnan(
        mcfw_validation._jaccard(
            np.array([False, False]),
            np.array([False, False]),
        )
    )
    assert np.isnan(
        mcfw_validation._cohen_kappa(
            np.array([True, True]),
            np.array([True, True]),
        )
    )
    assert mcfw_validation._symmetric_relative_difference(1.0, 3.0) == pytest.approx(
        1.0
    )
    assert np.isnan(mcfw_validation._symmetric_relative_difference(0.0, 0.0))


def test_mcfw_process_file_materializes_complete_planned_rows(tmp_path: Path):
    root, path = _write_mcfw_trial(tmp_path)
    events, pairs, failures, _warnings = mcfw_validation.process_file(
        path,
        dataset_root=root,
        manifest_row=_manifest_row(root, path),
    )

    assert len(events) == 6
    assert len(pairs) == 6
    assert failures.empty
    assert set(events["eye"]) == {"left", "right"}
    assert set(events["detector_id"]) == set(mcfw_validation.DETECTOR_IDS)
    assert set(zip(pairs["detector_a"], pairs["detector_b"], strict=True)) == set(
        mcfw_validation.DETECTOR_PAIRS
    )
    assert events["status"].eq("ok").all()
    assert pairs["status"].eq("ok").all()
    assert pairs["n_samples_usable"].eq(47).all()
    assert pairs["agreement_proportion"].between(0.0, 1.0).all()


def test_mcfw_nonmonotonic_file_is_retained_as_non_evaluable(tmp_path: Path):
    root, path = _write_mcfw_trial(tmp_path, nonincreasing=True)
    events, pairs, failures, _warnings = mcfw_validation.process_file(
        path,
        dataset_root=root,
        manifest_row=_manifest_row(root, path),
    )

    assert len(events) == 6
    assert len(pairs) == 6
    assert len(failures) == 2
    assert events["status"].str.startswith("adapter_or_geometry_failed").all()
    assert pairs["status"].str.startswith("adapter_or_geometry_failed").all()
    assert failures["stage"].eq("adapter_or_geometry").all()


def test_mcfw_quality_association_reports_descriptive_rho_without_p_values():
    rows = []
    for index, usable in enumerate([0.5, 0.6, 0.7, 0.8, 0.9], start=1):
        rows.append(
            {
                "status": "ok",
                "detector_a": "a",
                "detector_b": "b",
                "eye": "left",
                "trial_family": "natural_image",
                "analysis_usable_fraction": usable,
                "agreement_proportion": usable,
                "fixation_jaccard": usable,
                "fixation_count_symmetric_relative_difference": 1 - usable,
                "fixation_rate_symmetric_relative_difference": 1 - usable,
                "median_duration_symmetric_relative_difference": 1 - usable,
                "fixation_time_proportion_symmetric_relative_difference": 1 - usable,
                "trial": f"t{index}",
                "participant": "p1",
            }
        )
    result = mcfw_summary.quality_association(pd.DataFrame(rows))
    assert "p_value" not in result.columns
    overall = result.loc[
        result["grouping"].eq("detector_a+detector_b+eye")
        & result["outcome"].eq("agreement_disagreement")
    ].iloc[0]
    assert overall["n"] == 5
    assert overall[
        "spearman_rho_usable_fraction_vs_disagreement"
    ] == pytest.approx(-1.0)
