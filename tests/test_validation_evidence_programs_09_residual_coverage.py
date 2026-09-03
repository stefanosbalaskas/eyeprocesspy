from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.validation_evidence_programs_09 as ve


def _plan():
    return ep.eyeprocess_validation_plan(
        families=["recovery"],
        sample_size=40,
        n_items=4,
        missing_rate=0.0,
        noise_level="reference",
        specification="correct",
        replications=2,
        seed=123,
        label="residual",
    )


def test_private_sequence_class_unique_and_column_guards():
    obj = SimpleNamespace(eyeprocess_class="demo")
    assert ve._class_is(obj, "demo")
    assert not ve._class_is(obj, "other")

    assert ve._sequence(pd.Series([1, 2])) == [1, 2]
    assert ve._sequence(np.int64(3)) == [np.int64(3)]
    marker = object()
    assert ve._sequence(marker) == [marker]
    assert ve._unique(["a", "a", "b"]) == ["a", "b"]

    with pytest.raises(ep.EyeProcessValidationError, match="missing required columns"):
        ve._require_columns(pd.DataFrame({"a": [1]}), ["a", "b"], name="frame")

    assert ve._finite_numbers([1.2, "2.5"]) == [1.2, 2.5]
    assert ve._finite_numbers([1, "2"], integer=True) == [1, 2]


def test_utc_string_all_conversion_and_error_paths():
    nowish = ve._utc_string()
    assert nowish.endswith(" UTC")
    assert ve._utc_string(datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02 03:04:05 UTC"
    assert ve._utc_string(datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)) == "2026-01-02 03:04:05 UTC"
    assert ve._utc_string("2026-01-02 03:04:05") == "2026-01-02 03:04:05 UTC"
    assert ve._utc_string("2026-01-02 05:04:05+02:00") == "2026-01-02 03:04:05 UTC"
    with pytest.raises(ep.EyeProcessValidationError, match="datetime-like"):
        ve._utc_string(object())


def test_json_safe_restore_all_scalar_container_paths():
    frame = pd.DataFrame({"a": [1, np.nan], "b": [pd.NA, "x"]})
    frame.attrs["arr"] = np.array([1, 2])
    payload = {
        "frame": frame,
        "series": pd.Series([np.int64(2), pd.NA]),
        "tuple": (np.float64(1.5), np.inf),
        "array": np.array([3, 4]),
        "mapping": {1: np.float64(2.0)},
    }
    safe = ve._json_safe(payload)
    assert safe["tuple"][1] is None
    assert safe["series"][1] is None
    assert safe["mapping"]["1"] == 2.0

    restored = ve._json_restore(safe)
    assert isinstance(restored["frame"], pd.DataFrame)
    assert restored["frame"].attrs["arr"] == [1, 2]
    assert ve._json_restore([1, {"x": 2}]) == [1, {"x": 2}]
    assert ve._json_restore(7) == 7


def test_plan_remaining_validation_and_structural_paths():
    with pytest.raises(ep.EyeProcessValidationError, match="noise_level"):
        ep.eyeprocess_validation_plan(noise_level=[""])
    with pytest.raises(ep.EyeProcessValidationError, match="replications and seed"):
        ep.eyeprocess_validation_plan(replications=object())
    with pytest.raises(ep.EyeProcessValidationError, match="replications and seed"):
        ep.eyeprocess_validation_plan(seed=object())
    with pytest.raises(ep.EyeProcessValidationError, match="label"):
        ep.eyeprocess_validation_plan(label=3)

    broken = _plan()
    del broken["label"]
    with pytest.raises(ep.EyeProcessValidationError, match="missing fields"):
        ep.validate_eyeprocess_validation_plan(broken)

    with pytest.raises(ep.EyeProcessValidationError, match="positive scalar integers"):
        ep.eyeprocess_validation_seed(object(), 1)
    with pytest.raises(ep.EyeProcessValidationError, match="positive scalar integers"):
        ep.eyeprocess_validation_seed(1, 1, stream=-1)


def test_acceptance_rule_remaining_guards_and_unknown_direction():
    with pytest.raises(ep.EyeProcessValidationError, match="metric"):
        ep.validation_acceptance_rule(1, threshold=0.1)
    with pytest.raises(ep.EyeProcessValidationError, match="direction"):
        ep.validation_acceptance_rule("x", direction="sideways", threshold=0.1)
    with pytest.raises(ep.EyeProcessValidationError, match="threshold must be supplied"):
        ep.validation_acceptance_rule("x", threshold=None)
    with pytest.raises(ep.EyeProcessValidationError, match="between rules"):
        ep.validation_acceptance_rule("x", direction="between", threshold=0.0, upper=object())
    with pytest.raises(ep.EyeProcessValidationError, match="finite scalar"):
        ep.validation_acceptance_rule("x", threshold=np.inf)
    with pytest.raises(ep.EyeProcessValidationError, match="tolerance"):
        ep.validation_acceptance_rule("x", threshold=0.1, tolerance=object())
    with pytest.raises(ep.EyeProcessValidationError, match="tolerance"):
        ep.validation_acceptance_rule("x", threshold=0.1, tolerance=-1)

    rule = ep.validation_acceptance_rule("x", threshold=0.1)
    assert pd.isna(ep.evaluate_validation_acceptance(object(), rule))
    bad = dict(rule)
    bad["direction"] = "sideways"
    with pytest.raises(ep.EyeProcessValidationError, match="Unknown rule direction"):
        ep.evaluate_validation_acceptance(0.1, bad)


def test_acceptance_matrix_list_blank_name_between_and_validation_paths():
    frame = pd.DataFrame({"id": ["A"], "x": [0.5]})
    rule = ep.validation_acceptance_rule("x", direction="between", threshold=0.0, upper=1.0)

    with pytest.raises(ep.EyeProcessValidationError, match="data.frame"):
        ep.validation_acceptance_matrix({}, [rule])
    with pytest.raises(ep.EyeProcessValidationError, match="rules"):
        ep.validation_acceptance_matrix(frame, "bad")
    with pytest.raises(ep.EyeProcessValidationError, match="rules"):
        ep.validation_acceptance_matrix(frame, [])
    with pytest.raises(ep.EyeProcessValidationError, match="rules"):
        ep.validation_acceptance_matrix(frame, [{}])
    with pytest.raises(ep.EyeProcessValidationError, match="missing required columns"):
        ep.validation_acceptance_matrix(frame, [ep.validation_acceptance_rule("missing", threshold=1)])

    listed = ep.validation_acceptance_matrix(frame, [rule], id_cols=["", "id"])
    assert listed.loc[0, "rule_id"] == "rule_1"
    assert listed.loc[0, "threshold"] == "0.0:1.0"
    blank_named = ep.validation_acceptance_matrix(frame, {"": rule}, id_cols="id")
    assert blank_named.loc[0, "rule_id"] == "rule_1"


def test_acceptance_summary_and_mcse_empty_single_and_multigroup_paths():
    unevaluable = pd.DataFrame({"group": ["A", "A"], "sub": [1, 2], "pass": [pd.NA, pd.NA]})
    overall = ep.summarise_validation_acceptance(unevaluable)
    assert overall.loc[0, "n_evaluable"] == 0
    assert np.isnan(overall.loc[0, "pass_fraction"])
    grouped = ep.summarise_validation_acceptance(unevaluable, by=["group", "sub"])
    assert len(grouped) == 2

    with pytest.raises(ep.EyeProcessValidationError, match="missing required columns"):
        ep.summarise_validation_acceptance(pd.DataFrame({"x": [1]}), by="group")

    empty_values = pd.DataFrame({"group": ["A", "B"], "sub": [1, 1], "value": [np.nan, np.nan]})
    mcse = ep.validation_mcse_profile(empty_values, "value")
    assert mcse.loc[0, "n"] == 0
    assert np.isnan(mcse.loc[0, "mean"])
    one = ep.validation_mcse_profile(pd.DataFrame({"value": [3.0]}), "value")
    assert one.loc[0, "n"] == 1
    assert np.isnan(one.loc[0, "sd"])
    multi = ep.validation_mcse_profile(empty_values, "value", by=["group", "sub"])
    assert len(multi) == 2
    with pytest.raises(ep.EyeProcessValidationError, match="missing required columns"):
        ep.validation_mcse_profile(pd.DataFrame({"x": [1]}), "value")


def test_replication_budget_all_invalid_contracts():
    with pytest.raises(ep.EyeProcessValidationError, match="scalar numeric"):
        ep.validation_replication_budget(object(), 0.1)
    with pytest.raises(ep.EyeProcessValidationError, match="pilot_sd"):
        ep.validation_replication_budget(np.inf, 0.1)
    with pytest.raises(ep.EyeProcessValidationError, match="pilot_sd"):
        ep.validation_replication_budget(-1, 0.1)
    with pytest.raises(ep.EyeProcessValidationError, match="target_mcse"):
        ep.validation_replication_budget(1, 0)
    with pytest.raises(ep.EyeProcessValidationError, match="replication limits"):
        ep.validation_replication_budget(1, 0.1, minimum=0)
    with pytest.raises(ep.EyeProcessValidationError, match="replication limits"):
        ep.validation_replication_budget(1, 0.1, minimum=20, maximum=10)
    assert ep.validation_replication_budget(0.01, 1.0, minimum=5, maximum=10) == 5


def test_manifest_remaining_generation_write_and_read_errors(tmp_path):
    plan = _plan()
    manifest = ep.validation_scenario_manifest(plan)
    assert manifest["source_commit"] is None
    assert manifest["generated_at"].endswith(" UTC")

    naive = ep.validation_scenario_manifest(plan, generated_at=datetime(2026, 2, 3, 4, 5, 6))
    assert naive["generated_at"] == "2026-02-03 04:05:06 UTC"

    with pytest.raises(ep.EyeProcessValidationError, match="validation scenario manifest"):
        ep.write_validation_scenario_manifest({}, tmp_path / "bad.json")

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("not-json", encoding="utf-8")
    with pytest.raises(ep.EyeProcessValidationError, match="Could not read"):
        ep.read_validation_scenario_manifest(invalid_json)

    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"eyeprocess_class": "wrong"}), encoding="utf-8")
    with pytest.raises(ep.EyeProcessValidationError, match="not an eyeprocess"):
        ep.read_validation_scenario_manifest(wrong)


def test_evidence_grade_sequence_none_and_duplicate_requirements():
    grade = ep.eyeprocess_validation_evidence_grade(
        components=pd.Series(["design", None, "execution"]),
        required=["design", "design", "execution", None],
    )
    assert grade["required"] == ["design", "execution"]
    assert grade["grade"] == "complete"
    assert grade["coverage"] == 1.0
