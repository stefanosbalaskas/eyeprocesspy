from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.operational_validation_08 as ov


def test_streaming_scoring_guards_successful_adapter_shapes_and_history(monkeypatch):
    with pytest.raises(ep.EyeProcessValidationError, match="method must"):
        ov.score_partial_response_pattern(object(), [1], method="ML")
    with pytest.raises(ep.EyeProcessValidationError, match="at least one"):
        ov.score_partial_response_pattern(object(), [], method="MAP")

    with pytest.raises(ep.EyeProcessValidationError, match="method must"):
        ep.score_response_stream(object(), [1], method="ML")
    with pytest.raises(ep.EyeProcessValidationError, match="at least one"):
        ep.score_response_stream(object(), [], method="MAP")
    with pytest.raises(ep.EyeProcessValidationError, match="No observed responses"):
        ep.score_response_stream(object(), [np.nan, np.nan])
    with pytest.raises(ep.EyeProcessValidationError, match="No observed responses"):
        ep.score_response_stream(object(), [1.0], observed_order=[])
    for order in ([0], [2], [1, 1]):
        with pytest.raises(ep.EyeProcessValidationError, match="unique valid"):
            ep.score_response_stream(object(), [1.0], observed_order=order)

    calls = {"n": 0}

    def fake_score(model, pattern, method="MAP", **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return pd.DataFrame({"F1": [0.25], "SE_F1": [0.10]})
        if calls["n"] == 2:
            return pd.DataFrame({"theta": [0.50]})
        return pd.DataFrame()

    monkeypatch.setattr(ov, "score_partial_response_pattern", fake_score)
    scored = ep.score_response_stream(object(), [1.0, 0.0, 1.0], method="EAP")
    assert scored.observed_order.tolist() == [1, 2, 3]
    assert scored.history.loc[0, "theta"] == pytest.approx(0.25)
    assert scored.history.loc[0, "theta_se"] == pytest.approx(0.10)
    assert scored.history.loc[1, "theta"] == pytest.approx(0.50)
    assert math.isnan(scored.history.loc[1, "theta_se"])
    assert math.isnan(scored.history.loc[2, "theta"])

    with pytest.raises(ep.EyeProcessValidationError, match="eye_streaming_score"):
        ep.streaming_score_history({})
    hist = ep.streaming_score_history(scored)
    hist.loc[0, "theta"] = 99
    assert scored.history.loc[0, "theta"] != 99


def test_update_person_score_guards_and_success(monkeypatch):
    with pytest.raises(ep.EyeProcessValidationError, match="out of range"):
        ep.update_person_score(object(), [np.nan, np.nan], 0, 1)
    with pytest.raises(ep.EyeProcessValidationError, match="out of range"):
        ep.update_person_score(object(), [np.nan, np.nan], 3, 1)
    with pytest.raises(ep.EyeProcessValidationError, match="non-missing"):
        ep.update_person_score(object(), [np.nan, np.nan], 1, np.nan)

    expected = pd.DataFrame({"F1": [0.2]})
    monkeypatch.setattr(ov, "score_partial_response_pattern", lambda *a, **k: expected)
    out = ep.update_person_score(object(), [np.nan, 0.0], 1, [1.0], method="EAP")
    np.testing.assert_allclose(out["pattern"], [1.0, 0.0])
    assert out["score"].equals(expected)


def test_validation_bundle_collection_status_manifest_and_report_branches():
    with pytest.raises(ep.EyeProcessValidationError, match="must be named"):
        ep.collect_validation_evidence(1)

    with pytest.warns(UserWarning, match="Non-standard"):
        custom = ep.collect_validation_evidence(
            model_name=None,
            notes="audit note",
            custom_slot={"x": 1},
        )
    assert custom.model_name == "unnamed_model"

    bundle = ep.collect_validation_evidence(
        model_name="full",
        notes="note",
        recovery=pd.DataFrame({"x": [1]}),
        bias=pd.DataFrame(),
        rmse=RuntimeError("failed"),
        coverage={"ok": True},
        convergence={"ok": True},
        process_ablation={"ok": True},
        negative_controls={"ok": True},
        preflight={"ok": True},
        drift={"ok": True},
        external_validation={"ok": True},
    )
    manifest = ep.validation_bundle_manifest(bundle)
    status = dict(zip(manifest.slot, manifest.status))
    assert status["recovery"] == "available"
    assert status["bias"] == "empty"
    assert status["rmse"] == "error"
    assert status["sbc"] == "missing"

    with pytest.raises(ep.EyeProcessValidationError, match="eye_validation_bundle"):
        ep.validation_bundle_manifest({})
    with pytest.raises(ep.EyeProcessValidationError, match="eye_validation_bundle"):
        ep.validation_report({})

    lines = ep.validation_report(bundle, include_session=True)
    text = "\n".join(lines)
    for phrase in (
        "Parameter recovery evidence",
        "Interval-coverage evidence",
        "Convergence evidence",
        "Process-channel ablation evidence",
        "Negative-control evidence",
        "Biometric pre-flight evidence",
        "Deployment-drift evidence",
        "External/transportability evidence",
        "Notes",
        "Python version:",
        "Platform:",
    ):
        assert phrase in text

    empty_bundle = ep.collect_validation_evidence()
    empty_text = "\n".join(ep.validation_report(empty_bundle, include_session=False))
    assert "Available: none" in empty_text

    everything = {slot: 1 for slot in ov._VALIDATION_SLOTS}
    full_bundle = ep.collect_validation_evidence(**everything)
    full_text = "\n".join(ep.validation_report(full_bundle, include_session=False))
    assert "Missing/not supplied: none" in full_text


def test_jsonable_and_export_bundle_residual_paths(tmp_path: Path):
    assert ov._jsonable(pd.DataFrame({"x": [1]})) == [{"x": 1}]
    assert ov._jsonable(np.array([1, 2])) == [1, 2]
    assert ov._jsonable(np.int64(3)) == 3
    assert ov._jsonable(np.float64(1.5)) == pytest.approx(1.5)
    assert ov._jsonable({"x": np.int64(2)}) == {"x": 2}
    assert ov._jsonable((np.int64(1), "a")) == [1, "a"]
    assert ov._jsonable(None) is None
    fallback = ov._jsonable(object())
    assert fallback["class"] == "object" and "repr" in fallback

    with pytest.raises(ep.EyeProcessValidationError, match="eye_validation_bundle"):
        ep.export_validation_bundle({}, tmp_path / "bad")

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("x", encoding="utf-8")
    bundle = ep.collect_validation_evidence(
        recovery=pd.DataFrame({"x": [1]}),
        bias=np.array([1.0, 2.0]),
        notes={"arr": np.array([3])},
    )
    with pytest.raises(ep.EyeProcessValidationError, match="not empty"):
        ep.export_validation_bundle(bundle, occupied)

    out = ep.export_validation_bundle(bundle, occupied, overwrite=True, include_rds=False)
    assert Path(out.files["recovery"]).suffix == ".csv"
    assert "bias" not in out.files
    assert "bundle" not in out.files

    out2 = ep.export_validation_bundle(bundle, tmp_path / "with_json", include_rds=True)
    assert Path(out2.files["bias"]).suffix == ".json"
    assert Path(out2.files["bundle"]).exists()


def test_preaction_feature_window_optional_channel_and_group_skip_paths():
    base = pd.DataFrame(
        {
            "person_id": ["P1"] * 3,
            "trial_id": ["T1"] * 3,
            "time_ms": [800.0, 900.0, 1000.0],
            "response_time_ms": [1000.0] * 3,
        }
    )
    with pytest.raises(ep.EyeProcessValidationError, match="positive finite window"):
        ep.preaction_process_features(base, windows_ms=[np.nan, -1, 0])

    no_optional = ep.preaction_process_features(
        base,
        windows_ms=[np.nan, -1, 500, 500],
    )
    assert len(no_optional.data) == 1
    row = no_optional.data.iloc[0]
    assert math.isnan(row["pupil_mean"])
    assert math.isnan(row["blink_prop"])

    mixed = pd.DataFrame(
        {
            "person_id": ["P1"] * 3 + ["P2"] * 3 + ["P3"] * 2,
            "trial_id": ["T1"] * 8,
            "time_ms": [800, 900, 1000, 800, 900, 1000, 900, 1000],
            "response_time_ms": [1000, 1000, 1000, 1000, 1000, 1000, np.nan, np.nan],
            "aoi": ["target", "target", "target", None, None, None, "target", "target"],
            "pupil_bc": [1, 2, 3, 2, 3, 4, 1, 2],
            "blink": [False, True, False, False, False, True, False, False],
        }
    )
    feat = ep.preaction_process_features(mixed, windows_ms=[500])
    assert set(feat.data.person_id) == {"P1", "P2"}
    p2 = feat.data.loc[feat.data.person_id.eq("P2")].iloc[0]
    assert math.isnan(p2["aoi_prop__target"])

    too_short = pd.DataFrame(
        {
            "person_id": ["P4"] * 2,
            "trial_id": ["T1"] * 2,
            "time_ms": [900.0, 1000.0],
            "response_time_ms": [1000.0, 1000.0],
        }
    )
    assert ep.preaction_process_features(too_short, windows_ms=[500]).data.empty


def test_addm_proxy_empty_and_feature_family_registry_residual_paths():
    empty = pd.DataFrame(columns=["person_id", "trial_id", "time_ms", "aoi"])
    proxy = ep.addm_glam_proxy_features(empty)
    assert proxy.features.empty

    default_registry = ep.process_feature_family_registry()
    assert {"pattern", "family", "warning"}.issubset(default_registry.columns)

    custom = pd.DataFrame({"pattern": ["alpha"], "family": ["Custom"]})
    fam = ep.assign_process_feature_family(["alpha_signal", "other"], registry=custom)
    assert fam.tolist() == ["Custom", "Other"]
    assert ep.assign_process_feature_family([], registry=custom).size == 0

    with pytest.raises(ep.EyeProcessValidationError):
        ep.assign_process_feature_family(["x"], registry=pd.DataFrame({"pattern": ["x"]}))


def test_process_feature_stability_guards_custom_columns_and_selection_paths():
    with pytest.raises(ep.EyeProcessValidationError, match="top_n"):
        ep.process_feature_stability(
            pd.DataFrame({"feature": ["x"], "split": [1], "importance": [1.0]}),
            top_n=0,
        )
    with pytest.raises(ep.EyeProcessValidationError, match="No complete"):
        ep.process_feature_stability(
            pd.DataFrame({"feature": ["x"], "split": [1], "importance": [np.nan]})
        )

    data = pd.DataFrame(
        {
            "name": ["pupil_mean", "other", "pupil_mean", "other"],
            "fold": ["A", "A", "B", "B"],
            "score": [0.9, 0.1, 0.2, 0.8],
        }
    )
    out = ep.process_feature_stability(
        data,
        feature="name",
        split="fold",
        importance="score",
        top_n=1,
    )
    rates = dict(zip(out.feature, out.top_n_selection_rate))
    assert rates["pupil_mean"] == pytest.approx(0.5)
    assert rates["other"] == pytest.approx(0.5)
    assert dict(zip(out.feature, out.feature_family))["pupil_mean"] == "Pupil dynamics"
