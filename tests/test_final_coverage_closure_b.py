from __future__ import annotations

import builtins

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.grouped_validation_10 as gv
import eyeprocesspy.software_paper_evidence_09 as sp


def _group_data():
    rows = []
    for p in range(6):
        for item in range(4):
            x = float((p + item) % 3 - 1)
            rows.append(
                {
                    "participant_id": f"P{p}",
                    "item_id": f"I{item}",
                    "x": x,
                    "y": int((p + item) % 2 == 0),
                }
            )
    return pd.DataFrame(rows)


def test_grouped_private_validation_backend_and_name_guards(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "patsy":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as ctx:
        ctx.setattr(builtins, "__import__", blocked)
        with pytest.raises(ep.EyeProcessValidationError, match="psychometrics"):
            gv._load_model_backend()

    with pytest.raises(ep.EyeProcessValidationError, match="pandas DataFrame"):
        gv._require_frame([])
    assert gv._names("participant_id", "group") == ("participant_id",)
    with pytest.raises(ep.EyeProcessValidationError, match="non-empty column names"):
        gv._names(1, "group")
    with pytest.raises(ep.EyeProcessValidationError, match="at least 2"):
        gv._names(["participant_id"], "groups", minimum=2)
    with pytest.raises(ep.EyeProcessValidationError, match="non-empty column names"):
        gv._names([""], "group")
    with pytest.raises(ep.EyeProcessValidationError, match="Missing required columns"):
        gv._require_columns(pd.DataFrame({"a": [1]}), ["a", "b"])

    with pytest.raises(ep.EyeProcessValidationError, match="integer of at least two"):
        gv._fold_count("x")
    with pytest.raises(ep.EyeProcessValidationError, match="integer of at least two"):
        gv._fold_count(2.5)
    assert gv._fold_count("2") == 2

    keyed = gv._group_key(
        pd.DataFrame({"a": ["x", None], "b": ["y", "z"]}), ["a", "b"]
    )
    assert keyed.notna().tolist() == [True, False]


def test_grouped_family_metric_response_and_scoring_residuals():
    sm_bin = gv._family(None)
    assert sm_bin.__class__.__name__ == "Binomial"
    assert gv._family("logistic").__class__.__name__ == "Binomial"
    assert gv._family("normal").__class__.__name__ == "Gaussian"
    sentinel = object()
    assert gv._family(sentinel) is sentinel
    with pytest.raises(ep.EyeProcessValidationError, match="Unsupported"):
        gv._family("poisson")
    with pytest.raises(ep.EyeProcessValidationError, match="metric"):
        gv._metric("auc")

    data = _group_data()
    with pytest.raises(ep.EyeProcessValidationError, match="Could not evaluate"):
        gv._response_values("not_a_column ~ x", data)
    with pytest.raises(ep.EyeProcessValidationError, match="scalar response"):
        gv._response_values("y + x ~ 1", data)
    missing = data.copy()
    missing.loc[0, "x"] = np.nan
    with pytest.raises(ep.EyeProcessValidationError, match="removed rows"):
        gv._response_values("y ~ x", missing)

    assert np.isnan(gv._score(np.array([np.nan]), np.array([np.nan]), "accuracy"))
    with pytest.raises(ValueError, match="empty analysis"):
        gv._fit_and_score(data, "y ~ x", None, np.array([], dtype=int), np.array([0]), "log_loss")


def test_grouped_fold_and_crossed_error_paths(monkeypatch):
    data = _group_data()
    with pytest.raises(ep.EyeProcessValidationError, match="fewer independent groups"):
        ep.grouped_folds(data, group="item_id", v=5)

    broken = ep.grouped_cv(
        data,
        "not_a_column ~ x",
        group="participant_id",
        v=3,
        metric="log_loss",
        seed=1,
    )
    assert broken["results"].score.isna().all()
    assert broken["results"].error.notna().all()

    few = data.loc[data.item_id.isin(["I0", "I1"])].copy()
    with pytest.raises(ep.EyeProcessValidationError, match="fewer levels"):
        ep.crossed_grouped_folds(few, groups=["participant_id", "item_id"], v=3)

    fake_folds = gv.EyeCrossedGroupedFolds(
        folds=[
            {
                "analysis": np.array([], dtype=int),
                "assessment": np.array([0], dtype=int),
                "buffer": np.array([], dtype=int),
            }
        ],
        assignments={},
        groups=("participant_id", "item_id"),
        v=1,
    )
    with monkeypatch.context() as ctx:
        ctx.setattr(gv, "crossed_grouped_folds", lambda *args, **kwargs: fake_folds)
        result = ep.crossed_grouped_cv(data, "y ~ x", v=2)
    assert result["results"].error.iloc[0] == "empty analysis or assessment set"

    real_crossed = gv.crossed_grouped_folds
    with monkeypatch.context() as ctx:
        ctx.setattr(gv, "_fit_and_score", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("forced")))
        failed = ep.crossed_grouped_cv(data, "y ~ x", groups=["participant_id", "item_id"], v=2)
    assert failed["results"].error.notna().all()
    assert real_crossed is not None


def test_quantify_process_leakage_single_group_and_failure_reference(monkeypatch):
    data = _group_data()
    one_group = ep.quantify_process_leakage(
        data, "y ~ x", group="participant_id", v=3, seed=2
    )
    assert "cross_classified" not in set(one_group.scheme)

    def fail_cv(*args, **kwargs):
        raise RuntimeError("forced grouped failure")

    with monkeypatch.context() as ctx:
        ctx.setattr(gv, "grouped_cv", fail_cv)
        failed = ep.quantify_process_leakage(
            data, "y ~ x", group="participant_id", v=3, seed=2
        )
    assert failed.error.notna().all()
    assert failed.mean_log_loss.isna().all()
    assert failed.optimistic_difference.isna().all()


def test_software_evidence_private_coercion_and_json_helpers():
    assert sp._class_is(type("Tagged", (), {"eyeprocess_class": "x"})(), "x")
    assert sp._as_list(None) == []
    assert sp._as_list("x") == ["x"]
    assert sp._as_list(pd.Series([1, 2])) == [1, 2]
    assert sp._as_list(3) == [3]
    assert sp._recycle([], 3) == [None, None, None]

    series_frame = sp._as_frame(pd.Series([1, 2], name="v"), name="x")
    assert series_frame.columns.tolist() == ["v"]
    scalar_mapping = sp._as_frame({"a": 1}, name="x")
    assert scalar_mapping.to_dict("records") == [{"a": 1}]
    with pytest.raises(ep.EyeProcessValidationError, match="coercible"):
        sp._as_frame(object(), name="x")
    with pytest.raises(ep.EyeProcessValidationError, match="missing required"):
        sp._require_columns(pd.DataFrame({"a": [1]}), ["b"], name="x")

    assert sp._length_nonzero(None) is False
    assert sp._length_nonzero(pd.DataFrame({"a": [1]})) is True
    assert sp._length_nonzero({}) is False
    assert sp._length_nonzero("x") is True
    assert sp._length_nonzero([]) is False
    assert sp._length_nonzero(1) is True

    payload = {
        "frame": pd.DataFrame({"x": [np.nan]}),
        "series": pd.Series([pd.NA]),
        "mapping": {"a": np.int64(2)},
        "tuple": (np.array([1.0, np.inf]),),
    }
    safe = sp._json_safe(payload)
    assert safe["frame"][0]["x"] is None
    assert safe["series"] == [None]
    assert safe["mapping"]["a"] == 2
    assert safe["tuple"][0][1] is None


def test_software_claim_validation_coverage_and_readiness_residuals(tmp_path, monkeypatch):
    with pytest.raises(ep.EyeProcessValidationError, match="claim"):
        ep.software_paper_claim_matrix(["valid", ""])
    claims = ep.software_paper_claim_matrix(
        ["A", "B", "C"], evidence_id=["E1"], status=["supported"]
    )
    assert claims.evidence_id.tolist() == ["E1", "E1", "E1"]

    bundle_default = ep.software_paper_evidence_bundle(claims=claims)
    assert bundle_default["metadata"] == {}

    empty_claims = pd.DataFrame({"status": pd.Series(dtype=str)})
    coverage = ep.software_paper_coverage(empty_claims)
    assert coverage.n_claims.iloc[0] == 0 and np.isnan(coverage.coverage.iloc[0])
    assert sp._format_coverage(np.nan) == "NA"

    bad_claim_bundle = ep.software_paper_evidence_bundle(claims=object())
    readiness = ep.software_paper_readiness(
        bad_claim_bundle,
        require_validation=False,
        require_reproducibility=False,
        require_examples=False,
        require_articles=False,
    )
    assert readiness["ready"] is False

    no_status = ep.software_paper_evidence_bundle(claims=pd.DataFrame({"claim": ["A"]}))
    gaps = ep.software_paper_gap_analysis(no_status)
    assert gaps["claim_gaps"].empty

    import eyeprocesspy.governance_09 as gov
    with monkeypatch.context() as ctx:
        ctx.setattr(gov, "summarise_process_validation", lambda x: pd.DataFrame({"ok": [1]}))
        summarized = ep.software_paper_validation_table(
            {"eyeprocess_class": "eye_process_validation_result"}
        )
    assert summarized.ok.iloc[0] == 1

    with pytest.raises(ep.EyeProcessValidationError, match="must be named"):
        ep.software_paper_validation_table({})
    with pytest.raises(ep.EyeProcessValidationError, match="must be named"):
        ep.software_paper_validation_table({"": pd.DataFrame({"x": [1]})})

    with pytest.raises(ep.EyeProcessValidationError, match="eye_software_paper_evidence"):
        ep.software_paper_coverage({})

    report = tmp_path / "no-coverage.md"
    ep.write_software_paper_evidence(no_status, report)
    text = report.read_text(encoding="utf-8")
    assert "## Claim coverage" not in text


def test_software_paper_manifest_empty_paths_and_json_numpy(tmp_path):
    fingerprint = ep.eye_reproducibility_fingerprint(data=[1], result={"x": 1})
    claims = ep.software_paper_claim_matrix("A", status="supported")
    bundle = ep.software_paper_evidence_bundle(
        claims=claims,
        validation=[1],
        examples=[1],
        articles=[1],
        reproducibility=fingerprint,
        metadata={"array": np.array([1.0, np.inf]), "scalar": np.float64(2.0)},
    )
    frozen = tmp_path / "nested" / "evidence.json"
    manifest = ep.freeze_software_paper_evidence(bundle, frozen)
    assert manifest.loc[0, "path"].endswith("evidence.json")

    paper = ep.paper_reproducibility_manifest(
        bundle,
        manuscript="",
        figures=[None, ""],
        tables=None,
    )
    assert paper["files"].empty
