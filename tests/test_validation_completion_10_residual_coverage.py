from __future__ import annotations

import builtins
from pathlib import Path
import types

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.validation_completion_10 as vc

FIX = Path(__file__).parent / "fixtures" / "gazepoint"


def _eye():
    return ep.read_gazepoint_folder(FIX, recording_id="R1", quiet=True)


def _corpus_summary():
    vendors = ["gazepoint", "tobii", "pupillabs", "eyelink", "smi"]
    return pd.DataFrame(
        {
            "case_id": [f"{v}-{i}" for v in vendors for i in (1, 2)],
            "vendor": [v for v in vendors for _ in (1, 2)],
            "status": "pass",
            "software_version": "1.0",
            "device_model": "device",
            "independent_source": True,
            "licence_reviewed": True,
        }
    )


def test_validation_completion_private_frames_containers_and_attribute_contracts():
    frame = pd.DataFrame({"x": [1]})
    assert vc._frame(frame).equals(frame)
    series = pd.Series({"x": 1, "y": 2})
    assert vc._frame(series).shape == (1, 2)
    assert vc._frame({"x": 1}).x.iloc[0] == 1
    assert vc._frame([[1, 2]]).shape == (1, 2)
    with pytest.raises(ep.EyeProcessValidationError, match="coercible"):
        vc._frame(object())

    assert vc._bind_rows([]).empty
    assert len(vc._bind_rows([frame, None, pd.DataFrame({"x": [2]})])) == 2
    tagged = vc._set_frame_class(frame.copy(), "demo")
    assert tagged.attrs["eyeprocess_class"] == "demo"

    m = vc.EyeMultiverse(results=frame)
    p = vc.EyeValidationProgram(corpus={"ok": True})
    assert m.results.equals(frame)
    assert p.corpus == {"ok": True}
    with pytest.raises(AttributeError):
        _ = m.missing
    with pytest.raises(AttributeError):
        _ = p.missing
    assert vc._is_multiverse(m)
    assert not vc._is_multiverse({})


def test_preprocessing_multiverse_sequence_guards_and_extraction_failure():
    seq = ep.preprocessing_multiverse(
        [1, 2, 3],
        [1, 2],
        transform=lambda x, spec: np.asarray(x) * spec,
        analyse=lambda x: {"estimate": float(np.mean(x))},
        extract=lambda z: z,
    )
    assert set(seq.results.specification) == {"spec_1", "spec_2"}

    for specs in ([], (), "bad", None):
        with pytest.raises(ep.EyeProcessValidationError, match="non-empty"):
            ep.preprocessing_multiverse([1], specs, lambda x, s: x, lambda x: x)

    with pytest.raises(ep.EyeProcessValidationError, match="callable"):
        ep.preprocessing_multiverse([1], {"a": 1}, None, lambda x: x)

    failed = ep.preprocessing_multiverse(
        [1],
        {"a": 1},
        transform=lambda x, s: x,
        analyse=lambda x: x,
        extract=lambda x: (_ for _ in ()).throw(RuntimeError("extract broke")),
    )
    assert failed.results.error.iloc[0] == "extract broke"


def test_deep_size_benchmark_and_reporting_guards(tmp_path):
    shared = [1, 2, 3]
    payload = {
        "mapping": {"a": shared},
        "list": [shared, (1, 2), {1, 2}, frozenset({3, 4})],
        "frame": pd.DataFrame({"x": ["a", "b"]}),
        "series": pd.Series([1, 2]),
        "array": np.arange(4),
    }
    assert vc._deep_size(payload) > 0

    with pytest.raises(ep.EyeProcessValidationError, match="expr"):
        ep.benchmark_eyeprocess(1)
    for n in ("bad", 0):
        with pytest.raises(ep.EyeProcessValidationError, match="positive integer"):
            ep.benchmark_eyeprocess(lambda: 1, iterations=n)
    bench = ep.benchmark_eyeprocess(lambda: payload, iterations=1, label="payload")
    assert bench.result_size_bytes.iloc[0] > 0

    eye = _eye()
    quality = eye["quality"].copy()
    if quality.empty:
        quality = pd.DataFrame({"recording_id": ["R1"], "metric": ["missing_fraction"], "value": [0.1]})
    else:
        quality = quality.copy()
        quality.loc[quality.index[0], "metric"] = "missing_fraction"
    eye["quality"] = quality
    audit = ep.reporting_guideline_audit(
        eye,
        model={"model": "declared"},
        sensitivity=vc.EyeMultiverse(results=pd.DataFrame({"estimate": [1.0]})),
    )
    by = audit.set_index("section")
    assert bool(by.loc["model", "covered"])
    assert bool(by.loc["sensitivity", "covered"])
    assert bool(by.loc["exclusions", "covered"])

    with pytest.raises(ep.EyeProcessValidationError, match="eye_reporting_audit"):
        ep.write_reporting_guideline_report(pd.DataFrame(), tmp_path / "bad.md")


def test_public_benchmark_max_participants_and_overwrite_guards(tmp_path):
    eye = _eye()
    with pytest.raises(ep.EyeProcessValidationError, match="positive integer"):
        ep.create_public_benchmark(eye, tmp_path / "bad", max_participants="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="positive integer"):
        ep.create_public_benchmark(eye, tmp_path / "bad2", max_participants=0)

    target = tmp_path / "benchmark"
    ep.create_public_benchmark(eye, target, include_samples=True, overwrite=True)
    roundtrip = ep.import_canonical(target)
    assert len(roundtrip["gaze_samples"]) > 0
    with pytest.raises(ep.EyeProcessValidationError, match="already exists"):
        ep.create_public_benchmark(eye, target, overwrite=False)
    ep.create_public_benchmark(eye, target, include_samples=False, overwrite=True)
    assert ep.import_canonical(target)["gaze_samples"].empty


def test_json_jobs_sanitize_and_plot_helper_contracts(tmp_path, monkeypatch):
    frame = pd.DataFrame({"a": [1.0, np.nan]})
    converted = vc._jsonable(
        {
            "na": pd.NA,
            "frame": frame,
            "series": pd.Series([1, pd.NA]),
            "array": np.array([1, 2]),
            "tuple": (1, 2),
            "scalar": np.int64(3),
            "path": Path("x"),
            "other": object(),
        }
    )
    assert converted["na"] is None
    assert converted["scalar"] == 3
    assert converted["path"] == "x"
    assert converted["other"]["class"] == "object"
    snap = tmp_path / "snapshot.json"
    vc._write_snapshot(snap, converted)
    assert snap.exists()

    jobs = vc._jobs({"": 1, "same": 2}, "job-")
    assert set(jobs) == {"job-1", "same"}
    assert vc._jobs([1, 2], "job-") == {"job-1": 1, "job-2": 2}
    assert vc._jobs(None, "job-") == {}
    with pytest.raises(ep.EyeProcessValidationError, match="mapping or sequence"):
        vc._jobs(1, "job-")

    assert vc._sanitize("a / b") == "a-b"
    assert vc._sanitize("///") == "parameter"

    vc._save_bar(tmp_path / "bar.png", ["a", "b"], [0.2, 0.8], "Y", "T", (0, 1))
    vc._save_line(tmp_path / "line.png", [1, 2], [2, 3], "Y", "T")
    vc._save_benchmark_plot(
        tmp_path / "bench.png",
        pd.DataFrame({"label": ["a", "a", "b"], "elapsed_seconds": [0.1, 0.2, 0.3]}),
    )
    assert all((tmp_path / name).exists() for name in ["bar.png", "line.png", "bench.png"])

    real_import = builtins.__import__
    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib.pyplot":
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ep.EyeProcessValidationError, match="optional plotting"):
        vc._get_plt()


def test_evidence_corpus_and_result_helpers(monkeypatch):
    evidence = {}
    vc._merge_evidence(evidence, "", "x", 1)
    vc._merge_evidence(evidence, "m", "x", None)
    vc._merge_evidence(evidence, "m", "x", 1)
    vc._merge_evidence(evidence, "m", "x", 2)
    assert evidence == {"m": {"x": 1}}

    summary = _corpus_summary()
    mapping = {"summary": summary, "manifest": summary.copy(), "status": "pass"}
    assert vc._corpus_result(mapping)["status"] == "pass"

    obj = types.SimpleNamespace(summary=summary, manifest=summary.copy(), status="pass")
    assert vc._corpus_result(obj)["status"] == "pass"

    monkeypatch.setattr(vc, "validate_eye_corpus", lambda x: {"summary": summary, "status": "validated"})
    assert vc._corpus_result(["raw"])["status"] == "validated"

    class Result:
        comparison = pd.DataFrame({"x": [1]})
        results = pd.DataFrame({"y": [2]})
    assert not vc._comparison_frame({"comparison": pd.DataFrame({"x": [1]})}).empty
    assert not vc._comparison_frame(Result()).empty
    assert vc._comparison_frame({}).empty
    assert not vc._result_frame({"results": pd.DataFrame({"x": [1]})}, "results").empty
    assert not vc._result_frame(Result(), "results").empty
    assert vc._result_frame({}, "results").empty


def test_validation_program_exercises_all_orchestration_job_families(tmp_path, monkeypatch):
    summary = _corpus_summary()
    monkeypatch.setattr(vc, "audit_vendor_validation", lambda corpus: pd.DataFrame(
        {"vendor": ["gazepoint"], "pass_rate": [1.0], "cases": [2]}
    ))
    monkeypatch.setattr(vc, "write_vendor_validation_report", lambda x, path: Path(path).write_text("vendor", encoding="utf-8") or str(path))

    model_result = {"recovery": pd.DataFrame({"parameter": ["b"], "estimate": [0.1]})}
    monkeypatch.setattr(vc, "run_model_validation", lambda **job: model_result)
    monkeypatch.setattr(vc, "model_validation_summary", lambda result: pd.DataFrame({"parameter": ["b"], "bias": [0.0]}))

    sbc_result = {
        "ranks": pd.DataFrame(
            {"parameter": ["a", "a", None], "normalized_rank": [0.2, 0.8, 0.5]}
        )
    }
    monkeypatch.setattr(vc, "simulation_based_calibration", lambda **job: sbc_result)
    monkeypatch.setattr(vc, "sbc_summary", lambda result: pd.DataFrame({"parameter": ["a"], "n": [2]}))

    engine_result = {"estimates": pd.DataFrame({"engine": ["a"], "estimate": [0.1]})}
    monkeypatch.setattr(vc, "compare_model_engines", lambda **job: engine_result)

    reproduction = {"comparison": pd.DataFrame({"metric": ["x"], "delta": [0.0]})}
    monkeypatch.setattr(vc, "run_raven_reproduction", lambda **job: reproduction)

    grouped_result = {
        "results": pd.DataFrame({"fold": [1, 2], "score": [0.2, 0.3]}),
        "metric": "log_loss",
    }
    monkeypatch.setattr(vc, "grouped_cv", lambda **job: grouped_result)
    monkeypatch.setattr(vc, "crossed_grouped_cv", lambda **job: grouped_result)

    leakage = pd.DataFrame({"scheme": ["grouped", "rowwise"], "mean_log_loss": [0.3, 0.2]})
    monkeypatch.setattr(vc, "quantify_process_leakage", lambda **job: leakage)

    multiverse = vc.EyeMultiverse(
        results=pd.DataFrame({"specification": ["a", "b"], "estimate": [0.1, 0.2]}),
        specifications={"a": 1, "b": 2},
    )
    monkeypatch.setattr(vc, "preprocessing_multiverse", lambda **job: multiverse)

    reporting = pd.DataFrame(
        {"section": ["hardware"], "item": ["device"], "covered": [True], "status": ["pass"]}
    )
    reporting.attrs["eyeprocess_class"] = "eye_reporting_audit"
    monkeypatch.setattr(vc, "reporting_guideline_audit", lambda *args, **kwargs: reporting)
    monkeypatch.setattr(vc, "write_reporting_guideline_report", lambda x, path: Path(path).write_text("report", encoding="utf-8") or str(path))
    monkeypatch.setattr(vc, "create_public_benchmark", lambda x, path, **kwargs: str(Path(path)))

    evidence_audit = pd.DataFrame(
        {"model": ["m"], "completed": [2], "required": [2]}
    )
    monkeypatch.setattr(vc, "audit_advanced_model_evidence", lambda evidence, spec: evidence_audit)
    monkeypatch.setattr(vc, "write_advanced_model_evidence_report", lambda x, path: Path(path).write_text("evidence", encoding="utf-8") or str(path))

    result = ep.run_eyeprocess_validation_program(
        {"summary": summary, "manifest": summary.copy(), "status": "pass"},
        tmp_path / "program",
        model_jobs=[{"x": 1}],
        sbc_jobs={"sbc": {"x": 1}},
        engine_jobs=[{"x": 1}],
        reproduction_jobs={"rep": {"x": 1}},
        grouped_jobs={
            "g": {"x": 1},
            "cross": {"x": 1, "crossed": True},
        },
        leakage_jobs=[{"x": 1}],
        multiverse_jobs=[{"x": 1}],
        benchmark_jobs={
            "callable": lambda: 1,
            "mapping": {"expr": lambda: 2, "iterations": 1},
        },
        reporting_dataset={"placeholder": True},
        public_benchmark_dataset={"placeholder": True},
        advanced_evidence={"m": {"existing": {"ok": True}}},
        evidence_spec={"spec": True},
        overwrite=True,
    )
    assert result.eyeprocess_class == "eye_validation_program"
    assert set(result.grouped_validation) == {"g", "cross"}
    assert set(result.benchmarks) == {"callable", "mapping"}
    assert result.public_benchmark is not None
    assert (tmp_path / "program" / "plots" / "vendor-pass-rate.png").exists()
    assert (tmp_path / "program" / "plots" / "sbc-sbc-a.png").exists()
    assert (tmp_path / "program" / "plots" / "grouped-validation-cross.png").exists()
    assert (tmp_path / "program" / "plots" / "multiverse-multiverse-1.png").exists()
    assert (tmp_path / "program" / "plots" / "advanced-model-evidence.png").exists()
