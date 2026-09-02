from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy.validation_orchestration_10 as vo
from eyeprocesspy.exceptions import EyeProcessBackendError, EyeProcessValidationError


def _plan(**kwargs):
    return vo.validation_job_plan(
        {"n": [4]},
        replications=1,
        base_seed=17,
        model_family="residual",
        **kwargs,
    )


def _job(plan=None):
    plan = _plan() if plan is None else plan
    return plan["jobs"].iloc[0].to_dict()


def _simulator(n=4, seed=1, **kwargs):
    del kwargs
    return {"values": np.arange(int(n), dtype=float), "truth": {"mu": 1.5}, "seed": seed}


def _fitter(simulation, **kwargs):
    del kwargs
    return {
        "estimate": float(np.mean(simulation["values"])),
        "se": 0.25,
        "converged": True,
        "iterations": 3,
    }


def _extractor(fit):
    return pd.DataFrame(
        {
            "parameter": ["mu"],
            "estimate": [fit["estimate"]],
            "std_error": [fit["se"]],
        }
    )


def _truth(simulation):
    return simulation["truth"]


def _runner_args(**overrides):
    args = {
        "simulator": _simulator,
        "fitter": _fitter,
        "extractor": _extractor,
        "truth_extractor": _truth,
        "simulation_args": {},
        "fit_args": {},
        "diagnostics_extractor": None,
        "draws_extractor": None,
        "predictions_extractor": None,
        "confidence": 0.95,
    }
    args.update(overrides)
    return args


def test_scalar_int_float_conversion_exception_branch():
    class Odd:
        def __int__(self):
            return 2

        def __float__(self):
            raise ValueError("no float")

    with pytest.raises(EyeProcessValidationError, match="single integer"):
        vo._scalar_int(Odd(), "value", 0)


def test_manifest_write_read_and_split_guard_residuals(tmp_path):
    with pytest.raises(EyeProcessValidationError, match="validation_job_plan"):
        vo.write_validation_job_manifest({}, tmp_path / "bad")

    plan = _plan()
    output = Path(vo.write_validation_job_manifest(plan, tmp_path / "manifest"))
    with pytest.raises(EyeProcessValidationError, match="already exists"):
        vo.write_validation_job_manifest(plan, output)
    marker = output / "marker.txt"
    marker.write_text("old", encoding="utf-8")
    rewritten = Path(vo.write_validation_job_manifest(plan, output, overwrite=True))
    assert rewritten == output
    assert not marker.exists()

    with pytest.raises(EyeProcessValidationError, match="does not exist"):
        vo.read_validation_job_manifest(tmp_path / "missing")

    missing_plan = tmp_path / "missing-plan"
    missing_plan.mkdir()
    with pytest.raises(EyeProcessValidationError, match="missing `plan.json`"):
        vo.read_validation_job_manifest(missing_plan)

    invalid = tmp_path / "invalid-payload"
    invalid.mkdir()
    (invalid / "plan.json").write_text("[]", encoding="utf-8")
    with pytest.raises(EyeProcessValidationError, match="unsupported class"):
        vo.read_validation_job_manifest(invalid)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "plan.json").write_text("{}", encoding="utf-8")
    with pytest.raises(EyeProcessValidationError, match="unsupported class"):
        vo.read_validation_job_manifest(incomplete)

    required = {
        "plan_id": "p",
        "model_family": "m",
        "jobs": [],
        "grid": pd.DataFrame({"n": [1]}),
        "replications": 1,
        "base_seed": 1,
        "chunk_size": 1,
        "metadata": {},
        "schema_version": "1",
        "plan_fingerprint": "x",
    }
    bad_jobs = tmp_path / "bad-jobs"
    bad_jobs.mkdir()
    vo._atomic_json(bad_jobs / "plan.json", required)
    with pytest.raises(EyeProcessValidationError, match="jobs table"):
        vo.read_validation_job_manifest(bad_jobs)

    required["jobs"] = pd.DataFrame({"job_id": ["j"]})
    required["grid"] = []
    bad_grid = tmp_path / "bad-grid"
    bad_grid.mkdir()
    vo._atomic_json(bad_grid / "plan.json", required)
    with pytest.raises(EyeProcessValidationError, match="design grid"):
        vo.read_validation_job_manifest(bad_grid)

    with pytest.raises(EyeProcessValidationError, match="eye_validation_job_plan"):
        vo.split_validation_plan({})
    with pytest.raises(EyeProcessValidationError, match="No requested chunks"):
        vo.split_validation_plan(plan, chunks="C99999")

    pieces = vo.split_validation_plan(plan, chunks=[plan["jobs"]["chunk_id"].iloc[0]])
    piece = next(iter(pieces.values()))
    assert piece["jobs"] is not plan["jobs"]
    assert piece["metadata"]["parent_plan_id"] == plan["plan_id"]


def test_function_fingerprint_signature_fallback_and_call_signature_failure(monkeypatch):
    assert vo._function_fingerprint(len).startswith("fn-")

    def target(value=0, **kwargs):
        return value + kwargs.get("extra", 0)

    real_signature = vo.inspect.signature
    monkeypatch.setattr(vo.inspect, "signature", lambda function: (_ for _ in ()).throw(ValueError("no signature")))
    assert vo._call_supported(target, named={"value": 2, "extra": 3}) == 5
    monkeypatch.setattr(vo.inspect, "signature", real_signature)


def test_standardize_explicit_intervals_series_and_default_diagnostics():
    series = pd.Series({"mu": 2.0})
    out = vo._standardize_estimates(series, truth={"mu": 0.0})
    assert out.parameter.tolist() == ["mu"]
    assert np.isnan(out.relative_bias.iloc[0])

    explicit = vo._standardize_estimates(
        pd.DataFrame(
            {
                "parameter": ["a", "b"],
                "estimate": [1.0, 3.0],
                "std_error": [0.1, 0.2],
                "lower": [0.8, 2.8],
                "upper": [1.2, 3.2],
            }
        ),
        truth={"a": 1.0, "b": 2.0},
    )
    assert bool(explicit.covered.iloc[0])
    assert not bool(explicit.covered.iloc[1])
    assert explicit.interval_width.tolist() == pytest.approx([0.4, 0.4])

    diagnostics = vo._default_diagnostics(
        {
            "converged": False,
            "iterations": 11,
            "diagnostics": {
                "converged": True,
                "divergences": 2,
                "max_rhat": 1.03,
                "min_ess_bulk": 80,
                "min_ess_tail": 70,
            },
        }
    ).iloc[0]
    assert bool(diagnostics.converged)
    assert diagnostics.iterations == 11
    assert diagnostics.divergences == 2
    assert diagnostics.max_rhat == pytest.approx(1.03)
    assert diagnostics.min_ess_bulk == 80
    assert diagnostics.min_ess_tail == 70

    default = vo._job_result(_job(), "complete", "complete", time.time())
    assert default["diagnostics"]["converged"].iloc[0]
    assert default["estimates"].empty
    assert default["artifacts"] == {}


@pytest.mark.parametrize(
    ("stage", "simulator", "fitter", "extractor", "truth_extractor"),
    [
        ("simulation", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("sim")), _fitter, _extractor, _truth),
        ("fit", _simulator, lambda simulation: (_ for _ in ()).throw(RuntimeError("fit")), _extractor, _truth),
        ("extract", _simulator, _fitter, lambda fit: (_ for _ in ()).throw(RuntimeError("extract")), _truth),
        ("truth", _simulator, _fitter, _extractor, lambda simulation: (_ for _ in ()).throw(RuntimeError("truth"))),
    ],
)
def test_run_job_core_failure_stages(stage, simulator, fitter, extractor, truth_extractor):
    result = vo._run_job_core(
        _job(),
        simulator,
        fitter,
        extractor,
        truth_extractor,
    )
    assert result["status"] == "failed"
    assert result["stage"] == stage
    assert stage in result["error"]


def test_run_job_core_standardize_memory_and_optional_extractor_branches(monkeypatch):
    bad = vo._run_job_core(
        _job(),
        _simulator,
        _fitter,
        lambda fit: [1, 2],
        _truth,
    )
    assert bad["status"] == "failed"
    assert bad["stage"] == "standardize"

    monkeypatch.setattr(vo, "_deep_size", lambda value: 2 * 1024**2)
    sim_memory = vo._run_job_core(
        _job(),
        _simulator,
        _fitter,
        _extractor,
        _truth,
        memory_limit_mb=1,
    )
    assert sim_memory["stage"] == "memory"
    assert "Simulation object size" in sim_memory["error"]

    sizes = iter([1, 2 * 1024**2])
    monkeypatch.setattr(vo, "_deep_size", lambda value: next(sizes))
    fit_memory = vo._run_job_core(
        _job(),
        _simulator,
        _fitter,
        _extractor,
        _truth,
        memory_limit_mb=1,
    )
    assert fit_memory["stage"] == "memory"
    assert "Fit object size" in fit_memory["error"]

    monkeypatch.setattr(vo, "_deep_size", lambda value: 1)
    result = vo._run_job_core(
        _job(),
        _simulator,
        _fitter,
        _extractor,
        _truth,
        diagnostics_extractor=lambda fit: (_ for _ in ()).throw(RuntimeError("diag")),
        draws_extractor=lambda fit: (_ for _ in ()).throw(RuntimeError("draw")),
        predictions_extractor=lambda fit, simulation: (_ for _ in ()).throw(RuntimeError("pred")),
    )
    assert result["status"] == "nonconverged"
    assert result["diagnostics"]["diagnostic_error"].iloc[0] == "diag"
    assert result["draws"] == {"error": "draw"}
    assert result["predictions"] == {"error": "pred"}

    result = vo._run_job_core(
        _job(),
        _simulator,
        _fitter,
        _extractor,
        _truth,
        diagnostics_extractor=lambda fit: pd.DataFrame({"metric": [1]}),
        draws_extractor=lambda fit: pd.DataFrame({"draw": [1.0]}),
        predictions_extractor=lambda fit, simulation: pd.DataFrame({"predicted": [1.0]}),
    )
    assert result["status"] == "complete"
    assert result["diagnostics"]["converged"].iloc[0]
    assert isinstance(result["draws"], pd.DataFrame)
    assert isinstance(result["predictions"], pd.DataFrame)


def test_lock_stale_existing_owner_error_and_release(tmp_path, monkeypatch):
    lock = tmp_path / "checkpoints" / "job.lock"
    lock.mkdir(parents=True)
    old = time.time() - 100
    os.utime(lock, (old, old))
    assert vo._acquire_lock(lock, stale_after_seconds=1)
    assert (lock / "owner.csv").exists()
    vo._release_lock(lock)
    assert not lock.exists()

    lock.mkdir(parents=True)
    assert not vo._acquire_lock(lock, stale_after_seconds=math.inf)
    vo._release_lock(lock)

    original = pd.DataFrame.to_csv

    def fail_csv(self, *args, **kwargs):
        raise OSError("owner unavailable")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_csv)
    assert vo._acquire_lock(lock, stale_after_seconds=1)
    assert not (lock / "owner.csv").exists()
    vo._release_lock(lock)
    monkeypatch.setattr(pd.DataFrame, "to_csv", original)


def test_run_and_checkpoint_existing_corrupt_locked_callr_and_fail_fast(tmp_path, monkeypatch):
    plan = _plan()
    job = _job(plan)
    output = tmp_path / "run"
    (output / "checkpoints").mkdir(parents=True)
    (output / "logs").mkdir()

    checkpoint = vo._checkpoint_path(output, job["job_id"])
    vo._atomic_json(checkpoint, {"status": "complete", "job": job})
    cached = vo._run_and_checkpoint(
        job,
        output,
        _runner_args(),
        isolation="in_process",
        timeout_seconds=math.inf,
        memory_limit_mb=math.inf,
        overwrite=False,
        fail_fast=False,
        stale_lock_seconds=1,
    )
    assert cached["status"] == "complete"

    checkpoint.write_text("{bad", encoding="utf-8")
    corrupt = vo._run_and_checkpoint(
        job,
        output,
        _runner_args(),
        isolation="in_process",
        timeout_seconds=math.inf,
        memory_limit_mb=math.inf,
        overwrite=False,
        fail_fast=False,
        stale_lock_seconds=1,
    )
    assert corrupt["status"] == "corrupt"
    checkpoint.unlink()

    lock = vo._lock_path(output, job["job_id"])
    lock.mkdir()
    locked = vo._run_and_checkpoint(
        job,
        output,
        _runner_args(),
        isolation="in_process",
        timeout_seconds=math.inf,
        memory_limit_mb=math.inf,
        overwrite=True,
        fail_fast=False,
        stale_lock_seconds=math.inf,
    )
    assert locked["status"] == "locked"
    vo._release_lock(lock)

    with pytest.raises(EyeProcessBackendError, match="callr"):
        vo._run_and_checkpoint(
            job,
            output,
            _runner_args(),
            isolation="callr",
            timeout_seconds=1,
            memory_limit_mb=math.inf,
            overwrite=True,
            fail_fast=False,
            stale_lock_seconds=1,
        )
    assert not vo._lock_path(output, job["job_id"]).exists()

    monkeypatch.setattr(
        vo,
        "_run_job_core",
        lambda *args, **kwargs: vo._job_result(job, "failed", "fit", time.time(), error="boom"),
    )
    with pytest.raises(EyeProcessValidationError, match="Validation job failed"):
        vo._run_and_checkpoint(
            job,
            output,
            _runner_args(),
            isolation="in_process",
            timeout_seconds=math.inf,
            memory_limit_mb=math.inf,
            overwrite=True,
            fail_fast=True,
            stale_lock_seconds=1,
        )
    assert not vo._lock_path(output, job["job_id"]).exists()


def test_backend_isolation_and_status_table_residuals(tmp_path, monkeypatch):
    assert vo._backend(("auto", "future"), 1) == "sequential"
    assert vo._backend("auto", 2) == "future"
    assert vo._backend("sequential", 4) == "sequential"
    with pytest.raises(EyeProcessValidationError, match="backend"):
        vo._backend("bad", 1)

    assert vo._isolation(("auto", "callr"), math.inf) == "in_process"
    assert vo._isolation("auto", 1) == "callr"
    assert vo._isolation("in_process", 1) == "in_process"
    with pytest.raises(EyeProcessValidationError, match="isolation"):
        vo._isolation("bad", 1)

    plan = _plan()
    output = tmp_path / "status"
    (output / "logs").mkdir(parents=True)
    vo._status_table(output, plan)
    assert not (output / "job-status.csv").exists()

    (output / "logs" / "job-bad.csv").write_text('"unterminated', encoding="utf-8")
    vo._status_table(output, plan)
    assert not (output / "job-status.csv").exists()

    pd.DataFrame({"job_id": ["unknown"], "status": ["complete"]}).to_csv(
        output / "logs" / "job-unknown.csv", index=False
    )
    vo._status_table(output, plan)
    status = pd.read_csv(output / "job-status.csv")
    assert status.empty


def test_run_validation_jobs_input_output_filter_progress_and_timeout_guards(tmp_path, capsys):
    plan = _plan()

    with pytest.raises(EyeProcessValidationError, match="validation plan"):
        vo.run_validation_jobs({}, _simulator, _fitter, _extractor, _truth, tmp_path / "bad-plan")
    with pytest.raises(EyeProcessValidationError, match="must be functions"):
        vo.run_validation_jobs(plan, None, _fitter, _extractor, _truth, tmp_path / "bad-function")
    with pytest.raises(EyeProcessValidationError, match="confidence"):
        vo.run_validation_jobs(
            plan, _simulator, _fitter, _extractor, _truth, tmp_path / "bad-confidence", confidence=1
        )
    with pytest.raises(EyeProcessBackendError, match="finite timeout"):
        vo.run_validation_jobs(
            plan,
            _simulator,
            _fitter,
            _extractor,
            _truth,
            tmp_path / "timeout",
            timeout_seconds=1,
            isolation="auto",
        )

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "other.txt").write_text("x", encoding="utf-8")
    with pytest.raises(EyeProcessValidationError, match="not a validation manifest"):
        vo.run_validation_jobs(plan, _simulator, _fitter, _extractor, _truth, occupied)

    empty = tmp_path / "empty"
    empty.mkdir()
    run = vo.run_validation_jobs(
        plan,
        _simulator,
        _fitter,
        _extractor,
        _truth,
        empty,
        progress=True,
    )
    assert run["results"][0]["status"] == "complete"
    assert "[1/1]" in capsys.readouterr().out

    with pytest.raises(EyeProcessValidationError, match="No jobs were selected"):
        vo.run_validation_jobs(
            plan,
            _simulator,
            _fitter,
            _extractor,
            _truth,
            tmp_path / "no-id",
            job_ids="missing",
        )
    with pytest.raises(EyeProcessValidationError, match="No jobs were selected"):
        vo.run_validation_jobs(
            plan,
            _simulator,
            _fitter,
            _extractor,
            _truth,
            tmp_path / "no-chunk",
            chunks="missing",
        )


def test_resume_path_corrupt_retry_and_invalid_plan(tmp_path, monkeypatch):
    with pytest.raises(EyeProcessValidationError, match="Expected a validation plan"):
        vo.resume_validation_jobs({}, tmp_path)

    plan = _plan()
    output = Path(vo.write_validation_job_manifest(plan, tmp_path / "manifest"))
    job_id = plan["jobs"]["job_id"].iloc[0]
    checkpoint = vo._checkpoint_path(output, job_id)
    checkpoint.write_text("{bad", encoding="utf-8")

    captured = {}

    def fake_run(plan_arg, **kwargs):
        captured["job_ids"] = kwargs["job_ids"]
        return vo.EyeValidationRun(results=[{"status": "complete"}])

    monkeypatch.setattr(vo, "run_validation_jobs", fake_run)
    resumed = vo.resume_validation_jobs(
        output,
        output,
        retry=["corrupt"],
        simulator=_simulator,
        fitter=_fitter,
        extractor=_extractor,
        truth_extractor=_truth,
    )
    assert resumed["results"][0]["status"] == "complete"
    assert captured["job_ids"] == [job_id]


def test_bind_annotate_collect_corrupt_malformed_duplicate_and_autoplan(tmp_path):
    assert vo._bind_rows([pd.DataFrame(), None]).empty

    result = {
        "job": {"job_id": "j1", "scenario_id": "s1", "extra": 3},
        "status": "complete",
        "stage": "complete",
        "elapsed_seconds": 0.1,
        "warnings": ["w"],
        "error": None,
    }
    assert vo._annotate_frame(None, result).empty
    annotated = vo._annotate_frame(pd.DataFrame({"parameter": ["a"]}), result)
    assert annotated.extra.iloc[0] == 3
    assert annotated.warning_count.iloc[0] == 1

    with pytest.raises(EyeProcessValidationError, match="does not exist"):
        vo.collect_validation_jobs(tmp_path / "missing")

    plan = _plan()
    root = Path(vo.write_validation_job_manifest(plan, tmp_path / "root"))
    bad = root / "checkpoints" / "job-bad.json"
    bad.write_text("{bad", encoding="utf-8")
    malformed = root / "checkpoints" / "job-malformed.json"
    vo._atomic_json(malformed, {"status": "complete"})
    collection = vo.collect_validation_jobs(root)
    assert len(collection["corrupt"]) == 2
    assert collection["plan"]["plan_id"] == plan["plan_id"]

    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory in (first, second):
        (directory / "checkpoints").mkdir(parents=True)
    payload = vo._job_result(_job(plan), "complete", "complete", time.time())
    vo._atomic_json(first / "checkpoints" / f"job-{payload['job']['job_id']}.json", payload)
    changed = dict(payload)
    changed["status"] = "failed"
    vo._atomic_json(second / "checkpoints" / f"job-{payload['job']['job_id']}.json", changed)

    with pytest.raises(EyeProcessValidationError, match="Conflicting checkpoints"):
        vo.collect_validation_jobs([first, second], strict=True)

    dedup = vo.collect_validation_jobs([first, second], strict=False)
    assert len(dedup["results"]) == 1
    assert dedup["results"][0]["status"] == "failed"


def test_prune_invalid_custom_status_and_delete(tmp_path):
    with pytest.raises(EyeProcessValidationError, match="does not exist"):
        vo.prune_validation_checkpoints(tmp_path / "missing")

    root = tmp_path / "prune"
    (root / "checkpoints").mkdir(parents=True)
    job = _job()
    checkpoint = root / "checkpoints" / f"job-{job['job_id']}.json"
    vo._atomic_json(checkpoint, vo._job_result(job, "complete", "complete", time.time()))

    dry = vo.prune_validation_checkpoints(root, statuses="complete", dry_run=True)
    assert bool(dry.remove.iloc[0])
    assert checkpoint.exists()
    vo.prune_validation_checkpoints(root, statuses=["complete"], dry_run=False)
    assert not checkpoint.exists()
