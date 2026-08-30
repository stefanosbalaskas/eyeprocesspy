from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "validation_seed",
    "validation_job_plan",
    "write_validation_job_manifest",
    "read_validation_job_manifest",
    "split_validation_plan",
    "run_validation_jobs",
    "resume_validation_jobs",
    "collect_validation_jobs",
    "prune_validation_checkpoints",
]


def _simulator(n, seed, fail=False):
    rng = np.random.default_rng(seed)
    return {
        "values": rng.normal(0.4, 1.0, int(n)),
        "truth": {"mu": 0.4},
        "fail": bool(fail),
    }


def _fitter(simulation):
    if simulation["fail"]:
        raise RuntimeError("declared failure")
    values = simulation["values"]
    return {
        "estimate": float(np.mean(values)),
        "se": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        "converged": True,
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


def test_public_r023_execution_exports():
    assert len(TARGETS) == 9
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_validation_seed_is_deterministic_and_key_order_invariant():
    a = ep.validation_seed(
        {"n": 20, "effect": 0.3},
        replication=2,
        base_seed=42,
    )
    b = ep.validation_seed(
        {"effect": 0.3, "n": 20},
        replication=2,
        base_seed=42,
    )
    c = ep.validation_seed(
        pd.DataFrame([{"effect": 0.3, "n": 20}]),
        replication=2,
        base_seed=42,
    )
    assert a == b == c
    assert a > 0


def test_validation_plan_has_frozen_grid_cardinality_and_unique_jobs():
    grid = {"n": [20, 40], "effect": [0.0, 0.3]}
    first = ep.validation_job_plan(
        grid,
        replications=3,
        base_seed=42,
        model_family="demo",
        chunk_size=2,
    )
    second = ep.validation_job_plan(
        grid,
        replications=3,
        base_seed=42,
        model_family="demo",
        chunk_size=2,
    )
    assert first.eyeprocess_class == "eye_validation_job_plan"
    assert len(first["jobs"]) == 12
    assert first["jobs"]["job_id"].is_unique
    assert (first["jobs"]["seed"] > 0).all()
    assert first["jobs"]["seed"].tolist() == second["jobs"]["seed"].tolist()
    assert first["plan_fingerprint"] == second["plan_fingerprint"]
    assert first["jobs"]["expected_checkpoint"].str.endswith(".json").all()
    assert first["jobs"]["chunk_id"].nunique() == 6


def test_manifest_roundtrip_and_split_use_transparent_json(tmp_path):
    plan = ep.validation_job_plan(
        {"n": [20, 30]},
        replications=2,
        base_seed=7,
        model_family="mean",
        chunk_size=2,
    )
    directory = Path(
        ep.write_validation_job_manifest(
            plan,
            tmp_path / "manifest",
        )
    )
    assert (directory / "plan.json").exists()
    assert (directory / "manifest.json").exists()
    assert (directory / "jobs.csv").exists()
    assert (directory / "design-grid.csv").exists()
    assert (directory / "serialization-boundary.md").exists()
    assert not list(directory.rglob("*.rds"))

    restored = ep.read_validation_job_manifest(directory)
    assert restored.eyeprocess_class == "eye_validation_job_plan"
    assert restored["plan_fingerprint"] == plan["plan_fingerprint"]
    assert restored["jobs"]["job_id"].tolist() == plan["jobs"]["job_id"].tolist()

    pieces = ep.split_validation_plan(restored)
    assert set(pieces) == set(plan["jobs"]["chunk_id"])
    for chunk, piece in pieces.items():
        assert piece["metadata"]["parent_plan_id"] == plan["plan_id"]
        assert piece["metadata"]["selected_chunk"] == chunk
        assert piece["jobs"]["chunk_id"].eq(chunk).all()


def test_sequential_jobs_checkpoint_collect_and_resume(tmp_path):
    plan = ep.validation_job_plan(
        {"n": [20, 30]},
        replications=2,
        base_seed=7,
        model_family="mean",
    )
    directory = tmp_path / "execution"
    run = ep.run_validation_jobs(
        plan,
        _simulator,
        _fitter,
        _extractor,
        _truth,
        directory,
        progress=False,
    )
    assert run.eyeprocess_class == "eye_validation_run"
    assert len(run["results"]) == 4
    assert all(result["status"] == "complete" for result in run["results"])
    assert len(list((directory / "checkpoints").glob("job-*.json"))) == 4
    assert not list(directory.rglob("*.rds"))
    assert (directory / "runner-manifest.json").exists()
    assert (directory / "job-status.csv").exists()

    collection = ep.collect_validation_jobs(directory, plan)
    assert collection.eyeprocess_class == "eye_validation_collection"
    assert len(collection["estimates"]) == 4
    assert collection["estimates"]["parameter"].eq("mu").all()
    assert collection["jobs"]["status"].eq("complete").all()

    resumed = ep.resume_validation_jobs(
        plan,
        directory,
        simulator=_simulator,
        fitter=_fitter,
        extractor=_extractor,
        truth_extractor=_truth,
        progress=False,
    )
    assert resumed.eyeprocess_class == "eye_validation_run"
    assert resumed["results"] == []
    assert resumed["backend"] == "none"


def test_failures_are_checkpointed_and_retained(tmp_path):
    plan = ep.validation_job_plan(
        pd.DataFrame(
            {
                "n": [20, 20],
                "fail": [False, True],
            }
        ),
        replications=1,
        base_seed=10,
        model_family="failure",
    )
    directory = tmp_path / "failure"
    run = ep.run_validation_jobs(
        plan,
        _simulator,
        _fitter,
        _extractor,
        _truth,
        directory,
    )
    assert [result["status"] for result in run["results"]].count("failed") == 1

    collection = ep.collect_validation_jobs(
        directory,
        plan,
        strict=False,
    )
    failed = collection["jobs"].loc[collection["jobs"]["status"].eq("failed")]
    assert len(failed) == 1
    assert "declared failure" in failed.iloc[0]["error"]


def test_future_backend_runs_selected_chunks_without_pickling_user_functions(tmp_path):
    plan = ep.validation_job_plan(
        {"n": [20, 30, 40, 50]},
        replications=1,
        base_seed=11,
        model_family="future",
        chunk_size=1,
    )
    selected = plan["jobs"]["chunk_id"].iloc[:2].tolist()
    run = ep.run_validation_jobs(
        plan,
        _simulator,
        _fitter,
        _extractor,
        _truth,
        tmp_path / "future",
        workers=2,
        backend="future",
        chunks=selected,
    )
    assert run["backend"] == "future"
    assert len(run["results"]) == 2
    assert all(result["status"] == "complete" for result in run["results"])


def test_runner_and_plan_revision_guards_are_enforced(tmp_path):
    plan = ep.validation_job_plan(
        {"n": [20]},
        replications=1,
        base_seed=12,
        model_family="guard",
    )
    directory = tmp_path / "guard"
    ep.run_validation_jobs(
        plan,
        _simulator,
        _fitter,
        _extractor,
        _truth,
        directory,
    )

    revised = ep.validation_job_plan(
        {"n": [21]},
        replications=1,
        base_seed=12,
        model_family="guard",
        plan_id=plan["plan_id"],
    )
    with pytest.raises(
        ep.EyeProcessValidationError,
        match="different validation plan",
    ):
        ep.run_validation_jobs(
            revised,
            _simulator,
            _fitter,
            _extractor,
            _truth,
            directory,
        )

    with pytest.raises(ep.EyeProcessValidationError, match="Runner functions"):
        ep.run_validation_jobs(
            plan,
            _simulator,
            lambda simulation: _fitter(simulation),
            _extractor,
            _truth,
            directory,
        )


def test_corrupt_checkpoint_is_reported_and_pruned(tmp_path):
    plan = ep.validation_job_plan(
        {"n": [20]},
        replications=1,
        base_seed=13,
        model_family="corrupt",
    )
    directory = Path(
        ep.write_validation_job_manifest(
            plan,
            tmp_path / "corrupt",
        )
    )
    bad = directory / "checkpoints" / "job-S99999-R000001.json"
    bad.write_text("{not-json", encoding="utf-8")

    audit = ep.prune_validation_checkpoints(
        directory,
        dry_run=True,
    )
    row = audit.loc[audit["file"].eq(str(bad.resolve()))].iloc[0]
    assert row["status"] == "corrupt"
    assert bool(row["remove"])
    assert bad.exists()

    ep.prune_validation_checkpoints(
        directory,
        dry_run=False,
    )
    assert not bad.exists()


def test_callr_isolation_remains_an_explicit_r_backend_boundary(tmp_path):
    plan = ep.validation_job_plan(
        {"n": [20]},
        replications=1,
        base_seed=14,
    )
    with pytest.raises(ep.EyeProcessBackendError, match="callr"):
        ep.run_validation_jobs(
            plan,
            _simulator,
            _fitter,
            _extractor,
            _truth,
            tmp_path / "callr",
            isolation="callr",
        )
