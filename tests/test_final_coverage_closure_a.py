from __future__ import annotations

import builtins
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.benchmark_reproducibility_10 as br
import eyeprocesspy.process_quality_09 as pq


def _benchmark_copy(tmp_path: Path) -> Path:
    source = Path(ep.eyeprocess_benchmark_study()["path"])
    target = tmp_path / "benchmark"
    shutil.copytree(source, target)
    return target


def test_benchmark_private_mapping_root_and_coercion_guards(tmp_path: Path):
    study = ep.eyeprocess_benchmark_study()
    with pytest.raises(AttributeError):
        _ = study.this_key_does_not_exist

    with pytest.raises(ep.EyeProcessValidationError, match="directory is unavailable"):
        ep.eyeprocess_benchmark_study(tmp_path / "missing")

    root = _benchmark_copy(tmp_path)
    (root / "manifest.csv").unlink()
    with pytest.raises(ep.EyeProcessValidationError, match="manifest is missing"):
        ep.eyeprocess_benchmark_study(root)

    study = ep.eyeprocess_benchmark_study()
    assert br._coerce_study(study) is study
    assert br._coerce_study(study["path"])["path"] == study["path"]
    mapped = br._coerce_study({"path": study["path"], "manifest": study["manifest"]})
    assert mapped.eyeprocess_class == "eye_benchmark_study"
    with pytest.raises(TypeError, match="benchmark study"):
        br._coerce_study(object())


def test_benchmark_boolean_normalization_and_table_guards(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "logical": [" true ", "FALSE", None],
            "complete": ["TRUE", "false", "true"],
            "text": ["x", "y", "z"],
            "numeric": [1, 2, 3],
        }
    )
    out = br._normalise_csv_types(frame)
    assert str(out["logical"].dtype) == "boolean"
    assert pd.api.types.is_bool_dtype(out["complete"].dtype)
    assert out["text"].tolist() == ["x", "y", "z"]

    study = ep.eyeprocess_benchmark_study()
    with pytest.raises(ep.EyeProcessValidationError, match="non-empty"):
        ep.read_benchmark_table(study, "")
    with pytest.raises(ep.EyeProcessValidationError, match="absent"):
        ep.read_benchmark_table(study, "not_a_table")

    root = _benchmark_copy(tmp_path)
    manifest = pd.read_csv(root / "manifest.csv")
    first = root / str(manifest.iloc[0]["file"])
    first.unlink()
    broken = ep.eyeprocess_benchmark_study(root)
    with pytest.raises(ep.EyeProcessValidationError, match="file is missing"):
        ep.read_benchmark_table(broken, str(manifest.iloc[0]["table"]))

    root2 = _benchmark_copy(tmp_path / "second")
    (root2 / "expected_outputs.csv").unlink()
    with pytest.raises(ep.EyeProcessValidationError, match="expected-output file is missing"):
        ep.benchmark_expected_outputs(ep.eyeprocess_benchmark_study(root2))


def test_benchmark_import_fallback_validation_without_hashes_and_broken_files(tmp_path: Path, monkeypatch):
    import eyeprocesspy.dataset as dataset_mod

    def fail_dataset(**kwargs):
        raise RuntimeError("force benchmark-table fallback")

    monkeypatch.setattr(dataset_mod, "new_eye_dataset", fail_dataset)
    imported = ep.import_benchmark_study()
    assert imported.eyeprocess_class == "eye_benchmark_tables"

    validation = ep.validate_benchmark_study(verify_hashes=False)
    assert validation["valid"] is True
    assert validation["files"]["hash_match"].isna().all()

    root = _benchmark_copy(tmp_path)
    manifest = pd.read_csv(root / "manifest.csv")
    (root / str(manifest.iloc[0]["file"])).unlink()
    invalid = ep.validate_benchmark_study(ep.eyeprocess_benchmark_study(root), verify_hashes=False)
    assert invalid["valid"] is False
    assert invalid["relations"].empty


def test_benchmark_reproduction_validation_guards(tmp_path: Path):
    root = _benchmark_copy(tmp_path)
    gaze_path = root / "gaze_samples.csv"
    gaze = pd.read_csv(gaze_path)
    gaze.loc[0, "valid"] = np.nan
    gaze.to_csv(gaze_path, index=False)
    with pytest.raises(ep.EyeProcessValidationError, match="complete logical"):
        ep.run_benchmark_reproduction(ep.eyeprocess_benchmark_study(root))

    root2 = _benchmark_copy(tmp_path / "expected")
    expected_path = root2 / "expected_outputs.csv"
    expected = pd.read_csv(expected_path).iloc[:-1]
    expected.to_csv(expected_path, index=False)
    with pytest.raises(ep.EyeProcessValidationError, match="must match completely"):
        ep.run_benchmark_reproduction(ep.eyeprocess_benchmark_study(root2))


def test_reproducibility_path_manifest_and_scaffold_residuals(tmp_path: Path, monkeypatch):
    folder = tmp_path / "payload"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    a = folder / "a.txt"
    b = nested / "b.txt"
    a.write_text("a\n", encoding="utf-8")
    b.write_text("b\n", encoding="utf-8")

    paths = br._coerce_paths([folder, a, tmp_path / "absent"])
    assert {p.name for p in paths} == {"a.txt", "b.txt"}
    assert br._coerce_paths(a) == [a.resolve()]

    manifest = ep.package_reproducibility_manifest(folder, include_session=False)
    assert manifest["session"] == []

    with pytest.raises(TypeError, match="reproducibility manifest"):
        ep.verify_reproducibility_manifest({})
    with pytest.raises(TypeError, match="pandas DataFrame"):
        ep.verify_reproducibility_manifest({"files": []})

    fake_files = pd.DataFrame(
        [{"path": str(tmp_path / "gone.txt"), "bytes": 1, "md5": "x"}]
    )
    checked = ep.verify_reproducibility_manifest({"files": fake_files})
    assert checked.loc[0, "exists"] == False
    assert checked.loc[0, "unchanged"] == False

    root = _benchmark_copy(tmp_path / "with-dir")
    extra = root / "nested_resource"
    extra.mkdir()
    (extra / "note.txt").write_text("nested\n", encoding="utf-8")
    target = tmp_path / "scaffold"
    ep.write_software_paper_reproduction(target, ep.eyeprocess_benchmark_study(root))
    assert (target / "data" / "nested_resource" / "note.txt").is_file()
    ep.write_software_paper_reproduction(target, ep.eyeprocess_benchmark_study(root), overwrite=True)


def test_benchmark_release_invalid_branch(tmp_path: Path):
    root = _benchmark_copy(tmp_path)
    manifest = pd.read_csv(root / "manifest.csv")
    (root / str(manifest.iloc[0]["file"])).unlink()
    audit = ep.audit_benchmark_release(ep.eyeprocess_benchmark_study(root))
    assert audit["ready"] is False
    assert audit["reproduction"] is None


def test_process_quality_private_registry_and_search_guards():
    with pytest.raises(ep.EyeProcessValidationError, match="coercible"):
        pq._df(object())
    with pytest.raises(ep.EyeProcessValidationError, match="missing required columns"):
        pq._req(pd.DataFrame({"x": [1]}), ["x", "y"])
    assert np.isnan(pq._q([], 0.5))

    compact = ep.process_measure_registry(include_experimental=False)
    assert not (compact["status"] == "experimental").any()

    reg = ep.process_measure_registry()
    duplicate = pd.concat([reg, reg.iloc[[0]]], ignore_index=True)
    with pytest.raises(ep.EyeProcessValidationError, match="duplicate"):
        ep.validate_process_measure_registry(duplicate)
    blank = reg.iloc[[0]].copy()
    blank.loc[:, "guardrail"] = ""
    with pytest.raises(ep.EyeProcessValidationError, match="non-missing"):
        ep.validate_process_measure_registry(blank)

    with pytest.raises(ep.EyeProcessValidationError, match="must be scalar"):
        ep.register_process_measure(
            name=["a", "b"], channel="gaze", unit="ms", level="trial",
            interpretation="x", guardrail="y"
        )
    added = ep.register_process_measure(
        registry=compact,
        name=["custom"], channel=["gaze"], unit=["ms"], level=["trial"],
        interpretation=["custom interpretation"], guardrail=["custom guardrail"],
        status=["user_defined"],
    )
    assert "custom" in set(added.name)

    assert len(ep.find_process_measures(reg, channel=["gaze"])) > 0
    assert len(ep.find_process_measures(reg, level=["trial"])) > 0
    assert len(ep.find_process_measures(reg, status=["reference"])) > 0
    assert len(ep.find_process_measures(reg, query="attention")) > 0

    with pytest.raises(ep.EyeProcessValidationError, match="non-empty"):
        ep.process_measure_card("")
    with pytest.raises(ep.EyeProcessValidationError, match="exactly one"):
        ep.process_measure_card("does-not-exist")

    cov = ep.process_measure_coverage(pd.DataFrame({"fixation_count": [1.0, np.nan]}), reg)
    hit = cov.loc[cov.name == "fixation_count"].iloc[0]
    assert bool(hit.present) and hit.nonmissing_fraction == pytest.approx(0.5)
    empty_cov = ep.process_measure_coverage(pd.DataFrame(columns=["fixation_count"]), reg)
    assert np.isnan(empty_cov.loc[empty_cov.name == "fixation_count", "nonmissing_fraction"].iloc[0])


def _reliability_data(n_person: int = 4) -> pd.DataFrame:
    rows = []
    for p in range(n_person):
        for trial in range(1, 5):
            rows.append(
                {
                    "person": f"P{p+1}",
                    "trial": trial,
                    "session": "A" if trial <= 2 else "B",
                    "measure": float(p + trial),
                }
            )
    return pd.DataFrame(rows)


def test_process_reliability_degenerate_and_alternate_paths():
    data = _reliability_data()
    with pytest.raises(ep.EyeProcessValidationError, match="split must"):
        ep.split_half_process_reliability(data, "person", "trial", "measure", split="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="aggregate_fun"):
        ep.split_half_process_reliability(data, "person", "trial", "measure", aggregate_fun=1)
    with pytest.raises(ep.EyeProcessValidationError, match="positive integer"):
        ep.split_half_process_reliability(data, "person", "trial", "measure", repetitions=0)
    with pytest.raises(ep.EyeProcessValidationError, match="non-negative"):
        ep.split_half_process_reliability(data, "person", "trial", "measure", seed=-1)

    def axis_only(values, axis=None):
        if axis is None:
            raise TypeError("axis required")
        return np.mean(values, axis=axis)

    alt = ep.split_half_process_reliability(
        data, "person", "trial", "measure", split="odd_even", aggregate_fun=axis_only
    )
    assert len(alt) == 1
    random = ep.split_half_process_reliability(
        data, "person", "trial", "measure", split="random", repetitions=2, seed=2
    )
    assert len(random) == 2

    sparse = data.loc[data.person.isin(["P1", "P2"])].copy()
    sparse.loc[sparse.trial.eq(1), "measure"] = np.nan
    half = ep.split_half_process_reliability(sparse, "person", "trial", "measure")
    assert np.isnan(half.raw_r.iloc[0])

    invalid_icc = pq._icc_a1(np.ones((2, 1)))
    assert np.isnan(invalid_icc["icc"])
    zero_icc = pq._icc_a1(np.ones((3, 2)))
    assert np.isnan(zero_icc["icc"])

    with pytest.raises(ep.EyeProcessValidationError, match="exactly two"):
        ep.process_bland_altman(data, "person", "session", "measure", sessions=["A"])
    one = data.loc[data.person.eq("P1")]
    ba = ep.process_bland_altman(one, "person", "session", "measure")
    assert np.isnan(ba["summary"].sd_difference.iloc[0])

    one_session = data.loc[data.session.eq("A")]
    profile = ep.process_reliability_profile(one_session, "person", "session", "measure")
    assert profile["bland_altman"] is None

    with pytest.raises(ep.EyeProcessValidationError, match="method must"):
        ep.process_temporal_stability(data, "person", "session", "measure", method="bad")
    short = ep.process_temporal_stability(sparse, "person", "session", "measure")
    assert np.isnan(short.correlation.iloc[0])

    with pytest.raises(ep.EyeProcessValidationError, match="positive integer"):
        ep.bootstrap_process_reliability(data, "person", "session", "measure", replications=0)
    with pytest.raises(ep.EyeProcessValidationError, match="non-negative"):
        ep.bootstrap_process_reliability(data, "person", "session", "measure", seed=-1)
    no_ids = data.copy()
    no_ids["person"] = np.nan
    with pytest.raises(ep.EyeProcessValidationError, match="participant identifiers"):
        ep.bootstrap_process_reliability(no_ids, "person", "session", "measure", replications=1)
    boot = ep.bootstrap_process_reliability(sparse, "person", "session", "measure", replications=2, seed=1)
    assert len(boot) == 1


def test_process_quality_group_calibration_and_sampling_residuals(monkeypatch):
    base = pd.DataFrame(
        {
            "grp": ["A", "A", "B"],
            "gaze_x": [np.nan, 0.2, 0.3],
            "gaze_y": [np.nan, 0.2, 0.4],
            "target_x": [0.1, 0.1, 0.2],
            "target_y": [0.1, 0.1, 0.2],
            "timestamp_ms": [0.0, 10.0, 10.0],
        }
    )
    assert pq._groups(base, [])[0][0] == {".group": "all"}
    grouped = pq._groups(base, "grp")
    assert len(grouped) == 2

    all_nan = pd.DataFrame(
        {"gaze_x": [np.nan], "gaze_y": [np.nan], "target_x": [np.nan], "target_y": [np.nan]}
    )
    cal = ep.estimate_calibration_error(all_nan)
    assert cal.loc[0, "n"] == 0 and np.isnan(cal.loc[0, "mean_radial_error"])

    one = ep.gaze_precision_rms_s2s(pd.DataFrame({"gaze_x": [0.1], "gaze_y": [0.2], "t": [1]}), time="t")
    assert one.loc[0, "n_steps"] == 0
    with pytest.raises(ep.EyeProcessValidationError, match="unit must"):
        ep.effective_sampling_frequency(pd.DataFrame({"timestamp_ms": [0, 1]}), unit="bad")
    eff = ep.effective_sampling_frequency(pd.DataFrame({"timestamp_ms": [1.0]}))
    assert eff.loc[0, "n_intervals"] == 0 and np.isnan(eff.loc[0, "effective_hz"])
    with pytest.raises(ep.EyeProcessValidationError, match="cv_threshold"):
        ep.audit_sampling_irregularity(pd.DataFrame({"timestamp_ms": [0, 1]}), cv_threshold=-0.1)

    with pytest.raises(ep.EyeProcessValidationError, match="three complete"):
        ep.calibration_error_model(
            pd.DataFrame({"gaze_x": [0, 1], "gaze_y": [0, 1], "target_x": [0, 0], "target_y": [0, 0]})
        )
    fit_data = pd.DataFrame(
        {"gaze_x": [0.1, 0.2, 0.3, 0.4], "gaze_y": [0.2, 0.3, 0.4, 0.5], "target_x": [0, 0, 0, 0], "target_y": [0, 0, 0, 0]}
    )
    model = ep.calibration_error_model(fit_data)
    with pytest.raises(ep.EyeProcessValidationError, match="calibration_error_model"):
        ep.gaze_uncertainty_ellipse({})
    with pytest.raises(ep.EyeProcessValidationError, match="\(0,1\)"):
        ep.gaze_uncertainty_ellipse(model, level=1.0)
    with pytest.raises(ep.EyeProcessValidationError, match="two finite"):
        ep.gaze_uncertainty_ellipse(model, center=[np.nan, 0])
    ellipse = ep.gaze_uncertainty_ellipse(model, center=[0.0, 0.0])
    assert len(ellipse) == 1

    with pytest.raises(ep.EyeProcessValidationError, match="calibration_error_model"):
        ep.propagate_calibration_uncertainty(fit_data, {})
    with pytest.raises(ep.EyeProcessValidationError, match="positive integer"):
        ep.propagate_calibration_uncertainty(fit_data, model, draws=0)
    with pytest.raises(ep.EyeProcessValidationError, match="non-negative"):
        ep.propagate_calibration_uncertainty(fit_data, model, seed=-1)
    bad_model = dict(model)
    bad_model["covariance"] = np.array([[np.nan, 0], [0, 1]])
    with pytest.raises(ep.EyeProcessValidationError, match="non-finite"):
        ep.propagate_calibration_uncertainty(fit_data, bad_model)
    empty_draws = ep.propagate_calibration_uncertainty(fit_data.iloc[0:0], model, draws=1)
    assert empty_draws.empty

    empty_aoi = pd.DataFrame(columns=["aoi", "x_min", "x_max", "y_min", "y_max"])
    with pytest.raises(ep.EyeProcessValidationError, match="at least one rectangle"):
        pq._aois(empty_aoi)
    dup_aoi = pd.DataFrame(
        {"aoi": ["A", "A"], "x_min": [0, 0], "x_max": [1, 1], "y_min": [0, 0], "y_max": [1, 1]}
    )
    with pytest.raises(ep.EyeProcessValidationError, match="unique"):
        pq._aois(dup_aoi)
    bad_bounds = pd.DataFrame(
        {"aoi": ["A"], "x_min": [2], "x_max": [1], "y_min": [0], "y_max": [1]}
    )
    with pytest.raises(ep.EyeProcessValidationError, match="finite and ordered"):
        pq._aois(bad_bounds)

    aoi = pd.DataFrame({"aoi": ["A"], "x_min": [0], "x_max": [1], "y_min": [0], "y_max": [1]})
    probs = ep.aoi_membership_probability(
        pd.DataFrame({"sample_id": [1], "gaze_x": [np.nan], "gaze_y": [np.nan]}), aoi
    )
    assert np.isnan(probs.probability.iloc[0])

    with pytest.raises(ep.EyeProcessValidationError, match="min_probability"):
        ep.probabilistic_aoi_assignment(fit_data.iloc[:1], aoi, model, min_probability=2)
    uncertain = ep.probabilistic_aoi_assignment(
        pd.DataFrame({"gaze_x": [np.nan], "gaze_y": [np.nan]}), aoi, model, draws=2
    )
    assert pd.isna(uncertain.assignments.aoi.iloc[0])
    with pytest.raises(ep.EyeProcessValidationError, match="probabilistic"):
        ep.compare_hard_probabilistic_aoi(fit_data.iloc[:1], aoi, {})
    compared = ep.compare_hard_probabilistic_aoi(
        pd.DataFrame({"gaze_x": [np.nan], "gaze_y": [np.nan]}), aoi, uncertain
    )
    assert bool(compared.agreement.iloc[0])

    with pytest.raises(ep.EyeProcessValidationError, match="finite values"):
        ep.calibration_sensitivity_grid([np.nan], [0])
    boundary = ep.fixation_boundary_uncertainty(
        pd.DataFrame({"gaze_x": [np.nan, 0.5, 2.0], "gaze_y": [0.2, 0.5, 2.0]}), aoi
    )
    assert pd.isna(boundary.nearest_aoi.iloc[0])
    assert boundary.signed_boundary_distance.iloc[1] >= 0
    assert boundary.signed_boundary_distance.iloc[2] < 0

    monkeypatch.setattr(pq, "estimate_calibration_error", lambda *args, **kwargs: pd.DataFrame())
    with pytest.raises(ep.EyeProcessValidationError, match="No calibration groups"):
        ep.calibration_drift_profile(base, "grp")


def test_gaze_quality_profile_and_reporting_guards():
    with pytest.raises(ep.EyeProcessValidationError, match="supplied together"):
        ep.gaze_data_quality_profile(
            pd.DataFrame({"gaze_x": [0.1], "gaze_y": [0.1], "timestamp_ms": [0]}),
            target_x="target_x",
        )
    empty = ep.gaze_data_quality_profile(
        pd.DataFrame(columns=["gaze_x", "gaze_y", "timestamp_ms"])
    )
    assert empty["table"].empty

    data = pd.DataFrame(
        {
            "gaze_x": [0.1, np.nan, 0.3],
            "gaze_y": [0.1, 0.2, 0.3],
            "timestamp_ms": [0.0, 10.0, 20.0],
            "valid": [True, True, False],
        }
    )
    profile = ep.gaze_data_quality_profile(data, valid="valid")
    assert profile["table"].valid_fraction.iloc[0] == pytest.approx(1 / 3)
    with pytest.raises(ep.EyeProcessValidationError, match="eye_data_quality_profile"):
        ep.data_quality_reporting_table({})
    assert len(ep.data_quality_reporting_table(profile)) == 1
