from __future__ import annotations

import inspect

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.exceptions import EyeProcessValidationError

FROZEN_PROCESS_QUALITY_EXPORTS = [
    "process_measure_registry", "validate_process_measure_registry", "register_process_measure",
    "find_process_measures", "process_measure_card", "process_measure_guardrails",
    "process_measure_coverage", "process_measure_lineage", "process_measure_units",
    "split_half_process_reliability", "process_icc", "process_bland_altman",
    "process_reliability_profile", "process_temporal_stability", "bootstrap_process_reliability",
    "estimate_calibration_error", "gaze_precision_rms_s2s", "effective_sampling_frequency",
    "audit_sampling_irregularity", "calibration_error_model", "gaze_uncertainty_ellipse",
    "propagate_calibration_uncertainty", "aoi_membership_probability", "probabilistic_aoi_assignment",
    "compare_hard_probabilistic_aoi", "calibration_sensitivity_grid", "fixation_boundary_uncertainty",
    "calibration_drift_profile", "gaze_data_quality_profile", "data_quality_reporting_table",
]


def _reliability_data(seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    people = np.arange(1, 31)
    u = rng.normal(size=len(people))
    rows = []
    for i, person in enumerate(people):
        for session in (1, 2):
            rows.append({"person_id": person, "session": session, "value": u[i] + rng.normal(scale=.2)})
    return pd.DataFrame(rows)


def _calibration_data(seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tx = np.repeat([.2, .5, .8], 10)
    ty = np.repeat([.2, .5, .8], 10)
    return pd.DataFrame({
        "target_x": tx, "target_y": ty,
        "gaze_x": tx + rng.normal(0, .01, 30),
        "gaze_y": ty + rng.normal(0, .01, 30),
    })


def test_frozen_exports_resolve_and_signatures_match_argument_names():
    import json
    signatures = json.load(open("reference/R_SIGNATURES.json", encoding="utf-8"))
    for name in FROZEN_PROCESS_QUALITY_EXPORTS:
        fn = getattr(ep, name)
        assert callable(fn), name
        py_names = list(inspect.signature(fn).parameters)
        r_names = [a["name"] for a in signatures[name]["args"]]
        assert py_names == r_names, (name, py_names, r_names)


def test_registry_contract_and_guardrails_match_r_tests():
    reg = ep.process_measure_registry()
    assert reg.attrs["eyeprocess_class"] == "eye_process_measure_registry"
    assert "pupil_peak" in set(reg.name)
    card = ep.process_measure_card("pupil_peak")
    assert card.eyeprocess_class == "eye_process_measure_card"
    assert isinstance(card.guardrail, str) and card.guardrail
    assert ep.validate_process_measure_registry(reg) is True

    bad = reg.copy()
    bad.loc[0, "guardrail"] = pd.NA
    with pytest.raises(EyeProcessValidationError, match="requires non-missing"):
        ep.validate_process_measure_registry(bad)
    with pytest.raises(EyeProcessValidationError, match="must be scalar"):
        ep.register_process_measure(
            reg, name=["x", "y"], channel="gaze", unit="a.u.", level="trial",
            interpretation="Neutral process feature.", guardrail="Not a psychological diagnosis.",
        )


def test_registry_search_coverage_lineage_and_units():
    reg = ep.register_process_measure(
        name="custom_signal", channel="gaze", unit="a.u.", level="trial",
        interpretation="Observed custom signal.", guardrail="Not a psychological construct.",
    )
    hit = ep.find_process_measures(reg, channel="gaze", query="custom signal")
    assert "custom_signal" in set(hit.name)
    coverage = ep.process_measure_coverage(pd.DataFrame({"custom_signal": [1.0, np.nan, 3.0]}), reg)
    row = coverage.loc[coverage.name == "custom_signal"].iloc[0]
    assert bool(row.present) is True
    assert row.nonmissing_fraction == pytest.approx(2 / 3)
    lineage = ep.process_measure_lineage("custom_signal", ["gaze_x", "gaze_y", "gaze_x"], ["center", "aggregate"], "trial")
    assert lineage.inputs == ["gaze_x", "gaze_y"]
    assert len(lineage.lineage_hash) == 64
    assert {"channel", "unit"} == set(ep.process_measure_units(reg).columns)
    assert set(ep.process_measure_guardrails(reg).columns) == {"name", "channel", "interpretation", "guardrail", "status"}


def test_reliability_contracts_match_frozen_r_tests_and_smoke():
    d = _reliability_data()
    icc = ep.process_icc(d, "person_id", "session", "value")
    assert np.isfinite(icc.icc_a1.iloc[0])
    ba = ep.process_bland_altman(d, "person_id", "session", "value")
    assert ba.eyeprocess_class == "eye_process_bland_altman"
    assert len(ba.pairs) == 30
    profile = ep.process_reliability_profile(d, "person_id", "session", "value")
    assert profile.eyeprocess_class == "eye_process_reliability_profile"
    stab = ep.process_temporal_stability(d, "person_id", "session", "value")
    assert len(stab) == 1 and np.isfinite(stab.correlation.iloc[0])
    boot = ep.bootstrap_process_reliability(d, "person_id", "session", "value", replications=20, seed=4)
    assert boot.replications.iloc[0] == 20
    assert np.isfinite(boot.estimate.iloc[0])


def test_split_half_reliability_handles_nonconsecutive_index_and_random_replications():
    rng = np.random.default_rng(3)
    rows = []
    for person in range(1, 16):
        trait = rng.normal()
        for trial in range(1, 9):
            rows.append({"person": person, "trial": trial, "measure": trait + rng.normal(scale=.25)})
    d = pd.DataFrame(rows)
    d.index = np.arange(100, 100 + len(d))
    odd = ep.split_half_process_reliability(d, "person", "trial", "measure")
    assert odd.split.iloc[0] == "odd_even"
    rnd = ep.split_half_process_reliability(d, "person", "trial", "measure", split="random", repetitions=5, seed=2)
    assert len(rnd) == 5
    assert rnd.n_persons.eq(15).all()


def test_calibration_quality_contracts_match_frozen_r_tests():
    d = _calibration_data()
    model = ep.calibration_error_model(d)
    assert model.eyeprocess_class == "eye_calibration_error_model"
    ellipse = ep.gaze_uncertainty_ellipse(model)
    assert len(ellipse) == 1 and ellipse.major_axis.iloc[0] >= ellipse.minor_axis.iloc[0]

    rng = np.random.default_rng(4)
    g = pd.DataFrame({
        "timestamp_ms": np.arange(0, 1000, 10),
        "gaze_x": rng.normal(size=100), "gaze_y": rng.normal(size=100), "valid": pd.Series([True] * 100, dtype="boolean"),
    })
    q = ep.gaze_data_quality_profile(g, valid="valid")
    assert q.eyeprocess_class == "eye_data_quality_profile"
    assert np.isfinite(q.table.effective_hz.iloc[0])
    assert q.table.effective_hz.iloc[0] == pytest.approx(100.0)
    g.loc[0, "valid"] = pd.NA
    q2 = ep.gaze_data_quality_profile(g, valid="valid")
    assert np.isfinite(q2.table.valid_fraction.iloc[0])

    aois = pd.DataFrame({"aoi": ["A"], "x_min": [0.0], "x_max": [1.0], "y_min": [0.0], "y_max": [1.0]})
    bd = ep.fixation_boundary_uncertainty(pd.DataFrame({"gaze_x": [.5, np.nan], "gaze_y": [.5, .2]}), aois)
    assert np.isnan(bd.signed_boundary_distance.iloc[1])


def test_calibration_metrics_irregularity_drift_and_reporting():
    d = _calibration_data()
    d["session"] = np.repeat([1, 2, 3], 10)
    err = ep.estimate_calibration_error(d, by="session")
    assert len(err) == 3
    drift = ep.calibration_drift_profile(d, "session")
    assert drift.eyeprocess_class == "eye_calibration_drift_profile"
    assert drift.table.delta_from_first.iloc[0] == pytest.approx(0.0)

    t = pd.DataFrame({"timestamp_ms": [0, 10, 20, 31, 40], "gaze_x": [0, .1, .2, .3, .4], "gaze_y": [0, 0, .1, .1, .2]})
    ef = ep.effective_sampling_frequency(t)
    assert np.isfinite(ef.effective_hz.iloc[0])
    ir = ep.audit_sampling_irregularity(t, cv_threshold=.01)
    assert ir.eyeprocess_class == "eye_sampling_irregularity_audit"
    assert bool(ir.table.irregularity_flag.iloc[0]) is True
    pr = ep.gaze_precision_rms_s2s(t, time="timestamp_ms")
    assert np.isfinite(pr.rms_s2s.iloc[0])
    prof = ep.gaze_data_quality_profile(t)
    pd.testing.assert_frame_equal(ep.data_quality_reporting_table(prof), prof.table)


def test_probabilistic_aoi_uncertainty_preserves_missing_draw_semantics():
    draws = pd.DataFrame({
        "sample_id": [1, 1, 2, 2],
        "gaze_x": [.5, .6, np.nan, np.nan],
        "gaze_y": [.5, .6, .2, np.nan],
    })
    aois = pd.DataFrame({"aoi": ["A", "B"], "x_min": [0, .6], "x_max": [.59, 1], "y_min": [0, 0], "y_max": [1, 1]})
    p = ep.aoi_membership_probability(draws, aois)
    assert p.loc[(p.sample_id == 1) & (p.aoi == "A"), "probability"].iloc[0] == pytest.approx(.5)
    assert p.loc[p.sample_id == 2, "probability"].isna().all()

    model = ep.calibration_error_model(_calibration_data())
    gaze = pd.DataFrame({"gaze_x": [.25, .75], "gaze_y": [.5, .5]})
    pa = ep.probabilistic_aoi_assignment(gaze, aois, model, draws=100, seed=2, min_probability=.4)
    assert pa.eyeprocess_class == "eye_probabilistic_aoi_assignment"
    assert len(pa.assignments) == 2
    cmp = ep.compare_hard_probabilistic_aoi(gaze, aois, pa)
    assert set(["hard_aoi", "probabilistic_aoi", "agreement"]).issubset(cmp.columns)
    sens = ep.calibration_sensitivity_grid()
    assert len(sens) == 9 and sens.calibration_spec_id.iloc[0] == "CAL001"


def test_process_quality_plot_counterparts_expose_scientific_plot_data():
    d = _reliability_data()
    profile = ep.process_reliability_profile(d, "person_id", "session", "value")
    ax = ep.plot_eye_process_reliability_profile(profile)
    assert len(ax.eyeprocess_plot_data) == 30

    cal = _calibration_data()
    model = ep.calibration_error_model(cal)
    ax2 = ep.plot_eye_calibration_error_model(model)
    assert len(ax2.eyeprocess_plot_data) == 30

    cal["session"] = np.repeat([1, 2, 3], 10)
    drift = ep.calibration_drift_profile(cal, "session")
    assert len(ep.plot_eye_calibration_drift_profile(drift).eyeprocess_plot_data) == 3

    g = pd.DataFrame({"timestamp_ms": np.arange(0, 100, 10), "gaze_x": np.linspace(0, 1, 10), "gaze_y": np.linspace(0, 1, 10)})
    quality = ep.gaze_data_quality_profile(g)
    assert len(ep.plot_eye_data_quality_profile(quality).eyeprocess_plot_data) == 1
    irr = ep.audit_sampling_irregularity(g)
    assert len(ep.plot_eye_sampling_irregularity_audit(irr).eyeprocess_plot_data) == 1

    aois = pd.DataFrame({"aoi": ["A", "B"], "x_min": [0, .5], "x_max": [.49, 1], "y_min": [0, 0], "y_max": [1, 1]})
    pa = ep.probabilistic_aoi_assignment(g[["gaze_x", "gaze_y"]].iloc[:3], aois, model, draws=40, seed=1, min_probability=.2)
    ax3 = ep.plot_eye_probabilistic_aoi_assignment(pa)
    assert ax3.eyeprocess_plot_matrix.shape == (2, 3)
    assert len(ax3.eyeprocess_plot_data) == 6
