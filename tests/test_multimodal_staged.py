from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

import eyeprocesspy as ep

STAGED_EXPORTS = [
    "prepare_multimodal_irt_data", "audit_multimodal_measurement", "multimodal_irt_spec",
    "simulate_multimodal_irt", "process_information", "ablate_multimodal_channels",
    "multimodal_backend_status", "multimodal_ppc", "validate_multimodal_irt",
    "audit_multimodal_identifiability", "multimodal_m2_spec", "fit_multimodal_m2",
    "audit_multimodal_m2_identifiability", "multimodal_m2_ppc", "validate_multimodal_m2",
    "multimodal_m2_ablation", "multimodal_m2_process_information", "multimodal_m2_negative_controls",
    "simulate_multimodal_m2", "multimodal_m2_recovery", "multimodal_m3_spec", "fit_multimodal_m3",
    "audit_multimodal_m3_identifiability", "multimodal_m3_ppc", "multimodal_m3_ablation",
    "multimodal_m3_process_information", "multimodal_m3_negative_controls", "multimodal_m3_functional_bridge",
    "validate_multimodal_m3", "simulate_multimodal_m3", "multimodal_m3_recovery", "multimodal_m4_spec",
    "fit_multimodal_m4", "audit_multimodal_m4_identifiability", "multimodal_m4_state_diagnostics",
    "multimodal_m4_ppc", "multimodal_m4_ablation", "multimodal_m4_process_information",
    "multimodal_m4_negative_controls", "multimodal_m4_sensitivity", "validate_multimodal_m4",
    "simulate_multimodal_m4", "multimodal_m4_recovery",
]


def test_staged_export_smoke():
    assert len(STAGED_EXPORTS) == 43
    assert all(callable(getattr(ep, name, None)) for name in STAGED_EXPORTS)


def test_multimodal_measurement_contract_and_simulation():
    a = ep.simulate_multimodal_irt(n_person=20, n_item=6, seed=1)
    b = ep.simulate_multimodal_irt(n_person=20, n_item=6, seed=1)
    pd.testing.assert_frame_equal(a.data, b.data)
    pd.testing.assert_frame_equal(a.truth["persons"], b.truth["persons"])
    assert len(a.data) == 120
    assert set(a.measurement.channels) == {"response", "rt", "gaze", "pupil"}
    audit = ep.audit_multimodal_measurement(a.measurement)
    assert audit.valid and audit.key_unique
    assert set(a.data.response.unique()) <= {0, 1}
    assert (a.data.rt > 0).all()
    assert (a.data.gaze_fixation_count >= 0).all()
    assert np.isfinite(a.data.pupil_response).all()
    assert ep.validate_multimodal_irt(a).valid

    d = pd.DataFrame({"person": [1, 1], "item": [1, 1], "response": [1, 0]})
    with pytest.raises(ep.EyeProcessValidationError, match="not unique|duplicat"):
        ep.prepare_multimodal_irt_data(d, person="person", item="item", response="response")

    small = ep.simulate_multimodal_irt(n_person=10, n_item=4, seed=2)
    ident = ep.audit_multimodal_identifiability(small.measurement)
    assert not ident.supported
    assert {"few_persons", "few_items"} <= set(ident.issues)


def test_m2_spec_simulation_identifiability_and_controls():
    spec = ep.multimodal_m2_spec()
    assert spec.eyeprocess_class == "eye_multimodal_m2_spec"
    assert spec.model == "M2" and spec.backend == "cmdstanr"
    assert set(spec.channels) == {"response", "rt", "gaze"}
    assert spec.reference["doi"] == "10.1177/01466216221089344"
    assert "Rasch" in spec.fidelity["response"]
    with pytest.raises(ep.EyeProcessValidationError, match="ignorable"):
        ep.multimodal_m2_spec(missingness="MNAR")

    a = ep.simulate_multimodal_m2(n_person=30, n_item=8, seed=42)
    b = ep.simulate_multimodal_m2(n_person=30, n_item=8, seed=42)
    pd.testing.assert_frame_equal(a.data, b.data)
    np.testing.assert_array_equal(a.truth["theta"], b.truth["theta"])
    assert len(a.data) == 240 and len(a.truth["theta"]) == 30 and len(a.truth["b"]) == 8
    assert set(a.data.response.unique()) <= {0, 1}
    assert (a.data.rt > 0).all() and (a.data.gaze >= 0).all()

    drop = ep.simulate_multimodal_m2(n_person=40, n_item=8, dropout=(.10, .20, .30), seed=91)
    assert drop.data[["response", "rt", "gaze"]].isna().any().all()
    assert not drop.complete_data[["response", "rt", "gaze"]].isna().any().any()

    ident = ep.audit_multimodal_m2_identifiability(ep.simulate_multimodal_m2(40, 8, seed=12).data)
    assert ident.supported
    assert ident.response_design["components"] == ident.rt_design["components"] == ident.gaze_design["components"] == 1
    assert ident.checks["pass"].all()
    dup = pd.concat([a.data, a.data.iloc[[0]]], ignore_index=True)
    with pytest.raises(ep.EyeProcessValidationError, match="at most one row"):
        ep.audit_multimodal_m2_identifiability(dup)

    nc = ep.multimodal_m2_negative_controls(ep.simulate_multimodal_m2(40, 8, seed=23), seed=99)
    assert set(nc.datasets) == {"observed", "gaze_within_item", "rt_within_item", "response_within_item"}
    for item in nc.datasets["observed"].item_id.unique():
        x = np.sort(nc.datasets["observed"].loc[lambda z: z.item_id == item, "gaze"].to_numpy())
        y = np.sort(nc.datasets["gaze_within_item"].loc[lambda z: z.item_id == item, "gaze"].to_numpy())
        np.testing.assert_array_equal(x, y)
    assert len(nc.diagnostics) == 12


def test_m3_spec_simulation_identifiability_negative_controls_and_bridge():
    spec = ep.multimodal_m3_spec()
    assert spec.eyeprocess_class == "eye_multimodal_m3_spec"
    assert spec.model == "M3" and spec.backend == "cmdstanr"
    assert set(spec.channels) == {"response", "rt", "gaze", "pupil"}
    assert "not automatically interpreted" in spec.interpretation
    with pytest.raises(ep.EyeProcessValidationError, match="no fallback"):
        ep.multimodal_m3_spec(backend="brms")
    with pytest.raises(ep.EyeProcessValidationError, match="ignorable"):
        ep.multimodal_m3_spec(missingness="MNAR")

    a = ep.simulate_multimodal_m3(30, 8, seed=20260815, pupil_missingness="none", dropout=(0, 0, 0, 0))
    b = ep.simulate_multimodal_m3(30, 8, seed=20260815, pupil_missingness="none", dropout=(0, 0, 0, 0))
    pd.testing.assert_frame_equal(a.data, b.data)
    assert len(a.data) == 240
    expected = {"pupil_baseline", "luminance", "gaze_x", "gaze_y", "pupil_quality", "pupil_blink", "pupil_interpolated", "time_on_task", "device", "session"}
    assert expected <= set(a.data.columns)
    assert {"theta", "tau", "omega", "rho", "b", "beta", "m", "kappa"} <= set(a.truth)
    assert a.data.pupil.isna().sum() == 0

    null = ep.simulate_multimodal_m3(25, 6, pupil_signal="null", pupil_missingness="none", dropout=(0,0,0,0), seed=1)
    conf = ep.simulate_multimodal_m3(25, 6, pupil_signal="confounded", pupil_missingness="none", dropout=(0,0,0,0), seed=2)
    assert np.std(null.truth["rho"], ddof=1) > 0 and np.std(null.truth["kappa"], ddof=1) > 0
    assert np.all(np.abs(null.truth["cor_person"][3, :3]) < 1e-12)
    assert np.all(conf.truth["rho"] == 0) and np.all(conf.truth["kappa"] == 0)
    assert np.std(conf.complete_data.pupil_nuisance_effect, ddof=1) > 0

    ident = ep.audit_multimodal_m3_identifiability(ep.simulate_multimodal_m3(40, 8, pupil_missingness="none", dropout=(0,0,0,0), seed=4))
    assert ident.supported and ident.variation.all()
    assert list(ident.missing_fraction.index) == ["response", "rt", "gaze", "pupil"]
    bad = ep.simulate_multimodal_m3(25, 6, pupil_missingness="none", dropout=(0,0,0,0), seed=5).data.copy()
    bad.loc[0, "pupil_baseline"] = np.nan
    with pytest.raises(ep.EyeProcessValidationError, match="does not silently impute"):
        ep.audit_multimodal_m3_identifiability(bad)

    sim = ep.simulate_multimodal_m3(30, 12, pupil_missingness="none", dropout=(0,0,0,0), seed=61)
    nca = ep.multimodal_m3_negative_controls(sim, seed=62)
    ncb = ep.multimodal_m3_negative_controls(sim, seed=62)
    assert set(nca.datasets) >= {"observed", "pupil_within_item", "pupil_within_person", "pupil_phase_randomized", "luminance_only_pupil", "irrelevant_pupil"}
    pd.testing.assert_frame_equal(nca.datasets["pupil_phase_randomized"], ncb.datasets["pupil_phase_randomized"])
    orig = nca.datasets["observed"].pupil.to_numpy()
    rnd = nca.datasets["pupil_phase_randomized"].pupil.to_numpy()
    np.testing.assert_allclose(np.mean(rnd), np.mean(orig), atol=1e-8)
    np.testing.assert_allclose(np.std(rnd, ddof=1), np.std(orig, ddof=1), atol=1e-6)
    assert not np.array_equal(rnd, orig)

    d = ep.simulate_multimodal_m3(25, 6, seed=7).data.copy()
    d["trajectory_score"] = np.arange(1, len(d)+1) / len(d)
    bridge = ep.multimodal_m3_functional_bridge(d, "trajectory_score", provenance="score from preregistered trajectory basis")
    np.testing.assert_allclose(bridge.data.pupil, d.trajectory_score)
    assert bridge.representation == "functional_score" and "does not claim" in bridge.boundary
    with pytest.raises(ep.EyeProcessValidationError, match="one finite-or-NA value per trial"):
        ep.multimodal_m3_functional_bridge(d, [1, 2])


def test_m4_spec_simulation_sequences_evidence_and_gate():
    spec = ep.multimodal_m4_spec()
    assert spec.eyeprocess_class == "eye_multimodal_m4_spec"
    assert spec.model == "M4" and spec.n_states == 2
    assert ep.multimodal_m4_spec(n_states=1, trait_conditioning=()).state_null
    with pytest.raises(ep.EyeProcessValidationError, match="1 through 4"):
        ep.multimodal_m4_spec(n_states=5)
    with pytest.raises(ep.EyeProcessValidationError, match="requires `rt`"):
        ep.multimodal_m4_spec(n_states=2, state_channels=("gaze", "pupil"))

    a = ep.simulate_multimodal_m4(10, 6, seed=101)
    b = ep.simulate_multimodal_m4(10, 6, seed=101)
    pd.testing.assert_frame_equal(a.data, b.data)
    np.testing.assert_array_equal(a.truth["state"], b.truth["state"])
    assert ep.simulate_multimodal_m4(10, 6, scenario="null", seed=102).truth["n_states"] == 1

    ident = ep.audit_multimodal_m4_identifiability(ep.simulate_multimodal_m4(12, 6, seed=104), include_posterior=False)
    assert set(["domain", "criterion", "status", "severity", "value", "threshold", "message", "recommendation"]) <= set(ident.checks.columns)
    assert ident.overall in {"PASS", "PASS_WITH_CAUTION", "REVIEW", "FAIL", "NOT_EVALUATED"}
    assert isinstance(ident.supported, (bool, np.bool_))

    bad = ep.simulate_multimodal_m4(8, 6, seed=11).data.copy()
    bad = pd.concat([bad.iloc[[1, 0]], bad.iloc[2:]], ignore_index=True)
    with pytest.raises(ep.EyeProcessValidationError, match="order|sorted|trial|sequence"):
        ep.audit_multimodal_m4_identifiability(bad, include_posterior=False)

    sim = ep.simulate_multimodal_m4(10, 6, seed=106)
    nca = ep.multimodal_m4_negative_controls(sim, seed=99, run=False)
    ncb = ep.multimodal_m4_negative_controls(sim, seed=99, run=False)
    expected = ["order_shuffle", "process_shuffle", "state_independent", "nuisance_pseudostate", "device_session_pseudostate", "overfit_state_count"]
    assert not nca.executed and nca.controls == expected == ncb.controls
    pd.testing.assert_frame_equal(nca.data, ncb.data)
    rec = ep.multimodal_m4_recovery()
    assert not rec.executed and len(rec.design) == 5

    review_spec = ep.multimodal_m4_spec(n_states=2, transition_structure="markov", trait_conditioning=("theta", "tau"), initial_trait_conditioning=True)
    aud = ep.audit_multimodal_m4_identifiability(ep.simulate_multimodal_m4(30, 10, n_states=2, scenario="clear", seed=20260821), spec=review_spec, include_posterior=False)
    z = aud.checks.loc[aud.checks.criterion == "trait_conditioned_markov"]
    assert len(z) == 1 and z.iloc[0].status == "REVIEW" and aud.overall == "REVIEW" and aud.supported
    unconditional = ep.multimodal_m4_spec(n_states=2, transition_structure="markov", trait_conditioning=(), initial_trait_conditioning=False)
    aud2 = ep.audit_multimodal_m4_identifiability(ep.simulate_multimodal_m4(30, 10, n_states=2, scenario="clear", seed=20260822), spec=unconditional, include_posterior=False)
    assert aud2.checks.loc[aud2.checks.criterion == "trait_conditioned_markov", "status"].iloc[0] == "PASS"
    with pytest.raises(ep.EyeProcessGovernanceError, match="gated for REVIEW"):
        ep.fit_multimodal_m4(sim, spec=review_spec)


def test_staged_stan_contracts_and_m4_identification_metadata():
    from importlib.resources import files
    root = files("eyeprocesspy.resources.stan")
    m2 = root.joinpath("m2-man2022-response-rt-gaze-0-10.stan").read_text()
    assert "bernoulli_logit" in m2
    assert "normal_lpdf" in m2
    assert "neg_binomial_2_log_lpmf" in m2
    assert "corr_person" in m2 and "corr_item" in m2

    m3 = root.joinpath("m3-response-rt-gaze-pupil-0-10.stan").read_text()
    subset = root.joinpath("m3-ablation-subset-0-10.stan").read_text()
    assert "cholesky_factor_corr[4] L_person" in m3
    assert "cholesky_factor_corr[4] L_item" in m3
    assert "matrix[4, 4] corr_person" in m3
    assert "vector[8] gamma_pupil" in m3
    assert "matrix[N_pupil, 8] X_pupil" in m3
    assert "log_lik_pupil" in m3
    assert "use_pupil" in subset

    m4 = root.joinpath("m4-trait-conditioned-state-0-11.stan").read_text()
    assert "ordered[K] state_rt_raw;" in m4
    assert "vector[K] delta_rt = state_rt_raw - mean(state_rt_raw);" in m4
    assert "mean(state_rt_raw) ~ normal(0, 0.45);" in m4
    assert "log_sum_exp" in m4
    assert "matrix[N, K] state_prob" in m4
    assert "int<lower=1, upper=K> z" not in m4

    spec = ep.multimodal_m4_spec(n_states=2)
    note = spec.identification["m4_state"]["note"]
    assert "adjacent-gap prior" in note
    assert "does not order psychological meaning" in note


def test_staged_plot_counterparts_return_data_bearing_axes():
    sim0 = ep.simulate_multimodal_irt(n_person=12, n_item=5, seed=41)
    sim2 = ep.simulate_multimodal_m2(n_person=12, n_item=5, seed=42)
    sim3 = ep.simulate_multimodal_m3(n_person=12, n_item=5, seed=43)
    sim4 = ep.simulate_multimodal_m4(n_person=12, n_item=6, seed=44)

    def check(ax):
        try:
            assert ax.figure is not None
            assert hasattr(ax, "gp3_data")
        finally:
            plt.close(ax.figure)

    generic_validation = ep.validate_multimodal_irt(sim0)
    generic_information = ep.process_information(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([[0.4], [1.0], [1.6], [2.2]]))
    for fn, obj in [
        (ep.plot_eye_multimodal_measurement, sim0.measurement),
        (ep.plot_eye_multimodal_simulation, sim0),
        (ep.plot_eye_process_information, generic_information),
        (ep.plot_eye_multimodal_validation, generic_validation),
    ]:
        check(fn(obj))

    m2_validation = ep.validate_multimodal_m2(sim2)
    m2_information = ep.multimodal_m2_process_information(sim2)
    m2_controls = ep.multimodal_m2_negative_controls(sim2)
    m2_recovery = ep.multimodal_m2_recovery(n_rep=2, n_person=12, n_item=5)
    for fn, obj in [
        (ep.plot_eye_multimodal_m2_simulation, sim2),
        (ep.plot_eye_multimodal_m2_ppc, ep.multimodal_m2_ppc(sim2)),
        (ep.plot_eye_multimodal_m2_information, m2_information),
        (ep.plot_eye_multimodal_m2_validation, m2_validation),
        (ep.plot_eye_multimodal_m2_recovery, m2_recovery),
        (ep.plot_eye_multimodal_m2_negative_controls, m2_controls),
    ]:
        check(fn(obj))

    m3_validation = ep.validate_multimodal_m3(sim3)
    m3_information = ep.multimodal_m3_process_information(sim3)
    m3_controls = ep.multimodal_m3_negative_controls(sim3)
    m3_recovery = ep.multimodal_m3_recovery(reps=2, n_person=12, n_item=5)
    m3_ident = ep.audit_multimodal_m3_identifiability(sim3)
    for fn, obj in [
        (ep.plot_eye_multimodal_m3_simulation, sim3),
        (ep.plot_eye_multimodal_m3_ppc, ep.multimodal_m3_ppc(sim3)),
        (ep.plot_eye_multimodal_m3_information, m3_information),
        (ep.plot_eye_multimodal_m3_validation, m3_validation),
        (ep.plot_eye_multimodal_m3_recovery, m3_recovery),
        (ep.plot_eye_multimodal_m3_negative_controls, m3_controls),
        (ep.plot_eye_multimodal_m3_identifiability, m3_ident),
    ]:
        check(fn(obj))

    states = ep.multimodal_m4_state_diagnostics(sim4)
    ident4 = ep.audit_multimodal_m4_identifiability(sim4, include_posterior=False)
    validation4 = ep.validate_multimodal_m4(sim4)
    controls4 = ep.multimodal_m4_negative_controls(sim4)
    sensitivity4 = ep.multimodal_m4_sensitivity(sim4)
    recovery4 = ep.multimodal_m4_recovery()
    for fn, obj in [
        (ep.plot_eye_multimodal_m4_simulation, sim4),
        (ep.plot_eye_multimodal_m4_states, states),
        (ep.plot_eye_multimodal_m4_identifiability, ident4),
        (ep.plot_eye_multimodal_m4_ppc, ep.multimodal_m4_ppc(sim4)),
        (ep.plot_eye_multimodal_m4_information, ep.multimodal_m4_process_information(sim4)),
        (ep.plot_eye_multimodal_m4_negative_controls, controls4),
        (ep.plot_eye_multimodal_m4_sensitivity, sensitivity4),
        (ep.plot_eye_multimodal_m4_recovery, recovery4),
        (ep.plot_eye_multimodal_m4_validation, validation4),
    ]:
        check(fn(obj))


def test_m4_session_truth_and_state_diagnostic_structure():
    sim = ep.simulate_multimodal_m4(10, 6, n_session=2, seed=1701)
    assert sim.data.sequence_id.nunique() == 20
    assert sim.truth["initial_prob_person"].shape == (10, 2)
    assert sim.truth["transition_prob_person"].shape == (10, 2, 2)
    np.testing.assert_allclose(sim.truth["initial_prob_person"].sum(axis=1), 1.0)
    np.testing.assert_allclose(sim.truth["transition_prob_person"].sum(axis=2), 1.0)

    states = ep.multimodal_m4_state_diagnostics(sim)
    pcols = [c for c in states.probability if c.startswith("state_") and c.endswith("_probability")]
    np.testing.assert_allclose(states.probability[pcols].sum(axis=1), 1.0)
    assert len(states.occupancy) == 2
    assert states.transition_matrix.shape == (2, 2)
    assert len(states.transition) == 4
    assert {"mean_entropy", "median_entropy", "mean_run_length", "switching_rate"} <= set(states.summary)

    assert ep.simulate_multimodal_m4(10, 6, scenario="nuisance_confounded", seed=1702).truth["n_states"] == 1
    assert ep.simulate_multimodal_m4(10, 6, scenario="device_confounded", seed=1703).truth["n_states"] == 1
    with pytest.raises(ep.EyeProcessValidationError, match="state_dependent"):
        ep.simulate_multimodal_m4(10, 6, scenario="null", missingness="state_dependent", seed=1704)


def test_m4_rejects_noncontiguous_sequence_blocks():
    sim = ep.simulate_multimodal_m4(8, 6, n_session=2, seed=1801)
    d = sim.data.copy()
    # Move one row from the first sequence into the middle of another block.
    row = d.iloc[[0]]
    d = pd.concat([d.iloc[1:4], row, d.iloc[4:]], ignore_index=True)
    with pytest.raises(ep.EyeProcessValidationError, match="contiguous|order|sorted|trial|sequence"):
        ep.audit_multimodal_m4_identifiability(d, include_posterior=False)


def test_staged_signatures_preserve_frozen_r_argument_names():
    import inspect
    import json
    from pathlib import Path

    signatures = json.loads(
        (Path(__file__).resolve().parents[1] / "reference" / "R_SIGNATURES.json").read_text()
    )
    for name in STAGED_EXPORTS:
        r_names = [arg["name"] for arg in signatures[name]["args"]]
        r_names = ["kwargs" if n == "..." else n for n in r_names]
        py_names = list(inspect.signature(getattr(ep, name)).parameters)
        assert py_names == r_names, f"{name}: R={r_names}, Python={py_names}"
