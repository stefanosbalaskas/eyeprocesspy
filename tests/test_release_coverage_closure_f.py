from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.irt_validation_07 as iv
import eyeprocesspy.pupil_missingness as pm


def _recovery(with_intervals: bool = False):
    d = pd.DataFrame(
        {
            "replicate": [1, 2, 3],
            "parameter": ["b", "b", "b"],
            "truth": [-0.2, 0.0, 0.2],
            "estimate": [-0.1, 0.1, 0.3],
            "converged": [True, True, True],
        }
    )
    if with_intervals:
        d["lower"] = d["estimate"] - 0.3
        d["upper"] = d["estimate"] + 0.3
    return d


def _close(ax):
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_irt_validation_summary_identifiability_and_mcse_residuals():
    assert iv._bind([None, pd.DataFrame()]).empty

    # No lower/upper columns and no grouping hit the explicit fallback paths.
    s = iv.summarize_parameter_recovery(_recovery(), by=())
    assert len(s) == 1 and np.isnan(s.coverage.iloc[0])

    corr = np.array([[1.0, 0.9999], [0.9999, 1.0]])
    ident = iv.audit_identifiability(_recovery(), correlation_matrix=corr, max_abs_correlation=0.99)
    assert ident.attrs["correlation_issue"] is True
    assert not bool(ident["pass"].all())

    with pytest.raises(ep.EyeProcessValidationError, match="metric"):
        iv.validation_mcse(_recovery(), metric="bad")
    cov = iv.validation_mcse(_recovery(with_intervals=True), metric="coverage")
    assert cov.n.iloc[0] == 3
    assert iv.validation_mcse(_recovery(), metric="bias").n.iloc[0] == 3
    assert iv.validation_mcse(_recovery(), metric="rmse").n.iloc[0] == 3

    with pytest.raises(ep.EyeProcessValidationError, match="coverage or mean"):
        iv.recommended_validation_replications(metric="bad")
    assert iv.recommended_validation_replications(metric="mean", target_mcse=0.5, anticipated_sd=1, minimum=1) == 4


def test_irt_sbc_failure_paths_and_ppc_stress_guards():
    with pytest.raises(ep.EyeProcessValidationError, match="must be functions"):
        iv.run_sbc(None, lambda x: x, lambda x: x)

    sbc = iv.run_sbc(
        simulator=lambda r: {"bad": 1},
        fitter=lambda x: x,
        posterior_draws=lambda x: x,
        replications=1,
    )
    assert len(sbc.failures) == 1

    with pytest.raises(ep.EyeProcessValidationError, match="replication"):
        iv.posterior_sbc_contract(1)
    with pytest.raises(ep.EyeProcessValidationError, match="posterior_sbc_contract"):
        iv.run_posterior_sbc({}, {})

    contract = iv.posterior_sbc_contract(lambda r, observed: {"truth": {"b": 0.0}, "draws": {"x": [0.1]}})
    psbc = iv.run_posterior_sbc({}, contract, replications=1)
    assert len(psbc.failures) == 1

    with pytest.raises(ep.EyeProcessValidationError, match="contain datasets"):
        iv.posterior_predictive_discrepancies([1], [])

    with pytest.raises(ep.EyeProcessValidationError, match="runner"):
        iv.stress_test_misspecification([{"scenario": "x"}], runner=1)
    failed = iv.stress_test_misspecification(
        {"x": 1},
        runner=lambda sc, r: (_ for _ in ()).throw(RuntimeError("failure")),
        replications=1,
    )
    assert bool(failed.failed.iloc[0])


def test_irt_external_group_incremental_calibration_and_grade_residuals():
    # External validation exception path.
    ext = iv.external_validate_irt(
        [1, 2], [3],
        fitter=lambda train: (_ for _ in ()).throw(RuntimeError("fit failed")),
        predictor=lambda model, test: model,
        scorer=lambda test, pred: pd.DataFrame({"score": [1]}),
    )
    assert bool(ext.failed.iloc[0])

    # Group-out exception branch in every fold.
    data = pd.DataFrame({"site": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]})
    group = iv.leave_site_out_validation(
        data, "site",
        fitter=lambda train: (_ for _ in ()).throw(RuntimeError("fit failed")),
        predictor=lambda model, test: model,
        scorer=lambda test, pred: pd.DataFrame({"score": [1]}),
    )
    assert group.failed.all()

    # Incremental-information exception branch.
    inc = iv.audit_channel_incremental_information(
        data.rename(columns={"site": "fold"}), "fold",
        baseline_fitter=lambda train: (_ for _ in ()).throw(RuntimeError("fail")),
        process_fitter=lambda train: None,
        predictor=lambda model, test: model,
        scorer=lambda test, pred: [1.0],
    )
    assert inc.failed.all()

    # Both short-group and ordinary least-squares calibration paths.
    cal = pd.DataFrame(
        {
            "g": ["short", "short", "long", "long", "long"],
            "y": [0.0, 1.0, 1.0, 2.0, 3.0],
            "p": [0.2, 0.8, 1.1, 2.1, 2.9],
        }
    )
    out = iv.calibration_transfer_audit(cal, "g", "y", "p")
    assert out.loc[out.g.eq("short"), "slope"].isna().all()
    assert out.loc[out.g.eq("long"), "slope"].notna().all()

    recovery = iv.summarize_parameter_recovery(_recovery(with_intervals=True))
    spec = iv.irt_validation_spec("m", thresholds={"min_external_folds": 1})
    sbc_audit = pd.DataFrame({"pass_screen": [True]})
    sbc_audit.attrs["eyeprocess_class"] = "eye_sbc_audit"
    ppc = pd.DataFrame({"p_two_sided": [0.5]})
    external = pd.DataFrame({"failed": [False]})
    semantic = pd.DataFrame({"status": ["SUPPORTED"]})
    grade = iv.grade_model_evidence(
        recovery,
        spec=spec,
        external_validation=external,
        sbc=sbc_audit,
        ppc=ppc,
        semantic_roundtrip=semantic,
    )
    assert set(["sbc_screen", "ppc_extremes", "external_folds", "semantic_roundtrip"]).issubset(set(grade.checks.criterion))


def test_pupil_missingness_alignment_plot_and_pattern_mixture_residuals():
    class BadFrame:
        def __iter__(self):
            raise RuntimeError("bad")
    with pytest.raises(ep.EyeProcessValidationError, match="coercible"):
        pm._df(BadFrame())

    # One short curve hits registration's insufficient-point branch; another
    # complete curve allows the object to be inspected/plot-tested.
    raw = pd.DataFrame(
        {
            "person_id": ["P1", "P1", "P2", "P2", "P2"],
            "time": [0, 1, 0, 1, 2],
            "pupil": [1.0, 1.1, 1.0, 1.5, 1.2],
        }
    )
    reg = pm.register_pupil_curves(raw, "time", "pupil", grid_size=5)
    assert np.isnan(reg.raw[0]).all()
    ax = pm.plot_registered_pupil_effects(reg)
    assert not ax.eyeprocess_plot_data.empty
    _close(ax)

    # Explicit score-alignment failure and unkeyed score concatenation path.
    with pytest.raises(ep.EyeProcessValidationError, match="align"):
        pm.fit_phase_amplitude_irt([1, 0], pd.DataFrame({"phase": [0.1]}))
    fit = pm.fit_phase_amplitude_irt(
        [1, 0],
        pd.DataFrame({"phase": [0.1, 0.2]}),
        amplitude_scores=pd.DataFrame({"amp": [0.3, 0.4]}),
    )
    assert {"phase", "amp"}.issubset(fit.data.columns)

    # DataFrame metric inference plus estimand TypeError fallback.
    frame = pd.DataFrame({"value": [1.0, np.nan, 3.0]})
    def axis_only(x, axis):
        return np.mean(x, axis=axis)
    sens = pm.process_pattern_mixture(frame, delta=[0], metric=None, estimand=axis_only)
    assert sens.metric == "value"
    assert len(sens.table) == 1

    # Wrapper-only lines and alternate plotting selectors.
    ax = pm.plot_eye_pupil_registration(reg, type="warping")
    _close(ax)
    ax = pm.plot_eye_phase_amplitude_irt(fit, type="registered_effects")
    _close(ax)
