from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.exceptions import EyeProcessBackendError, EyeProcessValidationError

EXPORTS = [
    "apply_preflight_decision","audit_biometric_preflight","audit_multivariate_process_quality",
    "audit_presentation_accessibility","audit_process_anomalies","compare_presentation_fairness",
    "preflight_decisions","preflight_exclusion_manifest","preflight_failures","preflight_passed",
    "process_anomaly_distance","process_preflight_spec","simulate_presentation_variants",
    "audit_process_drift","compare_deployment_batches","drift_by_device","drift_by_site",
    "drift_by_stimulus_version","drift_by_vendor","process_drift_alerts","process_drift_spec",
    "aoi_trajectory_features","audit_process_window_sensitivity","bind_process_windows",
    "compare_aoi_trajectories","extract_process_windows","fit_aoi_growth_curve",
    "predict_aoi_trajectory","process_window_spec","summarize_process_windows","validate_process_windows",
    "adjust_pupil_confounds","audit_pupil_fatigue_drift","audit_pupil_frequency_stability",
    "audit_signal_filter","compare_pupil_kernels","compare_raw_adjusted_pupil","compare_signal_filters",
    "filter_eye_signal","filter_pupil_signal","fit_pupil_confound_model","fit_pupil_event_deconvolution",
    "pupil_activity_index","pupil_band_power","pupil_confound_effects","pupil_event_effects",
    "pupil_event_regressor","pupil_frequency_features","pupil_response_kernel","pupil_velocity_activity",
]


def test_all_50_exports_are_public_callables():
    assert len(EXPORTS) == 50
    missing = [n for n in EXPORTS if not callable(getattr(ep, n, None))]
    assert missing == []


def test_key_frozen_signature_names():
    expected = {
        "process_preflight_spec": ["min_gaze_validity","min_pupil_validity","max_gaze_missingness","max_pupil_missingness","min_valid_trial_fraction","trial_gaze_validity_threshold","min_rt_ms","max_rt_ms","sampling_rate_tolerance","blink_quantile","caution_flags","review_flags"],
        "audit_biometric_preflight": ["data","by","spec","valid_gaze_prop","valid_pupil_prop","missing_gaze","missing_pupil","rt_ms","blink_cluster_count","sampling_rate_hz"],
        "process_drift_spec": ["baseline","difficulty_limit","discrimination_limit","gaze_validity_drop","luminance_limit","relative_metric_quantile","min_batches"],
        "process_window_spec": ["width_ms","step_ms","start_ms","end_ms","align","min_samples"],
        "extract_process_windows": ["data","person","trial","time","spec","align_time","pupil","pupil_tonic","pupil_phasic","gaze_x","gaze_y","aoi","valid_gaze","valid_pupil","blink","trackloss"],
        "pupil_activity_index": ["y","time_ms","sampling_rate_hz","method","low_band","high_band","fast_window_ms","slow_window_ms"],
        "fit_pupil_confound_model": ["data","pupil","luminance","trial_order","theta","person","item","engine"],
    }
    for name, args in expected.items():
        assert list(inspect.signature(getattr(ep, name)).parameters) == args


def test_preflight_governance_returns_review_decisions():
    rng=np.random.default_rng(1)
    d=pd.DataFrame({
        "person_id":np.repeat([f"P{i}" for i in range(1,9)],6),
        "valid_gaze_prop":np.r_[np.repeat(.95,36),np.repeat(.55,12)],
        "valid_pupil_prop":rng.uniform(.75,.98,48),
        "missing_gaze":np.r_[np.repeat(0,36),np.repeat(1,12)],
        "missing_pupil":0,"rt_ms":900,"blink_cluster_count":rng.poisson(1,48),"sampling_rate_hz":60,
    })
    a=ep.audit_biometric_preflight(d)
    assert a.eyeprocess_class=="eye_biometric_preflight"
    assert len(ep.preflight_decisions(a))==8
    assert (ep.preflight_decisions(a).preflight_decision!="pass_preflight").any()
    assert len(ep.preflight_exclusion_manifest(a))==8
    kept=ep.apply_preflight_decision(d,a)
    assert len(kept)<len(d)


def test_multivariate_anomaly_is_review_oriented():
    rng=np.random.default_rng(2)
    d=pd.DataFrame({"person_id":[f"P{i}" for i in range(1,41)],
                    "dwell_ms":np.r_[rng.normal(700,50,39),2000],
                    "rt_ms":np.r_[rng.normal(1000,70,39),4000],
                    "valid_gaze_prop":np.r_[rng.uniform(.9,.99,39),.5]})
    a=ep.audit_process_anomalies(d,metrics=["dwell_ms","rt_ms","valid_gaze_prop"],aggregate=False)
    assert a.eyeprocess_class=="eye_process_anomaly_audit"
    assert "review_required" in a.table
    assert "not evidence" in a.caveat.lower()


def test_drift_detects_designed_difficulty_change():
    d=pd.MultiIndex.from_product([["i1","i2","i3"],range(1,5)],names=["item_id","deployment_batch"]).to_frame(index=False)
    # R expand.grid ordering: items vary fastest within batch
    d=d.sort_values(["deployment_batch","item_id"],kind="stable").reset_index(drop=True)
    d["irt_difficulty"]=np.r_[np.repeat(0.,3),np.repeat(.05,3),np.repeat(.10,3),[.8,.1,.1]]
    d["irt_discrimination"]=1.;d["dwell_ms"]=800.;d["valid_gaze_prop"]=.95
    a=ep.audit_process_drift(d,metrics=["irt_difficulty","irt_discrimination","dwell_ms","valid_gaze_prop"])
    assert a.eyeprocess_class=="eye_process_drift_audit"
    assert a.table.difficulty_drift_flag.any()
    assert len(ep.process_drift_alerts(a))>=1


def test_grouped_process_apis_reject_missing_identifiers():
    d=pd.DataFrame({"person_id":["P1",None],"valid_gaze_prop":[.95,.9]})
    with pytest.raises(EyeProcessValidationError,match="Grouping columns must not contain missing values"):
        ep.audit_biometric_preflight(d)


def synthetic_window_data():
    rng=np.random.default_rng(3)
    rows=[]
    for p in ["P1","P2"]:
        for tr in ["T1","T2"]:
            for s in range(60): rows.append((p,tr,s))
    d=pd.DataFrame(rows,columns=["person_id","trial_id","sample"]);d["time_ms"]=d["sample"]*50
    d["pupil_bc"]=np.sin(d.time_ms/600)+rng.normal(0,.05,len(d));d["x"]=rng.normal(500,30,len(d));d["y"]=rng.normal(400,30,len(d));d["aoi"]=np.resize(["target","text","button"],len(d));d["valid_gaze_prop"]=.95;d["valid_pupil_prop"]=.95;d["blink"]=False;d["trackloss"]=False
    return d


def test_process_windows_produce_temporal_features():
    d=synthetic_window_data();x=ep.extract_process_windows(d,spec=ep.process_window_spec(1000,500,0,3000),pupil="pupil_bc")
    assert x.eyeprocess_class=="eye_process_windows" and len(x.data)>0
    assert {"pupil_mean","aoi_entropy","gaze_path_length"}.issubset(x.data.columns)
    assert bool(ep.validate_process_windows(x).valid.iloc[0])
    assert len(ep.summarize_process_windows(x))>0


def test_pupil_activity_features_finite_on_oscillation():
    t=np.arange(0,5000+1e-9,1000/60);y=np.sin(2*np.pi*.3*t/1000)+.2*np.sin(2*np.pi*1.2*t/1000)
    lo=ep.pupil_band_power(y,60,.05,.5);hi=ep.pupil_band_power(y,60,.5,4)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert np.isfinite(ep.pupil_velocity_activity(y,t))
    assert np.isfinite(ep.pupil_activity_index(y,t,60,method="frequency_contrast"))


def test_pupil_event_deconvolution_positive_effect():
    rng=np.random.default_rng(4);t=np.arange(0,3000.1,20);k=ep.pupil_response_kernel(t-500)
    d=pd.DataFrame({"person_id":"P1","trial_id":"T1","time_ms":t,"pupil_bc":.8*k+rng.normal(0,.02,len(t)),"event_ms":500})
    fit=ep.fit_pupil_event_deconvolution(d,events={"stimulus":"event_ms"})
    assert fit.eyeprocess_class=="eye_pupil_deconvolution" and len(fit.effects)==1
    assert fit.effects.beta__stimulus.iloc[0]>0


def test_base_signal_filter_auditable_and_backends_explicit():
    rng=np.random.default_rng(5);y=np.sin(np.linspace(0,6,101))+rng.normal(0,.1,101);y[49]+=3
    f=ep.filter_eye_signal(y,width=9,method="runmed")
    assert f.eyeprocess_class=="eye_signal_filter_audit"
    assert np.isfinite(ep.audit_signal_filter(f).filtered_sd.iloc[0])
    with pytest.raises(EyeProcessBackendError): ep.filter_eye_signal(y,method="robfilter")


def test_pupil_confound_lm_reference_adjusts_values():
    rng=np.random.default_rng(51);n=60
    d=pd.DataFrame({"person_id":np.repeat([f"P{i}" for i in range(1,11)],6),"item_id":np.tile([f"I{i}" for i in range(1,7)],10),"pupil_peak":rng.normal(size=n),"screen_luminance":np.resize([80,160],n),"trial_sequence":np.resize([1,2,3],n)})
    fit=ep.fit_pupil_confound_model(d,engine="lm")
    assert fit.eyeprocess_class=="eye_pupil_confound_model" and fit.engine=="lm"
    assert np.isfinite(fit.data.pupil_confound_adjusted).all()
    with pytest.raises(EyeProcessBackendError): ep.fit_pupil_confound_model(d,engine="mgcv")


def test_empty_grouped_feature_inputs_fail_explicitly():
    d=pd.DataFrame({"person_id":pd.Series(dtype=str),"trial_id":pd.Series(dtype=str),"time_ms":pd.Series(dtype=float),"pupil_bc":pd.Series(dtype=float),"aoi":pd.Series(dtype=str)})
    with pytest.raises(EyeProcessValidationError,match="at least one row"):ep.pupil_frequency_features(d)
    with pytest.raises(EyeProcessValidationError,match="at least one row"):ep.aoi_trajectory_features(d)


def test_broader_contract_smoke():
    d=synthetic_window_data()
    x=ep.extract_process_windows(d,spec=ep.process_window_spec(1000,500,0,3000))
    b=ep.bind_process_windows(x,x); assert len(b.data)==2*len(x.data)
    sens=ep.audit_process_window_sensitivity(d,widths_ms=[500,1000],steps_ms=[250],spec=ep.process_window_spec(1000,500,0,3000));assert len(sens.table)==2
    traj=ep.aoi_trajectory_features(d,degree=2);assert len(traj.features)==4
    assert len(ep.compare_aoi_trajectories(traj))==1
    freq=ep.pupil_frequency_features(d);assert len(freq.features)==4
    stab=ep.audit_pupil_frequency_stability(d,windows_ms=[500,1000]);assert stab.eyeprocess_class=="eye_pupil_frequency_stability"

def test_governance_08_plot_counterparts_have_data_layers():
    import matplotlib.pyplot as plt
    d=synthetic_window_data()
    w=ep.extract_process_windows(d,spec=ep.process_window_spec(1000,500,0,3000))
    ax=ep.plot_eye_process_windows(w); assert hasattr(ax,'eyeprocess_plot_data'); plt.close(ax.figure)
    sens=ep.audit_process_window_sensitivity(d,widths_ms=[500,1000],steps_ms=[250],spec=ep.process_window_spec(1000,500,0,3000))
    ax=ep.plot_process_window_sensitivity(sens); assert len(ax.eyeprocess_plot_data)==2; plt.close(ax.figure)
    freq=ep.pupil_frequency_features(d)
    for fun in [ep.plot_eye_pupil_frequency_features,ep.plot_pupil_band_power,ep.plot_pupil_activity_windows]:
        ax=fun(freq); assert hasattr(ax,'eyeprocess_plot_data'); plt.close(ax.figure)
    t=np.arange(0,2000,1000/60); y=np.sin(2*np.pi*.3*t/1000)
    ax=ep.plot_pupil_spectrum(y,60); assert len(ax.eyeprocess_plot_data)>0; plt.close(ax.figure)


def test_standalone_plot_exports_from_r064_present():
    for n in ['plot_pupil_spectrum','plot_pupil_band_power','plot_pupil_activity_windows','plot_pupil_activity_sensitivity','plot_process_window_sensitivity']:
        assert callable(getattr(ep,n))
