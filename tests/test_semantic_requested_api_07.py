import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.irt import EyeResult

SEMANTIC_EXPORTS = [
    "validation_evidence_levels", "semantic_fidelity_spec", "field_fidelity_report",
    "timestamp_fidelity_audit", "coordinate_fidelity_audit", "pupil_unit_fidelity_audit",
    "eye_stream_fidelity_audit", "event_semantics_audit", "validate_hed_event_semantics",
    "validate_bids_eye_semantics", "semantic_roundtrip_audit", "semantic_loss_map",
    "public_validation_corpus", "compatibility_evidence_matrix", "validate_vendor_timestamp_semantics",
]
REQUESTED_EXPORTS = [
    "plot_distractor_information", "fit_gaze_informed_missingness_irt", "device_facet_effects",
    "session_facet_effects", "algorithm_facet_effects", "detect_process_changepoint",
    "plot_process_changepoint", "plot_person_item_space", "explain_latent_interaction",
    "plot_irf_uncertainty", "audit_latent_distribution", "compare_latent_distribution_models",
    "latent_distribution_stress_test", "fit_event_time_irt", "simulate_from_model",
    "extract_parameter_truth", "fit_validation_replicate", "vendor_schema_contract",
    "validate_vendor_semantics", "event_roundtrip_audit", "roundtrip_eye_bids",
    "cross_version_adapter_regression",
]


def test_new_export_surface_is_complete():
    missing = [n for n in SEMANTIC_EXPORTS + REQUESTED_EXPORTS if not callable(getattr(ep, n, None))]
    assert missing == []


def test_semantic_fidelity_matches_frozen_r_contracts():
    a = pd.DataFrame({
        "id": range(1, 6), "timestamp": range(1, 6), "x": [10, 20, 30, 40, 50],
        "y": [5, 6, 7, 8, 9], "pupil_size": [3, 3.1, 3.2, 3.3, 3.4],
    })
    rep = ep.field_fidelity_report(a, a.copy(), key="id")
    keep = rep.fields.field.isin(["timestamp", "x", "y", "pupil_size"])
    assert int(keep.sum()) == 4
    assert set(rep.fields.loc[keep, "status"]) == {"LOSSLESS"}

    c = a.copy(); c["x"] = c["x"] * 2 + 1
    rep2 = ep.field_fidelity_report(a, c, fields=["x"], key="id")
    assert rep2.fields.loc[0, "status"] == "UNIT_TRANSFORMED"
    assert rep2.fields.loc[0, "transform_intercept"] == pytest.approx(1, abs=1e-8)
    assert rep2.fields.loc[0, "transform_slope"] == pytest.approx(2, abs=1e-8)


def test_bids_semantics_and_public_validation_registry_match_r_contracts():
    d = pd.DataFrame({"timestamp": [1,2,3], "x_coordinate": [1,2,3], "y_coordinate": [4,5,6]})
    meta = {"Columns": list(d.columns), "SamplingFrequency": 60, "PhysioType": "eyetrack",
            "RecordedEye": "cyclopean", "SampleCoordinateSystem": "eye-in-head"}
    z = ep.validate_bids_eye_semantics(d, meta)
    assert bool(z.checks["pass"].all())
    corpus = ep.public_validation_corpus()
    assert {"Gazepoint", "EyeLink", "Tobii", "Pupil Labs"}.issubset(set(corpus.ecosystem))
    assert corpus.corpus.str.contains("GazeBase").any()
    assert corpus.corpus.str.contains("MCFW").any()


def test_explicit_completion_semantic_contracts_and_roundtrips():
    contract = ep.vendor_schema_contract(
        "Gazepoint", required_fields=["timestamp", "x", "y"], optional_fields=["pupil"],
        timestamp={"device_time": "timestamp"},
    )
    d = pd.DataFrame({"timestamp": range(1,6), "x": range(1,6), "y": range(6,11)})
    z = ep.validate_vendor_semantics(d, contract)
    assert z["pass"] is True

    ev = pd.DataFrame({"event_id":[1,2,3], "event":["start","stimulus","response"],
                       "timestamp":[0,1,2], "HED":["(Experiment-control)","(Sensory-event)","(Agent-action)"]})
    er = ep.event_roundtrip_audit(ev, ev, key="event_id", hed_column="HED")
    assert er.status == "LOSSLESS"

    rt = ep.roundtrip_eye_bids(d, exporter=lambda x: x, importer=lambda x: x,
                               audit_args={"key": None, "fields": ["timestamp","x","y"]})
    assert rt.status == "LOSSLESS"
    ar = ep.cross_version_adapter_regression(d, baseline_adapter=lambda x: x, candidate_adapter=lambda x: x,
                                              audit_args={"fields":["timestamp","x","y"]})
    assert ar.status == "LOSSLESS"


def test_validation_replicate_and_latent_distribution_contracts():
    def generator(replicate, scenario):
        return {"data": np.array([.8, 1.0, 1.2]), "truth": {"mu": 1.0}}
    def fitter(x): return float(np.mean(x))
    def extractor(fit, simulation):
        return pd.DataFrame({"parameter":["mu"], "estimate":[fit], "lower":[fit-.2], "upper":[fit+.2]})
    z = ep.fit_validation_replicate(1, generator, fitter, extractor)
    assert z.attrs.get("eyeprocess_class") == "eye_irt_recovery_results"
    assert z.loc[0, "truth"] == pytest.approx(1)
    assert z.loc[0, "estimate"] == pytest.approx(1)
    assert bool(z.loc[0, "converged"])

    rng = np.random.default_rng(42); x = rng.normal(size=200)
    aud = ep.audit_latent_distribution(x)
    assert np.isfinite(aud.loc[0, "normal_qq_correlation"])
    cmp = ep.compare_latent_distribution_models(x)
    assert cmp.eyeprocess_class == "eye_latent_distribution_comparison"
    assert {"normal","student_t","two_normal_mixture"}.issubset(set(cmp.comparison.model))


def test_gaze_informed_missingness_proxy_is_explicit():
    rng = np.random.default_rng(1)
    d = pd.DataFrame([(f"P{i}", f"I{j}") for i in range(1,13) for j in range(1,6)], columns=["participant_id","item_id"])
    d["gaze_exposure"] = rng.exponential(size=len(d)); d["response"] = rng.binomial(1,.65,size=len(d)).astype(float)
    d.loc[np.arange(2,len(d),11), "response"] = np.nan
    z = ep.fit_gaze_informed_missingness_irt(d)
    assert z.eyeprocess_class == "eye_gaze_informed_missingness_irt"
    assert z.theta_source == "smoothed-person-score-proxy"
    assert z.status == "reference-diagnostic"


def test_facet_aliases_event_time_and_simulation_helpers():
    d = pd.DataFrame({
        "participant_id": np.repeat(["P1","P2","P3","P4"], 4),
        "item_id": list("ABCD")*4,
        "device": ["D1","D1","D2","D2"]*4,
        "session": np.repeat(["S1","S2"],8),
        "algorithm": ["A1","A2"]*8,
        "response": [0,1,1,1,1,0,1,1,0,0,1,1,1,1,0,1],
        "process": np.linspace(.2,1.7,16),
    })
    mf = ep.fit_manyfacet_process_irt(d, process="process", device="device", session="session", algorithm="algorithm")
    assert ep.device_facet_effects(mf).facet == "device"
    assert ep.session_facet_effects(mf).facet == "session"
    assert ep.algorithm_facet_effects(mf).facet == "algorithm"

    et = d[["participant_id","item_id"]].copy(); et["event_time"] = np.linspace(.5,4,16); et["event"] = ([1,0,1,1]*4); et["theta"] = np.repeat([-1,-.3,.3,1],4)
    fit = ep.fit_event_time_irt(et)
    assert fit.eyeprocess_class == "eye_event_time_irt"
    assert fit.engine == "cox_reference"
    assert fit.theta_conditioned is True

    sim = ep.simulate_from_model(lambda n=3: {"data": np.arange(n), "truth":{"mu":2}}, n=4)
    assert len(sim["data"]) == 4
    truth = ep.extract_parameter_truth(sim)
    assert truth.to_dict("records") == [{"parameter":"mu","truth":2.0}]


def test_completion_plots_return_axes_with_data():
    z = pd.DataFrame({"option":["A","B"],"gaze_contrast":[-.2,.3],"choice_contrast":[.1,-.1]})
    ax = ep.plot_distractor_information(z); assert len(ax.gp3_data)==2; plt.close(ax.figure)

    cpdata=[]
    for p in ["P1","P2"]:
        for k in range(1,13): cpdata.append({"participant_id":p,"item_order":k,"response":int(k>6),"rt":1+k/10})
    cp=ep.detect_process_changepoint(pd.DataFrame(cpdata),min_segment=3)
    ax=ep.plot_process_changepoint(cp); assert hasattr(ax,"gp3_data"); plt.close(ax.figure)

    obj=EyeResult({"person_coordinates":np.array([[0.,0.],[1.,0.]]),"item_coordinates":np.array([[0.,1.],[1.,1.]]),"person_ids":["P1","P2"],"item_ids":["I1","I2"]},eyeprocess_class="eye_latent_space_irt")
    ax=ep.plot_person_item_space(obj); assert len(ax.gp3_data["person"])==2; plt.close(ax.figure)
    ex=ep.explain_latent_interaction(obj,top=2); assert len(ex)==2 and ex.distance.is_monotonic_increasing

    X=np.array([[0,0,0],[0,0,1],[0,1,1],[1,1,1],[1,1,1],[0,1,0],[1,0,1],[1,1,0]],float)
    gp=ep.fit_gpirt(X,spline_df=4); ax=ep.plot_irf_uncertainty(gp,item=1); assert len(ax.gp3_data)==101; plt.close(ax.figure)


def test_semantic_audit_components_and_plots():
    a=pd.DataFrame({"id":[1,2,3,4],"timestamp":[0.,1.,2.,3.],"x":[.1,.2,.3,.4],"y":[.2,.3,.4,.5],"pupil_size":[3.,3.1,3.2,3.3],"eye":["left","right","left","right"]})
    rt=ep.semantic_roundtrip_audit(a,a,key="id",pupil={"source_pupil":"pupil_size"},eye={"source_eye":"eye"})
    assert rt.overall == "LOSSLESS"
    loss=ep.semantic_loss_map(rt); assert set(loss.scope)=={"component","field"}
    ax=ep.plot_eye_semantic_roundtrip(rt); assert len(ax.gp3_data)==len(loss); plt.close(ax.figure)
    comp=pd.DataFrame({"ecosystem":["Gazepoint"],"device":["GP3"]})
    ev=pd.DataFrame({"ecosystem":["Gazepoint"],"device":["GP3"],"evidence_level":["synthetic-fixture"],"semantic_roundtrip_pass":[True]})
    cem=ep.compatibility_evidence_matrix(comp,ev); ax=ep.plot_eye_compatibility_evidence_matrix(cem); assert len(ax.gp3_data)==1; plt.close(ax.figure)
