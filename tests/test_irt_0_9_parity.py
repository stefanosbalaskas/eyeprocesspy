from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.exceptions import EyeProcessValidationError


def items(n=6):
    return pd.DataFrame({
        "item_id": [f"I{i+1}" for i in range(n)],
        "a": np.ones(n),
        "b": np.linspace(-1.5, 1.5, n),
        "c": np.zeros(n),
        "d": np.ones(n),
    })


def test_foundation_probability_information_contracts():
    th = np.linspace(-3, 3, 41)
    p2 = ep.eyeprocess_irt_2pl_probability(th, a=1.3, b=.2)
    assert np.all((p2 > 0) & (p2 < 1))
    assert np.all(np.diff(p2) > 0)
    grm = ep.eyeprocess_irt_grm_probability(th, a=1.2, thresholds=[-1, 0, 1])
    gpcm = ep.eyeprocess_irt_gpcm_probability(th, a=1.1, steps=[-.5, .5])
    nom = ep.eyeprocess_irt_nominal_probability(th, slopes=[-1, 0, 1], intercepts=[0, .3, -.2])
    np.testing.assert_allclose(grm.sum(axis=1), 1, atol=1e-10)
    np.testing.assert_allclose(gpcm.sum(axis=1), 1, atol=1e-10)
    np.testing.assert_allclose(nom.sum(axis=1), 1, atol=1e-10)
    it = pd.DataFrame({"item_id": [f"I{i}" for i in range(1,5)], "a": [.8,1,1.2,1.4], "b": [-1,-.2,.4,1], "c": 0, "d": 1})
    info = ep.eyeprocess_irt_test_information(th, it)
    assert info.attrs["eyeprocess_class"] == "eye_irt_information_profile"
    assert np.all(info.information >= 0)
    assert np.all(info.conditional_sem > 0)


def test_identification_sparse_and_information_validation():
    sp = ep.eyeprocess_irt_model_spec("2pl")
    a = ep.eyeprocess_irt_identification_audit(sp, constraints={"theta_mean_fixed": True, "theta_sd_fixed": True})
    assert a.valid
    d = pd.DataFrame({"person": [1,1,2,2,3], "item": ["A","B","A","C","B"], "response": [1,0,1,np.nan,1]})
    audit = ep.eyeprocess_irt_sparse_design_audit(d, person="person", item="item", response="response", min_person_items=1, min_item_persons=1)
    assert audit.eyeprocess_class == "eye_irt_sparse_design_audit"
    assert audit.n_observed < len(d)
    with pytest.raises(EyeProcessValidationError): ep.eyeprocess_irt_item_information(0, "3pl", c=1)
    with pytest.raises(EyeProcessValidationError): ep.eyeprocess_irt_item_information(0, "4pl", c=.8, d=.7)
    with pytest.raises(EyeProcessValidationError): ep.eyeprocess_irt_measurement_precision_profile([-4,-3,3,4], items(3), target=[-1,1])


def test_fit_diagnostics_and_local_dependence():
    rng = np.random.default_rng(1)
    p = rng.uniform(.15, .85, size=(20,6))
    y = rng.binomial(1, p)
    cols = [f"I{i}" for i in range(1,7)]
    y = pd.DataFrame(y, columns=cols); p = pd.DataFrame(p, columns=cols)
    assert len(ep.eyeprocess_irt_item_fit_residuals(y,p)) == 6
    assert len(ep.eyeprocess_irt_person_fit_residuals(y,p)) == 20
    q3 = ep.eyeprocess_irt_q3(y,p); assert q3.shape == (6,6)
    io = ep.eyeprocess_irt_infit_outfit(y,p,by="item"); assert np.all(np.isfinite(io.infit))
    lz = ep.eyeprocess_irt_person_fit_lz(y,p); assert np.all(np.isfinite(lz.lz))
    q = pd.DataFrame([[np.nan,.4,.1],[.4,np.nan,-.3],[.1,-.3,np.nan]], index=list("ABC"), columns=list("ABC"))
    z = ep.eyeprocess_irt_local_dependence_pairs(q, threshold=.25)
    assert len(z) == 2 and np.all(z.abs_q3 >= .25)


def test_native_scoring_and_adaptive_contracts():
    it = items(6); response = [1,1,1,0,0,0]
    eap = ep.eyeprocess_irt_eap_score(response,it); map_ = ep.eyeprocess_irt_map_score(response,it); mle = ep.eyeprocess_irt_mle_score(response,it)
    assert eap.eyeprocess_class == "eye_irt_score"
    assert np.all(np.isfinite([eap.estimate,map_.estimate,mle.estimate]))
    assert len(ep.eyeprocess_irt_plausible_values(eap,n=20,seed=3)) == 20
    with pytest.raises(EyeProcessValidationError): ep.eyeprocess_irt_eap_score(response,it,prior_mean=[0,1])
    with pytest.raises(EyeProcessValidationError): ep.eyeprocess_irt_plausible_values(eap,n=[2,3],seed=3)
    bank = ep.eyeprocess_irt_item_bank(pd.DataFrame({"item_id":[f"I{i}" for i in range(1,6)],"a":[.8,1,1.5,1.1,.9],"b":[-1,-.5,0,.5,1],"c":0,"d":1}), content=["A","A","B","B","B"])
    sel = ep.eyeprocess_irt_item_selection(bank, theta=0, administered="I3")
    assert sel.selected != "I3"
    assert ep.eyeprocess_irt_stopping_rule(10,se=.2,min_items=5)["stop"]
    trace = ep.eyeprocess_irt_adaptive_trace(["I1","I2"],[0,.1],[.1,.2],[.5,.4],[1,1.5],[1,0])
    assert trace.attrs["eyeprocess_class"] == "eye_irt_adaptive_trace"
    trace2 = ep.eyeprocess_irt_adaptive_trace(["I1","I2"],[0,.1],[.1,.2],[.5,.4],[1,1.5])
    assert len(trace2.response) == 2
    bal = ep.eyeprocess_irt_content_balance_audit(["I1","I3"],bank,target={"A":.4,"B":.6})
    assert np.isclose(bal.target.sum(), 1); assert sorted(bal.content.tolist()) == ["A","B"]
    assert len(ep.eyeprocess_irt_process_aware_selection_penalty([1,2],burden=.1)) == 2
    with pytest.raises(EyeProcessValidationError): ep.eyeprocess_irt_stopping_rule([1,2],se=.2)


def test_linking_dif_and_curve_weight_validation():
    ref = pd.DataFrame({"item_id":[f"I{i}" for i in range(1,9)],"a":np.linspace(.8,1.5,8),"b":np.linspace(-1.5,1.5,8),"c":0,"d":1})
    A=1.2; B=-.3; focal=ref.copy(); focal["a"]=ref.a*A; focal["b"]=(ref.b-B)/A
    link=ep.eyeprocess_irt_mean_sigma_link(ref,focal); linked=ep.eyeprocess_irt_apply_link(focal,link)
    assert np.isclose(link.A,A,atol=1e-8); assert np.isclose(link.B,B,atol=1e-8)
    np.testing.assert_allclose(linked.a,ref.a,atol=1e-8); np.testing.assert_allclose(linked.b,ref.b,atol=1e-8)
    r=pd.DataFrame({"item_id":["I1"],"a":[1.],"b":[0.],"c":[0.],"d":[1.]}); f=pd.DataFrame({"item_id":["I1"],"a":[1.1],"b":[.2],"c":[0.],"d":[1.]})
    s=ep.eyeprocess_irt_functioning_effect_summary(ep.eyeprocess_irt_dif_effect_curve(r,f)); assert np.isfinite(s["max_abs"])
    drift=ep.eyeprocess_irt_session_drift(pd.DataFrame({"item_id":["I1","I1","I2"],"session":[1,2,1],"b":[.1,.3,np.nan]})); assert np.isnan(drift.loc[drift.item_id=="I2","change"].iloc[0])
    ref4=items(4)
    with pytest.raises(EyeProcessValidationError): ep.eyeprocess_irt_stocking_lord_link(ref4,ref4,theta=[-1,0,1],weights=[1,1])
    with pytest.raises(EyeProcessValidationError): ep.eyeprocess_irt_haebara_link(ref4,ref4,theta=[-1,0,1],weights=[0,0,0])
    with pytest.raises(EyeProcessValidationError): ep.eyeprocess_irt_stocking_lord_link(ref4,ref4,start=[0,0])
    z=pd.DataFrame({"theta":[-1,0,1],"absolute_difference":[np.nan]*3,"signed_difference":[np.nan]*3})
    out=ep.eyeprocess_irt_functioning_effect_summary(z); assert np.isnan(out["max_abs"]) and np.isnan(out["mean_abs"]) and np.isnan(out["signed_area"])


def test_joint_process_and_engine_gating_contracts():
    s=ep.eyeprocess_joint_process_irt_spec(response_family="2pl",time_model="lognormal",process_channels=["pupil","gaze"])
    assert s.eyeprocess_class == "eye_joint_process_irt_spec"; assert ep.validate_eyeprocess_joint_process_irt_spec(s)
    m=ep.eyeprocess_multichannel_measurement_map(response="accuracy",channels=["rt","pupil","gaze"])
    assert {"channel","role","inference_boundary"}.issubset(m.columns)
    assert m.iloc[0].role == "item_response"
    reg=ep.eyeprocess_irt_engine_registry(); assert {"mirt","TAM","GDINA","LNIRT","eRm","equateIRT","catR","mirtCAT"}.issubset(set(reg.engine))
    with pytest.raises(EyeProcessValidationError,match="only accepts"): ep.fit_eyeprocess_mirt(np.array([[0,1],[1,0]]),engine="TAM")
    with pytest.raises(EyeProcessValidationError,match="only accepts"): ep.run_eyeprocess_equateirt("not_an_export",engine="wrong")
    g=ep.fit_eyeprocess_gdina(np.array([[0,1],[1,0]]),np.ones((2,1)))
    assert g.eyeprocess_class == "eye_gated_irt_engine" and g.fit is None
    assert ep.validate_eyeprocess_external_irt_fit(g,engine="GDINA")
    miss=ep.eyeprocess_process_missingness_pattern(pd.DataFrame({"accuracy":[1,np.nan,0],"rt":[1.,2.,np.nan],"pupil":[np.nan,3.,4.]}),"accuracy",["rt","pupil"])
    assert list(miss.columns)==["pattern","n","fraction"] and miss.n.sum()==3


def test_simulation_sbc_and_recovery_design_contracts():
    it=items(5)
    s1=ep.simulate_eyeprocess_irt_binary(50,it,seed=99); s2=ep.simulate_eyeprocess_irt_binary(50,it,seed=99)
    pd.testing.assert_frame_equal(s1.responses,s2.responses)
    rng=np.random.default_rng(4); truth=rng.normal(size=30); draws=rng.normal(size=(30,19)); ranks=ep.eyeprocess_irt_sbc_ranks(truth,draws,seed=8)
    assert np.all((ranks>=0)&(ranks<=19))
    ev=ep.eyeprocess_irt_sbc_summary(ranks,n_draws=19,bins=10); assert ev.eyeprocess_class=="eye_irt_sbc_evidence"
    ab=ep.run_eyeprocess_irt_ability_sbc(it,replications=20,posterior_draws=19,theta_grid=np.linspace(-5,5,201),seed=19)
    assert ab.eyeprocess_class=="eye_irt_sbc_evidence" and np.isfinite(ab.coverage) and len(ab.details)==20
    d=ep.eyeprocess_irt_recovery_design(sample_size=[50],n_items=[5],missing_rate=[0],testlet_sd=[0],replications=1,seed=1)
    assert d.attrs["eyeprocess_class"]=="eye_irt_recovery_design" and len(d)==1 and d.iloc[0].scenario_id=="IRTREC001"
    g=ep.run_eyeprocess_irt_recovery(d,verbose=False); assert g.eyeprocess_class=="eye_gated_irt_engine"
    suite=ep.eyeprocess_irt_misspecification_suite(); assert len(suite)==6 and list(suite.columns)==["scenario","perturbation","target"]
    fr=ep.freeze_eyeprocess_irt_reference(metadata={"x":1}); assert fr.eyeprocess_class=="eye_irt_validation_reference"


def test_multidimensional_cdm_contracts():
    L=np.array([[1,0],[1,0],[0,1],[0,1],[1,0],[0,1]],dtype=float)
    sp=ep.eyeprocess_mirt_loading_spec([f"I{i}" for i in range(1,7)],L,["D1","D2"],simple_structure=True)
    au=ep.eyeprocess_mirt_loading_audit(sp,min_items_per_dimension=2); assert au.meets_minimum.all()
    im=ep.eyeprocess_mirt_information_matrix([0,0],[1,.5]); assert im.shape==(2,2) and np.all(np.linalg.eigvalsh(im)>=-1e-10)
    Q=np.array([[1,0],[0,1],[1,1],[1,0]],dtype=int); q=ep.eyeprocess_cdm_qmatrix_audit(Q); assert q.complete_identity_block
    pr=ep.eyeprocess_cdm_attribute_profiles(2)[["A1","A2"]]; eta=ep.eyeprocess_cdm_dina_ideal_response(Q,pr); pp=ep.eyeprocess_cdm_dina_probability(eta); assert np.all((pp>=0)&(pp<=1))
    with pytest.raises(EyeProcessValidationError): ep.eyeprocess_cdm_classification_uncertainty(np.ones((3,1)))


def test_governance_advanced_missingness_and_targeting_contracts():
    it=pd.DataFrame({"item_id":[f"I{i}" for i in range(1,9)],"a":1.,"b":np.linspace(-2,2,8),"c":0.,"d":1.})
    cov=ep.eyeprocess_irt_bank_coverage(it,target_information=1); assert cov.eyeprocess_class=="eye_irt_bank_coverage"
    tg=ep.eyeprocess_irt_targeting_gap(np.linspace(-2,2,100),it); assert tg.eyeprocess_class=="eye_irt_targeting_gap"
    cp=ep.eyeprocess_irt_classification_precision([-.2,.2],[.2,.2],cut_score=0); assert len(cp)==2
    y=np.array([[1,0],[np.nan,1.]])  # equivalent to R's column-major matrix(c(1,NA,0,1),2)
    d=np.array([[1,1],[0,1]])
    ma=ep.eyeprocess_irt_missing_by_design_audit(y,d); assert ma.eyeprocess_class=="eye_irt_missing_design_audit"
    g=ep.eyeprocess_irt_prior_sensitivity_grid(); assert len(g)>1
    spec=ep.eyeprocess_irt_model_spec("2pl"); ident=ep.eyeprocess_irt_identification_audit(spec,constraints={"theta_mean_fixed":True,"theta_sd_fixed":True})
    card=ep.eyeprocess_irt_model_card(spec,identification=ident,intended_use="software validation"); assert card.eyeprocess_class=="eye_irt_model_card"
    audit=ep.eyeprocess_irt_model_card_audit(card); assert audit.loc[audit.field=="identification","present"].iloc[0]
    y2=np.array([[1,0],[np.nan,np.nan]],dtype=float); p2=np.full((2,2),.5)
    io=ep.eyeprocess_irt_infit_outfit(y2,p2,by="person"); assert io.n.tolist()==[2,0] and np.isnan(io.infit.iloc[1])
    lz=ep.eyeprocess_irt_person_fit_lz(y2,p2); assert lz.n_observed.tolist()==[2,0] and np.isnan(lz.lz.iloc[1])
    with pytest.raises(EyeProcessValidationError): ep.eyeprocess_irt_targeting_gap([-5,0,5],items(4),breaks=np.arange(-4,5))
