from pathlib import Path
import math

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.exceptions import EyeProcessBackendError


def _names(file):
    return [x.strip() for x in (Path(__file__).parent/'fixtures'/file).read_text().splitlines() if x.strip()]


def test_advanced_and_validation_frozen_exports_present():
    a=_names('advanced_process_irt_07_exports.txt'); v=_names('irt_validation_07_exports.txt')
    assert len(a)==29 and len(v)==34
    assert [n for n in a+v if not callable(getattr(ep,n,None))]==[]


def test_ngram_sequence_embedding_and_equating_match_r_contracts():
    seqs=[['stem','A','stem','B'],['stem','B','B'],['A','B','A']]
    x=ep.process_ngram_features(seqs,n=(1,2)); y=ep.process_ngram_features(seqs,n=(1,2))
    np.testing.assert_array_equal(x,y)
    emb=ep.process_sequence_embedding(seqs,n=(1,2),dimensions=2)
    assert emb.shape[0]==3 and emb.shape[1]<=2
    ref=pd.DataFrame({'a':[1,1.2,.9],'b':[-1,0,1]}); new=pd.DataFrame({'a':[.9,1.1,.8],'b':[-.8,.2,1.2]})
    z=ep.equate_irt_scales(ref,new,method='mean-sigma')
    assert np.isfinite(z.A) and np.isfinite(z.B) and len(z.transformed)==3


def test_exact_experimental_engines_fail_loudly():
    m=np.array([[0,1,1],[0,1,1]])
    with pytest.raises(EyeProcessBackendError,match='external_engine|validated'):
        ep.fit_flow_mirt(m)
    with pytest.raises(EyeProcessBackendError,match='external_engine|validated'):
        ep.fit_dynamic_gpirt(pd.DataFrame({'x':[1]}))
    with pytest.raises(EyeProcessBackendError,match='external_engine|validated'):
        ep.fit_continuous_time_irt(pd.DataFrame({'x':[1]}))
    with pytest.raises(EyeProcessBackendError): ep.fit_latent_space_irt(m)


def test_process_hmm_contract_and_transition_summary():
    rng=np.random.default_rng(3); rows=[]
    for s in range(8):
        pid=f'p{s%4}'; item=f'i{s%3}'; state=s%2
        for t in range(8):
            rows.append({'trial_id':f't{s}','timestamp':t,'x':rng.normal(2*state,.25),'y':rng.normal(-state,.25),
                         'response':int(rng.random()<(.7 if state==0 else .4)),'participant_id':pid,'item_id':item})
    d=pd.DataFrame(rows)
    fit=ep.fit_process_hmm_irt(d,n_states=2,max_iter=15,seed=2)
    assert fit.eyeprocess_class=='eye_process_hmm_irt'
    assert fit.posterior_state.shape==(len(d),2)
    assert np.allclose(fit.transition.sum(axis=1),1)
    occ=ep.process_state_occupancy(fit); tr=ep.process_state_transition_summary(fit)
    assert len(occ)==8 and len(tr)==4 and np.all((tr.probability>=0)&(tr.probability<=1))


def test_latent_class_crossclassified_embedding_and_gpirt_reference_contracts():
    rng=np.random.default_rng(4); n=80
    d=pd.DataFrame({'participant_id':[f'p{i%20}' for i in range(n)],'item_id':[f'i{i%5}' for i in range(n)]})
    d['f1']=rng.normal(size=n); d['f2']=rng.normal(size=n); d['response']=rng.binomial(1,1/(1+np.exp(-(.3*d.f1-.2*d.f2))))
    lc=ep.fit_latent_class_process_irt(d,process_features=['f1','f2'],n_classes=2,seed=1)
    assert len(lc.class_)==n and lc.centers.shape==(2,2)
    cc=ep.fit_crossclassified_process_irt(d.assign(outcome=d.f1),'outcome',fixed=['f2'])
    assert cc.status=='python-reference-estimator'
    seqs=[['a','b'] if i%2 else ['a','c'] for i in range(n)]
    em=ep.fit_response_process_embedding_irt(d,seqs,dimensions=2,n=(1,2))
    assert em.embedding.shape==(n,2)
    R=np.column_stack([rng.binomial(1,1/(1+np.exp(-(d.f1+b)))) for b in [-1,-.5,0,.5]])
    gp=ep.fit_gpirt(R,spline_df=5); cmp=ep.compare_parametric_nonparametric_irf(R,gp); aud=ep.audit_irf_shape(cmp)
    assert len(cmp)==4*101 and len(aud)==4 and {'mean_absolute_difference','max_absolute_difference','flag'}<=set(aud)


def test_process_information_cat_and_trajectory():
    info=ep.process_item_information([-.5,0,.5],[1,1.2],[0,.5],process_information=[.2,.3],weights={'response':1,'process':.5,'rt':0})
    assert info.utility.shape==(3,2); assert ep.expected_process_information(info).shape==(2,)
    bank=pd.DataFrame({'item_id':['a','b','c'],'a':[1,1.2,.9],'b':[-.5,0,.5],'process_information':[.1,.2,.3]})
    sel=ep.select_next_item_process(0,bank); assert sel.item_id in set(bank.item_id)
    sim=ep.simulate_process_cat(bank,true_theta=.4,n_items=3,seed=1); assert len(sim)==3 and sim.item_id.nunique()==3
    traj=ep.latent_trait_trajectory([0,1,2,3,4],[0,.2,.5,.7,1]); pr=ep.predict_theta_at_time(traj,[1.5,2.5]); assert len(pr)==2 and np.isfinite(pr.y).all()


def test_recovery_summary_convergence_and_mcse_r_contract():
    d=pd.DataFrame({'replicate':np.repeat(np.arange(1,5),2),'parameter':['a','b']*4,'truth':[1,0]*4,
                    'estimate':[1.1,.1,.9,-.1,1.05,.05,.95,-.05],
                    'lower':[.5,-.5]*4,'upper':[1.5,.5]*4,'converged':True})
    s=ep.summarize_parameter_recovery(d)
    assert {'bias','rmse','coverage','failure_rate'}<=set(s)
    assert np.isfinite(s.rmse).all() and np.allclose(s.coverage,1)
    assert ep.audit_convergence(d)['pass'].all()
    for metric in ['bias','rmse','coverage']:
        z=ep.validation_mcse(d,metric); assert len(z)==1 and np.isfinite(z.mcse.iloc[0])


def test_sbc_ppc_and_negative_control_r_contracts():
    rng=np.random.default_rng(1)
    def sim(r): return {'data':rng.normal(size=5),'truth':{'mu':0.0}}
    def fit(dat): return {'draws':pd.DataFrame({'mu':rng.normal(np.mean(dat),1,200)})}
    z=ep.run_sbc(sim,fit,lambda f:f['draws'],replications=8,seed=1)
    assert z.eyeprocess_class=='eye_irt_sbc' and ((z.ranks.normalized_rank>0)&(z.ranks.normalized_rank<1)).all()
    a=ep.audit_sbc(z,bins=4); assert 'pass_screen' in a
    obs=np.arange(1,11); reps=[obs+rng.normal(0,.2,10) for _ in range(20)]
    ppc=ep.posterior_predictive_discrepancies(obs,reps,{'mean':np.mean,'sd':lambda x:np.std(x,ddof=1)})
    assert len(ppc)==2 and ((ppc.p_two_sided>=0)&(ppc.p_two_sided<=1)).all()
    d=pd.DataFrame({'y':rng.normal(size=30),'process':rng.normal(size=30),'person':np.repeat(np.arange(10),3)})
    nc=ep.negative_control_process_test(d,'process',lambda x:float(x.y.corr(x.process)),within='person',permutations=10,seed=4)
    assert len(nc.null)==10 and 0<nc.p_value<=1


def test_stress_transport_incremental_and_evidence_contracts():
    runner=lambda sc,r: pd.DataFrame([{'value':float(r)+float(sc['strength'].iloc[0] if 'strength' in sc else 0)}])
    st=ep.stress_test_local_dependence(runner,strengths=[0,.5],replications=2,seed=1); assert len(st)==4 and not st.failed.any()
    data=pd.DataFrame({'group':np.repeat(['a','b','c'],6),'y':np.tile([0,1,0,1,0,1],3),'x':np.arange(18,dtype=float)})
    fitter=lambda tr:{'mean':tr.y.mean()}; predictor=lambda fit,te:np.repeat(fit['mean'],len(te)); scorer=lambda te,p:pd.DataFrame([{'brier':np.mean((te.y.to_numpy()-p)**2)}])
    lv=ep.leave_device_out_validation(data,'group',fitter,predictor,scorer); assert len(lv)==3 and not lv.failed.any()
    ta=ep.audit_measurement_transportability(lv,'brier',max_range=1); assert bool(ta['pass'].iloc[0])
    inc=ep.audit_channel_incremental_information(data,'group',fitter,fitter,predictor,lambda te,p:-np.mean((te.y.to_numpy()-p)**2)); assert len(inc)==3
    rec=pd.DataFrame({'replicate':[1,2,3,4],'parameter':'a','truth':1.,'estimate':[1.01,.99,1.02,.98],'lower':.5,'upper':1.5,'converged':True})
    grade=ep.grade_model_evidence(rec,ep.irt_validation_spec('m',replications=4)); assert grade.grade in {'moderate_validation_evidence','provisional_validation_evidence','insufficient_validation_evidence','strong_validation_evidence'}


def test_calibration_and_failure_taxonomy():
    d=pd.DataFrame({'g':np.repeat(['a','b'],20),'y':[0,1]*20,'p':np.tile([.2,.8],20)})
    z=ep.calibration_transfer_audit(d,'g','y','p'); assert len(z)==2 and np.isfinite(z.brier).all()
    t=ep.validation_failure_taxonomy(['model did not converge','No module named foo','nan overflow'])
    assert t.failure_type.tolist()==['nonconvergence','dependency','numerical']


def test_new_process_irt_plots_smoke():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rng=np.random.default_rng(9)
    eq=ep.equate_irt_scales(pd.DataFrame({'a':[1,1.1,.9],'b':[-1,0,1]}),pd.DataFrame({'a':[.9,1,.8],'b':[-.8,.1,1.1]}),method='mean-sigma')
    ax=ep.plot_eye_irt_equating(eq); assert len(ax.gp3_data)>0; plt.close(ax.figure)
    bank=pd.DataFrame({'item_id':['a','b'],'a':[1,1.2],'b':[0,.5]}); cat=ep.simulate_process_cat(bank,n_items=2,seed=2)
    ax=ep.plot_eye_process_cat_simulation(cat); assert len(ax.gp3_data)==2; plt.close(ax.figure)
    rec=pd.DataFrame({'replicate':[1,2,3,4],'parameter':'a','truth':1.,'estimate':[1.1,.9,1.05,.95],'lower':.5,'upper':1.5,'converged':True}); sm=ep.summarize_parameter_recovery(rec)
    ax=ep.plot_eye_irt_recovery_summary(sm); assert len(ax.gp3_data)==1; plt.close(ax.figure)
    sbc=ep.run_sbc(lambda r:{'data':[r],'truth':{'a':0}},lambda d:d,lambda f:pd.DataFrame({'a':rng.normal(size=100)}),replications=8)
    ax=ep.plot_eye_irt_sbc(sbc); assert len(ax.gp3_data)==8; plt.close(ax.figure)
    sa=ep.audit_sbc(sbc,bins=4); ax=ep.plot_eye_sbc_audit(sa); assert len(ax.gp3_data)==1; plt.close(ax.figure)
    ppc=ep.posterior_predictive_discrepancies(np.arange(5),[np.arange(5)+rng.normal(size=5) for _ in range(10)]); ax=ep.plot_eye_irt_ppc(ppc); assert len(ax.gp3_data)==3; plt.close(ax.figure)
    nc=ep.negative_control_process_test(pd.DataFrame({'x':rng.normal(size=20)}),'x',lambda d:float(d.x.mean()),permutations=10); ax=ep.plot_eye_process_negative_control(nc); assert len(ax.gp3_data)==10; plt.close(ax.figure)
