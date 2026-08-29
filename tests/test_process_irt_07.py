from pathlib import Path
import math

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.exceptions import EyeProcessBackendError, EyeProcessModelError, EyeProcessValidationError


def test_process_irt_07_frozen_exports_present():
    names=[x.strip() for x in (Path(__file__).parent/'fixtures'/'process_irt_07_exports.txt').read_text().splitlines() if x.strip()]
    assert len(names)==45
    missing=[n for n in names if not callable(getattr(ep,n,None))]
    assert missing==[]


def test_channels_registry_and_gating_contract():
    ch=ep.irt_count_channel('poisson',value='fixations')
    assert ch.superclass=='eye_irt_channel'
    models=ep.list_irt_models()
    assert {'id','status','channels'} <= set(models.columns)
    assert {'joint_gaze_rt','flow_mirt','bounded_continuous_process'} <= set(models.id)
    custom=ep.irt_model_spec(id='custom_validation_model',latent='ability',channels={'response':ep.irt_response_channel('2pl')})
    assert custom.status=='experimental'
    with pytest.raises(EyeProcessModelError):
        ep.fit_irt_model(custom,pd.DataFrame({'x':[1]}))
    with pytest.raises(EyeProcessBackendError):
        ep.fit_irt_model('flow_mirt',np.array([[0,1],[1,0]]),allow_experimental=True)


def test_missingness_classification_preserves_process_distinctions():
    d=pd.DataFrame({'response':[1,np.nan,np.nan,np.nan,np.nan], 'reached':[True,False,True,True,True],
                    'inspected':[True,False,False,True,True], 'started':[True,False,False,False,True]})
    z=list(ep.classify_item_missingness(d,inspected='inspected',started='started').astype(str))
    assert z==['answered','not_reached','reached_not_inspected','inspected_omission','started_unanswered']


def test_nominal_gaze_information_and_distractor_map():
    rng=np.random.default_rng(7); n=80
    d=pd.DataFrame({'participant_id':[f'p{i%20}' for i in range(n)], 'item_id':[f'i{i%4}' for i in range(n)]})
    d['gA']=rng.gamma(2,1,n); d['gB']=rng.gamma(2,1,n); d['gC']=rng.gamma(2,1,n)
    logits=np.column_stack([.4*d.gA-.2*d.gB, .3*d.gB, .3*d.gC]); probs=np.exp(logits-logits.max(axis=1)[:,None]); probs=probs/probs.sum(axis=1)[:,None]
    cats=np.array(['A','B','C']); d['response_option']=[rng.choice(cats,p=p) for p in probs]
    fit=ep.fit_nominal_gaze_irt(d,option_gaze=['gA','gB','gC'])
    info=ep.option_process_information(fit); mp=ep.distractor_process_map(fit)
    assert len(info)==n and np.isfinite(info.entropy_reduction).all()
    assert {'response_category','gaze_channel','coefficient'} <= set(mp.columns)
    aud=ep.audit_distractor_attention(d,option_gaze={'A':'gA','B':'gB','C':'gC'})
    assert aud.loc[0,'n']==n


def test_censored_normal_calibration_and_prediction_dimensions():
    rng=np.random.default_rng(11); theta=np.linspace(-2,2,60)
    X=pd.DataFrame({'item1':np.clip(.5+.12*theta+rng.normal(0,.08,60),0,1),
                    'item2':np.clip(.4+.18*theta+rng.normal(0,.10,60),0,1)})
    fit=ep.fit_censored_normal_process_irt(X,theta)
    assert fit.eyeprocess_class=='eye_censored_normal_process_irt'
    assert fit.coefficients.shape[0]==2
    assert np.isfinite(fit.coefficients.discrimination).all()
    assert (fit.coefficients.sigma>0).all()
    pr=ep.predict_eye_censored_normal_process_irt(fit,theta=[-1,0,1])
    assert pr.shape==(3,2) and np.all((pr>=0)&(pr<=1))


def test_channel_ablation_is_directional_and_explicit():
    d=pd.DataFrame({'y':range(1,7),'a':range(2,8),'b':range(3,9)})
    z=ep.process_channel_ablation(d,{'a':['a'],'b':['b']},lambda data,active_columns,channel_name:len(active_columns),baseline=['y'])
    assert len(z)==2 and np.allclose(z.information_loss,1)


def test_multiple_response_combinations_and_local_dependence():
    d=pd.DataFrame({'participant_id':np.repeat(['p1','p2'],4),'item_id':'i1','option_id':['A','B','C','D']*2,
                    'selected':[True,False,True,False,False,True,False,True]})
    z=ep.encode_response_combinations(d)
    assert set(z.response_combination)=={'A|C','B|D'}
    assert z.n_selected.tolist()==[2,2]
    rng=np.random.default_rng(7); r=rng.normal(size=(100,4)); p=r+rng.normal(scale=.25,size=(100,4))
    aud=ep.audit_process_local_dependence(pd.DataFrame(r,columns=['i1','i2','i3','i4']),p,threshold=.20)
    assert len(aud.pairs)==6
    assert {'response_flag','process_flag','concordant_direction'} <= set(aud.pairs.columns)


def test_multiple_response_reference_is_explicitly_non_exact():
    rng=np.random.default_rng(11); n_person=24
    rows=[]
    theta={f'p{i+1}':rng.normal() for i in range(n_person)}
    for pid,t in theta.items():
        for item in [f'i{x}' for x in range(1,5)]:
            for opt in ['A','B']:
                eta=-.3+.8*t+(.25 if opt=='A' else -.25)
                rows.append((pid,item,opt,t,rng.binomial(1,1/(1+np.exp(-eta)))))
    d=pd.DataFrame(rows,columns=['participant_id','item_id','option_id','theta','selected'])
    fit=ep.fit_multiple_response_process_irt(d)
    assert fit.eyeprocess_class=='eye_multiple_response_process_irt'
    assert fit.exact_multiple_response is False
    assert fit.status=='experimental-python-reference'


def test_changepoints_recalibration_and_manyfacet_contracts():
    rng=np.random.default_rng(4); rows=[]
    for p in ['p1','p2']:
        for order in range(1,31):
            shifted=order>15
            rows.append({'participant_id':p,'item_id':f'i{(order-1)%5+1}','item_order':order,
                         'response':rng.binomial(1,.75 if not shifted else .35),
                         'rt':rng.lognormal(1.0 if not shifted else 1.6,.15),
                         'fixation_count':rng.poisson(3 if not shifted else 7),'device':'d1' if p=='p1' else 'd2'})
    d=pd.DataFrame(rows)
    cp=ep.detect_irt_changepoints(d,gaze='fixation_count',min_segment=5,min_delta_sic=0)
    assert len(cp.results)==2 and cp.method=='sic-inspired-reference'
    rr=ep.recalibrate_after_changepoint(d,lambda x:{'n':len(x)},gaze='fixation_count',min_segment=5,min_delta_sic=0,policy='add_regime')
    assert '.process_regime' in rr.data and rr.fit['n']==len(d)
    mf=ep.fit_manyfacet_process_irt(d,response='response',process='fixation_count',device='device')
    ef=ep.facet_effects(mf,'process'); inv=ep.audit_process_measurement_invariance(mf,'process')
    assert len(ef.variance_components)>=3 and 'relative_sd' in inv.components


def test_process_dependent_discrimination_and_generalizability():
    rng=np.random.default_rng(8); n=180
    d=pd.DataFrame({'participant_id':[f'p{i%30}' for i in range(n)],'item_id':[f'i{i%6}' for i in range(n)]})
    d['theta']=rng.normal(size=n); d['process']=np.exp(.2*d.theta+rng.normal(scale=.4,size=n)); pr=1/(1+np.exp(-(-.2+.9*d.theta+.25*d.theta*np.log(d.process))))
    d['response']=rng.binomial(1,pr)
    fit=ep.process_dependent_discrimination_audit(d,'response','theta','process','participant_id','item_id',nonlinear=False)
    assert fit.eyeprocess_class=='eye_process_dependent_discrimination'
    assert np.isfinite(fit.interaction.Estimate.iloc[0])
    gs=ep.generalizability_process_study(d,'process',['participant_id','item_id'])
    assert math.isclose(float(gs.variance_components.proportion.sum()),1.0,rel_tol=1e-7,abs_tol=1e-7)


def test_process_irt_07_plots_return_axes_with_data():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ab=ep.process_channel_ablation(pd.DataFrame({'y':[1,2],'a':[2,3]}),{'a':['a']},lambda d,active,name:len(active),baseline=['y'])
    ax=ep.plot_eye_process_channel_ablation(ab); assert len(ax.gp3_data)==1; plt.close(ax.figure)
    rng=np.random.default_rng(2); R=pd.DataFrame(rng.normal(size=(40,3)),columns=['a','b','c']); ld=ep.audit_process_local_dependence(R)
    ax=ep.plot_eye_process_local_dependence_audit(ld); assert len(ax.gp3_data)==3; plt.close(ax.figure)
    d=pd.DataFrame({'participant_id':[f'p{i%10}' for i in range(80)],'item_id':[f'i{i%4}' for i in range(80)],'theta':rng.normal(size=80),'process':rng.lognormal(size=80)})
    d['response']=rng.binomial(1,1/(1+np.exp(-d.theta)))
    pdx=ep.process_dependent_discrimination_audit(d,'response','theta','process','participant_id','item_id',nonlinear=False)
    ax=ep.plot_eye_process_dependent_discrimination(pdx); assert len(ax.lines)==3; plt.close(ax.figure)
    gs=ep.generalizability_process_study(d,'process',['participant_id','item_id'])
    ax=ep.plot_eye_process_g_study(gs); assert len(ax.gp3_data)>=3; plt.close(ax.figure)
