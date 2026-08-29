from __future__ import annotations
import math
from typing import Any, Mapping
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from .irt import EyeResult
from .exceptions import EyeProcessValidationError


def _result(cls: str, **kw): return EyeResult(kw, eyeprocess_class=cls)
def _df(x,name='data'):
    if isinstance(x,pd.DataFrame): return x.copy()
    try:return pd.DataFrame(x)
    except Exception as e: raise EyeProcessValidationError(f'{name} must be coercible to a data frame.') from e
def _req(d,cols):
    miss=[c for c in cols if c not in d]
    if miss: raise EyeProcessValidationError(f"Missing required columns: {', '.join(miss)}")
def _ax(ax=None):
    import matplotlib.pyplot as plt
    return plt.subplots()[1] if ax is None else ax

def register_pupil_curves(x,time,pupil,anchor='stimulus',method='elastic',id_col='person_id',grid_size=101):
    d=_df(x); _req(d,[id_col,time,pupil]); ids=list(pd.unique(d[id_col].astype(str))); grid=np.linspace(0,1,int(grid_size)); raw=np.full((len(ids),len(grid)),np.nan); reg=np.full_like(raw,np.nan); peak=np.full(len(ids),np.nan); curves={}
    for i,idv in enumerate(ids):
        z=d[d[id_col].astype(str)==idv][[time,pupil]].copy(); z[time]=pd.to_numeric(z[time],errors='coerce'); z[pupil]=pd.to_numeric(z[pupil],errors='coerce'); z=z.dropna().sort_values(time)
        if len(z)<3: continue
        t=z[time].to_numpy(float); y=z[pupil].to_numpy(float); span=max(np.ptp(t),1e-12); tn=(t-t.min())/span; rc=np.interp(grid,tn,y); raw[i]=rc; peak[i]=grid[int(np.nanargmax(rc))]; curves[idv]=z
    ref=float(np.nanmedian(peak)) if np.isfinite(peak).any() else .5; shifts=ref-peak
    for i in range(len(ids)):
        if not np.isfinite(raw[i]).any():continue
        warped=grid-shifts[i]
        if method=='elastic': warped=grid-.75*shifts[i]*(4*grid*(1-grid))
        reg[i]=np.interp(np.clip(warped,0,1),grid,raw[i])
    summary=pd.DataFrame({'curve_id':ids,'peak_position':peak,'phase_shift':shifts})
    return _result('eye_pupil_registration',data=d,ids=ids,time_col=time,pupil_col=pupil,id_col=id_col,anchor=anchor,method=method,grid=grid,raw=raw,registered=reg,summary=summary,status='Pupil curves registered and separated from latency variation.')

def decompose_pupil_phase_amplitude(x,components=3):
    if getattr(x,'eyeprocess_class',None)!='eye_pupil_registration': raise EyeProcessValidationError('x must be an eye_pupil_registration object.')
    R=np.asarray(x['registered'],float); complete=np.sum(np.isfinite(R),axis=1)>=max(3,R.shape[1]/2); M=R[complete].copy(); ids=np.asarray(x['ids'])[complete]
    if len(M)<2: raise EyeProcessValidationError('At least two complete registered curves are required.')
    for i in range(len(M)):
        m=~np.isfinite(M[i]); M[i,m]=np.nanmean(M[i])
    X=M-M.mean(axis=0); U,S,Vt=np.linalg.svd(X,full_matrices=False); scores=U*S; k=min(int(components),scores.shape[1]); ad=pd.DataFrame(scores[:,:k],columns=[f'amplitude_pc{i}' for i in range(1,k+1)]); ad[x['id_col']]=ids
    ph=x['summary'][x['summary'].curve_id.astype(str).isin(ids.astype(str))][['curve_id','phase_shift','peak_position']].copy().rename(columns={'curve_id':x['id_col']}); sc=ph.merge(ad,on=x['id_col'],how='outer'); var=(S*S)/max(np.sum(S*S),1e-12)
    return _result('eye_pupil_phase_amplitude',registration=x,pca={'components':Vt,'singular_values':S},scores=sc,amplitude_variance=var,summary=sc,status='Pupil phase and amplitude components decomposed.')

def _linfit(X,y,binomial=False):
    X=np.asarray(X,float); y=np.asarray(y,float); ok=np.isfinite(y)&np.all(np.isfinite(X),axis=1); X=X[ok]; y=y[ok]
    if binomial:
        def f(b):
            p=np.clip(expit(X@b),1e-9,1-1e-9); return -np.sum(y*np.log(p)+(1-y)*np.log(1-p))
        b=minimize(f,np.zeros(X.shape[1]),method='BFGS').x; fitted=expit(X@b); resid=y-fitted
    else:
        b=np.linalg.lstsq(X,y,rcond=None)[0]; fitted=X@b; resid=y-fitted
    return {'coef':b,'fitted':fitted,'residuals':resid}

def fit_phase_amplitude_irt(responses,phase_scores,amplitude_scores=None,person_id=None,family='gaussian',**kwargs):
    scores=phase_scores['scores'].copy() if getattr(phase_scores,'eyeprocess_class',None)=='eye_pupil_phase_amplitude' else _df(phase_scores)
    if amplitude_scores is not None and getattr(phase_scores,'eyeprocess_class',None)!='eye_pupil_phase_amplitude': scores=pd.concat([scores.reset_index(drop=True),_df(amplitude_scores).reset_index(drop=True)],axis=1)
    R=np.asarray(responses,float); R=R[:,None] if R.ndim==1 else R; outcome=np.nanmean(R,axis=1) if R.shape[1]>1 else R[:,0]; person_id=np.asarray(person_id if person_id is not None else (getattr(responses,'index',np.arange(len(outcome))) if hasattr(responses,'index') else np.arange(len(outcome))),dtype=str)
    idcol=next((c for c in ['person_id','participant_id','curve_id'] if c in scores),None)
    if idcol:
        data=pd.DataFrame({'person_id':person_id,'outcome':outcome}).merge(scores.assign(person_id=scores[idcol].astype(str)),on='person_id')
    else:
        if len(scores)!=len(outcome): raise EyeProcessValidationError('Phase/amplitude scores must align with the response rows.')
        data=pd.concat([pd.DataFrame({'person_id':person_id,'outcome':outcome}),scores.reset_index(drop=True)],axis=1)
    predictors=[c for c in data.columns if c!='outcome' and pd.api.types.is_numeric_dtype(data[c])]; X=np.column_stack([np.ones(len(data))]+[pd.to_numeric(data[c],errors='coerce').to_numpy(float) for c in predictors]); fit=_linfit(X,data.outcome.to_numpy(float),family=='binomial'); names=['(Intercept)']+predictors; summary=pd.DataFrame({'Estimate':fit['coef']},index=names)
    return _result('eye_phase_amplitude_irt',model=fit,data=data,family=family,summary=summary,status='Phase-amplitude response model fitted. For confirmatory IRT, use a validated external engine or custom Stan model.')

def audit_pupil_registration(x):
    R=np.asarray(x['raw'],float); Q=np.asarray(x['registered'],float); rp=np.array([np.nan if not np.isfinite(r).any() else np.nanargmax(r)+1 for r in R],float); qp=np.array([np.nan if not np.isfinite(r).any() else np.nanargmax(r)+1 for r in Q],float); before=np.nanvar(rp,ddof=1); after=np.nanvar(qp,ddof=1); imp=(before-after)/max(before,1e-12)
    return _result('eye_pupil_registration_audit',summary=pd.DataFrame([{'peak_variance_before':before,'peak_variance_after':after,'relative_reduction':imp}]),raw_peak=rp,registered_peak=qp,status='Pupil registration audit completed.')

def _pupil_plot(x,kind,ax=None):
    ax=_ax(ax)
    if getattr(x,'eyeprocess_class',None)=='eye_pupil_registration':
        d=x['summary'];
        if kind=='registration':
            for r in np.asarray(x['raw'])[:20]: ax.plot(x['grid'],r,alpha=.35)
            for r in np.asarray(x['registered'])[:20]: ax.plot(x['grid'],r,ls='--',alpha=.35)
        elif kind=='warping': ax.scatter(d.peak_position,d.phase_shift)
        elif kind=='item_delay': ax.bar(d.curve_id,d.phase_shift); ax.tick_params(axis='x',rotation=90)
        else: ax.plot(x['grid'],np.nanmean(np.asarray(x['registered']),axis=0))
    elif getattr(x,'eyeprocess_class',None)=='eye_pupil_phase_amplitude':
        d=x['scores']; ax.scatter(d.phase_shift,d.amplitude_pc1 if 'amplitude_pc1' in d else np.zeros(len(d)))
    else:
        d=x['data']; coef=x['summary']['Estimate']; ax.bar(np.arange(len(coef)),coef)
    ax.set_title(kind.replace('_',' ').title()); ax.eyeprocess_plot_data=d.copy() if isinstance(d,pd.DataFrame) else pd.DataFrame(); return ax

def plot_pupil_registration(x,**kw):return _pupil_plot(x,'registration',**kw)
def plot_warping_functions(x,**kw):return _pupil_plot(x,'warping',**kw)
def plot_phase_amplitude_scores(x,**kw):return _pupil_plot(x,'scores',**kw)
def plot_item_phase_delay(x,**kw):return _pupil_plot(x,'item_delay',**kw)
def plot_registered_pupil_effects(x,**kw):return _pupil_plot(x,'registered_effects',**kw)

def fit_process_observation_model(x,observed,predictors,random=('person','item')):
    d=_df(x); obs=d[observed].to_numpy() if isinstance(observed,str) and observed in d else np.asarray(observed); predictors=[p for p in predictors if p in d]
    if len(obs)!=len(d): raise EyeProcessValidationError('observed must contain one value per row.')
    if not predictors: raise EyeProcessValidationError('At least one observation predictor is required.')
    work=d[predictors].copy(); work['.observed']=pd.Series(obs).astype(bool).astype(int).to_numpy(); X=np.column_stack([np.ones(len(work))]+[pd.to_numeric(work[c],errors='coerce').fillna(pd.to_numeric(work[c],errors='coerce').mean()).to_numpy(float) for c in predictors]); fit=_linfit(X,work['.observed'].to_numpy(float),True); prob=expit(X@fit['coef']); summary=pd.DataFrame({'Estimate':fit['coef']},index=['(Intercept)']+predictors)
    return _result('eye_process_observation_model',model=fit,data=work,probability=prob,random=list(random),summary=summary,status='Process-signal observation model fitted. Random group declarations are audit metadata in the dependency-free engine.')

def fit_joint_signal_missingness(outcome,observation,method='selection',x=None,predictors=None):
    d=_df(x) if x is not None else None; y=pd.to_numeric(d[outcome],errors='coerce').to_numpy(float) if isinstance(outcome,str) and d is not None else np.asarray(outcome,float); obs=observation['data']['.observed'].to_numpy(int) if getattr(observation,'eyeprocess_class',None)=='eye_process_observation_model' else (d[observation].astype(bool).astype(int).to_numpy() if isinstance(observation,str) and d is not None else np.asarray(observation,bool).astype(int))
    if len(y)!=len(obs): raise EyeProcessValidationError('Outcome and observation indicators must have equal length.')
    work=pd.DataFrame({'outcome':y,'observed':obs}); preds=[p for p in (predictors or []) if d is not None and p in d]
    for p in preds: work[p]=pd.to_numeric(d[p],errors='coerce'); X=np.column_stack([np.ones(len(work)),work.observed.to_numpy(float)]+[work[p].fillna(work[p].mean()).to_numpy(float) for p in preds]); fit=_linfit(X,y,False); summary=pd.DataFrame({'Estimate':fit['coef']},index=['(Intercept)','observed']+preds); corr=math.nan
    if getattr(observation,'eyeprocess_class',None)=='eye_process_observation_model':
        ok=np.isfinite(y); corr=float(np.corrcoef((y[ok]-(X[ok]@fit['coef'])),np.asarray(observation['probability'])[ok])[0,1]) if ok.sum()>2 else math.nan
    return _result('eye_joint_signal_missingness',model=fit,data=work,method=method,shared_correlation=corr,summary=summary,status='Joint signal-missingness approximation fitted; use a full shared-parameter model for confirmatory MNAR inference.')

def process_pattern_mixture(x,delta=np.arange(-1,1.0001,.1),metric=None,estimand=np.mean):
    if isinstance(x,pd.DataFrame):
        if metric is None: metric=next(c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])); values=pd.to_numeric(x[metric],errors='coerce').to_numpy(float)
    else: values=np.asarray(x,float)
    observed=values[np.isfinite(values)]; missing=~np.isfinite(values)
    if not len(observed): raise EyeProcessValidationError('At least one observed value is required.')
    center=float(observed.mean()); spread=float(observed.std(ddof=1)) if len(observed)>1 else 0.0; rows=[]
    for dd in np.atleast_1d(delta):
        c=values.copy(); c[missing]=center+float(dd)*spread
        try: est=float(estimand(c))
        except TypeError: est=float(estimand(c,axis=0))
        rows.append({'delta':float(dd),'estimate':est,'imputed_value':center+float(dd)*spread,'missing_fraction':float(missing.mean())})
    table=pd.DataFrame(rows); return _result('eye_mnar_sensitivity',table=table,summary=table,metric=metric,status='Pattern-mixture MNAR sensitivity completed.')

def sensitivity_mnar_process(x,estimand=np.mean,null=0,**kwargs):
    s=x if getattr(x,'eyeprocess_class',None)=='eye_mnar_sensitivity' else process_pattern_mixture(x,estimand=estimand,**kwargs); signs=np.sign(s['table'].estimate.to_numpy(float)-null); ch=np.where(np.diff(signs)!=0)[0]; tipping=float(np.mean(s['table'].delta.iloc[ch[0]:ch[0]+2])) if len(ch) else math.nan; summary=pd.DataFrame([{'tipping_delta':tipping,'stable_over_grid':not np.isfinite(tipping)}]); return _result('eye_mnar_tipping_point',sensitivity=s,tipping_delta=tipping,summary=summary,status='An MNAR tipping point was detected.' if np.isfinite(tipping) else 'The estimand did not cross the null over the tested delta grid.')

def _miss_plot(x,kind,time=None,aoi=None,ax=None):
    ax=_ax(ax)
    if getattr(x,'eyeprocess_class',None)=='eye_process_observation_model':
        p=np.asarray(x['probability']); d=pd.DataFrame({'probability':p})
        if kind=='missingness_time' and time is not None: d['time']=time; ax.plot(time,p)
        elif kind=='missingness_aoi' and aoi is not None: d['aoi']=aoi; g=d.groupby('aoi').probability.mean(); ax.bar(g.index.astype(str),g.values)
        else: ax.hist(p)
    else:
        s=x['sensitivity'] if getattr(x,'eyeprocess_class',None)=='eye_mnar_tipping_point' else x; d=s['table']; ax.plot(d.delta,d.estimate,marker='o'); ax.axhline(0,ls='--');
        if getattr(x,'eyeprocess_class',None)=='eye_mnar_tipping_point' and np.isfinite(x['tipping_delta']): ax.axvline(x['tipping_delta'],ls='--')
    ax.set_title(kind.replace('_',' ').title()); ax.eyeprocess_plot_data=d.copy(); return ax

def plot_observation_probability(x,**kw):return _miss_plot(x,'observation_probability',**kw)
def plot_missingness_by_time(x,**kw):return _miss_plot(x,'missingness_time',**kw)
def plot_missingness_by_aoi(x,**kw):return _miss_plot(x,'missingness_aoi',**kw)
def plot_mnar_tipping_point(x,**kw):return _miss_plot(x,'tipping_point',**kw)
def plot_complete_case_sensitivity(x,**kw):return _miss_plot(x,'complete_case_sensitivity',**kw)

__all__=[n for n in globals() if not n.startswith('_') and n not in {'math','np','pd','Any','Mapping','expit','minimize','EyeResult','EyeProcessValidationError'}]

def plot_eye_pupil_registration(x,type='registration',ax=None): return _pupil_plot(x,type,ax=ax)
def plot_eye_pupil_phase_amplitude(x,type='scores',ax=None): return _pupil_plot(x,type,ax=ax)
def plot_eye_phase_amplitude_irt(x,type='registered_effects',ax=None): return _pupil_plot(x,type,ax=ax)
def plot_eye_process_observation_model(x,type='observation_probability',time=None,aoi=None,ax=None): return _miss_plot(x,type,time=time,aoi=aoi,ax=ax)
def plot_eye_mnar_sensitivity(x,type='tipping_point',ax=None): return _miss_plot(x,type,ax=ax)
def plot_eye_mnar_tipping_point(x,type='tipping_point',ax=None): return _miss_plot(x,type,ax=ax)

__all__=[n for n in globals() if not n.startswith('_') and n not in {'math','np','pd','Any','Mapping','expit','minimize','EyeResult','EyeProcessValidationError'}]
