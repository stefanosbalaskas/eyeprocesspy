from __future__ import annotations
import math
from typing import Any, Callable, Mapping, Sequence
import numpy as np
import pandas as pd
from .irt import EyeResult
from .exceptions import EyeProcessValidationError


def _result(cls: str, **kw): return EyeResult(kw, eyeprocess_class=cls)
def _df(x,name='data'):
    if isinstance(x,pd.DataFrame): return x.copy()
    try: return pd.DataFrame(x)
    except Exception as e: raise EyeProcessValidationError(f"{name} must be coercible to a data frame.") from e
def _req(d, cols, name='data'):
    miss=[c for c in cols if c not in d.columns]
    if miss: raise EyeProcessValidationError(f"{name} is missing required columns: {', '.join(miss)}")
def _num(x): return pd.to_numeric(pd.Series(x),errors='coerce').to_numpy(float)
def _mean(x):
    a=np.asarray(x,float); a=a[np.isfinite(a)]; return float(a.mean()) if a.size else math.nan
def _sd(x):
    a=np.asarray(x,float); a=a[np.isfinite(a)]; return float(a.std(ddof=1)) if a.size>1 else math.nan
def _q(x,p,default=math.nan):
    a=np.asarray(x,float); a=a[np.isfinite(a)]; return float(np.quantile(a,p)) if a.size else default

def _ax(ax=None):
    import matplotlib.pyplot as plt
    return plt.subplots()[1] if ax is None else ax

def process_uncertainty_spec(calibration=True,aoi_assignment=True,preprocessing=True,sampling=True,model=True,source_sd=None,draws=1000,seed=20260807):
    names=['calibration','aoi_assignment','preprocessing','sampling','model']
    included=dict(zip(names,map(bool,[calibration,aoi_assignment,preprocessing,sampling,model])))
    full={n:math.nan for n in names}
    if source_sd is not None:
        if isinstance(source_sd,Mapping):
            for k,v in source_sd.items():
                if k in full: full[k]=float(v)
        else:
            vals=np.atleast_1d(source_sd)
            for k,v in zip(names,vals): full[k]=float(v)
    return _result('eye_process_uncertainty_spec',included=included,source_sd=full,draws=int(draws),seed=int(seed))

def estimate_process_uncertainty(x,spec=None,metrics=None,cluster=None):
    spec=spec or process_uncertainty_spec()
    if getattr(spec,'eyeprocess_class',None)!='eye_process_uncertainty_spec': raise EyeProcessValidationError('spec must be created by process_uncertainty_spec().')
    if np.ndim(x)==1 and not isinstance(x,pd.DataFrame): x=pd.DataFrame({'metric':x})
    d=_df(x)
    if metrics is None: metrics=[c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    metrics=[m for m in metrics if m in d]
    if not metrics: raise EyeProcessValidationError('No numeric process metrics were selected.')
    rows=[]
    for metric in metrics:
        vals=_num(d[metric]); finite=vals[np.isfinite(vals)]; n=len(finite); est=_mean(finite); raw_sd=_sd(finite)
        if cluster is not None and cluster in d:
            cm=d.assign(_v=pd.to_numeric(d[metric],errors='coerce')).groupby(cluster,dropna=False)['_v'].mean().to_numpy(float)
            sampling=_sd(cm)/math.sqrt(max(1,len(cm)))
        else: sampling=raw_sd/math.sqrt(max(1,n)) if np.isfinite(raw_sd) else math.nan
        base=(raw_sd if np.isfinite(raw_sd) else 1.0)
        defaults={'calibration':.08*base,'aoi_assignment':.10*base,'preprocessing':.07*base,'sampling':sampling,'model':.05*base}
        sds={k:(float(spec['source_sd'].get(k,math.nan)) if np.isfinite(spec['source_sd'].get(k,math.nan)) else defaults[k]) for k in defaults}
        sds['sampling']=sampling
        for k,v in spec['included'].items():
            if not v: sds[k]=0.0
        vars_={k:(v*v if np.isfinite(v) else 0.0) for k,v in sds.items()}; total_var=sum(vars_.values()); total_se=math.sqrt(total_var)
        for source in vars_:
            rows.append(dict(metric=metric,source=source,estimate=est,n=n,source_sd=sds[source],source_variance=vars_[source],variance_share=(vars_[source]/total_var if total_var>0 else math.nan),total_se=total_se,lower=est-1.96*total_se,upper=est+1.96*total_se))
    comp=pd.DataFrame(rows); summary=comp[['metric','estimate','n','total_se','lower','upper']].drop_duplicates().reset_index(drop=True)
    return _result('eye_process_uncertainty',components=comp,summary=summary,spec=spec,data=d,metrics=metrics,status='Measurement-uncertainty components estimated.')

def propagate_process_uncertainty(x,estimand=lambda data: np.nanmean(data),method='bootstrap',draws=None,seed=None):
    if method not in {'bootstrap','simulation','posterior'}: raise EyeProcessValidationError('Unknown propagation method.')
    unc=x if getattr(x,'eyeprocess_class',None)=='eye_process_uncertainty' else None
    data=unc['data'] if unc is not None else x; draws=int(draws or (unc['spec']['draws'] if unc else 1000)); seed=int(seed or (unc['spec']['seed'] if unc else 20260807)); rng=np.random.default_rng(seed)
    vals=[]
    if method=='posterior':
        posterior=data.get('draws') if isinstance(data,Mapping) and 'draws' in data else data
        p=np.asarray(posterior,float).ravel(); p=p[np.isfinite(p)]
        if not len(p): raise EyeProcessValidationError('Posterior propagation requires numeric draws.')
        vals=rng.choice(p,size=draws,replace=True).tolist()
    else:
        n=len(data)
        if n<1: raise EyeProcessValidationError('No observations are available for uncertainty propagation.')
        total_sd=float(pd.to_numeric(unc['summary']['total_se'],errors='coerce').mean()) if unc else 0.0
        for _ in range(draws):
            idx=rng.integers(0,n,size=n)
            sampled=data.iloc[idx].copy() if isinstance(data,pd.DataFrame) else np.asarray(data)[idx].copy()
            if method=='simulation' and unc is not None and np.isfinite(total_sd):
                if isinstance(sampled,pd.DataFrame):
                    for c in sampled.select_dtypes(include=[np.number]).columns: sampled[c]=sampled[c]+rng.normal(0,total_sd,len(sampled))
                else: sampled=sampled+rng.normal(0,total_sd,len(sampled))
            try: vals.append(float(np.asarray(estimand(sampled)).ravel()[0]))
            except Exception: vals.append(math.nan)
    a=np.asarray(vals,float); a=a[np.isfinite(a)]
    summary=pd.DataFrame([dict(method=method,draws=len(a),mean=_mean(a),sd=_sd(a),lower=_q(a,.025),median=_q(a,.5),upper=_q(a,.975))])
    return _result('eye_process_uncertainty_propagation',draws=a,summary=summary,method=method,source=x,status='Process uncertainty propagated to the requested estimand.')

def uncertainty_budget(x):
    if getattr(x,'eyeprocess_class',None)!='eye_process_uncertainty': raise EyeProcessValidationError('x must be an eye_process_uncertainty object.')
    return x['components'][['metric','source','source_sd','source_variance','variance_share']].reset_index(drop=True)

def compare_uncertainty_budgets(*args,**kwargs):
    objs=dict(kwargs) if kwargs else {f'budget_{i+1}':v for i,v in enumerate(args)}
    if not objs: raise EyeProcessValidationError('Supply at least one uncertainty object.')
    tabs=[]
    for label,obj in objs.items():
        if getattr(obj,'eyeprocess_class',None)!='eye_process_uncertainty': raise EyeProcessValidationError('All objects must be process-uncertainty results.')
        t=uncertainty_budget(obj); t.insert(0,'budget',label); tabs.append(t)
    combined=pd.concat(tabs,ignore_index=True); summary=combined.groupby(['budget','source'],as_index=False).variance_share.mean()
    return _result('eye_uncertainty_budget_comparison',budgets=objs,combined=combined,summary=summary,status='Uncertainty budgets compared.')

def _unc_plot(x,kind='waterfall',metric=None,ax=None):
    ax=_ax(ax); d=x['components']; metric=metric or str(d.metric.iloc[0]); s=d[d.metric==metric].copy()
    vals=s.source_sd if kind=='tornado' else s.source_variance
    if kind=='tornado': ax.barh(s.source,vals); ax.set_xlabel('Source standard deviation')
    else: ax.bar(s.source,vals); ax.tick_params(axis='x',rotation=45); ax.set_ylabel('Variance contribution')
    ax.set_title(f'Uncertainty {kind}: {metric}'); ax.eyeprocess_plot_data=s; return ax

def plot_uncertainty_waterfall(x,**kwargs): return _unc_plot(x,'waterfall',**kwargs)
def plot_uncertainty_tornado(x,**kwargs): return _unc_plot(x,'tornado',**kwargs)
def plot_uncertainty_by_item(x,**kwargs): return _unc_plot(x,'by_item',**kwargs)
def plot_uncertainty_by_stage(x,**kwargs): return _unc_plot(x,'by_stage',**kwargs)

def _first_col(d,cands,label):
    for c in cands:
        if c in d: return c
    raise EyeProcessValidationError(f'Missing {label}.')
def _reference_table(x,references,x_col,y_col):
    if references is None:
        tx=_first_col(x,['target_x','reference_x','expected_x'],'reference x coordinate'); ty=_first_col(x,['target_y','reference_y','expected_y'],'reference y coordinate')
        return pd.DataFrame({'observed_x':pd.to_numeric(x[x_col],errors='coerce'),'observed_y':pd.to_numeric(x[y_col],errors='coerce'),'reference_x':pd.to_numeric(x[tx],errors='coerce'),'reference_y':pd.to_numeric(x[ty],errors='coerce')})
    r=_df(references,'references')
    if len(r)==len(x):
        rx=_first_col(r,['reference_x','target_x','x'],'reference x coordinate'); ry=_first_col(r,['reference_y','target_y','y'],'reference y coordinate')
        return pd.DataFrame({'observed_x':pd.to_numeric(x[x_col],errors='coerce'),'observed_y':pd.to_numeric(x[y_col],errors='coerce'),'reference_x':pd.to_numeric(r[rx],errors='coerce'),'reference_y':pd.to_numeric(r[ry],errors='coerce')})
    raise EyeProcessValidationError('references must have one row per observation or share a target identifier with x.')
def _roll_groups(time,window):
    t=np.asarray(time,float)
    if isinstance(window,str):
        try: w=float(window.split()[0])
        except Exception: w=30.0
    else: w=float(window)
    if w<=0: w=30.0
    start=np.nanmin(t) if np.any(np.isfinite(t)) else 0
    return np.floor((t-start)/w).astype(int)+1

def detect_calibration_drift(x,references=None,window='30 sec',method='targets',x_col=None,y_col=None,time_col=None):
    d=_df(x); x_col=x_col or _first_col(d,['x','gaze_x','x_norm','x_px'],'gaze x coordinate'); y_col=y_col or _first_col(d,['y','gaze_y','y_norm','y_px'],'gaze y coordinate')
    if time_col is None: time_col=next((c for c in ['time','timestamp','time_sec','sample_time'] if c in d),None)
    ref=_reference_table(d,references,x_col,y_col); time=pd.to_numeric(d[time_col],errors='coerce').to_numpy(float) if time_col else np.arange(1,len(d)+1,dtype=float)
    ref['time']=time; ref['window_id']=_roll_groups(time,window); ref['error_x']=ref.observed_x-ref.reference_x; ref['error_y']=ref.observed_y-ref.reference_y; ref['error_distance']=np.sqrt(ref.error_x**2+ref.error_y**2)
    summary=ref.groupby('window_id',as_index=False)[['error_x','error_y','error_distance']].mean(); summary['n']=ref.groupby('window_id').size().reindex(summary.window_id).to_numpy(); summary['drift_from_first']=np.sqrt((summary.error_x-summary.error_x.iloc[0])**2+(summary.error_y-summary.error_y.iloc[0])**2)
    threshold=_q(summary.drift_from_first,.95,0); sd=_sd(ref.error_distance); summary['review_flag']=summary.drift_from_first > max(threshold,2*sd if np.isfinite(sd) else 0)
    return _result('eye_calibration_drift',observations=ref,summary=summary,method=method,coordinate_columns={'x':x_col,'y':y_col},time_col=time_col,status='Calibration drift summarized over time windows.')

def fit_offline_recalibration(x,method='translation',robust=True,x_col=None,y_col=None,reference_x_col=None,reference_y_col=None):
    if method not in {'translation','affine','polynomial'}: raise EyeProcessValidationError('Unknown recalibration method.')
    if getattr(x,'eyeprocess_class',None)=='eye_calibration_drift': d=x['observations'].copy()
    else:
        z=_df(x); x_col=x_col or _first_col(z,['observed_x','x','gaze_x'],'observed x'); y_col=y_col or _first_col(z,['observed_y','y','gaze_y'],'observed y'); reference_x_col=reference_x_col or _first_col(z,['reference_x','target_x','expected_x'],'reference x'); reference_y_col=reference_y_col or _first_col(z,['reference_y','target_y','expected_y'],'reference y')
        d=pd.DataFrame({'observed_x':pd.to_numeric(z[x_col],errors='coerce'),'observed_y':pd.to_numeric(z[y_col],errors='coerce'),'reference_x':pd.to_numeric(z[reference_x_col],errors='coerce'),'reference_y':pd.to_numeric(z[reference_y_col],errors='coerce')}).dropna()
    if len(d)<3: raise EyeProcessValidationError('At least three complete rows are required.')
    if method=='translation':
        offset={'x':_mean(d.reference_x-d.observed_x),'y':_mean(d.reference_y-d.observed_y)}; models=None
    else:
        ox=d.observed_x.to_numpy(float); oy=d.observed_y.to_numpy(float)
        X=np.column_stack([np.ones(len(d)),ox,oy]) if method=='affine' else np.column_stack([np.ones(len(d)),ox,oy,ox**2,oy**2,ox*oy])
        bx=np.linalg.lstsq(X,d.reference_x.to_numpy(float),rcond=None)[0]; by=np.linalg.lstsq(X,d.reference_y.to_numpy(float),rcond=None)[0]
        models={'x':bx,'y':by}; offset=None
    return _result('eye_recalibration_model',method=method,robust=bool(robust),offset=offset,models=models,training=d,status=f'{method.title()} offline recalibration fitted.')

def apply_offline_recalibration(x,model,x_col=None,y_col=None,suffix='_recalibrated'):
    d=_df(x); x_col=x_col or _first_col(d,['observed_x','x','gaze_x'],'observed x'); y_col=y_col or _first_col(d,['observed_y','y','gaze_y'],'observed y'); ox=pd.to_numeric(d[x_col],errors='coerce').to_numpy(float); oy=pd.to_numeric(d[y_col],errors='coerce').to_numpy(float)
    if model['method']=='translation': cx=ox+model['offset']['x']; cy=oy+model['offset']['y']
    else:
        X=np.column_stack([np.ones(len(d)),ox,oy]) if model['method']=='affine' else np.column_stack([np.ones(len(d)),ox,oy,ox**2,oy**2,ox*oy]); cx=X@model['models']['x']; cy=X@model['models']['y']
    out=d.copy(); out[f'{x_col}{suffix}']=cx; out[f'{y_col}{suffix}']=cy; out.attrs['eye_recalibration_model']=model; return out

def audit_recalibration(before,after,minimum_improvement=None):
    b=before['observations'].copy() if getattr(before,'eyeprocess_class',None)=='eye_calibration_drift' else _df(before); a=_df(after)
    bx=_first_col(b,['observed_x','x','gaze_x'],'observed x'); by=_first_col(b,['observed_y','y','gaze_y'],'observed y'); rx=_first_col(b,['reference_x','target_x','expected_x'],'reference x'); ry=_first_col(b,['reference_y','target_y','expected_y'],'reference y'); ax=_first_col(a,['x_recalibrated','gaze_x_recalibrated','observed_x_recalibrated','corrected_x'],'corrected x'); ay=_first_col(a,['y_recalibrated','gaze_y_recalibrated','observed_y_recalibrated','corrected_y'],'corrected y')
    refx=pd.to_numeric(b[rx],errors='coerce').to_numpy(float); refy=pd.to_numeric(b[ry],errors='coerce').to_numpy(float); be=np.sqrt((pd.to_numeric(b[bx],errors='coerce').to_numpy(float)-refx)**2+(pd.to_numeric(b[by],errors='coerce').to_numpy(float)-refy)**2); ae=np.sqrt((pd.to_numeric(a[ax],errors='coerce').to_numpy(float)-refx)**2+(pd.to_numeric(a[ay],errors='coerce').to_numpy(float)-refy)**2)
    br=math.sqrt(np.nanmean(be**2)); ar=math.sqrt(np.nanmean(ae**2)); imp=(br-ar)/max(br,1e-12); passed=ar<br if minimum_improvement is None else imp>=minimum_improvement
    return _result('eye_recalibration_audit',errors=pd.DataFrame({'before':be,'after':ae}),summary=pd.DataFrame([{'before_rmse':br,'after_rmse':ar,'relative_improvement':imp,'passed':passed}]),minimum_improvement=minimum_improvement,status='Recalibration improved spatial accuracy.' if passed else 'Recalibration did not meet the improvement criterion.')

def _cal_plot(x,kind,ax=None):
    ax=_ax(ax)
    if getattr(x,'eyeprocess_class',None)=='eye_calibration_drift':
        d=x['observations']; s=x['summary']
        if kind=='vector_field': ax.quiver(d.reference_x,d.reference_y,d.observed_x-d.reference_x,d.observed_y-d.reference_y,angles='xy',scale_units='xy',scale=1)
        elif kind=='error_ellipses': ax.scatter(d.error_x,d.error_y); ax.axhline(0,ls='--'); ax.axvline(0,ls='--')
        elif kind=='drift_over_time': ax.plot(s.window_id,s.drift_from_first,marker='o')
        else: ax.scatter(d.observed_x,d.observed_y)
        ax.eyeprocess_plot_data=d if kind!='drift_over_time' else s
    else:
        d=x['errors']; ax.boxplot([d.before,d.after],tick_labels=['before','after']); ax.eyeprocess_plot_data=d
    ax.set_title(kind.replace('_',' ').title()); return ax

def plot_calibration_vector_field(x,**kwargs): return _cal_plot(x,'vector_field',**kwargs)
def plot_calibration_error_ellipses(x,**kwargs): return _cal_plot(x,'error_ellipses',**kwargs)
def plot_drift_over_time(x,**kwargs): return _cal_plot(x,'drift_over_time',**kwargs)
def plot_recalibration_before_after(x,**kwargs): return _cal_plot(x,'before_after',**kwargs)
def plot_screen_coverage(x,**kwargs): return _cal_plot(x,'screen_coverage',**kwargs)

def fit_process_gstudy(x,metric,facets=('person','item','session','device'),design='crossed'):
    d=_df(x); _req(d,[metric]); facets=[f for f in facets if f in d]
    if not facets: raise EyeProcessValidationError('At least one facet column is required.')
    vals=pd.to_numeric(d[metric],errors='coerce'); data=d.loc[np.isfinite(vals),[metric]+facets].copy(); data[metric]=pd.to_numeric(data[metric],errors='coerce'); total=float(data[metric].var(ddof=1)) if len(data)>1 else 0.0
    comp={f:max(0.0,float(data.groupby(f)[metric].mean().var(ddof=1)) if data[f].nunique()>1 else 0.0) for f in facets}; residual=max(0.0,total-sum(comp.values())); names=list(comp)+['residual']; values=list(comp.values())+[residual]; denom=max(sum(values),1e-12); vc=pd.DataFrame({'component':names,'variance':values,'proportion':np.asarray(values)/denom}); counts={f:int(data[f].nunique()) for f in facets}; person='person' if 'person' in comp else facets[0]
    return _result('eye_process_gstudy',data=data,metric=metric,facets=facets,person_facet=person,design=design,variance_components=vc,counts=counts,summary=vc,status='Generalizability-study variance components estimated with a dependency-free method-of-moments approximation.')

def process_variance_components(x): return x['variance_components'].copy()
def design_process_dstudy(gstudy,persons=None,items=tuple(range(5,51,5)),sessions=tuple(range(1,6)),devices=(1,)):
    vc=dict(zip(gstudy['variance_components'].component,gstudy['variance_components'].variance)); pv=vc.get(gstudy['person_facet'],0); iv=vc.get('item',0); sv=vc.get('session',0); dv=vc.get('device',0); rv=vc.get('residual',0)
    grid=pd.MultiIndex.from_product([np.atleast_1d(items).astype(int),np.atleast_1d(sessions).astype(int),np.atleast_1d(devices).astype(int)],names=['items','sessions','devices']).to_frame(index=False); grid['relative_error']=iv/grid['items'].clip(lower=1)+sv/grid['sessions'].clip(lower=1)+rv/(grid['items']*grid['sessions']*grid['devices']).clip(lower=1); grid['absolute_error']=grid.relative_error+dv/grid.devices.clip(lower=1); grid['relative_dependability']=pv/np.maximum(pv+grid.relative_error,1e-12); grid['absolute_dependability']=pv/np.maximum(pv+grid.absolute_error,1e-12)
    return _result('eye_process_dstudy',gstudy=gstudy,design_grid=grid,persons=persons,summary=grid,status='Prospective D-study dependability coefficients computed.')
def _icc(values,groups):
    d=pd.DataFrame({'v':pd.to_numeric(pd.Series(values),errors='coerce'),'g':pd.Series(groups)}).dropna(); levels=d.g.unique();
    if len(levels)<2:return math.nan
    means=d.groupby('g').v.mean(); counts=d.groupby('g').size(); grand=d.v.mean(); between=float(np.sum(counts.loc[means.index]*(means-grand)**2)/max(1,len(levels)-1)); within=float(np.sum((d.v-d.g.map(means))**2)/max(1,len(d)-len(levels))); k=float(counts.mean()); return (between-within)/max(between+(k-1)*within,1e-12)
def audit_process_reliability(x,metrics,method='icc',person_col='person_id',item_col='item_id',draws=250):
    d=_df(x); metrics=[metrics] if isinstance(metrics,str) else list(metrics); _req(d,[person_col]+metrics); rows=[]
    for m in metrics:
        if method=='icc': est=_icc(d[m],d[person_col])
        elif method=='gtheory':
            facets=[f for f in [person_col,item_col] if f in d]; gs=fit_process_gstudy(d,m,facets=facets); est=float(design_process_dstudy(gs,items=[max(1,d[item_col].nunique()) if item_col in d else 1],sessions=[1])['design_grid'].relative_dependability.iloc[0])
        elif method=='split_half':
            _req(d,[item_col]); levels=list(pd.unique(d[item_col])); first=set(levels[::2]); a=d[d[item_col].isin(first)].groupby(person_col)[m].mean(); b=d[~d[item_col].isin(first)].groupby(person_col)[m].mean(); both=pd.concat([a,b],axis=1).dropna(); r=float(both.corr().iloc[0,1]) if len(both)>1 else math.nan; est=2*r/(1+r) if np.isfinite(r) and r!=-1 else math.nan
        else:
            rng=np.random.default_rng(20260807); levels=list(pd.unique(d[person_col])); boots=[]
            for _ in range(int(draws)):
                chunks=[]
                for i,g in enumerate(rng.choice(levels,len(levels),replace=True)): c=d[d[person_col]==g].copy(); c[person_col]=f'boot_{i}'; chunks.append(c)
                bd=pd.concat(chunks,ignore_index=True); boots.append(_icc(bd[m],bd[person_col]))
            est=_mean(boots)
        label='limited' if not np.isfinite(est) or est<.5 else ('moderate' if est<.75 else ('good' if est<.9 else 'excellent')); rows.append({'metric':m,'method':method,'estimate':est,'interpretation':label})
    summary=pd.DataFrame(rows); return _result('eye_process_reliability_audit',data=d,summary=summary,method=method,status='Process reliability audit completed.')
def _rel_plot(x,kind,ax=None):
    ax=_ax(ax)
    if getattr(x,'eyeprocess_class',None)=='eye_process_gstudy': d=x['variance_components']; ax.bar(d.component,d.variance)
    elif getattr(x,'eyeprocess_class',None)=='eye_process_dstudy': d=x['design_grid']; ax.plot(d['items'],d['absolute_dependability'],marker='o')
    else: d=x['summary']; ax.bar(d.metric,d.estimate)
    ax.set_title(kind.replace('_',' ').title()); ax.eyeprocess_plot_data=d.copy(); return ax
def plot_variance_components(x,**kw): return _rel_plot(x,'variance_components',**kw)
def plot_dependability_surface(x,**kw): return _rel_plot(x,'dependability_surface',**kw)
def plot_reliability_by_metric(x,**kw): return _rel_plot(x,'reliability_by_metric',**kw)
def plot_session_stability(x,**kw): return _rel_plot(x,'session_stability',**kw)
def plot_item_sampling_reliability(x,**kw): return _rel_plot(x,'item_sampling_reliability',**kw)

__all__=[n for n in globals() if not n.startswith('_') and n not in {'math','np','pd','Any','Callable','Mapping','Sequence','EyeResult','EyeProcessValidationError'}]

def plot_eye_process_uncertainty(x,type='waterfall',metric=None,ax=None): return _unc_plot(x,type,metric=metric,ax=ax)
def plot_eye_process_uncertainty_propagation(x,type='distribution',ax=None):
    ax=_ax(ax); d=pd.DataFrame({'draw':np.asarray(x['draws'],float)})
    if type=='sensitivity': ax.plot(np.sort(d['draw']))
    else: ax.hist(d['draw'].dropna())
    ax.eyeprocess_plot_data=d; ax.set_title('Process uncertainty propagation'); return ax
def plot_eye_uncertainty_budget_comparison(x,type='comparison',ax=None):
    ax=_ax(ax); d=x['summary'].copy(); piv=d.pivot(index='source',columns='budget',values='variance_share').fillna(0); piv.plot.bar(ax=ax); ax.eyeprocess_plot_data=d; return ax
def plot_eye_calibration_drift(x,type='drift_over_time',ax=None): return _cal_plot(x,type,ax=ax)
def plot_eye_recalibration_audit(x,type='before_after',ax=None): return _cal_plot(x,type,ax=ax)
def plot_eye_process_gstudy(x,type='variance_components',ax=None): return _rel_plot(x,type,ax=ax)
def plot_eye_process_dstudy(x,type='dependability_surface',ax=None): return _rel_plot(x,type,ax=ax)
def plot_eye_process_reliability_audit(x,type='reliability_by_metric',ax=None): return _rel_plot(x,type,ax=ax)

__all__=[n for n in globals() if not n.startswith('_') and n not in {'math','np','pd','Any','Callable','Mapping','Sequence','EyeResult','EyeProcessValidationError'}]
