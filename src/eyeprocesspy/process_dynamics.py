from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd
from .irt import EyeResult
from .exceptions import EyeProcessValidationError


def _result(cls: str, **kw): return EyeResult(kw, eyeprocess_class=cls)
def _df(x,name='data'):
    if isinstance(x,pd.DataFrame): return x.copy()
    try:return pd.DataFrame(x)
    except Exception as e: raise EyeProcessValidationError(f'{name} must be coercible to a data frame.') from e
def _ax(ax=None):
    import matplotlib.pyplot as plt
    return plt.subplots()[1] if ax is None else ax

def _run_lengths(vals):
    out=[]; c=0
    for v in vals:
        if bool(v): c+=1
        elif c: out.append(c); c=0
    if c:out.append(c)
    return out
def _diag_lengths(M,vertical=False):
    M=np.asarray(M,bool); out=[]
    if vertical:
        for j in range(M.shape[1]): out+=_run_lengths(M[:,j])
    else:
        for k in range(-M.shape[0]+1,M.shape[1]): out+=_run_lengths(np.diag(M,k=k))
    return np.asarray(out,int)
def _rec_matrix(a,b=None,representation='coordinates',radius=None):
    if b is None:b=a
    if representation=='aoi': return (np.asarray(a,dtype=str)[:,None]==np.asarray(b,dtype=str)[None,:]).astype(int)
    A=np.asarray(a,float); B=np.asarray(b,float); A=A[:,None] if A.ndim==1 else A; B=B[:,None] if B.ndim==1 else B
    dist=np.sqrt(np.nansum((A[:,None,:]-B[None,:,:])**2,axis=2)); positive=dist[np.isfinite(dist)&(dist>0)]; r=float(np.quantile(positive,.1)) if radius is None and positive.size else (float(radius) if radius is not None else 0.0); return (dist<=r).astype(int)
def recurrence_features(x,minimum_line=2):
    M=np.asarray(x['matrix'] if isinstance(x,dict) and 'matrix' in x else x); diag=_diag_lengths(M); vert=_diag_lengths(M,True); recurrent=int(np.nansum(M>0)); det=float(diag[diag>=minimum_line].sum()/recurrent) if recurrent else 0.; lam=float(vert[vert>=minimum_line].sum()/recurrent) if recurrent else 0.; vu=vert[vert>=minimum_line]; du=diag[diag>=minimum_line]
    if len(du):
        _,cnt=np.unique(du,return_counts=True); p=cnt/cnt.sum(); ent=float(-np.sum(p*np.log(p)))
    else:ent=0.
    return pd.DataFrame([{'recurrence_rate':float(np.nanmean(M>0)),'determinism':det,'laminarity':lam,'trapping_time':float(vu.mean()) if len(vu) else 0.,'diagonal_entropy':ent}])
def gaze_recurrence(x,representation='coordinates',x_col='x',y_col='y',aoi_col='aoi',radius=None):
    if isinstance(x,pd.DataFrame): series=x[aoi_col].astype(str).to_numpy() if representation=='aoi' else x[[x_col,y_col]].to_numpy(float)
    else: series=np.asarray(x)
    if representation=='velocity': series=np.vstack([np.zeros((1,series.shape[1])),np.diff(series,axis=0)])
    M=_rec_matrix(series,representation='aoi' if representation=='aoi' else 'coordinates',radius=radius); return _result('eye_recurrence',matrix=M,series=series,representation=representation,radius=radius,summary=recurrence_features(M),status='Gaze recurrence matrix calculated.')
def cross_recurrence(x,y,channels='gaze_pupil',radius=None):
    def std(v):
        a=np.asarray(v,float); a=a[:,None] if a.ndim==1 else a
        if a.size==0:raise EyeProcessValidationError('Cross-recurrence inputs must contain observations and at least one channel.')
        mu=np.nanmean(a,axis=0); sd=np.nanstd(a,axis=0); sd=np.where(sd>0,sd,1); return (a-mu)/sd
    A,B=std(x),std(y); k=min(A.shape[1],B.shape[1]); M=_rec_matrix(A[:,:k],B[:,:k],radius=radius); return _result('eye_cross_recurrence',matrix=M,x=x,y=y,channels=channels,radius=radius,summary=recurrence_features(M),status='Cross-recurrence matrix calculated.')
def windowed_recurrence(x,window,step):
    series=x['series'] if getattr(x,'eyeprocess_class',None)=='eye_recurrence' else np.asarray(x); n=len(series); window=int(window); step=int(step)
    if window<2 or step<1 or n<2:raise EyeProcessValidationError('Invalid window/step or insufficient observations.')
    window=min(window,n); starts=range(0,max(1,n-window+1),step); rows=[]
    for s in starts:
        sub=series[s:min(n,s+window)]; M=_rec_matrix(sub,representation='aoi' if np.asarray(series).dtype.kind in {'U','S','O'} and np.asarray(series).ndim==1 else 'coordinates'); f=recurrence_features(M).iloc[0].to_dict(); rows.append({'start':s+1,'end':min(n,s+window),**f})
    summary=pd.DataFrame(rows); return _result('eye_windowed_recurrence',series=series,window=window,step=step,summary=summary,status='Windowed recurrence features calculated.')
def _rec_plot(x,kind,ax=None):
    ax=_ax(ax)
    if getattr(x,'eyeprocess_class',None)=='eye_windowed_recurrence': d=x['summary']; ax.plot(d.start,d.recurrence_rate,marker='o')
    else:
        M=np.asarray(x['matrix']); d=pd.DataFrame(M)
        if kind in {'matrix','crossmodal'}: ax.imshow(M,aspect='auto',origin='lower')
        elif kind=='diagonal_profile': ax.plot([np.mean(np.diag(M,k)) for k in range(-M.shape[0]+1,M.shape[1])])
        else: ax.bar(np.arange(M.shape[0]),M.sum(axis=1))
    ax.set_title(kind.replace('_',' ').title()); ax.eyeprocess_plot_data=d; return ax
def plot_recurrence_matrix(x,**kw):return _rec_plot(x,'matrix',**kw)
def plot_windowed_recurrence(x,**kw):return _rec_plot(x,'windowed',**kw)
def plot_diagonal_recurrence_profile(x,**kw):return _rec_plot(x,'diagonal_profile',**kw)
def plot_crossmodal_recurrence(x,**kw):return _rec_plot(x,'crossmodal',**kw)
def plot_recurrence_network(x,**kw):return _rec_plot(x,'network',**kw)

# Point process

def _expand_range(v):
    lo,hi=np.nanmin(v),np.nanmax(v)
    if lo==hi:
        pad=max(abs(lo)*.01,.5); lo-=pad;hi+=pad
    return lo,hi
def fit_fixation_point_process(x,spatial_covariates=None,temporal_covariates=None,interaction='none',x_col='x',y_col='y',time_col='time',grid_size=20):
    d=_df(x); 
    if len(d)<5 or x_col not in d or y_col not in d:raise EyeProcessValidationError('At least five fixation observations with coordinates are required.')
    work=d.copy(); work['.x']=pd.to_numeric(work[x_col],errors='coerce'); work['.y']=pd.to_numeric(work[y_col],errors='coerce'); work['.time']=pd.to_numeric(work[time_col],errors='coerce') if time_col in work else np.arange(1,len(work)+1); work=work.dropna(subset=['.x','.y','.time'])
    if len(work)<5:raise EyeProcessValidationError('At least five complete fixation observations are required.')
    gs=int(grid_size); xr=_expand_range(work['.x'].to_numpy()); yr=_expand_range(work['.y'].to_numpy()); xb=np.linspace(*xr,gs+1); yb=np.linspace(*yr,gs+1); work['.x_bin']=np.clip(np.digitize(work['.x'],xb,right=True),1,gs); work['.y_bin']=np.clip(np.digitize(work['.y'],yb,right=True),1,gs)
    grid=work.groupby(['.x_bin','.y_bin'],as_index=False).size().rename(columns={'size':'count','.x_bin':'x_bin','.y_bin':'y_bin'}); grid['x_center']=[(xb[int(i)-1]+xb[int(i)])/2 for i in grid.x_bin]; grid['y_center']=[(yb[int(i)-1]+yb[int(i)])/2 for i in grid.y_bin]
    if interaction=='self_exciting':
        o=work.sort_values('.time'); lag=np.r_[np.nan,np.sqrt(np.diff(o['.x'])**2+np.diff(o['.y'])**2)]; scale=np.nanmean(lag); scale=scale if np.isfinite(scale) and scale>0 else 1; hist=pd.DataFrame({'x_bin':o['.x_bin'],'y_bin':o['.y_bin'],'history':np.exp(-lag/scale)}).groupby(['x_bin','y_bin'],as_index=False).history.mean(); grid=grid.merge(hist,on=['x_bin','y_bin'],how='left'); grid['history']=grid.history.fillna(0)
    requested=[]
    for c in (list(spatial_covariates or [])+list(temporal_covariates or [])):
        if c in work and c not in requested: requested.append(c)
    cmap=[]
    for j,c in enumerate(requested,1):
        internal=f'covariate_{j}'; vals=pd.to_numeric(work[c],errors='coerce'); tab=pd.DataFrame({'x_bin':work['.x_bin'],'y_bin':work['.y_bin'],internal:vals}).groupby(['x_bin','y_bin'],as_index=False)[internal].mean(); grid=grid.merge(tab,on=['x_bin','y_bin'],how='left'); grid[internal]=grid[internal].fillna(grid[internal].mean() if np.isfinite(grid[internal].mean()) else 0); cmap.append({'source':c,'internal':internal})
    features=[np.ones(len(grid)),grid.x_center,grid.y_center,grid.x_center**2,grid.y_center**2,grid.x_center*grid.y_center]
    if interaction=='self_exciting':features.append(grid.history)
    for c in [m['internal'] for m in cmap]:features.append(grid[c])
    X=np.column_stack(features); y=np.log(np.maximum(grid['count'].to_numpy(float),1e-6)); beta=np.linalg.lstsq(X,y,rcond=None)[0]; expected=np.exp(np.clip(X@beta,-30,30)); grid['expected']=expected; grid['residual']=grid['count']-expected; cmapdf=pd.DataFrame(cmap,columns=['source','internal']); defaults={c:float(grid[c].mean()) for c in cmapdf.internal} if len(cmapdf) else {}
    summary=pd.DataFrame({'Estimate':beta}); model={'coef':beta,'interaction':interaction,'covariates':list(cmapdf.internal)}
    return _result('eye_fixation_point_process',model=model,data=work,grid=grid,x_breaks=xb,y_breaks=yb,interaction=interaction,spatial_covariates=spatial_covariates,temporal_covariates=temporal_covariates,covariate_map=cmapdf,predictor_defaults=defaults,summary=summary,status='Fixation intensity fitted as a gridded Poisson point-process approximation.')
def fit_marked_gaze_process(x,marks=('duration','pupil','saccade_amplitude'),x_col='x',y_col='y',time_col='time'):
    d=_df(x); marks=[m for m in marks if m in d]
    if len(d)<5 or not marks:raise EyeProcessValidationError('No requested mark columns are available.')
    xx=pd.to_numeric(d[x_col],errors='coerce').to_numpy(float); yy=pd.to_numeric(d[y_col],errors='coerce').to_numpy(float); tt=pd.to_numeric(d[time_col],errors='coerce').to_numpy(float) if time_col in d else np.arange(len(d)); X=np.column_stack([np.ones(len(d)),xx,yy,tt,xx*yy]); models={}; rows=[]
    for m in marks:
        y=pd.to_numeric(d[m],errors='coerce').to_numpy(float); ok=np.isfinite(y)&np.all(np.isfinite(X),axis=1); b=np.linalg.lstsq(X[ok],y[ok],rcond=None)[0]; models[m]={'coef':b}; rows.extend({'mark':m,'term':f'b{i}','Estimate':v} for i,v in enumerate(b))
    return _result('eye_marked_gaze_process',models=models,data=d,marks=marks,summary=pd.DataFrame(rows),status='Marked gaze-process models fitted.')
def predict_fixation_intensity(model,new_stimulus=None):
    if getattr(model,'eyeprocess_class',None)!='eye_fixation_point_process':raise EyeProcessValidationError('model must be an eye_fixation_point_process object.')
    d=(model['grid'].copy() if new_stimulus is None else _df(new_stimulus));
    if 'x_center' not in d: d['x_center']=pd.to_numeric(d['x'] if 'x' in d else d['gaze_x'],errors='coerce')
    if 'y_center' not in d: d['y_center']=pd.to_numeric(d['y'] if 'y' in d else d['gaze_y'],errors='coerce')
    feats=[np.ones(len(d)),d.x_center,d.y_center,d.x_center**2,d.y_center**2,d.x_center*d.y_center]
    if model['interaction']=='self_exciting': feats.append(pd.to_numeric(d['history'],errors='coerce').fillna(0).to_numpy() if 'history' in d else np.zeros(len(d)))
    for row in model['covariate_map'].itertuples(index=False):
        if row.source in d:d[row.internal]=pd.to_numeric(d[row.source],errors='coerce')
        elif row.internal not in d:d[row.internal]=model['predictor_defaults'].get(row.internal,0)
        feats.append(pd.to_numeric(d[row.internal],errors='coerce').fillna(model['predictor_defaults'].get(row.internal,0)).to_numpy())
    X=np.column_stack(feats); d['predicted_intensity']=np.exp(np.clip(X@model['model']['coef'],-30,30)); return d
def diagnose_gaze_point_process(model):
    g=model['grid']; pear=(g['count']-g['expected'])/np.sqrt(np.maximum(g['expected'],1e-8)); corr=float(np.corrcoef(g['count'],g['expected'])[0,1]) if len(g)>1 else math.nan; over=float(np.sum(pear**2)/max(1,len(g)-len(model['model']['coef']))); summary=pd.DataFrame([{'mean_pearson':float(np.mean(pear)),'sd_pearson':float(np.std(pear,ddof=1)) if len(pear)>1 else math.nan,'overdispersion':over,'correlation_observed_expected':corr}]); return _result('eye_gaze_point_process_diagnostics',model=model,residuals=np.asarray(pear),summary=summary,status='Point-process diagnostics calculated.')
def _pp_plot(x,kind,ax=None):
    ax=_ax(ax); g=x['grid'] if 'grid' in x else x['model']['grid'];
    if kind=='observed_expected':ax.scatter(g.expected,g['count'])
    elif kind in {'intensity','covariate_surface'}:ax.scatter(g.x_center,g.y_center,c=g.expected)
    elif kind=='excitation':ax.plot(g.get('history',pd.Series(np.zeros(len(g)))))
    else:ax.scatter(g.x_center,g.y_center,s=20+50*np.abs(g.residual)/max(np.max(np.abs(g.residual)),1e-9))
    ax.eyeprocess_plot_data=g.copy();ax.set_title(kind.replace('_',' ').title());return ax
def plot_fixation_intensity(x,**kw):return _pp_plot(x,'intensity',**kw)
def plot_spatial_residuals(x,**kw):return _pp_plot(x,'spatial_residuals',**kw)
def plot_temporal_excitation_kernel(x,**kw):return _pp_plot(x,'excitation',**kw)
def plot_covariate_effect_surface(x,**kw):return _pp_plot(x,'covariate_surface',**kw)
def plot_observed_expected_fixations(x,**kw):return _pp_plot(x,'observed_expected',**kw)

# Scanpaths

def _edit(a,b):
    a=list(a);b=list(b); dp=np.zeros((len(a)+1,len(b)+1),int);dp[:,0]=np.arange(len(a)+1);dp[0,:]=np.arange(len(b)+1)
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):dp[i,j]=min(dp[i-1,j]+1,dp[i,j-1]+1,dp[i-1,j-1]+(a[i-1]!=b[j-1]))
    return float(dp[-1,-1])
def _paths(x,id_col='person_id',aoi_col='aoi',x_col='x',y_col='y'):
    if isinstance(x,list):return x
    if isinstance(x,dict) and not isinstance(x,pd.DataFrame):return list(x.values())
    d=_df(x); groups=[]
    for _,z in d.groupby(id_col):groups.append(z[aoi_col].astype(str).tolist() if aoi_col in d else z[[x_col,y_col]].to_numpy(float))
    return groups
def _dist(a,b,distance='edit'):
    if isinstance(a,(list,tuple,np.ndarray)) and np.asarray(a).ndim==1 and np.asarray(a).dtype.kind in {'U','S','O'}:return _edit(a,b)
    A=np.asarray(a,float);B=np.asarray(b,float)
    if distance=='transport':
        q=np.linspace(0,1,25);return float(np.sum(np.abs(np.quantile(A[:,0],q)-np.quantile(B[:,0],q)))+np.sum(np.abs(np.quantile(A[:,1],q)-np.quantile(B[:,1],q))))
    # lightweight DTW
    D=np.full((len(A)+1,len(B)+1),np.inf);D[0,0]=0
    for i in range(1,len(A)+1):
        for j in range(1,len(B)+1):D[i,j]=np.linalg.norm(A[i-1]-B[j-1])+min(D[i-1,j],D[i,j-1],D[i-1,j-1])
    return float(D[-1,-1])
def representative_scanpath(x,method='medoid',id_col='person_id',aoi_col='aoi',x_col='x',y_col='y',distance='multimatch'):
    if isinstance(x,dict) and not isinstance(x,pd.DataFrame): names=list(x.keys()); paths=list(x.values())
    else: paths=_paths(x,id_col,aoi_col,x_col,y_col); names=[str(i+1) for i in range(len(paths))]
    n=len(paths); M=np.zeros((n,n));
    for i in range(n):
        for j in range(i+1,n):M[i,j]=M[j,i]=_dist(paths[i],paths[j],distance)
    med=int(np.argmin(M.mean(axis=1))); rep=paths[med]; sequence=np.asarray(paths[0]).ndim==1
    if method!='medoid':
        if sequence:
            mx=max(map(len,paths)); out=[]
            for k in range(mx):
                vals=[p[k] for p in paths if len(p)>k]; out.append(pd.Series(vals).mode().iloc[0])
            rep=out
        else:
            L=max(2,int(round(np.median([len(p) for p in paths]))));grid=np.linspace(0,1,L); aligned=[]
            for p in paths:
                p=np.asarray(p,float);t=np.linspace(0,1,len(p));aligned.append(np.column_stack([np.interp(grid,t,p[:,0]),np.interp(grid,t,p[:,1])]))
            rep=np.mean(aligned,axis=0)
    summary=pd.DataFrame([{'n_paths':n,'mean_dispersion':float(M[np.triu_indices(n,1)].mean()) if n>1 else 0.,'representative_length':len(rep)}]); return _result('eye_scanpath_representative',paths=paths,path_names=names,representative=rep,medoid_id=names[med],medoid_index=med,distance_matrix=M,method=method,distance=distance,sequence=sequence,summary=summary,status='Representative scanpath derived.')
def scanpath_dispersion(x):
    o=x if getattr(x,'eyeprocess_class',None)=='eye_scanpath_representative' else representative_scanpath(x);M=o['distance_matrix'];m=o['medoid_index'];return pd.DataFrame({'scanpath':o['path_names'],'mean_distance':M.mean(axis=1),'distance_to_representative':M[:,m]})
def compare_scanpath_distributions(x,group,distance='multimatch',permutations=499):
    o=representative_scanpath(x,distance=distance);g=np.asarray(group);M=o['distance_matrix'];tri=np.triu(np.ones_like(M,dtype=bool),1)
    def stat(labels):
        same=(labels[:,None]==labels[None,:])&tri;diff=(labels[:,None]!=labels[None,:])&tri;return float(np.mean(M[diff])-np.mean(M[same])) if same.any() and diff.any() else 0.
    obs=stat(g);rng=np.random.default_rng(20260807);null=np.array([stat(rng.permutation(g)) for _ in range(int(permutations))]);p=(1+np.sum(null>=obs))/(len(null)+1);return _result('eye_scanpath_comparison',representative=o,groups=g,observed=obs,null=null,p_value=float(p),summary=pd.DataFrame([{'statistic':obs,'p_value':p,'permutations':permutations}]),status='Scanpath distributions compared by permutation.')
def bootstrap_representative_scanpath(x,draws=250,seed=20260807):
    o=x if getattr(x,'eyeprocess_class',None)=='eye_scanpath_representative' else representative_scanpath(x);rng=np.random.default_rng(seed); meds=[]
    for _ in range(int(draws)):
        idx=rng.integers(0,len(o['paths']),len(o['paths'])); sampled={f'{o["path_names"][i]}_{j}':o['paths'][i] for j,i in enumerate(idx)};meds.append(representative_scanpath(sampled,method='medoid',distance=o['distance'])['medoid_id'])
    s=pd.Series(meds).value_counts();summary=pd.DataFrame({'medoid':s.index.astype(str),'count':s.values,'probability':s.values/s.sum()});return _result('eye_scanpath_bootstrap',source=o,medoid_frequencies=s,summary=summary,status='Representative scanpath bootstrapped.')
def _sp_plot(x,kind,ax=None):
    ax=_ax(ax);o=x['representative'] if getattr(x,'eyeprocess_class',None)=='eye_scanpath_comparison' else x
    if getattr(x,'eyeprocess_class',None)=='eye_scanpath_comparison': d=pd.DataFrame({'null':x['null']});ax.hist(x['null']);ax.axvline(x['observed'])
    elif getattr(x,'eyeprocess_class',None)=='eye_scanpath_bootstrap':d=x['summary'];ax.bar(d.medoid,d.probability)
    elif kind=='similarity_matrix':d=pd.DataFrame(o['distance_matrix']);ax.imshow(o['distance_matrix'])
    elif kind=='dispersion':d=scanpath_dispersion(o);ax.bar(d.scanpath,d.mean_distance)
    else:
        r=np.asarray(o['representative']);d=pd.DataFrame({'position':np.arange(len(r))})
        if o['sequence']:
            ax.scatter(np.arange(len(r)),np.ones(len(r)));[ax.text(i,1,str(v)) for i,v in enumerate(r)]
        else:ax.plot(r[:,0],r[:,1],marker='o')
    ax.eyeprocess_plot_data=d;ax.set_title(kind.replace('_',' ').title());return ax
def plot_scanpath_atlas(x,**kw):return _sp_plot(x,'atlas',**kw)
def plot_representative_scanpath(x,**kw):return _sp_plot(x,'representative',**kw)
def plot_scanpath_dispersion(x,**kw):return _sp_plot(x,'dispersion',**kw)
def plot_group_scanpath_transport(x,**kw):return _sp_plot(x,'group_transport',**kw)
def plot_scanpath_similarity_matrix(x,**kw):return _sp_plot(x,'similarity_matrix',**kw)

# Episodes

def detect_process_changepoints(x,channels=('gaze_velocity','aoi','pupil','eda'),time_col=None,window=10,threshold_quantile=.9,min_segment=5):
    d=_df(x);window=int(window);min_req=max(10,2*window+1)
    if window<2 or len(d)<min_req:raise EyeProcessValidationError(f'At least {min_req} observations are required.')
    chans=[c for c in channels if c in d and pd.api.types.is_numeric_dtype(d[c])]
    if not chans:raise EyeProcessValidationError('At least one numeric process channel is required.')
    Z=np.column_stack([(pd.to_numeric(d[c],errors='coerce').to_numpy(float)-np.nanmean(pd.to_numeric(d[c],errors='coerce')))/(np.nanstd(pd.to_numeric(d[c],errors='coerce')) or 1) for c in chans]);n=len(d);score=np.full(n,np.nan)
    for i in range(window,n-window+1):score[i]=np.sqrt(np.nansum((np.nanmean(Z[i:i+window],axis=0)-np.nanmean(Z[i-window:i],axis=0))**2))
    th=float(np.nanquantile(score,threshold_quantile));cand=np.where(score>=th)[0];selected=[]
    for c in cand[np.argsort(score[cand])[::-1]]:
        if all(abs(c-s)>=min_segment for s in selected):selected.append(int(c))
    selected=sorted(selected); time=d[time_col].to_numpy() if time_col and time_col in d else np.arange(1,n+1);summary=pd.DataFrame({'index':np.asarray(selected)+1,'time':time[selected] if selected else [],'score':score[selected] if selected else []});return _result('eye_process_changepoints',data=d,channels=chans,score=score,threshold=th,changepoints=np.asarray(selected,dtype=int),summary=summary,status='Multichannel process change points detected.')
def segment_process_episodes(x,**kwargs):
    c=x if getattr(x,'eyeprocess_class',None)=='eye_process_changepoints' else detect_process_changepoints(x,**kwargs);n=len(c['data']);breaks=[0]+list(c['changepoints'])+[n];eid=np.zeros(n,int)
    for i,(a,b) in enumerate(zip(breaks[:-1],breaks[1:]),1):eid[a:b]=i
    data=c['data'].copy();data['episode_id']=eid;levels=sorted(set(eid));summary=pd.DataFrame([{'episode_id':i,'n_observations':int(np.sum(eid==i)),'start_index':int(np.where(eid==i)[0].min()+1),'end_index':int(np.where(eid==i)[0].max()+1)} for i in levels]);return _result('eye_process_episodes',changepoints=c,data=data,summary=summary,status='Process observations segmented into episodes.')
def label_process_episodes(x,rules=None,model=None):
    if getattr(x,'eyeprocess_class',None)!='eye_process_episodes':raise EyeProcessValidationError('x must be an eye_process_episodes object.')
    n=len(x['summary']);defaults=['orientation','initial_encoding','option_inspection','comparison','reconsideration','commitment']
    if callable(model):labels=list(map(str,model(x['summary'],x['data'])))
    elif isinstance(rules,dict) and rules:
        labels=np.array(['unclassified']*n,dtype=object)
        for name,fn in rules.items():labels[np.asarray(fn(x['summary'],x['data']),bool)]=name
        labels=labels.tolist()
    else:
        idx=np.round(np.linspace(0,len(defaults)-1,n)).astype(int);labels=[defaults[i] for i in idx];
        if n>2:labels[-1]='commitment'
    out=_result('eye_process_episodes',**dict(x));out['summary']=x['summary'].copy();out['data']=x['data'].copy();out['summary']['episode_label']=labels;mapper=dict(zip(out['summary'].episode_id,labels));out['data']['episode_label']=out['data'].episode_id.map(mapper);out['status']='Process episodes labelled using transparent rules or an external classifier.';return out
def compare_episode_structure(x,group):
    if getattr(x,'eyeprocess_class',None)=='eye_process_episodes':
        d=x['data'].copy();
        if len(group)!=len(d):raise EyeProcessValidationError('group must align with episode-level observations.')
        d['.group']=group
    else:raise EyeProcessValidationError('x must be an episode object or list.')
    label='episode_label' if 'episode_label' in d else 'episode_id';tab=d.groupby(['.group',label]).size().rename('Freq').reset_index().rename(columns={'.group':'group',label:'episode'});tot=tab.groupby('group').Freq.transform('sum');tab['Freq_total']=tot;tab['proportion']=tab.Freq/np.maximum(tot,1);return _result('eye_episode_comparison',table=tab,summary=tab,status='Episode structures compared across groups.')
def _ep_plot(x,kind,ax=None):
    ax=_ax(ax)
    if getattr(x,'eyeprocess_class',None)=='eye_process_changepoints':d=pd.DataFrame({'score':x['score']});ax.plot(x['score']);ax.axhline(x['threshold'],ls='--')
    elif getattr(x,'eyeprocess_class',None)=='eye_episode_comparison':d=x['table'];p=d.pivot(index='episode',columns='group',values='proportion').fillna(0);p.plot.bar(ax=ax)
    else:
        d=x['summary'];
        if kind in {'waterfall','duration'}:ax.bar(d.episode_id,d.n_observations)
        elif kind=='transition_graph':ax.plot(d.episode_id,np.ones(len(d)),marker='o')
        else:
            ch=x['changepoints']['channels'][0];ax.plot(x['data'][ch]);[ax.axvline(c,ls='--') for c in x['changepoints']['changepoints']]
    ax.eyeprocess_plot_data=d.copy();ax.set_title(kind.replace('_',' ').title());return ax
def plot_process_episodes(x,**kw):return _ep_plot(x,'episodes',**kw)
def plot_changepoint_ribbons(x,**kw):return _ep_plot(x,'ribbons',**kw)
def plot_episode_waterfall(x,**kw):return _ep_plot(x,'waterfall',**kw)
def plot_episode_transition_graph(x,**kw):return _ep_plot(x,'transition_graph',**kw)
def plot_episode_duration_distribution(x,**kw):return _ep_plot(x,'duration',**kw)

__all__=[n for n in globals() if not n.startswith('_') and n not in {'math','np','pd','Any','EyeResult','EyeProcessValidationError'}]

def plot_eye_recurrence(x,type='matrix',ax=None): return _rec_plot(x,type,ax=ax)
def plot_eye_cross_recurrence(x,type='crossmodal',ax=None): return _rec_plot(x,'crossmodal',ax=ax)
def plot_eye_windowed_recurrence(x,type='windowed',metric='recurrence_rate',ax=None):
    ax=_ax(ax); d=x['summary']; m=metric if metric in d else 'recurrence_rate'; ax.plot(d.start,d[m],marker='o'); ax.eyeprocess_plot_data=d.copy(); return ax
def plot_eye_fixation_point_process(x,type='intensity',ax=None): return _pp_plot(x,type,ax=ax)
def plot_eye_gaze_point_process_diagnostics(x,type='diagnostics',ax=None): return _pp_plot(x['model'], 'observed_expected' if type=='observed_expected' else 'spatial_residuals', ax=ax)
def plot_eye_scanpath_representative(x,type='representative',ax=None): return _sp_plot(x,type,ax=ax)
def plot_eye_scanpath_comparison(x,type='group_transport',ax=None): return _sp_plot(x,type,ax=ax)
def plot_eye_scanpath_bootstrap(x,type='stability',ax=None): return _sp_plot(x,type,ax=ax)
def plot_eye_process_changepoints(x,type='ribbons',ax=None): return _ep_plot(x,type,ax=ax)
def plot_eye_process_episodes(x,type='episodes',ax=None): return _ep_plot(x,type,ax=ax)
def plot_eye_episode_comparison(x,type='structure',ax=None): return _ep_plot(x,type,ax=ax)

__all__=[n for n in globals() if not n.startswith('_') and n not in {'math','np','pd','Any','EyeResult','EyeProcessValidationError'}]
