from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd
from .irt import EyeResult
from .exceptions import EyeProcessValidationError
from .pupil_missingness import fit_process_observation_model
from .process_dynamics import cross_recurrence, recurrence_features


def _result(cls: str, **kw): return EyeResult(kw, eyeprocess_class=cls)
def _ax(ax=None):
    import matplotlib.pyplot as plt
    return plt.subplots()[1] if ax is None else ax

def _nodes(value,stage):
    if value is None:return pd.DataFrame(columns=['node_id','label','stage'])
    if isinstance(value,pd.DataFrame) and {'node_id','label'}.issubset(value.columns):
        out=value.copy();
        if 'stage' not in out:out['stage']=stage
        return out
    if isinstance(value,dict): labels=list(value.keys())
    elif isinstance(value,str):labels=[value]
    else:labels=list(map(str,value))
    seen={}; ids=[]
    for lab in labels:
        seen[lab]=seen.get(lab,0)+1; suffix='' if seen[lab]==1 else f'.{seen[lab]-1}';ids.append(f'{stage}::{lab}{suffix}')
    return pd.DataFrame({'node_id':ids,'label':labels,'stage':stage})
def build_evidence_graph(raw_data,transformations=None,metrics=None,models=None,diagnostics=None,decisions=None,edges=None):
    stages={'raw_data':raw_data,'transformations':transformations,'metrics':metrics,'models':models,'diagnostics':diagnostics,'decisions':decisions};nodes=pd.concat([_nodes(v,k) for k,v in stages.items()],ignore_index=True)
    if nodes.empty:raise EyeProcessValidationError('At least one evidence node is required.')
    if edges is None:
        rows=[];active=[k for k,v in stages.items() if v is not None and len(v)]
        for a,b in zip(active[:-1],active[1:]):
            frm=list(nodes.loc[nodes.stage==a,'node_id']);to=list(nodes.loc[nodes.stage==b,'node_id']);parents=frm[:3]
            for t in to:
                for f in parents:rows.append({'from':f,'to':t,'relation':'supports'})
        ed=pd.DataFrame(rows,columns=['from','to','relation'])
    else:
        ed=pd.DataFrame(edges).copy();
        if not {'from','to'}.issubset(ed):raise EyeProcessValidationError('edges must contain from and to.')
        if 'relation' not in ed:ed['relation']='supports'
    return _result('eye_evidence_graph',nodes=nodes,edges=ed,stages=stages,summary=pd.DataFrame([{'nodes':len(nodes),'edges':len(ed),'decisions':int((nodes.stage=='decisions').sum())}]),status='Evidence provenance graph built.')
def _ancestors(edges,node):
    found={node};front={node}
    while True:
        parents=set(edges.loc[edges['to'].isin(front),'from'])-found
        if not parents:break
        found|=parents;front=parents
    return found
def trace_item_decision(graph,item_id):
    n=graph['nodes'];cand=n[(n.node_id.astype(str)==str(item_id))|n.label.astype(str).str.contains(str(item_id),regex=False)];dec=cand[cand.stage=='decisions'];target=(dec if not dec.empty else cand)
    if target.empty:raise EyeProcessValidationError('No matching decision node was found.')
    target_id=target.node_id.iloc[0];keep=_ancestors(graph['edges'],target_id);sn=n[n.node_id.isin(keep)].copy();se=graph['edges'][graph['edges']['from'].isin(keep)&graph['edges']['to'].isin(keep)].copy();return _result('eye_decision_trace',graph=graph,target=target_id,nodes=sn,edges=se,summary=sn,status='Decision evidence path traced.')
def compare_decision_provenance(graph_a,graph_b):
    na=set(graph_a['nodes'].node_id);nb=set(graph_b['nodes'].node_id)
    def keys(e):return set(e.apply(lambda r:f"{r['from']} -> {r['to']} -> {r['relation']}",axis=1))
    ea,eb=keys(graph_a['edges']),keys(graph_b['edges']);addedn=sorted(nb-na);removedn=sorted(na-nb);addede=sorted(eb-ea);removede=sorted(ea-eb);summary=pd.DataFrame({'feature':['nodes_added','nodes_removed','edges_added','edges_removed'],'count':[len(addedn),len(removedn),len(addede),len(removede)]});return _result('eye_provenance_comparison',graph_a=graph_a,graph_b=graph_b,summary=summary,nodes_added=addedn,nodes_removed=removedn,edges_added=addede,edges_removed=removede,status='Decision provenance graphs compared.')
def _cycle(nodes,edges):
    children={n:list(edges.loc[edges['from']==n,'to']) for n in nodes};state={n:0 for n in nodes}
    def visit(n):
        if state[n]==1:return True
        if state[n]==2:return False
        state[n]=1
        for c in children.get(n,[]):
            if c in state and visit(c):return True
        state[n]=2;return False
    return any(visit(n) for n in nodes if state[n]==0)
def audit_evidence_dependencies(graph):
    nodes=list(graph['nodes'].node_id);ed=graph['edges'];mf=sorted(set(ed['from'])-set(nodes));mt=sorted(set(ed['to'])-set(nodes));inc={n:0 for n in nodes};out={n:0 for n in nodes}
    for f,t,*_ in ed[['from','to','relation']].itertuples(index=False,name=None):
        if t in inc: inc[t]+=1
        if f in out: out[f]+=1
    orphan=[n for n in nodes if inc[n]==0 and out[n]==0];cyc=_cycle(nodes,ed) if len(ed) else False;summary=pd.DataFrame([{'missing_source_nodes':len(mf),'missing_target_nodes':len(mt),'orphan_nodes':len(orphan),'has_cycle':cyc,'passed':not mf and not mt and not cyc}]);return _result('eye_evidence_dependency_audit',graph=graph,missing_from=mf,missing_to=mt,orphan=orphan,summary=summary,status='Evidence dependency audit completed.')
def _graph_plot(nodes,edges,title,ax=None):
    ax=_ax(ax);stages=list(pd.unique(nodes.stage));pos={};
    for i,s in enumerate(stages):
        z=nodes[nodes.stage==s];ys=np.linspace(.1,.9,len(z)) if len(z)>1 else [.5]
        for y,row in zip(ys,z.itertuples(index=False)):pos[row.node_id]=(i,y);ax.scatter(i,y,s=180,facecolors='white',edgecolors='black');ax.text(i,y,str(row.label),fontsize=7,ha='center')
    for f,t,*_ in edges[['from','to','relation']].itertuples(index=False,name=None):
        if f in pos and t in pos:
            a=pos[f]; b=pos[t]; ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':.8})
    ax.set_xticks(range(len(stages)),stages,rotation=45,ha='right');ax.set_yticks([]);ax.set_title(title);ax.eyeprocess_plot_data=nodes.copy();return ax
def plot_evidence_graph(x,**kw):return _graph_plot(x['nodes'],x['edges'],'Evidence and decision provenance',**kw)
def plot_item_decision_path(x,**kw):return _graph_plot(x['nodes'],x['edges'],f"Decision evidence path: {x['target']}",**kw)
def plot_metric_dependency_graph(x,**kw):return _graph_plot(x['nodes'],x['edges'],'Metric dependency graph',**kw)
def plot_model_decision_impact(x,**kw):return _graph_plot(x['nodes'],x['edges'],'Model-to-decision impact graph',**kw)
def fit_process_missingness_model(x,observed,predictors,random=('person','item')):return fit_process_observation_model(x,observed,predictors,random=random)
def crossmodal_recurrence_model(x,y,outcome=None,channels='gaze_pupil',radius=None,covariates=None):
    rec=cross_recurrence(x,y,channels=channels,radius=radius);feat=recurrence_features(rec);model=None;data=None
    if outcome is not None:
        yy=np.asarray(outcome,float);data=pd.DataFrame({'outcome':yy}) if covariates is None else pd.concat([pd.DataFrame({'outcome':yy}),pd.DataFrame(covariates).reset_index(drop=True)],axis=1)
        if len(data)!=len(yy):raise EyeProcessValidationError('covariates must align with outcome.')
        for c in feat.columns:data[c]=float(feat[c].iloc[0])
        preds=[c for c in data.columns if c!='outcome' and pd.api.types.is_numeric_dtype(data[c])];X=np.column_stack([np.ones(len(data))]+[pd.to_numeric(data[c],errors='coerce').fillna(0).to_numpy(float) for c in preds]);b=np.linalg.lstsq(X,yy,rcond=None)[0];model={'coef':b,'predictors':preds,'fitted':X@b,'residuals':yy-X@b};summary=pd.DataFrame({'Estimate':b},index=['(Intercept)']+preds)
    else:summary=feat
    return _result('eye_crossmodal_recurrence_model',recurrence=rec,features=feat,model=model,data=data,summary=summary,status='Cross-modal recurrence features calculated.' if model is None else 'Cross-modal recurrence outcome model fitted.')

def plot_crossmodal_recurrence_model(x,type='crossmodal',ax=None):
    if type=='crossmodal':
        from .process_dynamics import plot_crossmodal_recurrence
        return plot_crossmodal_recurrence(x['recurrence'],ax=ax)
    ax=_ax(ax)
    if x['model'] is None:ax.text(.5,.5,'No downstream outcome model was fitted.',ha='center');d=pd.DataFrame()
    elif type=='diagnostics':d=pd.DataFrame({'fitted':x['model']['fitted'],'residual':x['model']['residuals']});ax.scatter(d.fitted,d.residual);ax.axhline(0,ls='--')
    else:d=x['summary'].reset_index(names='term');ax.bar(d.term,d.Estimate);ax.tick_params(axis='x',rotation=90)
    ax.eyeprocess_plot_data=d;return ax

__all__=[n for n in globals() if not n.startswith('_') and n not in {'math','np','pd','Any','EyeResult','EyeProcessValidationError'}]

def plot_eye_evidence_graph(x,type='graph',ax=None): return _graph_plot(x['nodes'],x['edges'],'Evidence and decision provenance',ax=ax)
def plot_eye_decision_trace(x,type='decision_path',ax=None): return _graph_plot(x['nodes'],x['edges'],f"Decision evidence path: {x['target']}",ax=ax)
def plot_eye_provenance_comparison(x,type='comparison',ax=None):
    ax=_ax(ax); d=x['summary']; ax.bar(d.feature,d['count']); ax.tick_params(axis='x',rotation=45); ax.eyeprocess_plot_data=d.copy(); return ax

__all__=[n for n in globals() if not n.startswith('_') and n not in {'math','np','pd','Any','EyeResult','EyeProcessValidationError'}]

def plot_eye_crossmodal_recurrence_model(x,type='crossmodal',ax=None): return plot_crossmodal_recurrence_model(x,type=type,ax=ax)

__all__=[n for n in globals() if not n.startswith('_') and n not in {'math','np','pd','Any','EyeResult','EyeProcessValidationError'}]
