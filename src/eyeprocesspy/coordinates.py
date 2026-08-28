from __future__ import annotations
import numpy as np
import pandas as pd
from .dataset import _assert_eye_dataset, add_provenance
from .schema import standardize_eye_table
from .exceptions import EyeProcessCoordinateError

_SUPPORTED={"display_normalized_top_left","display_pixels_top_left","surface_normalized_bottom_left","world_camera_pixels","reference_image_pixels"}

def register_coordinate_space(x, space, overwrite=False):
    _assert_eye_dataset(x)
    if not isinstance(space,pd.DataFrame): raise TypeError("`space` must be a pandas DataFrame.")
    out=x.copy(); space=standardize_eye_table(space,'coordinate_spaces'); ids=set(space['coordinate_space_id'].dropna().astype(str)); existing=set(out['coordinate_spaces']['coordinate_space_id'].dropna().astype(str)); dup=sorted(ids&existing)
    if dup and not overwrite: raise EyeProcessCoordinateError(f"Coordinate-space id already exists: {', '.join(dup)}.")
    if overwrite and ids: out['coordinate_spaces']=out['coordinate_spaces'][~out['coordinate_spaces']['coordinate_space_id'].astype(str).isin(ids)]
    out['coordinate_spaces']=standardize_eye_table(pd.concat([out['coordinate_spaces'],space],ignore_index=True,sort=False),'coordinate_spaces')
    return add_provenance(out,'register_coordinate_space','coordinate_spaces',','.join(sorted(ids)))

def coordinate_space(x,id):
    _assert_eye_dataset(x); ids=[id] if isinstance(id,str) else list(id); d=x['coordinate_spaces']; rows=[]
    for v in ids:
        m=d[d['coordinate_space_id']==v]
        if m.empty: raise EyeProcessCoordinateError(f"Unknown coordinate-space id: {v}.")
        rows.append(m.iloc[[0]])
    return pd.concat(rows,ignore_index=True)

def _convert_xy(x,y,from_,to,from_width=np.nan,from_height=np.nan,to_width=np.nan,to_height=np.nan,clip=False):
    if from_ not in _SUPPORTED or to not in _SUPPORTED: raise EyeProcessCoordinateError("Automatic conversion is limited to supported 2D coordinate spaces.")
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    if from_ in {"display_pixels_top_left","world_camera_pixels","reference_image_pixels"}:
        if not np.isfinite(from_width) or not np.isfinite(from_height): raise EyeProcessCoordinateError("Source width and height are required for pixel conversion.")
        xn=x/from_width; yn=y/from_height
    elif from_=="surface_normalized_bottom_left": xn=x; yn=1-y
    else: xn=x; yn=y
    if to in {"display_pixels_top_left","world_camera_pixels","reference_image_pixels"}:
        if not np.isfinite(to_width) or not np.isfinite(to_height): raise EyeProcessCoordinateError("Destination width and height are required for pixel conversion.")
        xo=xn*to_width; yo=yn*to_height
    elif to=="surface_normalized_bottom_left": xo=xn; yo=1-yn
    else: xo=xn; yo=yn
    if clip:
        if to in {"display_normalized_top_left","surface_normalized_bottom_left"}: xo=np.clip(xo,0,1); yo=np.clip(yo,0,1)
        else: xo=np.clip(xo,0,to_width); yo=np.clip(yo,0,to_height)
    return pd.DataFrame({'x':xo,'y':yo})

def convert_coordinates(x, from_, to, components=("gaze_samples","episodes","aoi_geometry"), clip=False, overwrite=False):
    _assert_eye_dataset(x); out=x.copy(); fr=coordinate_space(out,from_).iloc[0]; tr=coordinate_space(out,to).iloc[0]; comps=[components] if isinstance(components,str) else list(components)
    for comp in comps:
        d=out[comp].copy()
        if d.empty or 'coordinate_space_id' not in d: continue
        idx=d.index[d['coordinate_space_id']==from_];
        if not len(idx): continue
        def cv(xc,yc): return _convert_xy(d.loc[idx,xc],d.loc[idx,yc],fr.space_type,tr.space_type,float(fr.width),float(fr.height),float(tr.width),float(tr.height),clip)
        if comp=='gaze_samples':
            xy=cv('gaze_x','gaze_y')
            if overwrite:
                d.loc[idx,'gaze_x']=xy.x.to_numpy(); d.loc[idx,'gaze_y']=xy.y.to_numpy(); d.loc[idx,'coordinate_space_id']=to
            else:
                cp=d.loc[idx].copy(); cp['sample_id']=cp['sample_id'].astype(str)+'_'+str(to); cp['gaze_x']=xy.x.to_numpy(); cp['gaze_y']=xy.y.to_numpy(); cp['coordinate_space_id']=to; d=pd.concat([d,cp],ignore_index=True)
        elif comp=='episodes':
            for xc,yc in [('centroid_x','centroid_y'),('start_x','start_y'),('end_x','end_y')]: xy=cv(xc,yc); d.loc[idx,xc]=xy.x.to_numpy(); d.loc[idx,yc]=xy.y.to_numpy()
            d.loc[idx,'coordinate_space_id']=to
        elif comp=='aoi_geometry':
            xy=cv('x','y'); d.loc[idx,'x']=xy.x.to_numpy(); d.loc[idx,'y']=xy.y.to_numpy()
            if all(np.isfinite([fr.width,fr.height,tr.width,tr.height])): d.loc[idx,'width']=pd.to_numeric(d.loc[idx,'width'],errors='coerce')/float(fr.width)*float(tr.width); d.loc[idx,'height']=pd.to_numeric(d.loc[idx,'height'],errors='coerce')/float(fr.height)*float(tr.height)
            d.loc[idx,'coordinate_space_id']=to
        out[comp]=standardize_eye_table(d,comp)
    return add_provenance(out,'convert_coordinates',','.join(comps),f"{from_} -> {to}; clip={str(bool(clip)).upper()}; overwrite={str(bool(overwrite)).upper()}",reversible=not clip)

def audit_coordinate_spaces(x):
    _assert_eye_dataset(x); used=[]
    for n in ['gaze_samples','episodes','aoi_geometry']:
        if 'coordinate_space_id' in x[n]: used.extend(x[n]['coordinate_space_id'].dropna().tolist())
    used=list(dict.fromkeys(used)); reg=set(x['coordinate_spaces']['coordinate_space_id'].dropna())
    rows=[]
    for cid in used:
        rows.append(dict(coordinate_space_id=cid,registered=cid in reg,n_gaze=int((x['gaze_samples']['coordinate_space_id']==cid).sum()),n_episodes=int((x['episodes']['coordinate_space_id']==cid).sum()),n_aoi_geometry=int((x['aoi_geometry']['coordinate_space_id']==cid).sum()),status='ok' if cid in reg else 'error'))
    return pd.DataFrame(rows)
