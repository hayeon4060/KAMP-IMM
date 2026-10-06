import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/common.py
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
from final_model import raw, CLIP
from rg3_short_scan import feature_bank

def robust(Xr):
    Xr=np.asarray(Xr,float)
    med=np.nanmedian(Xr,0); q75,q25=np.nanpercentile(Xr,[75,25],0); sd=np.nanstd(Xr,0)
    iqr=np.where(q75-q25>1e-9,q75-q25,np.where(sd>1e-9,sd,1.0))
    return lambda Z: np.nan_to_num(np.clip((np.asarray(Z,float)-med)/iqr,-CLIP,CLIP),nan=0.0)

def maha(Xr, Xl):
    z=robust(Xr); lw=LedoitWolf().fit(z(Xr))
    return np.sqrt(lw.mahalanobis(z(Xl))), np.sqrt(lw.mahalanobis(z(Xr)))

def pct(ref_s, s):
    r=np.sort(np.asarray(ref_s,float)); r=r[np.isfinite(r)]
    s=np.asarray(s,float); out=np.searchsorted(r,np.where(np.isfinite(s),s,-np.inf),side='right')/(len(r)+1)
    return out

def std_k(d, v, k=10):
    return d[raw(v)].groupby(d['run_id'],sort=False).transform(lambda s: s.rolling(k,min_periods=3).std()).values

def bank(d):
    d=d.copy(); d['prev_gap_sec']=d['gap']
    X=feature_bank(d); return X[[c for c in X.columns if '미래' not in c]]
