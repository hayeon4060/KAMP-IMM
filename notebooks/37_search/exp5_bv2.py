import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp5_bv2.py
from common import *
from harness import load, evaluate
from final_model import ALL_RAW, resolution, SEED
from sklearn.ensemble import HistGradientBoostingRegressor
pd.set_option('display.width',250)
TEMPV=[v for v in ALL_RAW if 'Temperature' in v]
def rel(d, rc):
    """같은 Run 직전 20샷 중앙값 대비 상대값 (현재 샷 제외 → 인과적)"""
    R=pd.DataFrame(index=d.index)
    for c in rc:
        med=d[c].groupby(d['run_id'],sort=False).transform(lambda s: s.shift(1).rolling(20,min_periods=3).median())
        R[c]=d[c]-med
    return R
def time_split(d, frac=0.7):
    pos=d.groupby('run_id',sort=False).cumcount(); n=d.groupby('run_id',sort=False)['run_id'].transform('size')
    return (pos>=frac*n).values   # True = 검증
rows=[]
for p in ('cn7','rg3'):
    ref, lab, rc = load(p)
    rows.append(evaluate(p,lab,lab['score'].values,'final'))
    Rr, Rl = rel(ref,rc), rel(lab,rc)
    val=time_split(ref)
    targets=[c for c in rc if c.replace('_raw','') not in TEMPV]
    Zr=[];Zl=[]; names=[]
    for c in targets:
        Xr=Rr.drop(columns=c); Xl=Rl.drop(columns=c)
        yr=Rr[c]; ok=(~val)&yr.notna().values
        if yr[ok].nunique()<3: continue
        m=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=30,random_state=SEED).fit(Xr[ok],yr[ok])
        res_v=(yr-m.predict(Xr))[val & yr.notna().values]
        sc=max(1.4826*np.median(np.abs(res_v-np.median(res_v))), resolution(ref[c]),1e-6)
        Zr.append(((yr-m.predict(Xr))/sc).values); Zl.append(((Rl[c]-m.predict(Xl))/sc).values); names.append(c)
    Zr=np.nan_to_num(np.clip(np.column_stack(Zr),-20,20)); Zl=np.nan_to_num(np.clip(np.column_stack(Zl),-20,20))
    from sklearn.covariance import LedoitWolf
    lw=LedoitWolf().fit(Zr[val])
    bm_l=np.sqrt(lw.mahalanobis(Zl)); bm_r=np.sqrt(lw.mahalanobis(Zr[val]))
    pl=pct(bm_r,bm_l)
    m_=lab['eval'].values&(lab['PassOrFail'].values==0)
    print(p,'targets',len(names),'| q0.99 경보 비율(labeled 정상):',round(float((pl[m_]>=0.99).mean()),3),'| 이전 B 57.9%/5.6%')
    rows.append(evaluate(p,lab,bm_l,'B_v2_rel_maha'))
    bx=np.abs(Zl).max(1); rows.append(evaluate(p,lab,bx,'B_v2_rel_max'))
    fr=np.maximum(ref['pct'].values[val],pct(bm_r,bm_r)); fl=np.maximum(lab['pct'].values,pl)
    rows.append(evaluate(p,lab,pct(fr,fl),'max(final,B_v2)'))
print(pd.DataFrame(rows).drop(columns='_c').round(3).to_string())
