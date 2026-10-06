import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp6_validate_E.py
from common import *
from harness import load, evaluate, circ_null, topk_counts
from scipy.stats import rankdata
import sys
pd.set_option('display.width',250)
p='cn7'; ref, lab, rc = load(p)
POOL=['hopper_slope20','mold_slope20','hopper_rise','mold_rise','dev_Hopper_Temperature','dev_Mold_Temperature_3']
Z=robust(ref[POOL].values)(lab[POOL].values)   # reference 기준 robust z
def interleave(m, *ss):
    R=[]
    for s in ss:
        s=np.where(np.isfinite(s),s,-1e18); r=np.zeros(len(s)); r[m]=rankdata(s[m])/m.sum(); R.append(r)
    return np.max(np.column_stack(R),1)
def cliff(a,b):
    if len(a)==0 or len(b)==0: return 0.0
    return float(((a[:,None]>b[None,:]).mean()-(a[:,None]<b[None,:]).mean()))
def design(L, k=3, win=10):
    """26~29번 절차를 라벨만 바꿔 재실행: 가스 vs 시간맞춘 정상(±10샷) δ 상위 k 성분 → 부호 맞춘 합 → 최근 5샷 평균"""
    gas=(L['type'].values=='가스')&L['eval'].values
    nrm=(L['type'].values=='정상')&L['eval'].values
    near=np.zeros(len(L),bool)
    runs=L['run_id'].values; seq=L['seq'].values
    for i in np.flatnonzero(gas):
        near|=(runs==runs[i])&(np.abs(seq-seq[i])<=win)
    nm=nrm&near
    d=np.array([cliff(Z[gas,j],Z[nm,j]) for j in range(len(POOL))])
    top=np.argsort(-np.abs(d))[:k]
    c=np.zeros(len(L))
    for j in top: c+=np.clip(np.sign(d[j])*Z[:,j],0,10)
    e=pd.Series(c).groupby(L['run_id'].values,sort=False).transform(lambda s: s.rolling(5,min_periods=1).mean()).values
    e=np.where(L['since_restart'].values>15,e,-np.inf)
    return e,[(POOL[j],round(d[j],2)) for j in top]
m=lab['eval'].values
e_obs,sel=design(lab)
s_obs=interleave(m,lab['score'].values,e_obs)
r=evaluate(p,lab,s_obs,'interleave(final, E_재도출)')
print('관측 선택:',sel); print({k:v for k,v in r.items() if k!='_c'})
c_obs=r['_c']['가스']; ap_obs=r['② AP']
rng=np.random.default_rng(1); N=int(sys.argv[1]) if len(sys.argv)>1 else 300
null=[]
for b in range(N):
    L=circ_null(lab,rng)
    e,_=design(L); s=interleave(L['eval'].values,L['score'].values,e)
    rr=evaluate(p,L,s,frame1=False); null.append((rr['_c']['가스'],rr['_c']['미성형'],rr['② AP']))
null=np.array(null)
print(f'N={len(null)} | 가스 포착: 관측 {c_obs}, 무작위 평균 {null[:,0].mean():.2f}, 95% {np.quantile(null[:,0],.95)}, p={(1+(null[:,0]>=c_obs).sum())/(len(null)+1):.3f}')
print(f'② AP: 관측 {ap_obs:.3f}, 무작위 평균 {null[:,2].mean():.3f}, 95% {np.quantile(null[:,2],.95):.3f}, p={(1+(null[:,2]>=ap_obs).sum())/(len(null)+1):.3f}')
print('목표(가스≥6 & 미성형≥3) 무작위 달성 비율:', float(((null[:,0]>=6)&(null[:,1]>=3)).mean()))
np.save('null_E.npy',null)
