import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp14_shortnull.py
from common import *
from harness import load
from scipy.stats import rankdata
import sys
p=sys.argv[1]; ref, lab, rc = load(p)
X=bank(lab).replace([np.inf,-np.inf],np.nan)
m=lab['eval'].values; n=m.sum(); K=int(np.ceil(0.1*n)); t=lab['type'].values[m]; runs=lab['run_id'].values[m]
jit=np.random.default_rng(0).random(n)*1e-9; SEL=[]
for c in X.columns:
    for sg in (1,-1):
        v=sg*X[c].values[m]; v=np.where(np.isfinite(v),v,np.nanmin(v)-1); sel=np.zeros(n,int); sel[np.argsort(-(rankdata(v)+jit))[:K]]=1; SEL.append(sel)
SEL=np.array(SEL)
ty=sys.argv[2]
obs=(SEL@(t==ty).astype(int)).max()
grp=[np.flatnonzero(runs==r) for r in pd.unique(runs)]; rng=np.random.default_rng(9); nu=[]
for b in range(3000):
    tt=t.copy()
    for idx in grp: tt[idx]=np.roll(t[idx],rng.integers(0,len(idx)))
    nu.append((SEL@(tt==ty).astype(int)).max())
nu=np.array(nu)
print(f'{p} {ty}: 관측 최고 {obs}/{(t==ty).sum()} | 무작위 최고 평균 {nu.mean():.2f}, 95% {np.quantile(nu,.95)} | 탐색보정 p={(1+(nu>=obs).sum())/(len(nu)+1):.3f}')
