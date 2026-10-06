import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp7_bankscan.py
from common import *
from harness import load, evaluate
from scipy.stats import rankdata
import sys
pd.set_option('display.width',250)
p=sys.argv[1]; ref, lab, rc = load(p)
X=bank(lab)
for f in ['hopper_slope20','mold_slope20','hopper_rise','mold_rise','fill_anomaly','ev_max_abs_dev','dev_flow_res']: X['F|'+f]=lab[f].values
X=X.replace([np.inf,-np.inf],np.nan)
m=lab['eval'].values; n=m.sum(); K=int(np.ceil(0.1*n))
t=lab['type'].values[m]; runs=lab['run_id'].values[m]
fin=lab['score'].values[m]
def rk(s):
    s=np.where(np.isfinite(s),s,np.nanmin(s[np.isfinite(s)])-1 if np.isfinite(s).any() else 0); return rankdata(s)/len(s)
rf=rk(fin)
jit=np.random.default_rng(0).random(n)*1e-9
BYRUN = len(sys.argv)>2 and sys.argv[2]=='run'
grp=[np.flatnonzero(runs==r) for r in pd.unique(runs)]
def topsel(sc):
    sc=sc+jit; sel=np.zeros(n,bool)
    if BYRUN:
        for idx in grp: sel[idx[np.argsort(-sc[idx])[:int(np.ceil(0.1*len(idx)))]]]=True
    else: sel[np.argsort(-sc)[:K]]=True
    return sel
names=[];SEL=[];SEL_I=[]
for c in X.columns:
    for sg in (1,-1):
        v=sg*X[c].values[m]
        r=rk(v)
        sel=topsel(r); si=topsel(np.maximum(rf,r))
        names.append(f'{c} ({"+" if sg>0 else "-"})'); SEL.append(sel); SEL_I.append(si)
SEL=np.array(SEL).astype(int); SEL_I=np.array(SEL_I).astype(int)
def counts(S,tt): return S@(tt=="가스").astype(int), S@(tt=="미성형").astype(int)
g,s_=counts(SEL,t); gi,si_=counts(SEL_I,t)
d=pd.DataFrame({'후보':names,'단독 가스':g,'단독 미성형':s_,'interleave 가스':gi,'interleave 미성형':si_})
tg={'cn7':(6,3),'rg3':(7,4)}[p]
print(d.sort_values(['interleave 가스','interleave 미성형'],ascending=False).head(12).to_string())
print('목표 달성 후보 수 (interleave):',int(((gi>=tg[0])&(si_>=tg[1])).sum()),'/ (단독):',int(((g>=tg[0])&(s_>=tg[1])).sum()), '/ 전체',len(names))
# 탐색 보정 null: Run 안에서 eval 샷끼리 라벨 원형 이동
def stat(gg,ss):
    ok=ss>=tg[1]
    return gg[ok].max() if ok.any() else 0
obs_i=stat(gi,si_); obs_s=stat(g,s_)
rng=np.random.default_rng(3); N=2000; nu_i=[];nu_s=[]; hit=0
groups=[np.flatnonzero(runs==r) for r in pd.unique(runs)]
for b in range(N):
    tt=t.copy()
    for idx in groups: tt[idx]=np.roll(t[idx],rng.integers(0,len(idx)))
    a,b_=counts(SEL_I,tt); c,d_=counts(SEL,tt)
    nu_i.append(stat(a,b_)); nu_s.append(stat(c,d_)); hit+=int(((a>=tg[0])&(b_>=tg[1])).any())
nu_i=np.array(nu_i); nu_s=np.array(nu_s)
print(f'[interleave] 관측 최고 가스(미성형≥{tg[1]}): {obs_i} | 무작위 최고 평균 {nu_i.mean():.2f}, 95% {np.quantile(nu_i,.95)} | 탐색보정 p={(1+(nu_i>=obs_i).sum())/(N+1):.4f}')
print(f'[단독] 관측 {obs_s} | 무작위 평균 {nu_s.mean():.2f}, 95% {np.quantile(nu_s,.95)} | p={(1+(nu_s>=obs_s).sum())/(N+1):.4f}')
print('무작위 라벨에서도 목표를 달성하는 후보가 하나라도 나오는 비율:',hit/N)
d.to_csv(f'bankscan_{p}_{"run" if BYRUN else "pool"}.csv',index=False)
