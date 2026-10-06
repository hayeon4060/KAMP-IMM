import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp21_std4_null.py
from common import *
from harness import load
pd.set_option('display.width',250)
STD4=['Barrel_Temperature_2','Hopper_Temperature','Cushion_Position','Barrel_Temperature_5']
p='cn7'; ref, lab, rc = load(p); fin=lab['score'].values; pc=lab['pct'].values
m=lab['eval'].values; idx=np.flatnonzero(m); n=len(idx); K=int(np.ceil(.1*n))
j=np.random.default_rng(0).random(n)*1e-9
runs=lab['run_id'].values[idx]; t=lab['type'].values[idx]
alarm=pc[idx]>=0.999
print('최종 모델 q0.999 경보 샷 수:',alarm.sum(),'(',round(alarm.mean()*100,2),'% ) — 이 중 미성형',int((alarm&(t=="미성형")).sum()))
for k in (5,10,20):
    Xr=np.column_stack([std_k(ref,v,k) for v in STD4]); Xl=np.column_stack([std_k(lab,v,k) for v in STD4]); g=maha(Xr,Xl)[0][idx]+j
    sel=alarm.copy()
    for i in np.argsort(-g):
        if sel.sum()>=K: break
        sel[i]=True
    obs=(int((sel&(t=='가스')).sum()), int((sel&(t=='미성형')).sum()))
    # 원형 이동 null (선택 집합은 라벨과 무관 → 라벨만 이동)
    rng=np.random.default_rng(7); nu=[]
    grp=[np.flatnonzero(runs==r) for r in pd.unique(runs)]
    for b in range(5000):
        tt=t.copy()
        for g_ in grp: tt[g_]=np.roll(t[g_],rng.integers(0,len(g_)))
        nu.append((sel&(tt=='가스')).sum())
    nu=np.array(nu)
    print(f'std4_k{k}: 경보(최종 q0.999)+나머지 std4로 10% 채움 → 가스 {obs[0]}/11, 미성형 {obs[1]}/4 | 무작위 위치 가스 평균 {nu.mean():.2f}, 95% {np.quantile(nu,.95)}, p={(1+(nu>=obs[0]).sum())/(len(nu)+1):.3f}')
