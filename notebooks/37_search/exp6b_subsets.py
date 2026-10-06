import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp6b_subsets.py
from exp6_validate_E import *   # reuses Z, POOL, design pieces (runs its main once, fine)
import itertools
def score_subset(L, subset, signs):
    c=np.zeros(len(L))
    for j,sg in zip(subset,signs): c+=np.clip(sg*Z[:,j],0,10)
    e=pd.Series(c).groupby(L['run_id'].values,sort=False).transform(lambda s: s.rolling(5,min_periods=1).mean()).values
    return np.where(L['since_restart'].values>15,e,-np.inf)
def all_subsets(L):
    gas=(L['type'].values=='가스')&L['eval'].values; nrm=(L['type'].values=='정상')&L['eval'].values
    near=np.zeros(len(L),bool); runs=L['run_id'].values; seq=L['seq'].values
    for i in np.flatnonzero(gas): near|=(runs==runs[i])&(np.abs(seq-seq[i])<=10)
    d=np.array([cliff(Z[gas,j],Z[nrm&near,j]) for j in range(len(POOL))])
    out=[]
    for k in (2,3):
        for sub in itertools.combinations(range(len(POOL)),k):
            e=score_subset(L,sub,np.sign(d[list(sub)]))
            s=interleave(L['eval'].values,L['score'].values,e)
            c=topk_counts(L,s); out.append((sub,c['가스'],c['미성형']))
    return out
obs=all_subsets(lab)
ok=[o for o in obs if o[1]>=6 and o[2]>=3]
print('부분집합 수',len(obs),'| 목표 달성 부분집합:',[( [POOL[j] for j in o[0]],o[1],o[2]) for o in ok])
best_obs=max(o[1] for o in obs if o[2]>=3)
rng=np.random.default_rng(2); null=[]
for b in range(200):
    L=circ_null(lab,rng); res=all_subsets(L)
    null.append(max([o[1] for o in res if o[2]>=3] or [0]))
null=np.array(null)
print(f'탐색 보정: 관측 최고 가스 {best_obs}, 무작위 최고 평균 {null.mean():.2f}, 95% {np.quantile(null,.95)}, p={(1+(null>=best_obs).sum())/(len(null)+1):.3f}')
