import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp10c_perm_std10new.py
from common import *
from harness import load, evaluate, topk_counts
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score
from final_model import FEATURES
import sys, json
p='rg3'; ref, lab, rc = load(p)
NEW=['Switch_Over_Position_raw','Average_Screw_RPM_raw','Max_Back_Pressure_raw']
ll=pd.read_csv('linked/linked_labeled.csv',parse_dates=['TimeStamp'],usecols=['product','TimeStamp']+NEW)
lab=lab.merge(ll[ll['product']==p].drop_duplicates('TimeStamp')[['TimeStamp']+NEW],on='TimeStamp',how='left')
X=pd.DataFrame({f'{v}|std10':lab[v].groupby(lab['run_id'],sort=False).transform(lambda s:s.rolling(10,min_periods=3).std()) for v in rc+NEW})
STD=list(X.columns)
Xs=SimpleImputer(strategy='median').fit_transform(X[STD])
ev=lab['eval'].values; runs=lab['run_id'].values; t0=lab['type'].values.copy()
def loro_gas(t):
    s=np.zeros(len(lab)); y=(t=='가스').astype(int); use=ev&((t=='가스')|(t=='정상'))
    for r in pd.unique(runs):
        te=runs==r; tr=(~te)&use
        if y[tr].sum()<2: continue
        m=RandomForestClassifier(300,max_depth=4,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=0).fit(Xs[tr],y[tr])
        s[te]=m.predict_proba(Xs[te])[:,1]
    return s
def stats(t,s):
    L=lab.copy(); L['type']=t
    c=topk_counts(L,s); mm=ev&((t=='가스')|(t=='정상'))
    return c['가스'], average_precision_score((t[mm]=='가스').astype(int), s[mm])
obs=stats(t0,loro_gas(t0)); print('관측',obs,flush=True)
rng=np.random.default_rng(5); N=int(sys.argv[1]); null=[]
evi=np.flatnonzero(ev)
for b in range(N):
    t=t0.copy()
    for r in pd.unique(runs):
        idx=evi[runs[evi]==r]
        if len(idx)==0: continue
        t[idx]=np.roll(t0[idx],rng.integers(0,len(idx)))
    null.append(stats(t,loro_gas(t)))
    if b%20==0: print(b,flush=True)
null=np.array(null)
res={'관측 가스 포착':obs[0],'관측 가스AP':obs[1],'null 포착 평균':float(null[:,0].mean()),'null 포착 95%':float(np.quantile(null[:,0],.95)),
     'p_포착':float((1+(null[:,0]>=obs[0]).sum())/(N+1)),'null AP 평균':float(null[:,1].mean()),'null AP 95%':float(np.quantile(null[:,1],.95)),
     'p_AP':float((1+(null[:,1]>=obs[1]).sum())/(N+1)),'N':N}
print(res); json.dump(res,open('perm_rg3gas_std10new.json','w'),ensure_ascii=False)
