import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp19_gasnew_check.py
from common import *
from harness import load, evaluate
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from seq_ae import delta_ap
pd.set_option('display.width',250)
NEW=['Switch_Over_Position_raw','Average_Screw_RPM_raw','Max_Back_Pressure_raw']
ll=pd.read_csv('linked/linked_labeled.csv',parse_dates=['TimeStamp'],usecols=['product','TimeStamp']+NEW)
p='rg3'; ref, lab, rc = load(p)
L=lab.merge(ll[ll['product']==p].drop_duplicates('TimeStamp')[['TimeStamp']+NEW],on='TimeStamp',how='left')
X=pd.DataFrame({f'{v}|std10':L[v].groupby(L['run_id'],sort=False).transform(lambda s:s.rolling(10,min_periods=3).std()) for v in rc+NEW})
Xv=SimpleImputer(strategy='median').fit_transform(X)
ev=L['eval'].values; t=L['type'].values; runs=L['run_id'].values
rows=[]; m=ev; yy=(L['PassOrFail'].values[m]==1).astype(int)
for sd in range(5):
    s=np.zeros(len(L)); y=(t=='가스').astype(int); use=ev&((t=='가스')|(t=='정상'))
    for r in pd.unique(runs):
        te=runs==r; tr=(~te)&use
        if y[tr].sum()<2: continue
        mm=RandomForestClassifier(1000,max_depth=4,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=sd).fit(Xv[tr],y[tr]); s[te]=mm.predict_proba(Xv[te])[:,1]
    e=evaluate(p,L,s,f'seed{sd}'); d=delta_ap(yy,s[m],L['score'].values[m],n=1000)
    e['ΔAP']=f'{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] P{d[3]:.2f}'; rows.append(e)
print(pd.DataFrame(rows).drop(columns='_c').round(3).to_string())
