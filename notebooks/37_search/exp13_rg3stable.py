import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp13_rg3stable.py
from common import *
from harness import load, evaluate
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from seq_ae import delta_ap
from final_model import raw, ALL_RAW
pd.set_option('display.width',250)
p='rg3'; ref, lab, rc = load(p)
ev=lab['eval'].values; t=lab['type'].values; runs=lab['run_id'].values
def stdbank(k):
    return pd.DataFrame({f'{v}|std{k}':std_k(lab,v,k) for v in ALL_RAW if raw(v) in lab.columns})
rows=[]; S={}
for k in (5,10,20):
    X=SimpleImputer(strategy='median').fit_transform(stdbank(k))
    s=np.zeros(len(lab)); y=(t=='가스').astype(int); use=ev&((t=='가스')|(t=='정상'))
    for r in pd.unique(runs):
        te=runs==r; tr=(~te)&use
        if y[tr].sum()<2: continue
        m=RandomForestClassifier(3000,max_depth=4,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=0).fit(X[tr],y[tr])
        s[te]=m.predict_proba(X[te])[:,1]
    S[k]=s; e=evaluate(p,lab,s,f'gas_rf3000|std{k}')
    mm=ev; yy=(lab['PassOrFail'].values[mm]==1).astype(int); d=delta_ap(yy,s[mm],lab['score'].values[mm],n=2000)
    e['ΔAP']=f'{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] P{d[3]:.2f}'; rows.append(e)
print(pd.DataFrame(rows).drop(columns='_c').round(3).to_string())
