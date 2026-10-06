import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp18_newvars_sup.py
from common import *
from harness import load, evaluate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from final_model import raw, ALL_RAW
pd.set_option('display.width',250)
NEW=['Switch_Over_Position_raw','Average_Screw_RPM_raw','Max_Back_Pressure_raw']
ll=pd.read_csv('linked/linked_labeled.csv',parse_dates=['TimeStamp'],usecols=['product','TimeStamp']+NEW)
p='rg3'; ref, lab, rc = load(p)
L=lab.merge(ll[ll['product']==p].drop_duplicates('TimeStamp')[['TimeStamp']+NEW],on='TimeStamp',how='left')
F={}
for v in [c[:-4] for c in rc]+[c[:-4] for c in NEW]:
    x=L[v+'_raw']; g=x.groupby(L['run_id'],sort=False)
    F[f'{v}|std10']=g.transform(lambda s:s.rolling(10,min_periods=3).std())
    F[f'{v}|d1']=g.diff()
    F[f'{v}|dev20']=x-g.transform(lambda s:s.shift(1).rolling(20,min_periods=3).median())
X=pd.DataFrame(F).replace([np.inf,-np.inf],np.nan)
SETS={'std10+new':[c for c in X.columns if c.endswith('std10')],
      'd1+dev20+new':[c for c in X.columns if not c.endswith('std10')],
      'all3':list(X.columns)}
ev=L['eval'].values; t=L['type'].values; runs=L['run_id'].values
def loro(cols,target,mk):
    Xv=SimpleImputer(strategy='median').fit_transform(X[cols]); s=np.zeros(len(L)); y=(t==target).astype(int); use=ev&((t==target)|(t=='정상'))
    for r in pd.unique(runs):
        te=runs==r; tr=(~te)&use
        if y[tr].sum()<2: continue
        m=mk().fit(Xv[tr],y[tr]); s[te]=m.predict_proba(Xv[te])[:,1]
    return s
MK={'rf':lambda: RandomForestClassifier(2000,max_depth=4,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=0),
    'lr':lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.05,class_weight='balanced',max_iter=3000))}
rows=[evaluate(p,L,L['score'].values,'final')]
for sn,cols in SETS.items():
    for mn,mk in MK.items():
        for tg in ('가스','미성형'):
            rows.append(evaluate(p,L,loro(cols,tg,mk),f'{tg}|{mn}|{sn}'))
print(pd.DataFrame(rows).drop(columns='_c').round(3).to_string())
