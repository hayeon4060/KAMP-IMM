import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp9_rg3type.py
from common import *
from harness import load, evaluate
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from final_model import FEATURES
pd.set_option('display.width',250)
p='rg3'; ref, lab, rc = load(p)
X=bank(lab)
for f in FEATURES: X['F|'+f]=lab[f].values
X=X.replace([np.inf,-np.inf],np.nan)
ev=lab['eval'].values; t=lab['type'].values; runs=lab['run_id'].values
STD=[c for c in X.columns if 'std' in c]; NOLV=[c for c in X.columns if '수준' not in c and 'Run중앙값' not in c]
M={'lr':lambda: make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=0.05,class_weight='balanced',max_iter=3000)),
   'rf':lambda: make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(500,max_depth=4,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=0)),
   'lgbm':lambda: lgb.LGBMClassifier(n_estimators=200,learning_rate=0.03,num_leaves=7,min_child_samples=8,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,class_weight='balanced',verbose=-1,random_state=0)}
def loro(cols, target, mk):
    s=np.zeros(len(lab)); y=(t==target).astype(int); use=ev&((t==target)|(t=='정상'))
    for r in pd.unique(runs):
        te=runs==r; tr=(~te)&use
        if y[tr].sum()<2: continue
        m=mk(); m.fit(X.loc[tr,cols],y[tr]); s[te]=m.predict_proba(X.loc[te,cols])[:,1]
    return s
def rk(s):
    r=np.zeros(len(s)); r[ev]=rankdata(s[ev])/ev.sum(); return r
rows=[evaluate(p,lab,lab['score'].values,'final')]
G={}; S={}
for mn,mk in M.items():
    G[mn]=loro(STD,'가스',mk); S[mn+'|nolevel']=loro(NOLV,'미성형',mk); S[mn+'|std']=loro(STD,'미성형',mk)
    rows.append(evaluate(p,lab,G[mn],f'gas_{mn}|std'))
for k,v in S.items(): rows.append(evaluate(p,lab,v,f'short_{k}'))
for gk,gv in G.items():
    for sk,sv in S.items():
        rows.append(evaluate(p,lab,np.maximum(rk(gv),rk(sv)),f'interleave(gas_{gk}, short_{sk})'))
df=pd.DataFrame(rows).drop(columns='_c'); print(df.round(3).to_string())

print('---- 이웃(과거 k샷) 최대값 ----')
rows=[]
for k in (2,3,5,8):
    for nm,s in (('gas_rf',G['rf']),('gas_lgbm',G['lgbm'])):
        sm=pd.Series(s).groupby(runs,sort=False).transform(lambda x: x.rolling(k,min_periods=1).max()).values
        rows.append(evaluate(p,lab,sm,f'{nm}_pastmax{k}'))
        rows.append(evaluate(p,lab,np.maximum(rk(s),rk(sm)),f'interleave({nm},{nm}_pastmax{k})'))
print(pd.DataFrame(rows).drop(columns='_c').round(3).to_string())
import pickle; pickle.dump({'G':G,'S':S},open('rg3_type_scores.pkl','wb'))
