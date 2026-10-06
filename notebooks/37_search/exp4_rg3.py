import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp4_rg3.py
from common import *
from harness import load, evaluate
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
pd.set_option('display.width',250)
p='rg3'
ref, lab, rc = load(p)
X=bank(lab)
for f in ['hopper_slope20','mold_slope20','hopper_rise','mold_rise','fill_anomaly','ev_max_abs_dev','dev_flow_res']: X['F|'+f]=lab[f].values
X=X.replace([np.inf,-np.inf],np.nan)
ev=lab['eval'].values; y=(lab['PassOrFail'].values==1).astype(int); t=lab['type'].values
runs=lab['run_id'].values
def models():
    return {
     'lr_l2':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=0.05,class_weight='balanced',max_iter=2000)),
     'rf':make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(500,max_depth=5,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=0)),
     'et':make_pipeline(SimpleImputer(strategy='median'),ExtraTreesClassifier(500,max_depth=5,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=0)),
     'lgbm':lgb.LGBMClassifier(n_estimators=200,learning_rate=0.03,num_leaves=7,min_child_samples=10,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,class_weight='balanced',verbose=-1,random_state=0),
    }
FS={'bank_all':list(X.columns),
    'bank_nolevel':[c for c in X.columns if '|수준' not in c],
    'std_only':[c for c in X.columns if 'std' in c],
    'free17':None}
from final_model import FEATURES
rows=[evaluate(p,lab,lab['score'].values,'final')]
S={}
for fsn,cols in FS.items():
    Xf = lab[FEATURES] if cols is None else X[cols]
    for mn in models():
        s=np.zeros(len(lab))
        for r in pd.unique(runs):
            te=runs==r; tr=(~te)&ev
            if y[tr].sum()<2: continue
            m=models()[mn]; m.fit(Xf[tr],y[tr]); s[te]=m.predict_proba(Xf[te])[:,1]
        S[f'{mn}|{fsn}']=s
        rows.append(evaluate(p,lab,s,f'{mn}|{fsn}'))
df=pd.DataFrame(rows).drop(columns='_c'); print(df.round(3).to_string())
import pickle; pickle.dump(S,open('rg3_sup_scores.pkl','wb'))
