import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp11_seed_transfer.py
from common import *
from harness import load, evaluate
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from final_model import FEATURES
pd.set_option('display.width',250)
D={}
for p in ('cn7','rg3'):
    ref, lab, rc = load(p)
    Xl=bank(lab); Xr=bank(ref)
    STD=[c for c in Xl.columns if 'std' in c]
    z=robust(Xr[STD].values)
    D[p]=dict(ref=ref,lab=lab,X=pd.DataFrame(Xl[STD].values,columns=STD),Z=pd.DataFrame(z(Xl[STD].values),columns=STD))
# (a) RG3 gas_rf|std 시드 민감도 (LORO)
p='rg3'; lab=D[p]['lab']; X=D[p]['X']; ev=lab['eval'].values; t=lab['type'].values; runs=lab['run_id'].values
res=[]
for sd in range(10):
    s=np.zeros(len(lab)); y=(t=='가스').astype(int); use=ev&((t=='가스')|(t=='정상'))
    for r in pd.unique(runs):
        te=runs==r; tr=(~te)&use
        if y[tr].sum()<2: continue
        m=make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(500,max_depth=4,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=sd)).fit(X[tr],y[tr])
        s[te]=m.predict_proba(X[te])[:,1]
    e=evaluate(p,lab,s,f'seed{sd}',frame1=False); res.append((sd,e['② AP'],e['AP_가스'],e['상위10% 가스'],e['상위10% 미성형']))
print('RG3 gas_rf|std 시드별:'); print(pd.DataFrame(res,columns=['seed','②AP','AP가스','상위10%가스','상위10%미성형']).round(3).to_string())
# (b) 제품 간 이전: 한 제품 가스로 학습 → 다른 제품에 적용 (reference robust z로 척도 맞춤)
common=[c for c in D['cn7']['Z'].columns if c in D['rg3']['Z'].columns]
for src,dst in (('rg3','cn7'),('cn7','rg3')):
    L=D[src]['lab']; t=L['type'].values; ev=L['eval'].values; use=ev&((t=='가스')|(t=='정상'))
    m=RandomForestClassifier(500,max_depth=4,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=0).fit(D[src]['Z'].loc[use,common],(t[use]=='가스').astype(int))
    s=m.predict_proba(D[dst]['Z'][common])[:,1]
    print(f'이전 {src}→{dst}:', {k:v for k,v in evaluate(dst,D[dst]['lab'],s,f'{src}→{dst}').items() if k!='_c'})
    imp=pd.Series(m.feature_importances_,index=common).sort_values(ascending=False).head(5); print(' 중요 변수:',imp.round(3).to_dict())
