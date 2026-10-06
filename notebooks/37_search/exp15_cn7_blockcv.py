import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp15_cn7_blockcv.py
from common import *
from harness import load, evaluate
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from final_model import raw, ALL_RAW, FEATURES
pd.set_option('display.width',250)
p='cn7'; ref, lab, rc = load(p)
ev=lab['eval'].values; t=lab['type'].values; runs=lab['run_id'].values; seq=lab['seq'].values
S10=pd.DataFrame({f'{v}|std10':std_k(lab,v,10) for v in ALL_RAW if raw(v) in lab.columns})
FS={'std10':S10,'free17+std10':pd.concat([lab[FEATURES].reset_index(drop=True),S10],axis=1)}
# R1을 시간 블록 2개로: 가스 25·41 (앞) / 49·51~58 (뒤). 다른 Run은 Run 단위로 함께 나눔
blk=np.full(len(lab),-1)
blk[(runs=='cn7_R1')&(seq<=45)]=0; blk[(runs=='cn7_R1')&(seq>45)]=1
others=[r for r in pd.unique(runs) if r!='cn7_R1']
for i,r in enumerate(others): blk[runs==r]=i%2
rows=[evaluate(p,lab,lab['score'].values,'final')]
for fn,X in FS.items():
    Xv=SimpleImputer(strategy='median').fit_transform(X.replace([np.inf,-np.inf],np.nan))
    for target in ('가스','불량전체'):
        y=(t=='가스').astype(int) if target=='가스' else (lab['PassOrFail'].values==1).astype(int)
        use=ev&((y==1)|(t=='정상'))
        s=np.zeros(len(lab))
        for b in (0,1):
            te=blk==b; tr=(blk==1-b)&use
            m=RandomForestClassifier(2000,max_depth=4,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=0).fit(Xv[tr],y[tr])
            s[te]=m.predict_proba(Xv[te])[:,1]
        rows.append(evaluate(p,lab,s,f'blockCV rf|{fn}|{target}'))
print(pd.DataFrame(rows).drop(columns='_c').round(3).to_string())
