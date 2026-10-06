import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp20_budget.py
from common import *
from harness import load
from ensemble_relation import gas_expert_cn7
from final_model import raw
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pickle
pd.set_option('display.width',250)
def split_budget(lab, fin, gas, f, total=0.10):
    """최종 모델이 상위 f%를 먼저 가져가고, 남은 예산(total-f)을 가스 전문가 순위로 채움"""
    m=lab['eval'].values; idx=np.flatnonzero(m); n=len(idx); K=int(np.ceil(total*n)); kf=int(np.ceil(f*n)) if f>0 else 0
    j=np.random.default_rng(0).random(n)*1e-9
    a=np.where(np.isfinite(fin[idx]),fin[idx],-1e18)+j; b=np.where(np.isfinite(gas[idx]),gas[idx],-1e18)+j
    sel=set(np.argsort(-a)[:kf])
    for i in np.argsort(-b):
        if len(sel)>=K: break
        sel.add(i)
    sel=np.array(sorted(sel)); t=lab['type'].values[idx][sel]
    return int((t=='가스').sum()), int((t=='미성형').sum())
STD4=['Barrel_Temperature_2','Hopper_Temperature','Cushion_Position','Barrel_Temperature_5']
rows=[]
for p in ('cn7','rg3'):
    ref, lab, rc = load(p); fin=lab['score'].values
    G={}
    for k in (5,10):
        Xr=np.column_stack([std_k(ref,v,k) for v in STD4]); Xl=np.column_stack([std_k(lab,v,k) for v in STD4]); G[f'std4_k{k} (무라벨)']=maha(Xr,Xl)[0]
    if p=='cn7':
        G['E_thermal (R1에서 도출, 사후)']=gas_expert_cn7(ref,lab)[1]
    else:
        NEW=['Switch_Over_Position_raw','Average_Screw_RPM_raw','Max_Back_Pressure_raw']
        ll=pd.read_csv('linked/linked_labeled.csv',parse_dates=['TimeStamp'],usecols=['product','TimeStamp']+NEW)
        L=lab.merge(ll[ll['product']==p].drop_duplicates('TimeStamp')[['TimeStamp']+NEW],on='TimeStamp',how='left')
        X=SimpleImputer(strategy='median').fit_transform(pd.DataFrame({v:L[v].groupby(L['run_id'],sort=False).transform(lambda s:s.rolling(10,min_periods=3).std()) for v in rc+NEW}))
        ev=L['eval'].values; t=L['type'].values; runs=L['run_id'].values; s=np.zeros(len(L)); y=(t=='가스').astype(int); use=ev&((t=='가스')|(t=='정상'))
        for r in pd.unique(runs):
            te=runs==r; tr=(~te)&use
            if y[tr].sum()<2: continue
            s[te]=RandomForestClassifier(1000,max_depth=4,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=0).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
        G['gas_RF_std10+new (지도, LORO)']=s
    for gn,g in G.items():
        for f in (0,0.01,0.02,0.03,0.05,0.10):
            ga,sh=split_budget(lab,fin,g,f); rows.append({'제품':p,'가스 전문가':gn,'최종모델 몫':f'{int(f*100)}%','가스 전문가 몫':f'{int((0.10-f)*100)}%','가스':ga,'미성형':sh})
d=pd.DataFrame(rows)
for p in ('cn7','rg3'):
    print(f'===== {p.upper()} (가스 {11 if p=="cn7" else 17}, 미성형 {4 if p=="cn7" else 8}) =====')
    print(d[d['제품']==p].pivot_table(index='가스 전문가',columns='최종모델 몫',values=['가스','미성형'],aggfunc='first').to_string())
d.to_csv('budget_split.csv',index=False)
