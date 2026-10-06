import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp3.py
from common import *
from harness import load, evaluate
from scipy.stats import rankdata
pd.set_option('display.width',250)
TEMPS=['Barrel_Temperature_1','Barrel_Temperature_2','Barrel_Temperature_3','Barrel_Temperature_4','Barrel_Temperature_5','Barrel_Temperature_6','Hopper_Temperature','Mold_Temperature_3','Mold_Temperature_4']
STD4=['Barrel_Temperature_2','Hopper_Temperature','Cushion_Position','Barrel_Temperature_5']
def interleave(lab, *scores):
    m=lab['eval'].values
    R=[]
    for s in scores:
        s=np.asarray(s,float); s=np.where(np.isfinite(s),s,-1e18)
        r=np.zeros(len(s)); r[m]=rankdata(s[m])/m.sum(); R.append(r)
    return np.max(np.column_stack(R),1)
rows=[]
for p in ('cn7','rg3'):
    ref, lab, rc = load(p)
    fin=lab['score'].values
    rows.append(evaluate(p,lab,fin,'final'))
    cands={}
    for k in (5,10):
        for nm,vs in (('tempstd',TEMPS),('std4',STD4),('barrelstd',TEMPS[:6])):
            Xr=np.column_stack([std_k(ref,v,k) for v in vs]); Xl=np.column_stack([std_k(lab,v,k) for v in vs])
            sl,sr=maha(Xr,Xl); cands[f'{nm}_k{k}']=sl
    # thermal E (R1-derived, post-hoc reference)
    from ensemble_relation import gas_expert_cn7
    if p=='cn7':
        er,el=gas_expert_cn7(ref,lab); cands['E_thermal(사후)']=el
    for nm,s in cands.items():
        rows.append(evaluate(p,lab,s,nm))
        rows.append(evaluate(p,lab,interleave(lab,fin,s),f'interleave(final,{nm})'))
print(pd.DataFrame(rows).drop(columns='_c').round(3).to_string())
