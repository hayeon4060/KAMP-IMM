import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp2.py
from common import *
from harness import load, evaluate
pd.set_option('display.width',250)
STD4=['Barrel_Temperature_2','Hopper_Temperature','Cushion_Position','Barrel_Temperature_5']
rows=[]
for p in ('cn7','rg3'):
    ref, lab, rc = load(p)
    rows.append(evaluate(p,lab,lab['score'].values,'final'))
    for k in (5,10,20):
        Xr=np.column_stack([std_k(ref,v,k) for v in STD4]); Xl=np.column_stack([std_k(lab,v,k) for v in STD4])
        sl,sr=maha(Xr,Xl); pl=pct(sr,sl); pr_=pct(sr,sr)
        rows.append(evaluate(p,lab,sl,f'std4_k{k}'))
        f_l=np.maximum(lab['pct'].values,pl); f_r=np.maximum(ref['pct'].values,pr_)
        rows.append(evaluate(p,lab,pct(f_r,f_l),f'max(final,std4_k{k})'))
print(pd.DataFrame(rows).drop(columns='_c').round(3).to_string())
