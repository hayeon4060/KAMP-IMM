import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp12_rg3combo.py
from common import *
from harness import load, evaluate
from scipy.stats import rankdata
import pickle
pd.set_option('display.width',250)
p='rg3'; ref, lab, rc = load(p)
ev=lab['eval'].values
def rk(s):
    s=np.where(np.isfinite(s),s,-1e18); r=np.zeros(len(s)); r[ev]=rankdata(s[ev])/ev.sum(); return r
T=pickle.load(open('rg3_type_scores.pkl','rb')); G=T['G']; S=T['S']
STD4=['Barrel_Temperature_2','Hopper_Temperature','Cushion_Position','Barrel_Temperature_5']
Xr=np.column_stack([std_k(ref,v,5) for v in STD4]); Xl=np.column_stack([std_k(lab,v,5) for v in STD4])
std4k5=maha(Xr,Xl)[0]
fin=lab['score'].values
C={'gas_rf':G['rf'],'std4_k5(무라벨)':std4k5,'final':fin,'short_lgbm|std':S['lgbm|std'],'short_lr|std':S['lr|std']}
rows=[]
import itertools
for k in (2,3):
    for combo in itertools.combinations(C,k):
        if 'gas_rf' not in combo: continue
        s=np.max(np.column_stack([rk(C[c]) for c in combo]),1)
        rows.append(evaluate(p,lab,s,'interleave('+', '.join(combo)+')'))
print(pd.DataFrame(rows).drop(columns='_c').round(3).to_string())
