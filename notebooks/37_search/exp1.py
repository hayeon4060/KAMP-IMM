import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp1.py
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
from harness import load, evaluate
rows=[]
for p in ('cn7','rg3'):
    ref, lab, rc = load(p)
    print(p, lab.loc[lab['eval'],'type'].value_counts().to_dict(), 'eval', lab['eval'].sum())
    s0 = lab['score'].values
    rows.append(evaluate(p, lab, s0, 'final'))
    # EWMA accumulators of final pct (label-free, episode)
    for span in (5,10,20):
        e = lab.groupby('run_id',sort=False)['pct'].transform(lambda x: x.ewm(span=span).mean()).values
        rows.append(evaluate(p, lab, e, f'ewma_pct{span}'))
        rows.append(evaluate(p, lab, np.maximum(lab['pct'].values, e), f'max(final,ewma{span})'))
df=pd.DataFrame(rows).drop(columns='_c')
print(df.round(3).to_string())
