import sys,os; sys.path.insert(0,'.'); sys.path.insert(0,'37_search')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from harness import load
from final_model import FEATURES, FinalModel, raw
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
for p in ('cn7','rg3'):
    ref, lab, rc = load(p)
    base = pd.read_csv(f'base/{p}_shots.csv', parse_dates=['TimeStamp'])[['TimeStamp','LH_fail','RH_fail','after_stop','shots_since_stop']]
    lab = lab.merge(base, on='TimeStamp', how='left')
    # robust z of features vs reference (same as model prep)
    X0 = ref[FEATURES].replace([np.inf,-np.inf],np.nan).values.astype(float)
    med = np.nanmedian(X0,0); q75,q25=np.nanpercentile(X0,[75,25],0); sd=np.nanstd(X0,0)
    iqr=np.where(q75-q25>1e-9,q75-q25,np.where(sd>1e-9,sd,1.0))
    Z = np.nan_to_num(np.clip((lab[FEATURES].values.astype(float)-med)/iqr,-20,20))
    top = [', '.join(f'{FEATURES[j]}({Z[i,j]:+.1f})' for j in np.argsort(-np.abs(Z[i]))[:3]) for i in range(len(lab))]
    lab['top3'] = top
    ev = lab[lab['eval']]
    print(f'\n##### {p}  eval shots {len(ev)}')
    al = ev[ev['pct']>=0.999]
    print('--- q0.999 alarms'); print(al[['run_id','TimeStamp','seq','since_restart','gap','type','pct','top3']].to_string(index=False))
    al2 = ev[ev['pct']>=0.99]
    print('--- q0.99 alarms: type x (since_restart<=5)'); print(pd.crosstab(al2['type'], al2['since_restart']<=5))
    print('--- q0.99 FP since_restart distribution', al2.loc[al2['type']=='정상','since_restart'].describe().round(1).to_dict())
    # restart share in eval normal
    print('normal shots with since_restart<=5 share:', round((ev.loc[ev.type=='정상','since_restart']<=5).mean(),3))
    print('--- defects: side & rank')
    d = ev[ev.type!='정상'].copy(); d['rank']=ev['score'].rank(ascending=False,pct=True)
    print(d[['run_id','TimeStamp','type','LH_fail','RH_fail','seq','since_restart','rank','top3']].round(3).to_string(index=False))
    print('side counts', d[['LH_fail','RH_fail']].sum().to_dict())
    # per-run defect rate and mean pct
    g = ev.groupby('run_id').agg(n=('type','size'), defects=('PassOrFail','sum'), mold3=(raw('Mold_Temperature_3'),'mean'), mean_pct=('pct','mean'), alarm99=('pct',lambda s:(s>=0.99).mean()))
    print(g.round(3).to_string())
