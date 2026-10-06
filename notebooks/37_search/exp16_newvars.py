import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp16_newvars.py
from common import *
from harness import load, evaluate, topk_counts
from scipy.stats import rankdata
pd.set_option('display.width',250)
NEW=['Switch_Over_Position_raw','Clamp_Open_Position_raw','Average_Screw_RPM_raw','Max_Back_Pressure_raw']
ll=pd.read_csv('linked/linked_labeled.csv',parse_dates=['TimeStamp'],usecols=['product','TimeStamp']+NEW)
for p in ('cn7','rg3'):
    ref, lab, rc = load(p)
    a=ll[ll['product']==p].drop_duplicates('TimeStamp')[['TimeStamp']+NEW]
    L=lab.merge(a,on='TimeStamp',how='left')
    print(p,'새 변수 결측률:',L[NEW].isna().mean().round(3).to_dict(),'| 고유값 수:',L[NEW].nunique().to_dict())
    rows=[]
    for c in NEW:
        x=L[c]; g=x.groupby(L['run_id'],sort=False)
        feats={'수준':x,'직전변화':g.diff(),'std10':g.transform(lambda s:s.rolling(10,min_periods=3).std()),
               '편차20':x-g.transform(lambda s:s.shift(1).rolling(20,min_periods=3).median())}
        for fn,v in feats.items():
            for sg in (1,-1):
                k=topk_counts(L,sg*v.values); kr=topk_counts(L,sg*v.values,by_run=True)
                rows.append((c[:-4],fn,sg,k['가스'],k['미성형'],kr['가스'],kr['미성형']))
    d=pd.DataFrame(rows,columns=['변수','변환','부호','상위10%가스','상위10%미성형','Run별가스','Run별미성형'])
    print(d.sort_values(['상위10%미성형','Run별미성형'],ascending=False).head(8).to_string())
    print(d.sort_values(['Run별가스','상위10%가스'],ascending=False).head(5).to_string())
