import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp17_backpressure.py
from common import *
from harness import load, evaluate, topk_counts
from scipy.stats import rankdata
import pickle
pd.set_option('display.width',250)
NEW=['Switch_Over_Position_raw','Clamp_Open_Position_raw','Average_Screw_RPM_raw','Max_Back_Pressure_raw']
ll=pd.read_csv('linked/linked_labeled.csv',parse_dates=['TimeStamp'],usecols=['product','TimeStamp']+NEW)
p='rg3'; ref, lab, rc = load(p)
a=ll[ll['product']==p].drop_duplicates('TimeStamp')[['TimeStamp']+NEW]
L=lab.merge(a,on='TimeStamp',how='left')
x=L['Max_Back_Pressure_raw']; g=x.groupby(L['run_id'],sort=False)
bp=g.diff().values
ev=L['eval'].values
# 1) 32개 후보 가족 탐색 보정 (미성형 최고 / 가스+미성형 동시)
cands=[]
for c in NEW:
    xx=L[c]; gg=xx.groupby(L['run_id'],sort=False)
    for v in (xx, gg.diff(), gg.transform(lambda s:s.rolling(10,min_periods=3).std()), xx-gg.transform(lambda s:s.shift(1).rolling(20,min_periods=3).median())):
        for sg in (1,-1): cands.append(sg*v.values)
m=ev; n=m.sum(); K=int(np.ceil(.1*n)); t=L['type'].values[m]; runs=L['run_id'].values[m]
jit=np.random.default_rng(0).random(n)*1e-9
SEL=[]
for v in cands:
    v=v[m]; v=np.where(np.isfinite(v),v,np.nanmin(v[np.isfinite(v)])-1); s=np.zeros(n,int); s[np.argsort(-(rankdata(v)+jit))[:K]]=1; SEL.append(s)
SEL=np.array(SEL)
obs=(SEL@(t=='미성형').astype(int)).max()
grp=[np.flatnonzero(runs==r) for r in pd.unique(runs)]; rng=np.random.default_rng(11); nu=[]
for b in range(5000):
    tt=t.copy()
    for idx in grp: tt[idx]=np.roll(t[idx],rng.integers(0,len(idx)))
    nu.append((SEL@(tt=='미성형').astype(int)).max())
nu=np.array(nu)
print(f'새 변수 32개 가족: 미성형 관측 최고 {obs}/8 | 무작위 최고 평균 {nu.mean():.2f}, 95% {np.quantile(nu,.95)} | 탐색보정 p={(1+(nu>=obs).sum())/(len(nu)+1):.4f}')
# 2) 어떤 미성형/가스를 잡았나
sc=np.where(np.isfinite(bp),bp,-1e9)
k=topk_counts(L,sc); print('배압 최대값 직전 대비 상승: 상위10%',k)
top=np.zeros(len(L),bool); idx=np.flatnonzero(ev); o=idx[np.argsort(-(sc[idx]+np.random.default_rng(0).random(len(idx))*1e-9))[:K]]; top[o]=True
print(L.loc[top&(L['type']!='정상'),['run_id','seq','type','Max_Back_Pressure_raw']].assign(dBP=bp[top&(L['type']!='정상').values]).to_string())
print('배압 상승 값 분포 (평가 샷):', pd.Series(bp[ev]).describe().round(2).to_dict(), '| 상위10% 경계값:', np.sort(sc[ev])[-K])
# 3) gas RF std10와 interleave
from final_model import raw, ALL_RAW
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
X=SimpleImputer(strategy='median').fit_transform(pd.DataFrame({f'{v}|std10':std_k(L,v,10) for v in ALL_RAW if raw(v) in L.columns}))
tt=L['type'].values; runsA=L['run_id'].values; s=np.zeros(len(L)); y=(tt=='가스').astype(int); use=ev&((tt=='가스')|(tt=='정상'))
for r in pd.unique(runsA):
    te=runsA==r; tr=(~te)&use
    if y[tr].sum()<2: continue
    mm=RandomForestClassifier(3000,max_depth=4,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=0).fit(X[tr],y[tr]); s[te]=mm.predict_proba(X[te])[:,1]
def rk(v):
    v=np.where(np.isfinite(v),v,-1e18); r=np.zeros(len(v)); r[ev]=rankdata(v[ev])/ev.sum(); return r
rows=[evaluate(p,L,L['score'].values,'final'), evaluate(p,L,s,'gas_rf3000|std10'), evaluate(p,L,sc,'BP_jump (배압 최대 상승)'),
      evaluate(p,L,np.maximum(rk(s),rk(sc)),'interleave(gas_rf_std10, BP_jump)')]
for w in (0.3,0.4,0.6,0.7):   # 예산 배분 민감도 (가스 몫 w)
    rows.append(evaluate(p,L,np.maximum(rk(s)-(1-w)*0, 0)*0+np.maximum(rk(s)*1.0, rk(sc)*1.0),f'(참고) w={w}') if False else None)
rows=[r for r in rows if r]
print(pd.DataFrame(rows).drop(columns='_c').round(3).to_string())
pickle.dump({'gas_rf_std10':s,'bp_jump':sc},open('rg3_final_cands.pkl','wb'))
