import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp22_resample.py
from common import *
from harness import load, evaluate
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
import warnings, sys; warnings.filterwarnings('ignore')
pd.set_option('display.width',250)
NEW=['Switch_Over_Position_raw','Average_Screw_RPM_raw','Max_Back_Pressure_raw']
ll=pd.read_csv('linked/linked_labeled.csv',parse_dates=['TimeStamp'],usecols=['product','TimeStamp']+NEW)
p='rg3'; ref, lab, rc = load(p)
L=lab.merge(ll[ll['product']==p].drop_duplicates('TimeStamp')[['TimeStamp']+NEW],on='TimeStamp',how='left')
X=SimpleImputer(strategy='median').fit_transform(pd.DataFrame({v:L[v].groupby(L['run_id'],sort=False).transform(lambda s:s.rolling(10,min_periods=3).std()) for v in rc+NEW}))
ev=L['eval'].values; t=L['type'].values; runs=L['run_id'].values
def k_(y): return max(1,min(5,int(y.sum())-1))
def rus(r): return lambda y,s: RandomUnderSampler(sampling_strategy={0:min((y==0).sum(),int(r*y.sum()))},random_state=s)
METHODS={
 '없음(가중치만, 기준)':None,
 '언더|무작위 3:1':[rus(3)], '언더|무작위 10:1':[rus(10)],
 '언더|NearMiss-1 3:1':[lambda y,s: NearMiss(version=1,sampling_strategy={0:int(3*y.sum())})],
 '언더|Tomek Links':[lambda y,s: TomekLinks()], '언더|ENN':[lambda y,s: EditedNearestNeighbours()],
 '언더|ClusterCentroids 3:1':[lambda y,s: ClusterCentroids(sampling_strategy={0:int(3*y.sum())},random_state=s)],
 '업|무작위 복제 1:1':[lambda y,s: RandomOverSampler(random_state=s)],
 '업|SMOTE':[lambda y,s: SMOTE(k_neighbors=k_(y),random_state=s)],
 '업|Borderline-SMOTE':[lambda y,s: BorderlineSMOTE(k_neighbors=k_(y),random_state=s)],
 '업|ADASYN':[lambda y,s: ADASYN(n_neighbors=k_(y),random_state=s)],
 '업|SVM-SMOTE':[lambda y,s: SVMSMOTE(k_neighbors=k_(y),random_state=s)],
 '언더+업|무작위3:1+복제':[rus(3),lambda y,s: RandomOverSampler(random_state=s)],
 '언더+업|무작위3:1+SMOTE':[rus(3),lambda y,s: SMOTE(k_neighbors=k_(y),random_state=s)],
 '언더+업|무작위10:1+SMOTE':[rus(10),lambda y,s: SMOTE(k_neighbors=k_(y),random_state=s)],
 '언더+업|무작위10:1+ADASYN':[rus(10),lambda y,s: ADASYN(n_neighbors=k_(y),random_state=s)],
 '결합|SMOTE+ENN':[lambda y,s: SMOTEENN(smote=SMOTE(k_neighbors=k_(y),random_state=s),random_state=s)],
 '결합|SMOTE+Tomek':[lambda y,s: SMOTETomek(smote=SMOTE(k_neighbors=k_(y),random_state=s),random_state=s)],
 '앙상블|BalancedRandomForest':'brf', '앙상블|EasyEnsemble':'easy'}
def fit_pred(Xt,y,Xs,how,sd):
    if how=='brf': m=BalancedRandomForestClassifier(300,max_depth=4,min_samples_leaf=3,sampling_strategy='all',replacement=True,bootstrap=False,n_jobs=-1,random_state=sd)
    elif how=='easy': m=EasyEnsembleClassifier(n_estimators=20,n_jobs=-1,random_state=sd)
    else:
        if how:
            for f in how:
                try: Xt,y=f(y,sd).fit_resample(Xt,y)
                except Exception as e: return None
        m=RandomForestClassifier(300,max_depth=4,min_samples_leaf=3,class_weight='balanced_subsample' if how is None else None,n_jobs=-1,random_state=sd)
    return m.fit(Xt,y).predict_proba(Xs)[:,1]
rows=[]
for target in ('가스','미성형'):
    y0=(t==target).astype(int); use=ev&((y0==1)|(t=='정상'))
    for mn,how in METHODS.items():
        res=[]
        for sd in range(3):
            s=np.zeros(len(L)); ok=True
            for r in pd.unique(runs):
                te=runs==r; tr=(~te)&use
                if y0[tr].sum()<3: continue
                pr=fit_pred(X[tr],y0[tr],X[te],how,sd)
                if pr is None: ok=False; break
                s[te]=pr
            if not ok: break
            e=evaluate(p,L,s,frame1=False); res.append((e['② AP'],e[f'AP_{target}'],e['_c']['가스'],e['_c']['미성형']))
        if not res: rows.append({'타깃':target,'방식':mn,'비고':'실행 불가'}); continue
        a=np.array(res)
        rows.append({'타깃':target,'방식':mn,'② AP':a[:,0].mean(),f'타깃 AP':a[:,1].mean(),'가스 평균':a[:,2].mean(),'가스 범위':f'{a[:,2].min():.0f}~{a[:,2].max():.0f}','미성형 평균':a[:,3].mean(),'미성형 범위':f'{a[:,3].min():.0f}~{a[:,3].max():.0f}'})
        print(target,mn,'done',flush=True)
d=pd.DataFrame(rows); d.to_csv('resample_rg3.csv',index=False)
print(d.round(3).to_string())
