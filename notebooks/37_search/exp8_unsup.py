import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/exp8_unsup.py
from common import *
from harness import load, evaluate
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from final_model import FEATURES
import pickle
pd.set_option('display.width',250)
def fam(Xl, Xr):
    cols=[c for c in Xl.columns if Xr[c].notna().mean()>0.5 and Xr[c].nunique()>2]
    return cols
rows=[]; ALL={}
for p in ('cn7','rg3'):
    ref, lab, rc = load(p)
    Xl=bank(lab); Xr=bank(ref)
    for f in FEATURES: Xl['F|'+f]=lab[f].values; Xr['F|'+f]=ref[f].values
    Xl=Xl.replace([np.inf,-np.inf],np.nan); Xr=Xr.replace([np.inf,-np.inf],np.nan)
    good=fam(Xl,Xr)
    FAM={'std_all':[c for c in good if 'std' in c],
         'free17+std':[c for c in good if c.startswith('F|') or 'std' in c],
         'nolevel_all':[c for c in good if '수준' not in c and 'Run중앙값' not in c]}
    m=lab['eval'].values
    rows.append(evaluate(p,lab,lab['score'].values,'final'))
    for fn,cols in FAM.items():
        z=robust(Xr[cols].values); Zr=z(Xr[cols].values); Zl=z(Xl[cols].values)
        sc={}
        sc['maha']=maha(Xr[cols].values,Xl[cols].values)[0]
        sc['iforest']=-IsolationForest(n_estimators=300,random_state=0).fit(Zr).score_samples(Zl)
        nn=NearestNeighbors(n_neighbors=10).fit(Zr[np.random.default_rng(0).choice(len(Zr),min(4000,len(Zr)),replace=False)])
        sc['knn10']=nn.kneighbors(Zl)[0].mean(1)
        pca=PCA(n_components=0.9,random_state=0).fit(Zr); sc['pca_recon']=((Zl-pca.inverse_transform(pca.transform(Zl)))**2).sum(1)
        for mn,s in sc.items():
            nm=f'{fn}|{mn}'; ALL[(p,nm)]=s
            rows.append(evaluate(p,lab,s,nm))
            fr=lab['score'].values
            R=[np.zeros(len(s)),np.zeros(len(s))]
            R[0][m]=rankdata(fr[m])/m.sum(); R[1][m]=rankdata(s[m])/m.sum()
            ALL[(p,'interleave(final,'+nm+')')]=np.maximum(*R)
            rows.append(evaluate(p,lab,np.maximum(*R),'interleave(final,'+nm+')'))
df=pd.DataFrame(rows).drop(columns='_c')
print(df.round(3).to_string())
pickle.dump(ALL,open('unsup_scores.pkl','wb'))
