"""
guidebook_semi.py — KAMP 가이드북 2.3.2 준지도 학습을 '그대로' 재현 + 누수 요인만 하나씩 바꿔 보기

가이드북 방식 (semi_supervised_guidebook.ipynb와 같은 로직)
  - 데이터: moldset_labeled_{p}.csv / moldset_unlabeled_{p}.csv (2차 가공, 파일별 z-score)
  - 분할: 행 단위 StratifiedShuffleSplit 7:3
  - 루프: labeled로 학습 → unlabeled 예측 → 확신도(max 확률) 상위 10%에 예측값을 pseudo-label →
          labeled에 합쳐 재학습 → unlabeled 90% 쓸 때까지 반복
  - 모델: SVM(C=0.001, gamma=0.01, class_weight {0:100, 1:1}) / RF / GNB  (DNN은 기존 노트북 참고)
  - 평가: 30% 평가셋 — 정확도, 정밀도, 재현율, ROC-AUC(0/1 예측 기준), F1 (+ 확률 기반 PR-AUC 추가)

바꿔 보는 요인 (한 번에 하나씩)
  split   : 'guidebook'(행 단위) / 'shot'(LH·RH 같은 샷은 같은 쪽)
            → 행 단위면 평가 행의 '쌍둥이 행'(공정값 완전 동일)이 학습에 들어감
  overlap : 'keep'(가이드북) / 'remove'(labeled와 같은 샷을 unlabeled에서 제거)
            → 2차 labeled 샷은 unlabeled에도 들어 있음(CN7 606/606) → 평가 샷이 pseudo-label 학습에 들어감
  seed    : 분할 난수 — 평가 불량이 5~8개라 분할마다 결과가 크게 흔들림
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix)

WEIGHTS = {0: 100.0, 1: 1.0}          # 가이드북 [코드 50] 그대로


def model_factory(name, seed=42, rf_trees=1200):
    if name == 'SVM':
        return SVC(C=0.001, kernel='rbf', gamma=1e-2, class_weight=WEIGHTS, probability=True, random_state=seed)
    if name == 'RF':
        return RandomForestClassifier(bootstrap=False, max_depth=10, max_features='sqrt', min_samples_leaf=1,
                                      min_samples_split=5, n_estimators=rf_trees, class_weight=WEIGHTS,
                                      n_jobs=-1, random_state=seed)
    if name == 'GNB':
        return GaussianNB()
    raise ValueError(name)


# ====================================================================== 데이터
def load_moldset(p, data_dir='../data', linked_dir='./linked'):
    """반환: labeled(+shot_id), unlabeled(+TimeStamp, labeled 샷 여부), FEATURES"""
    lab = pd.read_csv(f'{data_dir}/moldset_labeled_{p}.csv', low_memory=False)
    unl = pd.read_csv(f'{data_dir}/moldset_unlabeled_{p}.csv', low_memory=False)
    feats = [c for c in lab.columns if c not in ('Unnamed: 0', 'PassOrFail')]
    lab['shot_id'] = lab.groupby(feats, sort=False).ngroup()

    # unlabeled 'Unnamed: 0' = linked_unlabeled.sec_index → TimeStamp 연결
    lu = pd.read_csv(f'{linked_dir}/linked_unlabeled_{p}.csv', usecols=['sec_index', 'TimeStamp'])
    unl = unl.merge(lu.rename(columns={'sec_index': 'Unnamed: 0'}), on='Unnamed: 0', how='left')
    ll = pd.read_csv(f'{linked_dir}/linked_labeled.csv', usecols=['product', 'TimeStamp'])
    lab_ts = set(ll.loc[ll['product'] == p, 'TimeStamp'])
    unl['labeled와_같은샷'] = unl['TimeStamp'].isin(lab_ts)
    print(f'[{p}] labeled {len(lab)}행 / {lab["shot_id"].nunique()}샷, 불량 {int(lab["PassOrFail"].sum())}행 | '
          f'unlabeled {len(unl)}행 중 labeled와 같은 샷 {int(unl["labeled와_같은샷"].sum())}행')
    return lab, unl, feats


def split_data(lab, unl, feats, split='guidebook', overlap='keep', test_size=0.3, seed=42, unl_sample=5000):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    if split == 'guidebook':
        tr_i, te_i = next(sss.split(lab[feats], lab['PassOrFail']))
    else:
        sy = lab.groupby('shot_id')['PassOrFail'].max()
        a, b = next(sss.split(sy.index.to_frame(), sy.values))
        tr_i = np.flatnonzero(lab['shot_id'].isin(sy.index[a]))
        te_i = np.flatnonzero(lab['shot_id'].isin(sy.index[b]))
    tr, te = lab.iloc[tr_i], lab.iloc[te_i]
    u = unl if overlap == 'keep' else unl[~unl['labeled와_같은샷']]
    n_overlap_in_pool = int(u['labeled와_같은샷'].sum())
    if unl_sample is not None and unl_sample < len(u):
        u = u.sample(unl_sample, random_state=seed)
    info = {'train 행': len(tr), 'train 불량': int(tr['PassOrFail'].sum()), 'test 행': len(te),
            'test 불량': int(te['PassOrFail'].sum()),
            '쌍둥이 누수(test 행 중 train에 같은 공정값)': int(te['shot_id'].isin(tr['shot_id']).sum()),
            'unlabeled 풀의 labeled 샷 행': n_overlap_in_pool,
            '사용 unlabeled 중 labeled 샷 행': int(u['labeled와_같은샷'].sum())}
    return (tr[feats].reset_index(drop=True), tr['PassOrFail'].reset_index(drop=True),
            te[feats].reset_index(drop=True), te['PassOrFail'].reset_index(drop=True),
            u[feats].reset_index(drop=True), info)


# ====================================================================== 가이드북 루프 ([코드 50])
def _scores(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'정확도': accuracy_score(y, pred), '정밀도': precision_score(y, pred, zero_division=0),
            '재현율': recall_score(y, pred, zero_division=0),
            'ROC-AUC(0/1)': roc_auc_score(y, pred) if len(set(y)) > 1 else np.nan,
            'F1': f1_score(y, pred, zero_division=0),
            'PR-AUC(확률)': average_precision_score(y, score) if y.sum() else np.nan,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}


def guidebook_loop(make_model, Xtr, ytr, Xte, yte, U, percentage=10, unlabeled_usage=90):
    X, y, pool = Xtr.copy(), ytr.copy(), U.copy()
    left = int(len(pool) * (100 - unlabeled_usage) * 0.01)
    hist = []
    while True:
        m = make_model()
        m.fit(X, y)
        hist.append({'학습 행': len(X), 'pseudo 불량': int(y.sum() - ytr.sum()),
                     **_scores(yte.values, m.predict(Xte), m.predict_proba(Xte)[:, 1])})
        if len(pool) < left:
            break
        conf = m.predict_proba(pool).max(axis=1)
        order = np.argsort(-conf)
        k = max(1, int(len(pool) * percentage * 0.01))
        chosen, pool = pool.iloc[order[:k]], pool.iloc[order[k:]]
        X = pd.concat([X, chosen], ignore_index=True)
        y = pd.concat([y, pd.Series(m.predict(chosen), name='PassOrFail')], ignore_index=True)
    return pd.DataFrame(hist)


def run_grid(products=('cn7', 'rg3'), splits=('guidebook', 'shot'), overlaps=('keep', 'remove'),
             models=('SVM', 'RF', 'GNB'), seeds=(42,), unl_sample=5000, rf_trees=1200,
             data_dir='../data', linked_dir='./linked'):
    rows, hists = [], {}
    for p in products:
        lab, unl, feats = load_moldset(p, data_dir, linked_dir)
        for sp in splits:
            for ov in overlaps:
                for sd in seeds:
                    Xtr, ytr, Xte, yte, U, info = split_data(lab, unl, feats, sp, ov, seed=sd, unl_sample=unl_sample)
                    for mn in models:
                        h = guidebook_loop(lambda: model_factory(mn, sd, rf_trees), Xtr, ytr, Xte, yte, U)
                        key = (p, sp, ov, sd, mn)
                        hists[key] = h
                        base = {'제품': p, '분할': sp, 'unlabeled 중복': ov, 'seed': sd, '모델': mn, **info}
                        rows.append({**base, '단계': '라벨만(반복0)', **h.iloc[0].to_dict()})
                        rows.append({**base, '단계': '준지도(최종)', **h.iloc[-1].to_dict()})
                        print(f'  {key}  F1 {h.iloc[0]["F1"]:.2f} → {h.iloc[-1]["F1"]:.2f}, '
                              f'pseudo 불량 {int(h.iloc[-1]["pseudo 불량"])}')
    return pd.DataFrame(rows), hists
