"""
f1_eval.py — 선택된 조합의 F1 평가 (제품별 LORO, 샷 단위)
임계값은 테스트 Run을 보지 않고 결정 (A: 정상 분위수, B: 내부 LORO 튜닝, C: 검사 예산)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from features import FoldFeaturizer, combo_columns, needs_fold
from loro_eval import make_model, UNSUPERVISED, _X

FRACTIONS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]   # 비율 임계값 후보
NORMAL_Q = [0.99, 0.995]                                               # A 방법 분위수


# ---------------------------------------------------------------- 모델 학습·점수
def _prep(tr, te, combo):
    if needs_fold(combo):
        ff = FoldFeaturizer().fit(tr)
        tr, te = ff.transform(tr), ff.transform(te)
    cols = combo_columns(combo)
    return _X(tr, cols), _X(te, cols), tr['PassOrFail'].values


def fit_and_score(tr, te, combo, model_name):
    """반환: (테스트 점수, 학습 양품 점수)  — 클수록 불량 의심"""
    Xtr, Xte, ytr = _prep(tr, te, combo)
    m = make_model(model_name)
    if model_name in UNSUPERVISED:
        m.fit(Xtr[ytr == 0])
        return -m.score_samples(Xte), -m.score_samples(Xtr[ytr == 0])
    if len(np.unique(ytr)) < 2:
        return np.zeros(len(Xte)), np.zeros(int((ytr == 0).sum()))
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m.predict_proba(Xtr[ytr == 0])[:, 1]


# ---------------------------------------------------------------- 판정 함수
def flag_by_fraction(scores, runs, frac):
    """Run마다 상위 frac 비율을 불량(1)으로 판정"""
    scores, runs = np.asarray(scores), np.asarray(runs)
    pred = np.zeros(len(scores), dtype=int)
    for r in np.unique(runs):
        idx = np.flatnonzero(runs == r)
        k = max(1, int(np.ceil(frac * len(idx))))
        pred[idx[np.argsort(-scores[idx])[:k]]] = 1
    return pred


def _f1(y, pred):
    return precision_recall_fscore_support(y, pred, average='binary', zero_division=0)[2]


def best_threshold(y, s):
    """점수 임계값 후보 중 F1 최대 (동점이면 높은 임계값 = 보수적으로)"""
    if y.sum() == 0:
        return np.inf
    cand = np.unique(s)
    best_t, best_f = np.inf, -1
    for t in cand[::-1]:
        f = _f1(y, (s >= t).astype(int))
        if f > best_f:
            best_t, best_f = t, f
    return best_t


def best_fraction(y, s, runs):
    if y.sum() == 0:
        return FRACTIONS[0]
    fs = [_f1(y, flag_by_fraction(s, runs, f)) for f in FRACTIONS]
    return FRACTIONS[int(np.argmax(fs))]


# ---------------------------------------------------------------- LORO + 내부 LORO
def run_f1_loro(d, combo, model_name):
    """
    반환: 샷별 DataFrame (run_id, y, Reason, score, pred_A99, pred_A995, pred_Bthr, pred_Bfrac)
    B 방법: 학습 Run들로 내부 LORO → 내부 OOF 점수로 임계값/비율 선택 → 바깥 테스트 Run에 적용
    """
    out = []
    runs = d['run_id'].unique()
    for test_run in runs:
        tr = d[d['run_id'] != test_run]
        te = d[d['run_id'] == test_run]

        # 바깥 모델
        s_te, s_tr_good = fit_and_score(tr, te, combo, model_name)

        # 내부 LORO (학습 Run 안에서만)
        inner = []
        for r in tr['run_id'].unique():
            itr, ite = tr[tr['run_id'] != r], tr[tr['run_id'] == r]
            s_in, _ = fit_and_score(itr, ite, combo, model_name)
            inner.append(pd.DataFrame({'run_id': r, 'y': ite['PassOrFail'].values, 's': s_in}))
        inner = pd.concat(inner, ignore_index=True)
        thr = best_threshold(inner['y'].values, inner['s'].values)
        frac = best_fraction(inner['y'].values, inner['s'].values, inner['run_id'].values)

        res = pd.DataFrame({'run_id': test_run, 'shot_seq': te['shot_seq_in_run'].values,
                            'y': te['PassOrFail'].values, 'Reason': te['Reason'].values,
                            'score': s_te,
                            'pred_Bthr': (s_te >= thr).astype(int),
                            'pred_Bfrac': flag_by_fraction(s_te, np.repeat(test_run, len(te)), frac),
                            'B_thr': thr, 'B_frac': frac})
        if model_name in UNSUPERVISED:                   # A: 학습 양품 점수 분위수
            for q in NORMAL_Q:
                res[f'pred_A{int(q * 1000)}'] = (s_te >= np.quantile(s_tr_good, q)).astype(int)
        out.append(res)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- 성능표
def f1_report(oof, eval_mask=None, label=''):
    o = oof if eval_mask is None else oof[eval_mask]
    y = o['y'].values
    rows = []
    methods = [c for c in o.columns if c.startswith('pred_')]
    for m in methods:
        p, r, f, _ = precision_recall_fscore_support(y, o[m].values, average='binary', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, o[m].values, labels=[0, 1]).ravel()
        rows.append({'평가셋': label, '방법': m, 'F1': f, 'Precision': p, 'Recall': r,
                     'TP': tp, 'FP': fp, 'FN': fn, '판정 불량 수': tp + fp})
    for k in [0.05, 0.10]:                              # C: 검사 예산
        pred = flag_by_fraction(o['score'].values, o['run_id'].values, k)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({'평가셋': label, '방법': f'C_top{int(k * 100)}%', 'F1': f, 'Precision': p, 'Recall': r,
                     'TP': tp, 'FP': fp, 'FN': fn, '판정 불량 수': tp + fp})
    # D: 사후 최적 (참고용, 보고 금지)
    t = best_threshold(y, o['score'].values)
    pred = (o['score'].values >= t).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rows.append({'평가셋': label, '방법': 'D_사후최적(참고)', 'F1': f, 'Precision': p, 'Recall': r,
                 'TP': tp, 'FP': fp, 'FN': fn, '판정 불량 수': tp + fp})
    return pd.DataFrame(rows)


def type_recall(oof, method):
    """판정 방법별 불량 유형 탐지 수"""
    o = oof[oof['y'] == 1]
    return (o.groupby('Reason')[method].agg(['sum', 'size'])
             .rename(columns={'sum': '탐지', 'size': '전체'}))


def run_detail(oof, method):
    """Run별: 불량 수, 탐지, 오탐"""
    g = oof.groupby('run_id')
    return pd.DataFrame({'샷': g.size(), '불량': g['y'].sum(),
                         '탐지(TP)': g.apply(lambda x: int(((x[method] == 1) & (x['y'] == 1)).sum())),
                         '오탐(FP)': g.apply(lambda x: int(((x[method] == 1) & (x['y'] == 0)).sum()))})