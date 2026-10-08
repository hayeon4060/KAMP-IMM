"""
op_metrics.py — 운영 기준별 Precision / Recall / F1 + 유형별 '몇 개 중 몇 개' (모든 노트북 공통)

'잡았다'의 기준 (둘 다 계산)
  - 검사 예산: 평가 대상 샷 중 점수 상위 k% (기본 5%·10%)를 검사했을 때 그 안에 들어온 불량
  - 경보     : 정상 기준 백분위 pct ≥ q (기본 0.99·0.999)인 샷 = 경보, 그중 불량 (pct가 있을 때만)
평가 대상: lab['eval'] (초기허용불량·RG3 R6 제외), 불량 = PassOrFail == 1, 유형 = lab['type'] (가스/미성형/기타)
"""
import numpy as np
import pandas as pd


def _fill(s):
    s = np.asarray(s, float).copy()
    ok = np.isfinite(s)
    s[~ok] = (s[ok].min() - 1) if ok.any() else 0
    return s


def _types(lab):
    if 'type' in lab.columns:
        return lab['type'].values
    r = lab['Reason'].fillna('정상').values
    return np.where(r == '가스', '가스', np.where(np.isin(r, ['미성형', '가스+미성형']), '미성형',
                                               np.where(lab['PassOrFail'].values == 1, '기타', '정상')))


def _row(name, sel, y, t):
    tp = int((sel & (y == 1)).sum())
    fp = int((sel & (y == 0)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / y.sum() if y.sum() else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    r = {'기준': name, '검사(경보) 샷': int(sel.sum()), 'TP': tp, 'FP': fp,
         'Precision': P, 'Recall': R, 'F1': F, '불량': f'{tp}/{int(y.sum())}'}
    for ty in ('가스', '미성형'):
        n = int((t == ty).sum())
        if n:
            r[ty] = f'{int((sel & (t == ty)).sum())}/{n}'
    return r


def op_table(lab, score, pct=None, budgets=(0.05, 0.10), qs=(0.99, 0.999), eval_col='eval'):
    """반환: 기준별 표 (DataFrame)"""
    m = lab[eval_col].values if eval_col in lab.columns else np.ones(len(lab), bool)
    idx = np.flatnonzero(m)
    y = (lab['PassOrFail'].values[idx] == 1).astype(int)
    t = _types(lab)[idx]
    s = _fill(score)[idx] + np.random.default_rng(0).random(len(idx)) * 1e-9
    rows = []
    for b in budgets:
        sel = np.zeros(len(idx), bool)
        sel[np.argsort(-s)[:int(np.ceil(b * len(idx)))]] = True
        rows.append(_row(f'검사 상위 {int(b * 100)}%', sel, y, t))
    if pct is not None:
        pc = np.asarray(pct, float)[idx]
        for q in qs:
            rows.append(_row(f'경보 pct≥{q}', pc >= q, y, t))
    return pd.DataFrame(rows)


def op_flat(lab, score, pct=None, budgets=(0.05, 0.10), qs=(0.99, 0.999)):
    """한 줄 요약용: {'상위10% P': .., '상위10% R': .., '상위10% F1': .., '상위10% 불량': 'a/b', ...}"""
    T = op_table(lab, score, pct, budgets, qs)
    out = {}
    for _, r in T.iterrows():
        k = r['기준'].replace('검사 상위 ', '상위').replace('경보 pct≥', 'q')
        out[f'{k} 불량'] = r['불량']
        out[f'{k} P'] = r['Precision']
        out[f'{k} R'] = r['Recall']
        out[f'{k} F1'] = r['F1']
    return out
