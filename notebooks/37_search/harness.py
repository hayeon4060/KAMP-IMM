import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 37_search/harness.py
"""
harness.py — 37번 탐색 공통 도구: 데이터 적재 / 평가 / 목표 판정 / 탐색 보정 순열검정

목표 (사용자 지정, '잡는다' = 검사 예산 10%: 평가 대상 샷 중 점수 상위 10%에 들어온 불량 수)
  CN7: 미성형 3~4/4 유지 + 가스 ≥ 6/11 (절반)
  RG3 (R6 제외): 가스 ≥ 7/17, 미성형 ≥ 4/8
  + 최종 모델보다 ② AP 향상
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from final_model import (ALL_RAW, raw, FEATURES, INITIAL, TYPE_GROUPS, load_labeled_shots, load_official_rows,
                         eval_frame1, apply_exclusions)

MD = './outputs/final_model'
TARGET = {'cn7': {'가스': 6, '미성형': 3}, 'rg3': {'가스': 7, '미성형': 4}}


def load(p):
    ref = pd.read_csv(f'{MD}/{p}_train_features.csv', parse_dates=['TimeStamp'])
    lab = pd.read_csv(f'{MD}/{p}_scores.csv', parse_dates=['TimeStamp'])
    lab['Reason'] = lab['Reason'].fillna('정상')
    lab = apply_exclusions(lab, p)
    rc = [raw(v) for v in ALL_RAW if raw(v) in ref.columns]
    sh = load_labeled_shots(p)[['TimeStamp'] + rc]
    lab = lab.merge(sh, on='TimeStamp', how='left')
    for d in (ref, lab):
        d.sort_values(['run_id', 'TimeStamp'], inplace=True)
        d.reset_index(drop=True, inplace=True)
        g = d.groupby('run_id', sort=False)
        gap = g['TimeStamp'].diff().dt.total_seconds()
        d['gap'] = gap
        rs = gap.isna() | (gap > 300)
        d['since_restart'] = d.groupby([d['run_id'], rs.groupby(d['run_id']).cumsum()]).cumcount() + 1
        d['seq'] = g.cumcount() + 1
    lab['eval'] = (lab['split'] != '제외') & (lab['Reason'] != INITIAL)
    lab['type'] = np.select([lab['Reason'].isin(TYPE_GROUPS['가스']), lab['Reason'].isin(TYPE_GROUPS['미성형'])],
                            ['가스', '미성형'], np.where(lab['PassOrFail'] == 1, '기타', '정상'))
    return ref, lab, rc


def _clean(x):
    x = np.asarray(x, float).copy()
    ok = np.isfinite(x)
    x[~ok] = (x[ok].min() - 1) if ok.any() else 0
    return x


def topk_counts(lab, s, k=0.10, by_run=False):
    m = lab['eval'].values
    s = _clean(s)[m] + np.random.default_rng(0).random(m.sum()) * 1e-9
    t = lab['type'].values[m]
    sel = np.zeros(m.sum(), bool)
    if by_run:
        runs = lab['run_id'].values[m]
        for r in pd.unique(runs):
            idx = np.flatnonzero(runs == r)
            sel[idx[np.argsort(-s[idx])[:int(np.ceil(k * len(idx)))]]] = True
    else:
        sel[np.argsort(-s)[:int(np.ceil(k * len(s)))]] = True
    return {ty: int((sel & (t == ty)).sum()) for ty in ('가스', '미성형')}


def evaluate(p, lab, s, name='', frame1=True):
    m = lab['eval'].values
    sc = _clean(s)
    y = (lab['PassOrFail'].values[m] == 1).astype(int)
    t = lab['type'].values[m]
    r = {'제품': p, '모델': name, '② AP': average_precision_score(y, sc[m])}
    for ty in ('가스', '미성형'):
        mm = (t == ty) | (y == 0)
        r[f'AP_{ty}'] = average_precision_score((t[mm] == ty).astype(int), sc[m][mm])
    c = topk_counts(lab, sc)
    cr = topk_counts(lab, sc, by_run=True)
    n = {ty: int((t == ty).sum()) for ty in ('가스', '미성형')}
    r['상위10% 가스'] = f"{c['가스']}/{n['가스']}"
    r['상위10% 미성형'] = f"{c['미성형']}/{n['미성형']}"
    r['Run별10% 가스'] = f"{cr['가스']}/{n['가스']}"
    r['Run별10% 미성형'] = f"{cr['미성형']}/{n['미성형']}"
    tg = TARGET[p]
    r['목표달성'] = 'O' if (c['가스'] >= tg['가스'] and c['미성형'] >= tg['미성형']) else '-'
    if frame1:
        f1 = eval_frame1(p, load_official_rows(p), pd.Series(sc, index=lab['TimeStamp'].values))
        r['① AP'] = f1['AP']
    r['_c'] = c
    return r


def circ_null(lab, rng):
    """라벨 원형 이동: Run마다 라벨 순서를 임의 오프셋만큼 돌림 → 불량의 '몰림 구조'는 유지하고 위치만 무작위"""
    out = lab.copy()
    for r, idx in out.groupby('run_id', sort=False).groups.items():
        idx = np.asarray(list(idx))
        sh = rng.integers(0, len(idx))
        for c in ('PassOrFail', 'Reason', 'type', 'eval'):
            out.loc[idx, c] = np.roll(out.loc[idx, c].values, sh)
    return out


def search_adjusted_p(p, lab, scores, n=500, seed=0, stat='② AP'):
    """
    탐색 보정 검정: 여러 점수 중 '최고'를 고른 과정 자체를 무작위 라벨(원형 이동)에서 반복
    → 관측 최고값이 무작위 최고값 분포의 몇 %인지 (선택 편향까지 반영한 p)
    """
    rng = np.random.default_rng(seed)
    obs = max(evaluate(p, lab, s, frame1=False)[stat] for s in scores.values())
    null = []
    for _ in range(n):
        L = circ_null(lab, rng)
        if L.loc[L['eval'], 'PassOrFail'].sum() == 0:
            continue
        null.append(max(evaluate(p, L, s, frame1=False)[stat] for s in scores.values()))
    null = np.asarray(null)
    return obs, float(np.mean(null)), float(np.quantile(null, 0.95)), (1 + (null >= obs).sum()) / (len(null) + 1)
