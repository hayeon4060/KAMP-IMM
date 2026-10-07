"""
slices.py — 비라벨 학습 구간 정의 (대상 제품 p 기준)

각 구간 = (이름, 축, 설명, 행 마스크 함수). 캐시 F(전 흐름 파생변수)에 마스크만 씌워 고른다.
라벨 시작일: CN7 10/16, RG3 10/21 (그 이전 = 과거만, 이후 = 같은 시기, 단 라벨 Run ±60분은 이미 제거됨)
"""
import numpy as np
import pandas as pd

import common as C

LAB_START = {'cn7': pd.Timestamp('2020-10-16'), 'rg3': pd.Timestamp('2020-10-21')}
OTHER = {'cn7': 'RG3', 'rg3': 'CN7'}


def _lot_rate(F):
    """로트별 불량률 대리값 = 로트 불량 수(측면 합) / 로트 샷 수"""
    n = F.groupby('lot')['lot'].transform('size')
    return F['lot_err'] / n


def define(F, p):
    P = p.upper()
    t = F['TimeStamp']
    w2 = F['equip'] == C.EQUIP_W2
    same = F['prod'] == P
    ws = w2 & same
    s0 = LAB_START[p]
    S = []

    def add(name, axis, desc, m):
        S.append((name, axis, desc, np.asarray(m, bool)))

    # --- 기간 (우진2 · 같은 제품)
    add('cur_0901', '기간', '현 제출: 우진2·같은제품·09-01 이후', ws & (t >= '2020-09-01'))
    add('w2_all', '기간', '우진2·같은제품·전체 기간', ws)
    add('w2_pre', '기간', '라벨 시작 이전 전체', ws & (t < s0))
    for wk in (2, 4, 8, 12):
        add(f'w2_last{wk}w', '기간', f'라벨 직전 {wk}주', ws & (t < s0) & (t >= s0 - pd.Timedelta(weeks=wk)))
    add('w2_concurrent', '기간', '라벨 시작 이후(같은 시기, 라벨 Run±60분 제외)', ws & (t >= s0))
    add('w2_H1', '기간', '3~6월', ws & (t < '2020-07-01'))
    add('w2_H2', '기간', '7~10월', ws & (t >= '2020-07-01'))
    for mo in range(3, 11):
        m = ws & (t.dt.month == mo)
        if m.sum() >= 300:
            add(f'w2_m{mo:02d}', '월별', f'{mo}월만', m)
    # --- 설비
    oth = same & ~w2
    if oth.sum() >= 300:
        add('eq_other', '설비', '같은제품·다른 설비만', oth)
    add('eq_all', '설비', '같은제품·전 설비', same)
    for e in sorted(F.loc[oth, 'equip'].unique()):
        m = same & (F['equip'] == e)
        if m.sum() >= 300:
            add(f'eq_{e}', '설비', f'같은제품·{e}만', m)
    # --- 제품
    add('pr_other', '제품', f'우진2·상대제품({OTHER[p]})만', w2 & (F['prod'] == OTHER[p]))
    add('pr_cn7rg3_w2', '제품', '우진2·CN7+RG3', w2 & F['prod'].isin(['CN7', 'RG3']))
    add('pr_cn7rg3_all', '제품', '전 설비·CN7+RG3', F['prod'].isin(['CN7', 'RG3']))
    add('pr_w2_allparts', '제품', '우진2·전 부품', w2)
    add('pr_allparts_exsame', '제품', '전 설비·같은제품 제외 전 부품', ~same)
    add('pr_everything', '제품', '전 설비·전 부품', np.ones(len(F), bool))
    # --- 정제 (우진2·같은제품·전체 기준)
    add('cl_skipstart20', '정제', '전체 기간, Run 첫 20샷 제외', ws & (F['seq_in_run'] > 20))
    add('cl_skipstart50', '정제', '전체 기간, Run 첫 50샷 제외', ws & (F['seq_in_run'] > 50))
    lr = _lot_rate(F)
    med = lr[ws].median()
    add('cl_lowerr_lot', '정제', '전체 기간, 로트 불량률 중앙값 이하 로트', ws & (lr <= med))
    add('cl_higherr_lot', '정제', '(대조) 로트 불량률 중앙값 초과 로트', ws & (lr > med))
    add('cl_zeroerr_lot', '정제', '전체 기간, 로트 불량 0 로트', ws & (F['lot_err'] == 0))
    dv = F[C.FEAT_DEV].abs()
    add('cl_quiet', '정제', '전체 기간, 모든 변수 |편차|≤5 샷만', ws & (dv.max(axis=1).fillna(0) <= 5))
    add('cl_cur_skipstart20', '정제', '현 제출 구간 + Run 첫 20샷 제외',
        ws & (t >= '2020-09-01') & (F['seq_in_run'] > 20))
    add('cl_cur_lowerr', '정제', '현 제출 구간 + 로트 불량률 중앙값 이하', ws & (t >= '2020-09-01') & (lr <= med))
    return S


def subsample_mask(base_mask, n, seed):
    idx = np.flatnonzero(base_mask)
    m = np.zeros(len(base_mask), bool)
    if len(idx) <= n:
        m[idx] = True
    else:
        m[np.random.default_rng(seed).choice(idx, n, replace=False)] = True
    return m
