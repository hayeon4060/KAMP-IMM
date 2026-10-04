"""
check_logzero.py — 18번 'divide by zero encountered in log' 경고 점검

경고 위치: compare_frames.physics_relations (팀원 물리 관계 R1~R5)
  R1 = log(Filling_Time) + log(Max_Injection_Speed)
  R2 = log(Plasticizing_Time) + log(Max_Screw_RPM) − log(Plasticizing_Position)
  R4 = Max_Switch_Over_Pressure / Max_Injection_Pressure
  → 값이 0이면 log(0) = −inf (또는 0으로 나눔) → inf를 NaN으로 바꿈

현재 처리 (18번 T_phys300)
  - 기준(과거 300샷)에서 NaN 행은 빠짐
  - 대상 샷 관계가 NaN이면 점수 0 = '가장 정상'으로 취급  ← 문제 가능성
    (속도·계량시간이 0인 샷은 오히려 이상 샷일 수 있음)

점검
  1) 어떤 데이터·변수에 0이 몇 개 있나
  2) labeled 샷 중 관계가 NaN인 샷 — 불량이 섞였나
  3) 처리 방식만 바꿔서 성능 비교
       skip    : 18번과 동일 (NaN 샷 점수 0)
       max     : NaN인 관계의 z를 상한값 20으로 (= '관계 계산 불가 = 최대 이상')
"""
import numpy as np
import pandas as pd

from features import raw
from compare_frames import physics_relations, robust_z, _rms

LOG_VARS = {'R1': ['Filling_Time', 'Max_Injection_Speed'],
            'R2': ['Plasticizing_Time', 'Max_Screw_RPM', 'Plasticizing_Position'],
            'R4': ['Max_Injection_Pressure']}
CHECK_VARS = list(dict.fromkeys(v for vs in LOG_VARS.values() for v in vs))


# ====================================================================== 1) 0 값 개수
def zero_table(datasets):
    """datasets: {이름: df} → 변수별 0 이하 개수"""
    rows = []
    for name, d in datasets.items():
        for v in CHECK_VARS:
            c = raw(v)
            if c not in d.columns:
                continue
            x = pd.to_numeric(d[c], errors='coerce')
            rows.append({'데이터': name, '변수': v, '샷 수': len(d), '0 이하': int((x <= 0).sum()),
                         '비율': float((x <= 0).mean()), 'NaN': int(x.isna().sum())})
    return pd.DataFrame(rows)


def nan_relation_table(datasets):
    """관계 R1~R5별 NaN 개수"""
    rows = []
    for name, d in datasets.items():
        R = physics_relations(d)
        r = {'데이터': name, '샷 수': len(d), '관계 하나라도 NaN': int(R.isna().any(axis=1).sum())}
        for c in R.columns:
            r[f'{c} NaN'] = int(R[c].isna().sum())
        rows.append(r)
    return pd.DataFrame(rows)


# ====================================================================== 2) 영향 받은 labeled 샷
def affected_labeled(sh):
    R = physics_relations(sh)
    bad = R.isna().any(axis=1)
    cols = ['TimeStamp', 'run_id', 'PassOrFail', 'Reason'] + [raw(v) for v in CHECK_VARS if raw(v) in sh.columns]
    out = sh.loc[bad, [c for c in cols if c in sh.columns]].copy()
    na = R[bad].isna()
    out['NaN 관계'] = [','.join(na.columns[row]) for row in na.values]   # 0행이어도 안전
    return out


# ====================================================================== 3) 처리 방식별 T_phys300
def phys_scores(targets, base, K=300, minp=10, nan_policy='skip', nan_z=20.0):
    """
    final_compare._past_scores(rel=True)와 같은 계산 + NaN 처리 방식 선택
    nan_policy: 'skip' (18번과 동일, 점수 0) / 'max' (NaN 관계 z = nan_z)
    """
    BX = physics_relations(base).values
    TX = physics_relations(targets).values
    pos = np.searchsorted(base['TimeStamp'].values, targets['TimeStamp'].values, side='left')   # 엄격히 과거
    sc = np.zeros(len(TX))
    for i, k in enumerate(pos):
        P = BX[max(0, k - K):k]
        P = P[~np.isnan(P).any(axis=1)]
        if len(P) < minp:
            continue
        x = TX[i]
        m = np.isnan(x)
        if m.any():
            if nan_policy == 'skip':
                continue
            x = np.where(m, np.median(P, axis=0), x)
            z = robust_z(x, P)
            z[m] = nan_z
        else:
            z = robust_z(x, P)
        sc[i] = _rms(z)
    return sc
