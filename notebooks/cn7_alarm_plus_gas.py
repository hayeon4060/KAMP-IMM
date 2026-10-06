"""
cn7_alarm_plus_gas.py — CN7: 최종 모델 경보 + 가스 전문가로 검사 예산 나누기 (37번 반복 20~21)

규칙 (검사 예산 = 평가 대상 샷의 10%)
  1) 최종 모델(M_maha_free) 경보 샷(pct ≥ 0.999, 라벨 없이 정해지는 운영 기준) → 먼저 검사 (미성형 담당)
  2) 남은 예산을 가스 전문가 점수 순서로 채움
가스 전문가 (선택)
  - 'std4'     : 배럴2·호퍼·쿠션·배럴5의 직전 k샷 std → 정상 reference로 학습한 Mahalanobis (라벨 미사용, RG3에서 온 가설)
  - 'E_thermal': 온도 교란 점수 (호퍼 기울기↑ + 금형 편차↓ + 0.5·호퍼 편차↑, 최근 5샷 평균, 재가동 후 15샷 제외)
                 ⚠ R1 가스를 보고 성분을 고른 규칙 (사후)
검증: 라벨을 Run 안에서 원형 이동(불량이 몰린 구조 유지, 위치만 무작위) 5,000번 → 같은 규칙이 우연히 잡는 가스 수와 비교

실행 (notebooks 폴더에서):  python cn7_alarm_plus_gas.py
  사전 준비: 20a·20b 실행 결과(outputs/final_model/cn7_train_features.csv, cn7_scores.csv)
결과: outputs/cn7_alarm_plus_gas/summary.csv, selected_shots_{전문가}.csv (검사 대상으로 뽑힌 샷 목록)
"""
import os
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from final_model import raw, CLIP, INITIAL, TYPE_GROUPS, apply_exclusions

MD = './outputs/final_model'
OUT = './outputs/cn7_alarm_plus_gas'
BUDGET, ALARM_Q = 0.10, 0.999
STD4 = ['Barrel_Temperature_2', 'Hopper_Temperature', 'Cushion_Position', 'Barrel_Temperature_5']


def load(p='cn7'):
    ref = pd.read_csv(f'{MD}/{p}_train_features.csv', parse_dates=['TimeStamp'])
    lab = pd.read_csv(f'{MD}/{p}_scores.csv', parse_dates=['TimeStamp'])
    lab['Reason'] = lab['Reason'].fillna('정상')
    lab = apply_exclusions(lab, p)
    from final_model import load_labeled_shots, ALL_RAW
    rc = [raw(v) for v in ALL_RAW if raw(v) in ref.columns]
    lab = lab.merge(load_labeled_shots(p)[['TimeStamp'] + rc], on='TimeStamp', how='left')
    for d in (ref, lab):
        d.sort_values(['run_id', 'TimeStamp'], inplace=True)
        d.reset_index(drop=True, inplace=True)
        g = d.groupby('run_id', sort=False)
        gap = g['TimeStamp'].diff().dt.total_seconds()
        rs = gap.isna() | (gap > 300)
        d['since_restart'] = d.groupby([d['run_id'], rs.groupby(d['run_id']).cumsum()]).cumcount() + 1
        d['seq'] = g.cumcount() + 1
    lab['eval'] = (lab['split'] != '제외') & (lab['Reason'] != INITIAL)
    lab['type'] = np.select([lab['Reason'].isin(TYPE_GROUPS['가스']), lab['Reason'].isin(TYPE_GROUPS['미성형'])],
                            ['가스', '미성형'], np.where(lab['PassOrFail'] == 1, '기타', '정상'))
    return ref, lab


def _robust(Xr):
    med = np.nanmedian(Xr, 0)
    q75, q25 = np.nanpercentile(Xr, [75, 25], 0)
    sd = np.nanstd(Xr, 0)
    iqr = np.where(q75 - q25 > 1e-9, q75 - q25, np.where(sd > 1e-9, sd, 1.0))
    return lambda Z: np.nan_to_num(np.clip((Z - med) / iqr, -CLIP, CLIP), nan=0.0)


def gas_std4(ref, lab, k=10):
    std = lambda d, v: d[raw(v)].groupby(d['run_id'], sort=False).transform(lambda s: s.rolling(k, min_periods=3).std())
    Xr = np.column_stack([std(ref, v) for v in STD4])
    Xl = np.column_stack([std(lab, v) for v in STD4])
    z = _robust(Xr)
    lw = LedoitWolf().fit(z(Xr))
    return np.sqrt(lw.mahalanobis(z(Xl)))


def gas_E(ref, lab, use_mold=True):
    hs = ref['hopper_slope20'].dropna()
    med, iqr = hs.median(), max(hs.quantile(0.75) - hs.quantile(0.25), 1e-6)
    c = ((lab['hopper_slope20'] - med) / iqr).fillna(0).clip(0, 10) \
        + 0.5 * lab['dev_Hopper_Temperature'].fillna(0).clip(0, 20)
    if use_mold:
        c = c + (-lab['dev_Mold_Temperature_3'].fillna(0)).clip(0, 10)
    e = c.groupby(lab['run_id'], sort=False).transform(lambda s: s.rolling(5, min_periods=1).mean())
    return e.where(lab['since_restart'] > 15, -np.inf).values


def select(lab, gas_score):
    """경보 샷 먼저 + 남은 예산을 가스 점수 순으로 → 검사 대상(bool, 평가 대상 샷 기준)"""
    m = lab['eval'].values
    idx = np.flatnonzero(m)
    K = int(np.ceil(BUDGET * len(idx)))
    sel = lab['pct'].values[idx] >= ALARM_Q
    g = np.where(np.isfinite(gas_score[idx]), gas_score[idx], -1e18) + np.random.default_rng(0).random(len(idx)) * 1e-9
    for i in np.argsort(-g):
        if sel.sum() >= K:
            break
        sel[i] = True
    return idx, sel


def circ_null(lab, idx, sel, n=5000, seed=7):
    t = lab['type'].values[idx]
    runs = lab['run_id'].values[idx]
    groups = [np.flatnonzero(runs == r) for r in pd.unique(runs)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        tt = t.copy()
        for g in groups:
            tt[g] = np.roll(t[g], rng.integers(0, len(g)))
        out.append((sel & (tt == '가스')).sum())
    return np.asarray(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    ref, lab = load('cn7')
    idx, base_sel = select(lab, lab['score'].values)            # 비교: 최종 모델 단독 (예산 전부 최종 점수)
    experts = {'최종 모델 단독': lab['score'].values,
               'std4_k5': gas_std4(ref, lab, 5), 'std4_k10': gas_std4(ref, lab, 10), 'std4_k20': gas_std4(ref, lab, 20),
               'E_thermal (금형편차 포함, 사후)': gas_E(ref, lab, True),
               'E_thermal (금형편차 제외)': gas_E(ref, lab, False)}
    t_all = lab['type'].values
    rows = []
    for name, g in experts.items():
        idx, sel = select(lab, g)
        t = t_all[idx]
        null = circ_null(lab, idx, sel)
        gas, short = int((sel & (t == '가스')).sum()), int((sel & (t == '미성형')).sum())
        yy = (lab['PassOrFail'].values[idx] == 1)
        tp = int((sel & yy).sum())
        P, Rc = tp / max(1, int(sel.sum())), tp / max(1, int(yy.sum()))
        rows.append({'가스 전문가': name, '검사 샷': int(sel.sum()), '경보 샷(최종 q0.999)': int((lab['pct'].values[idx] >= ALARM_Q).sum()),
                     '불량': f'{tp}/{int(yy.sum())}', 'Precision': round(P, 4), 'Recall': round(Rc, 3),
                     'F1': round(2 * P * Rc / (P + Rc), 4) if P + Rc else 0.0,
                     '가스': f'{gas}/{int((t == "가스").sum())}', '미성형': f'{short}/{int((t == "미성형").sum())}',
                     '무작위 위치 가스 평균': round(float(null.mean()), 2), '무작위 95%': float(np.quantile(null, 0.95)),
                     '순열 p': round(float((1 + (null >= gas).sum()) / (len(null) + 1)), 3)})
        picked = lab.iloc[idx[sel]][['run_id', 'seq', 'TimeStamp', 'Reason', 'pct']].copy()
        picked['gas_score'] = g[idx[sel]]
        picked.to_csv(f'{OUT}/selected_shots_{name.split(" ")[0]}.csv', index=False, encoding='utf-8-sig')
    S = pd.DataFrame(rows)
    S.to_csv(f'{OUT}/summary.csv', index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print(S.to_string())
    return S


if __name__ == '__main__':
    main()
