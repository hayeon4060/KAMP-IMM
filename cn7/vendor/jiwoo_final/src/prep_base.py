"""전처리 ② labeled 샷 단위 베이스 테이블 + 기준값(기록 단위)

base/{cn7,rg3}_shots.csv
    1차 labeled_data.csv (CN7·RG3, 650톤-우진2호기) 전체 → 완전 중복 제거 → 2차 범위 행 검증
    (z-score 재현 1e-6, 라벨 100%) → 2차 기준 z-score → LH/RH 2행을 샷 1행으로 → 끼어든 이상 기록 제거
    → Run 구분(60분 공백) → split(2차 / 추가 / 제외: 10샷 미만 Run)
base/reference_scales.json
    변수별 기록 단위·흐름저항 편차 하한 — 평가에 쓰지 않는 unlabeled(우진2호기)에서 계산
"""
import json
import numpy as np
import pandas as pd

from paths import DATA, BASE

PRODUCTS = ['cn7', 'rg3']
SECONDARY_PLAN_DATES = {'cn7': ['2020-10-16', '2020-10-20'], 'rg3': ['2020-10-21', '2020-10-22', '2020-10-23']}
EQUIP = '650톤-우진2호기'
RUN_GAP_MIN, STOP_GAP_MIN, MIN_RUN_SHOTS = 60, 5, 10
MIN_CYCLE_SEC, CYCLE_DEV_SEC = 50, 2
INITIAL_REASON = '초기허용불량'
DROP_VARS = ['Clamp_Open_Position', 'Average_Screw_RPM', 'Max_Back_Pressure']   # 상수·기록오류·중복

# 기준값 계산용 (features.py 와 동일)
EVENT_VARS = ['Max_Injection_Speed', 'Filling_Time', 'Max_Injection_Pressure', 'Max_Switch_Over_Pressure',
              'Average_Back_Pressure', 'Cushion_Position', 'Plasticizing_Time', 'Cycle_Time']
STATE_VARS = ['Mold_Temperature_3', 'Hopper_Temperature']
FLOW_VARS = ['Max_Injection_Speed', 'Max_Injection_Pressure', 'Max_Switch_Over_Pressure',
             'Cushion_Position', 'Average_Back_Pressure']
raw = lambda c: f'{c}_raw'


def build_shots():
    P = pd.read_csv(DATA / 'labeled_data.csv', low_memory=False)
    P['product'] = P['PART_NAME'].str[:3].str.lower()
    P['cavity'] = P['PART_NAME'].str[-2:]
    P['plan_date'] = pd.to_datetime(P['PART_FACT_PLAN_DATE'].str[:10])
    P['TimeStamp'] = pd.to_datetime(P['TimeStamp'])
    P = P[P['product'].isin(PRODUCTS) & (P['EQUIP_NAME'] == EQUIP)]
    P = P.drop_duplicates(subset=[c for c in P.columns if c != '_id'], keep='first').reset_index(drop=True)
    P['PassOrFail'] = P['PassOrFail'].map({'Y': 0, 'N': 1}).astype(int)
    P['Reason'] = P['Reason'].fillna('정상')
    rpm_x10 = P['Average_Screw_RPM'] > 100                      # 앞 기간 10배 기록 → ÷10 보정
    P.loc[rpm_x10, 'Average_Screw_RPM'] = P.loc[rpm_x10, 'Average_Screw_RPM'] / 10

    # 2차 범위 행 찾기 + 검증
    FEATURES = None
    P['in_secondary'] = False
    P['sec_index'] = np.nan
    SCALER = {}
    for p in PRODUCTS:
        sec = pd.read_csv(DATA / f'moldset_labeled_{p}.csv')
        if FEATURES is None:
            FEATURES = [c for c in sec.columns if c not in ('Unnamed: 0', 'PassOrFail')]
        mask = (P['product'] == p) & P['plan_date'].isin(pd.to_datetime(SECONDARY_PLAN_DATES[p]))
        prim = P[mask]
        assert len(prim) == len(sec), f'{p}: 행 수 불일치'
        mu, sd = prim[FEATURES].mean(), prim[FEATURES].std(ddof=0)
        z = ((prim[FEATURES] - mu) / sd).fillna(0.0)
        assert np.abs(z.values - sec[FEATURES].values).max() < 1e-6, f'{p}: z-score 재현 실패'
        assert (prim['PassOrFail'].values == sec['PassOrFail'].values).all(), f'{p}: 라벨 불일치'
        P.loc[mask, 'in_secondary'] = True
        P.loc[mask, 'sec_index'] = sec['Unnamed: 0'].values
        SCALER[p] = (mu, sd)
    RAW = [f'{c}_raw' for c in FEATURES]

    # 전체 행에 2차 기준 z-score
    for c in FEATURES:
        P[f'{c}_raw'] = P[c]
    for p in PRODUCTS:
        mu, sd = SCALER[p]
        m = P['product'] == p
        z = (P.loc[m, RAW].values - mu.values) / sd.replace(0, np.nan).values
        P.loc[m, FEATURES] = np.nan_to_num(z, nan=0.0)

    ROW_COLS = (['product', 'TimeStamp', 'plan_date', 'cavity', 'PassOrFail', 'Reason',
                 'in_secondary', 'sec_index', '_id'] + FEATURES + RAW)
    R = P[ROW_COLS].copy()

    # LH·RH 2행 → 샷 1행
    R['shot_key'] = R['product'] + '_' + R['TimeStamp'].dt.strftime('%Y%m%d%H%M%S')
    g = R.groupby('shot_key')
    assert not g['cavity'].agg(lambda s: s.duplicated().any()).any()
    assert int(g[RAW].nunique().gt(1).any(axis=1).sum()) == 0, 'LH·RH 공정값이 다른 샷 존재'

    def per_cav(col, cav):
        return R[R['cavity'] == cav].set_index('shot_key')[col]

    S = g[['product', 'TimeStamp', 'plan_date'] + FEATURES + RAW].first()
    S['in_secondary'] = g['in_secondary'].all()
    S['LH_fail'] = per_cav('PassOrFail', 'LH')
    S['RH_fail'] = per_cav('PassOrFail', 'RH')
    S['Reason_LH'] = per_cav('Reason', 'LH')
    S['Reason_RH'] = per_cav('Reason', 'RH')
    S['PassOrFail'] = S[['LH_fail', 'RH_fail']].max(axis=1).astype(int)

    def merge_reason(row):
        rs = sorted({r for r in (row['Reason_LH'], row['Reason_RH']) if isinstance(r, str) and r != '정상'})
        return '+'.join(rs) if rs else '정상'
    S['Reason'] = S.apply(merge_reason, axis=1)
    S['PassOrFail_process'] = ((S['PassOrFail'] == 1) & (S['Reason'] != INITIAL_REASON)).astype(int)
    S = S.reset_index()

    # 끼어든 이상 기록 제거 (앞/뒤 샷과 50초 미만 + Cycle_Time이 그날 중앙값과 2초 넘게 차이)
    S = S.sort_values(['product', 'TimeStamp']).reset_index(drop=True)
    gp = S.groupby('product')['TimeStamp'].diff().dt.total_seconds()
    gn = -S.groupby('product')['TimeStamp'].diff(-1).dt.total_seconds()
    near = np.fmin(gp, gn) < MIN_CYCLE_SEC
    day_med = S.groupby(['product', S['TimeStamp'].dt.date])['Cycle_Time_raw'].transform('median')
    odd = near & ((S['Cycle_Time_raw'] - day_med).abs() > CYCLE_DEV_SEC)
    S = S[~odd].reset_index(drop=True)

    # Run 구분
    S = S.sort_values(['product', 'TimeStamp']).reset_index(drop=True)
    gap = S.groupby('product')['TimeStamp'].diff().dt.total_seconds()
    new_run = gap.isna() | (gap > RUN_GAP_MIN * 60)
    S['run_no'] = new_run.groupby(S['product']).cumsum().astype(int)
    S['run_id'] = S['product'] + '_R' + S['run_no'].astype(str)
    S['prev_gap_sec'] = gap.where(~new_run)
    S['shot_seq_in_run'] = S.groupby('run_id').cumcount() + 1
    S['after_stop'] = S['prev_gap_sec'] > STOP_GAP_MIN * 60
    seg = (new_run | S['after_stop']).groupby(S['run_id']).cumsum()
    S['shots_since_stop'] = S.groupby([S['run_id'], seg]).cumcount() + 1

    run_size = S.groupby('run_id')['shot_key'].transform('size')
    run_sec = S.groupby('run_id')['in_secondary'].transform('all')
    assert (S.groupby('run_id')['in_secondary'].nunique() == 1).all(), '2차·추가가 섞인 Run 존재'
    S['split'] = np.where(run_size < MIN_RUN_SHOTS, '제외', np.where(run_sec, '2차', '추가'))

    KEEP_VARS = [c for c in FEATURES if c not in DROP_VARS]
    SHOT_COLS = (['run_id', 'split', 'TimeStamp', 'shot_seq_in_run', 'prev_gap_sec', 'after_stop',
                  'shots_since_stop', 'PassOrFail', 'PassOrFail_process', 'Reason', 'LH_fail', 'RH_fail']
                 + KEEP_VARS + [f'{c}_raw' for c in KEEP_VARS])
    for p in PRODUCTS:
        out = S.loc[S['product'] == p, SHOT_COLS].sort_values('TimeStamp')
        out.to_csv(BASE / f'{p}_shots.csv', index=False)
        print(f'  [{p}] 샷 {len(out):,} | split {out["split"].value_counts().to_dict()}')


def _resolution(x):
    v = np.unique(np.round(pd.Series(x).dropna().values.astype(float), 4))
    return float(np.round(np.min(np.diff(v)), 4)) if len(v) > 1 else 1.0


def build_reference_scales():
    """unlabeled(우진2호기) 제품별 행 → 변수별 기록 단위, 흐름저항(사출압/속도) 1차차분 5% 분위"""
    U = pd.read_csv(DATA / 'unlabeled_data.csv', low_memory=False)
    out = {}
    for p in PRODUCTS:
        ref = U[U['PART_NAME'].str.contains(p.upper(), na=False) & U['EQUIP_NAME'].str.contains('우진2', na=False)]
        ref = ref.drop_duplicates(subset=['TimeStamp'])
        cols = list(dict.fromkeys(EVENT_VARS + STATE_VARS + FLOW_VARS))
        ref = ref.rename(columns={v: raw(v) for v in cols})
        steps = {v: _resolution(ref[raw(v)]) for v in EVENT_VARS + STATE_VARS}
        fr = ref[raw('Max_Injection_Pressure')] / ref[raw('Max_Injection_Speed')]
        fr_step = fr.diff().abs().replace(0, np.nan).quantile(0.05)
        out[p] = {'steps': steps, 'fr_step': float(fr_step)}
    json.dump(out, open(BASE / 'reference_scales.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('  reference_scales.json', {p: round(v['fr_step'], 6) for p, v in out.items()})


def main():
    build_shots()
    build_reference_scales()


if __name__ == '__main__':
    main()
