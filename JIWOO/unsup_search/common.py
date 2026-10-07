"""
common.py — 비라벨 구간 탐색 공용: 경로, 비라벨 캐시 구축, 라벨 평가 세트, 평가 지표

설계
  - 비라벨 1차 원본(unlabeled_data.csv)을 (설비, 제품코드) 흐름으로 나눠 샷 단위로 정리
  - final/ 패키지와 같은 규칙: 이중기록 제거 → (우진2) 라벨 샷·라벨 Run ±60분 제거 → 끼어든 이상 기록 제거
    → Run 구분(60분 공백) → 10샷 미만 Run 제거 → 0 모드(배럴1 < 100℃) 제거
  - 인과 파생변수(같은 Run의 과거 샷만)는 흐름 전체에서 한 번 계산해 캐시 → 구간은 행 필터로만 고름
  - 라벨은 평가에만 사용
"""
import os
import sys
os.environ.setdefault('JOBLIB_TEMP_FOLDER', 'C:/Users/Public/joblib_tmp')   # 사용자 경로(한글)면 loky 오류
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent
FINAL = ROOT.parent / 'final'
sys.path.insert(0, str(FINAL / 'src'))
import final_model as fm  # noqa: E402

DATA = FINAL / 'data' / 'dataset'
FINAL_INTER = FINAL / 'outputs' / 'intermediate'
OUT = ROOT / 'outputs'
CACHE = OUT / 'cache'
for d in (OUT, CACHE):
    d.mkdir(parents=True, exist_ok=True)

PRODUCTS = ('cn7', 'rg3')
EQUIP_W2 = '650톤-우진2호기'
raw = fm.raw
ALL_RAW = fm.ALL_RAW                      # 21개 (상수·기록오류 변수 제외)
FEAT_C17 = list(fm.FEATURES)              # 제출 모델 인과 파생변수 17개 (dev_flow_res는 제품별 fr_step)
FEAT_RAW = [raw(v) for v in ALL_RAW]      # 원값 21개 (정적)
FEAT_DEV = [f'dv_{v}' for v in ALL_RAW]   # 전 변수 인과 편차 21개
REL = ['rel_fill_speed', 'rel_plast', 'rel_hold', 'rel_sw_ratio', 'rel_rest']   # 물리 관계 5개 (원값)
FEAT_REL = [f'dv_{r}' for r in REL]       # 물리 관계 인과 편차 5개
STEP_DEFAULT = {v: 0.01 for v in ['Injection_Time', 'Filling_Time', 'Plasticizing_Time', 'Cycle_Time',
                                  'Clamp_Close_Time', 'Cushion_Position', 'Plasticizing_Position']}
SCALES = json.load(open(FINAL_INTER / 'base' / 'reference_scales.json', encoding='utf-8'))


def step_of(v):
    return SCALES['cn7']['steps'].get(v, STEP_DEFAULT.get(v, 0.1))


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ====================================================================== 라벨 쪽
def labeled_shots(p):
    return fm.load_labeled_shots(p, str(FINAL_INTER))


def labeled_windows():
    """라벨 샷 시각 + Run별 (min-60분, max+60분) — 우진2 비라벨에서 제거"""
    ts, win = [], []
    pad = pd.Timedelta(minutes=fm.RUN_GAP_MIN)
    for p in PRODUCTS:
        s = labeled_shots(p)
        ts.append(s['TimeStamp'])
        w = s.groupby('run_id')['TimeStamp'].agg(['min', 'max'])
        win += [(a - pad, b + pad) for a, b in w.itertuples(index=False)]
    return pd.concat(ts), win


def add_relations(d):
    """물리 관계 5개 (노트북 10과 같은 정의, 원값)"""
    with np.errstate(all='ignore'):
        d['rel_fill_speed'] = np.log(d[raw('Filling_Time')]) + np.log(d[raw('Max_Injection_Speed')])
        d['rel_plast'] = (np.log(d[raw('Plasticizing_Time')]) + np.log(d[raw('Max_Screw_RPM')])
                          - np.log(d[raw('Plasticizing_Position')]))
        d['rel_hold'] = d[raw('Injection_Time')] - d[raw('Filling_Time')]
        d['rel_sw_ratio'] = d[raw('Max_Switch_Over_Pressure')] / d[raw('Max_Injection_Pressure')]
        d['rel_rest'] = (d[raw('Cycle_Time')] - d[raw('Injection_Time')] - d[raw('Plasticizing_Time')]
                         - d[raw('Clamp_Close_Time')])
    for r in REL:
        d[r] = d[r].replace([np.inf, -np.inf], np.nan)
    return d


def ext_features(d):
    """d: run_id, TimeStamp, *_raw (Run → 시간 순 정렬 가정). 제출 모델 17개 + 전 변수 편차 + 관계 편차 추가"""
    d = d.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
    out = {}
    for p in PRODUCTS:   # 제출 모델 파생변수 (dev_flow_res만 제품별 fr_step 차이)
        f = fm.make_features(d[['run_id', 'TimeStamp'] + FEAT_RAW], SCALES[p]['steps'], SCALES[p]['fr_step'])
        out[p] = f
    base = out['cn7']
    for c in FEAT_C17:
        d[c] = base[c].values
    d['dev_flow_res_rg3'] = out['rg3']['dev_flow_res'].values
    d['dev_flow_res_cn7'] = d['dev_flow_res']
    for v in ALL_RAW:
        d[f'dv_{v}'] = fm._by_run(d, raw(v), lambda a, s=step_of(v): fm.causal_robust_dev(a, s)).values
    d = add_relations(d)
    rel_step = {'rel_fill_speed': 1e-3, 'rel_plast': 1e-3, 'rel_hold': 0.01, 'rel_sw_ratio': 1e-3, 'rel_rest': 0.01}
    for r in REL:
        d[f'dv_{r}'] = fm._by_run(d, r, lambda a, s=rel_step[r]: fm.causal_robust_dev(a, s)).values
    return d


# ====================================================================== 비라벨 캐시
META = ['TimeStamp', 'PART_NAME', 'EQUIP_NAME', 'PART_FACT_PLAN_DATE', 'PART_FACT_SERIAL', 'ERR_FACT_QTY']


def _prep_stream(key, g, lab_ts, lab_win):
    """한 (설비, 제품코드) 흐름 → 정제·Run·파생변수"""
    equip, prod = key
    g = g.copy()
    g['t'] = pd.to_datetime(g['TimeStamp'])
    # 측면(LH/RH) 행 → 샷: 공정값은 첫 행, 로트 불량 수는 측면별 첫 값 합
    lot_err = g.groupby(['TimeStamp', 'PART_NAME'])['ERR_FACT_QTY'].first().groupby(level=0).sum()
    s = g.sort_values('t').drop_duplicates('TimeStamp').copy()
    s['lot_err'] = s['TimeStamp'].map(lot_err).values
    s['lot'] = s['PART_FACT_PLAN_DATE'].astype(str).str[:10] + '_' + s['PART_FACT_SERIAL'].astype(str)
    s = s.rename(columns={v: raw(v) for v in ALL_RAW})
    s['TimeStamp'] = s['t']
    n0 = len(s)
    if equip == EQUIP_W2:      # 라벨 샷·라벨 Run 시간창 제거 (평가 샷과 같은 시간대가 학습에 안 들어가게)
        s = s[~s['TimeStamp'].isin(lab_ts)]
        m = np.zeros(len(s), bool)
        for a, b in lab_win:
            m |= s['TimeStamp'].between(a, b).values
        s = s[~m]
    s = s.reset_index(drop=True)
    # 끼어든 이상 기록
    gp = s['TimeStamp'].diff().dt.total_seconds()
    gn = -s['TimeStamp'].diff(-1).dt.total_seconds()
    day_med = s.groupby(s['TimeStamp'].dt.date)[raw('Cycle_Time')].transform('median')
    odd = (np.fmin(gp, gn) < fm.MIN_CYCLE_SEC) & ((s[raw('Cycle_Time')] - day_med).abs() > fm.CYCLE_DEV_SEC)
    s = s[~odd].reset_index(drop=True)
    # Run
    gap = s['TimeStamp'].diff().dt.total_seconds()
    new = gap.isna() | (gap > fm.RUN_GAP_MIN * 60)
    tag = f'{equip}|{prod}'
    s['run_id'] = tag + '|R' + new.cumsum().astype(int).astype(str).str.zfill(4)
    s = s[s.groupby('run_id')['run_id'].transform('size') >= fm.MIN_RUN_SHOTS]
    s = s[s[raw('Barrel_Temperature_1')] >= fm.ZERO_MODE_BARREL].reset_index(drop=True)
    if len(s) < 50:
        return None
    s['equip'], s['prod'] = equip, prod
    s['seq_in_run'] = s.groupby('run_id').cumcount() + 1
    keep = ['equip', 'prod', 'run_id', 'TimeStamp', 'seq_in_run', 'PART_NAME', 'lot', 'lot_err'] + FEAT_RAW
    f = ext_features(s[keep].copy())
    f['n_raw_shots'] = n0
    return f


def build_cache(n_jobs=12, force=False):
    path = CACHE / 'unlabeled_feats.pkl'
    if path.exists() and not force:
        return pd.read_pickle(path)
    log('비라벨 원본 읽기')
    U = pd.read_csv(DATA / 'unlabeled_data.csv', low_memory=False, usecols=META + ALL_RAW)
    U['prod'] = U['PART_NAME'].astype(str).str[:3].str.strip()
    lab_ts, lab_win = labeled_windows()
    groups = [(k, g) for k, g in U.groupby(['EQUIP_NAME', 'prod']) if len(g) >= 100]
    groups.sort(key=lambda kg: -len(kg[1]))
    log(f'흐름 {len(groups)}개 파생변수 계산 (병렬 {n_jobs})')
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_prep_stream)(k, g, lab_ts, lab_win) for k, g in groups)
    F = pd.concat([r for r in res if r is not None], ignore_index=True)
    F.to_pickle(path)
    log(f'캐시 저장: {len(F):,}샷, 흐름 {F.groupby(["equip", "prod"]).ngroups}개 → {path.name}')
    return F


def build_labeled_feats(force=False):
    path = CACHE / 'labeled_feats.pkl'
    if path.exists() and not force:
        return pd.read_pickle(path)
    out = {}
    for p in PRODUCTS:
        s = labeled_shots(p)
        keep = ['run_id', 'TimeStamp', 'split', 'PassOrFail', 'Reason', 'LH_fail', 'RH_fail'] + FEAT_RAW
        f = ext_features(s[keep].copy())
        f['seq_in_run'] = f.groupby('run_id').cumcount() + 1
        out[p] = f
    pd.to_pickle(out, path)
    return out


# ====================================================================== 평가 세트 (행 단위)
def eval_rows(p):
    """
    행 단위 평가 표: TimeStamp, y, period(official/extra), Reason
      official = 대회 공식 2차 행 (linked_labeled.csv), extra = 추가 기간 샷의 LH/RH 행 (base shots)
    """
    l = pd.read_csv(FINAL_INTER / 'linked' / 'linked_labeled.csv', parse_dates=['TimeStamp'])
    l = l[(l['product'] == p) & l['in_secondary']]
    off = pd.DataFrame({'TimeStamp': l['TimeStamp'].values, 'y': l['PassOrFail'].astype(int).values,
                        'Reason': l['Reason'].fillna('정상').values, 'period': 'official'})
    s = labeled_shots(p)
    s = s[s['split'] == '추가']
    ex = []
    for side in ('LH', 'RH'):
        v = s[s[f'{side}_fail'].notna()]
        ex.append(pd.DataFrame({'TimeStamp': v['TimeStamp'].values, 'y': v[f'{side}_fail'].astype(int).values,
                                'Reason': np.where(v[f'{side}_fail'] == 1, v['Reason'], '정상'),
                                'period': 'extra'}))
    return pd.concat([off] + ex, ignore_index=True)


def _rand_tiebreak(s):
    return np.asarray(s, float) + np.random.RandomState(0).rand(len(s)) * 1e-9


def metrics(y, s, budget=0.05):
    y = np.asarray(y, int)
    s = _rand_tiebreak(s)
    if y.sum() == 0:
        return {}
    k = int(round(len(s) * budget))
    top = np.argsort(-s, kind='stable')[:k]
    tp = int(y[top].sum())
    return {'AP': average_precision_score(y, s), 'TP5': tp, 'F1_5': 2 * tp / (k + y.sum()), 'P': int(y.sum()),
            'n': len(y)}


def evaluate(p, shot_ts, shot_score, rows=None):
    """샷 점수 → 행에 붙여 공식/추가 기간 지표"""
    rows = eval_rows(p) if rows is None else rows
    sc = pd.Series(np.asarray(shot_score, float), index=pd.DatetimeIndex(shot_ts))
    sc = sc[~sc.index.duplicated()]
    r = {}
    for per in ('official', 'extra'):
        e = rows[rows['period'] == per]
        s = e['TimeStamp'].map(sc).fillna(np.nanmin(sc.values)).values
        for k, v in metrics(e['y'].values, s).items():
            r[f'{per[:3]}_{k}'] = v
    return r
