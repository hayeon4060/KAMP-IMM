"""
final_model.py — 최종 탐지기 (국소 Mahalanobis, dev26 · k=300) 학습·테스트

  1) 학습 데이터 : unlabeled 정상 생산 샷 (우진2호기, 그 제품의 '평가 기간 이전' 전부 — 첫 labeled 샷 60분 전까지)
                   labeled 샷·labeled Run 시간창(±60분)·끼어든 이상 기록·짧은 Run·0 모드(비가동) 제거
  2) 입력 변수   : 인과적 파생변수 26개(dev26) = 원변수 21개의 Run 내 강건 편차 + 물리 관계 5개(R1~R5)의 강건 편차
                   — 같은 Run의 '과거 샷'만 사용 (미래 정보 없음). 기존 17개(c17)도 계산해 출력(확률 층·설명용)
  3) 학습        : reference 중앙값·IQR로 스케일 → ±20 클리핑 → 최근접 정상 300샷 검색 구조 저장
  4) 점수        : 국소 Mahalanobis 거리 = 그 샷과 가장 비슷한 정상 300샷의 Ledoit-Wolf 평균·공분산 기준 거리
                   (클수록 불량 의심) → reference 자기 제외 점수 분포 대비 백분위(pct)
  5) 임계값      : pct ≥ q  (q = 1 − 정상 오경보율, 라벨 없이 결정; 운용점 q = 0.999)
라벨은 평가에만 사용.

입력 (outputs/intermediate/, 전처리 단계 산출물)
  linked/linked_unlabeled_{cn7,rg3}.csv, linked/linked_labeled.csv  ← prep_link.py
  base/{cn7,rg3}_shots.csv, base/reference_scales.json            ← prep_base.py
출력 (outputs/final_model/)
  {p}_final_model.pkl, {p}_scores.csv, data_log.csv, train_params.csv, frame1/2.csv, threshold.csv, check.csv
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score, precision_recall_curve

from paths import INTER, FINAL

BASE_DIR, OUT_DIR = str(INTER), str(FINAL)

# ====================================================================== 설정 (결과 보기 전에 고정한 값)
SEED = 42
PRODUCTS = ('cn7', 'rg3')
REF_PERIOD = None                                       # 고정 기간 대신 제품별 '평가 기간 이전' (ref_period)
REF_START = '2020-01-01'
K_LOCAL = 300                                           # 국소 기준 이웃 수 (사전 고정 판정 unsup_search2 C2, k=100~2000 평탄)
NN_DECIMALS = 9                                         # 이웃 검색용 반올림 자리 (부동소수 끝자리 차이로 이웃이 바뀌지 않게)
RUN_GAP_MIN, STOP_GAP_MIN, MIN_RUN_SHOTS = 60, 5, 10    # Run 구분: 60분 공백 = 새 Run, 10샷 미만 Run 제거
MIN_CYCLE_SEC, CYCLE_DEV_SEC = 50, 2                    # 끼어든 이상 기록 판정
ZERO_MODE_BARREL = 100                                  # 배럴1 온도 < 100℃ = 비가동(0 모드)
REF_MAX = 30000                                         # reference 상한 (넘으면 무작위 추출)
W, MINP, REJECT, RESET_AFTER = 20, 5, 5.0, 5            # 인과적 기준선: 직전 20샷, 최소 5샷, |z|>5 제외, 5연속 제외 시 재설정
CLIP = 20
QS = (0.95, 0.98, 0.99, 0.995, 0.999)
INITIAL = '초기허용불량'
TYPE_GROUPS = {'가스': ['가스'], '미성형': ['미성형', '가스+미성형']}
# 동결 기준 결과 (재현 확인용)
EXPECTED = {('cn7', '①'): 0.465, ('rg3', '①'): 0.022, ('cn7', '②'): 0.203, ('rg3', '②'): 0.054}

raw = lambda c: f'{c}_raw'
ALL_RAW = ['Injection_Time', 'Filling_Time', 'Plasticizing_Time', 'Cycle_Time', 'Clamp_Close_Time',
           'Cushion_Position', 'Plasticizing_Position', 'Max_Injection_Speed', 'Max_Screw_RPM',
           'Max_Injection_Pressure', 'Max_Switch_Over_Pressure', 'Average_Back_Pressure',
           'Barrel_Temperature_1', 'Barrel_Temperature_2', 'Barrel_Temperature_3',
           'Barrel_Temperature_4', 'Barrel_Temperature_5', 'Barrel_Temperature_6',
           'Hopper_Temperature', 'Mold_Temperature_3', 'Mold_Temperature_4']
EVENT_VARS = ['Max_Injection_Speed', 'Filling_Time', 'Max_Injection_Pressure', 'Max_Switch_Over_Pressure',
              'Average_Back_Pressure', 'Cushion_Position', 'Plasticizing_Time', 'Cycle_Time']
STATE_VARS = ['Mold_Temperature_3', 'Hopper_Temperature']

# 기존(제출 1차) 입력 변수 17개 — 출력·확률 층 설명용으로 계속 계산
C17 = ([f'dev_{v}' for v in EVENT_VARS]                                    # 이벤트 편차 8
            + ['fill_anomaly', 'ev_max_abs_dev']                                # 이벤트 요약 2
            + ['dev_Mold_Temperature_3', 'dev_Hopper_Temperature',              # 열 상태 6
               'mold_rise', 'hopper_rise', 'mold_slope20', 'hopper_slope20']
            + ['dev_flow_res'])                                                 # 흐름저항 1

# 물리 관계 5개 (원값으로 계산 → Run 내 강건 편차). step = 기록 단위 하한
REL_STEP = {'R1': 1e-3, 'R2': 1e-3, 'R3': 1e-2, 'R4': 1e-3, 'R5': 1e-2}
REL_NAME = {'R1': '충전시간×사출속도', 'R2': '계량(시간×RPM÷위치)', 'R3': '보압시간', 'R4': '절환압/사출압',
            'R5': '기타 시간'}
# 최종 입력 변수 26개 (dev26)
FEATURES = [f'dv_{v}' for v in ALL_RAW] + [f'dv_{r}' for r in REL_STEP]
SHORT_NAME = {'Injection_Time': '사출시간', 'Filling_Time': '충전시간', 'Plasticizing_Time': '계량시간',
              'Cycle_Time': '사이클', 'Clamp_Close_Time': '형체시간', 'Cushion_Position': '쿠션',
              'Plasticizing_Position': '계량위치', 'Max_Injection_Speed': '사출속도', 'Max_Screw_RPM': '스크루RPM',
              'Max_Injection_Pressure': '사출압력', 'Max_Switch_Over_Pressure': '절환압력',
              'Average_Back_Pressure': '평균배압', 'Hopper_Temperature': '호퍼온도',
              'Mold_Temperature_3': '금형온도3', 'Mold_Temperature_4': '금형온도4',
              **{f'Barrel_Temperature_{i}': f'배럴{i}온도' for i in range(1, 7)}}
FEAT_NAME = {**{f'dv_{v}': SHORT_NAME[v] for v in ALL_RAW}, **{f'dv_{r}': REL_NAME[r] for r in REL_STEP}}


# ====================================================================== 1. 데이터
def load_labeled_shots(p, base_dir=BASE_DIR):
    s = pd.read_csv(f'{base_dir}/base/{p}_shots.csv', parse_dates=['TimeStamp'])
    s['Reason'] = s['Reason'].fillna('정상')
    return s.sort_values('TimeStamp').reset_index(drop=True)


def load_official_rows(p, base_dir=BASE_DIR):
    """공식 2차 행 (LH/RH 행 단위 라벨)"""
    l = pd.read_csv(f'{base_dir}/linked/linked_labeled.csv', parse_dates=['TimeStamp'],
                    usecols=['product', 'in_secondary', 'TimeStamp', 'cavity', 'PassOrFail', 'Reason'])
    l = l[(l['product'] == p) & l['in_secondary']].copy()
    l['Reason'] = l['Reason'].fillna('정상')
    return l.reset_index(drop=True)


def build_reference(p, labeled_shots, base_dir=BASE_DIR, period=REF_PERIOD):
    """unlabeled → 정상 기준 샷 (학습 데이터)"""
    u = pd.read_csv(f'{base_dir}/linked/linked_unlabeled_{p}.csv', parse_dates=['TimeStamp'], low_memory=False)
    if 'EQUIP_NAME' in u.columns:
        u = u[u['EQUIP_NAME'].astype(str).str.contains('우진2', na=False)]
    u = u.drop_duplicates('TimeStamp').sort_values('TimeStamp').reset_index(drop=True)   # LH/RH 2행 → 샷 1행
    log = {'unlabeled 샷': len(u)}

    # (1) labeled 샷 + labeled Run 시간창(±60분) 제거 → 평가 샷과 같은 시간대가 학습에 안 들어감
    lab_ts = pd.to_datetime(labeled_shots['TimeStamp'])
    u = u[~u['TimeStamp'].isin(lab_ts)]
    pad = pd.Timedelta(minutes=RUN_GAP_MIN)
    win = labeled_shots.assign(TimeStamp=lab_ts).groupby('run_id')['TimeStamp'].agg(['min', 'max'])
    m = np.zeros(len(u), bool)
    for a, b in win.itertuples(index=False):
        m |= u['TimeStamp'].between(a - pad, b + pad).values
    u = u[~m]
    if period is not None:
        u = u[u['TimeStamp'].between(pd.Timestamp(period[0]), pd.Timestamp(period[1]))]
    u = u.reset_index(drop=True)
    log['labeled 시간대 제거·기간 적용 후'] = len(u)

    # (2) 끼어든 이상 기록 제거
    gp = u['TimeStamp'].diff().dt.total_seconds()
    gn = -u['TimeStamp'].diff(-1).dt.total_seconds()
    day_med = u.groupby(u['TimeStamp'].dt.date)[raw('Cycle_Time')].transform('median')
    odd = (np.fmin(gp, gn) < MIN_CYCLE_SEC) & ((u[raw('Cycle_Time')] - day_med).abs() > CYCLE_DEV_SEC)
    u = u[~odd].reset_index(drop=True)
    log['이상 기록 제거'] = int(odd.sum())

    # (3) Run 구분 (60분 공백), 10샷 미만 Run 제거
    gap = u['TimeStamp'].diff().dt.total_seconds()
    new = gap.isna() | (gap > RUN_GAP_MIN * 60)
    u['run_id'] = f'{p}_U' + new.cumsum().astype(int).astype(str).str.zfill(4)
    u['prev_gap_sec'] = gap.where(~new)
    u = u[u.groupby('run_id')['run_id'].transform('size') >= MIN_RUN_SHOTS]

    # (4) 0 모드(비가동 기록) 제거 — Run 구분 뒤에 제거
    z = u[raw('Barrel_Temperature_1')] < ZERO_MODE_BARREL
    u = u[~z]
    log['0 모드 제거'] = int(z.sum())

    keep = ['run_id', 'TimeStamp'] + [raw(v) for v in ALL_RAW if raw(v) in u.columns]
    u = u[keep].reset_index(drop=True)
    log['reference 샷'] = len(u)
    log['reference Run'] = u['run_id'].nunique()
    return u, log


def ref_period(labeled_shots):
    """평가 기간 이전 = 그 제품의 첫 labeled 샷 60분 전까지 (미래 정보 없음)"""
    end = pd.to_datetime(labeled_shots['TimeStamp']).min() - pd.Timedelta(minutes=RUN_GAP_MIN)
    return (REF_START, str(end))


# ====================================================================== 2. 파생변수 (인과적: 같은 Run의 과거 샷만)
def resolution(x):
    """기록 단위(0.01 / 0.1 등) = 고유값 간 최소 간격"""
    v = np.unique(np.round(pd.Series(x).dropna().values.astype(float), 4))
    return float(np.round(np.min(np.diff(v)), 4)) if len(v) > 1 else 1.0


def causal_robust_dev(x, step, w=W, minp=MINP, reject=REJECT, reset_after=RESET_AFTER):
    """
    dev_t = (x_t − 기준선 중앙값) / max(1.4826·MAD, 기록 단위)
    기준선 = 같은 Run 과거 샷 중 '정상으로 받아들인' 최근 w개 (라벨 미사용)
      |dev| > reject 샷은 기준선 제외 / reset_after 연속 제외되면 새 수준으로 보고 기준선 교체
    """
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    hist, rejected = [], []
    for i, xi in enumerate(x):
        if np.isnan(xi):
            continue
        if len(hist) < minp:
            hist.append(xi)
            continue
        h = np.asarray(hist[-w:])
        med = np.median(h)
        scale = max(1.4826 * np.median(np.abs(h - med)), step)
        z = (xi - med) / scale
        out[i] = z
        if abs(z) <= reject:
            hist.append(xi)
            rejected = []
        else:
            rejected.append(xi)
            if len(rejected) >= reset_after:
                hist = list(rejected)
                rejected = []
    return out


def causal_slope(y, w=W):
    """과거 w샷(현재 포함) 선형 기울기"""
    def _s(a):
        a = a[~np.isnan(a)]
        return np.nan if len(a) < 3 else np.polyfit(np.arange(len(a)), a, 1)[0]
    return pd.Series(y, dtype=float).rolling(w, min_periods=3).apply(_s, raw=True).values


def _by_run(d, col, func):
    out = pd.Series(np.nan, index=d.index)
    for _, idx in d.groupby('run_id', sort=False).groups.items():
        out.loc[idx] = func(d.loc[idx, col].values)
    return out


def relations(d):
    """물리 관계 5개 원값 (R1 충전×속도, R2 계량, R3 보압시간, R4 압력비, R5 기타 시간)"""
    r = lambda c: d[raw(c)].astype(float)
    lg = lambda c: np.log(r(c).where(r(c) > 0))
    out = pd.DataFrame({
        'R1': lg('Filling_Time') + lg('Max_Injection_Speed'),
        'R2': lg('Plasticizing_Time') + lg('Max_Screw_RPM') - lg('Plasticizing_Position'),
        'R3': r('Injection_Time') - r('Filling_Time'),
        'R4': r('Max_Switch_Over_Pressure') / r('Max_Injection_Pressure'),
        'R5': r('Cycle_Time') - r('Injection_Time') - r('Plasticizing_Time') - r('Clamp_Close_Time'),
    }, index=d.index)
    return out.replace([np.inf, -np.inf], np.nan)


def make_features(df, steps, fr_step, steps_all=None):
    """df: run_id, TimeStamp, *_raw (샷 단위). 반환: C17 + FEATURES(dev26)가 추가된 df (Run → 시간 순)
    steps_all: 원변수 21개 기록 단위(학습 reference에서 구해 모델에 저장한 값). 없으면 df에서 계산"""
    d = df.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True).copy()
    st = {v: resolution(d[raw(v)]) for v in EVENT_VARS + STATE_VARS}
    st.update({k: v for k, v in (steps or {}).items() if k in st})

    # 이벤트 편차
    for v in EVENT_VARS:
        d[f'dev_{v}'] = _by_run(d, raw(v), lambda a, s=st[v]: causal_robust_dev(a, s))
    dev_cols = [f'dev_{v}' for v in EVENT_VARS]
    d['ev_max_abs_dev'] = d[dev_cols].abs().max(axis=1, skipna=True)
    d['fill_anomaly'] = d['dev_Filling_Time'] - d['dev_Max_Injection_Speed']   # 음수 = 충전↓·속도↑ (미성형형)

    # 열 상태
    for v, name in [('Mold_Temperature_3', 'mold'), ('Hopper_Temperature', 'hopper')]:
        g = d.groupby('run_id', sort=False)[raw(v)]
        start = g.transform(lambda s: s.expanding().mean().where(np.arange(len(s)) < 5).ffill())  # Run 첫 5샷 평균
        d[f'{name}_rise'] = d[raw(v)] - start
        d[f'{name}_slope20'] = _by_run(d, raw(v), causal_slope)
        d[f'dev_{v}'] = _by_run(d, raw(v), lambda a, s=st[v]: causal_robust_dev(a, s))

    # 흐름저항 (사출압 / 사출속도 ≈ 점도 대리 지표)
    d['flow_res'] = d[raw('Max_Injection_Pressure')] / d[raw('Max_Injection_Speed')]
    fr = 1e-4 if (fr_step is None or pd.isna(fr_step)) else fr_step
    d['dev_flow_res'] = _by_run(d, 'flow_res', lambda a: causal_robust_dev(a, fr))

    # dev26: 원변수 21개 + 관계 5개의 Run 내 강건 편차
    sa = {v: resolution(d[raw(v)]) for v in ALL_RAW}
    sa.update({k: v for k, v in (steps_all or {}).items() if k in sa})
    for v in ALL_RAW:
        d[f'dv_{v}'] = _by_run(d, raw(v), lambda a, s=sa[v]: causal_robust_dev(a, s))
    rel = relations(d)
    for k, stp in REL_STEP.items():
        d[k] = rel[k].values
        d[f'dv_{k}'] = _by_run(d, k, lambda a, s=stp: causal_robust_dev(a, s))
    return d


def load_scales(p, ref_shots, base_dir=BASE_DIR):
    """변수별 기록 단위·흐름저항 하한 — json 우선, 없으면 reference로 계산"""
    path = f'{base_dir}/base/reference_scales.json'
    if os.path.exists(path):
        s = json.load(open(path, encoding='utf-8'))[p]
        return s['steps'], s['fr_step'], 'reference_scales.json'
    steps = {v: resolution(ref_shots[raw(v)]) for v in EVENT_VARS + STATE_VARS}
    fr = (ref_shots[raw('Max_Injection_Pressure')] / ref_shots[raw('Max_Injection_Speed')]).replace([np.inf, -np.inf], np.nan)
    fr_step = float(fr.diff().abs().replace(0, np.nan).quantile(0.05))
    print(f'[{p}] reference_scales.json 없음 → reference에서 계산 (기준 수치와 조금 다를 수 있음)')
    return steps, fr_step, 'reference에서 계산'


# ====================================================================== 3. 모델
class FinalModel:
    """정상 기준 국소 Mahalanobis 이상탐지 — 그 샷과 가장 비슷한 정상 k샷의 Ledoit-Wolf 평균·공분산으로 거리 계산"""

    def __init__(self, cols=FEATURES, k=K_LOCAL):
        self.cols = list(cols)
        self.k = int(k)

    def _X(self, df):
        return df[self.cols].replace([np.inf, -np.inf], np.nan).values.astype(float)

    def _prep(self, df):
        X = (self._X(df) - self.med) / self.iqr
        return np.nan_to_num(np.clip(X, -CLIP, CLIP), nan=0.0)        # Run 첫 5샷 dev NaN = 편차 없음(0)

    def fit(self, ref_feat):
        X0 = self._X(ref_feat)
        self.med = np.nanmedian(X0, axis=0)
        q75, q25 = np.nanpercentile(X0, [75, 25], axis=0)
        sd = np.nanstd(X0, axis=0)
        self.iqr = np.where(q75 - q25 > 1e-9, q75 - q25, np.where(sd > 1e-9, sd, 1.0))
        self.X = self._prep(ref_feat)
        self.nn = NearestNeighbors(n_neighbors=self.k + 1, algorithm='brute').fit(np.round(self.X, NN_DECIMALS))   # 이웃 검색은 반올림 값으로(플랫폼 간 동일)
        self.lw = LedoitWolf().fit(self.X)                              # (참고) 전역 축소계수 기록용
        self.s_ref = np.sort(self._local(self.X, drop_self=True)[0])     # 자기 제외 점수 = 정상 분포
        self.n_ref = len(self.X)
        return self

    def _local(self, Q, drop_self=False):
        """반환: 거리, 변수별 기여(z·Pz / d², 쇼트마다 합 1)"""
        idx = self.nn.kneighbors(np.round(Q, NN_DECIMALS), return_distance=False)
        idx = idx[:, 1:] if drop_self else idx[:, :self.k]
        d = np.empty(len(Q)); C = np.zeros((len(Q), Q.shape[1]))
        for i, q in enumerate(Q):
            lw = LedoitWolf().fit(self.X[idx[i]])
            z = q - lw.location_
            pz = lw.precision_ @ z
            d2 = float(z @ pz)
            d[i] = np.sqrt(max(d2, 0.0))
            if d2 > 0:
                C[i] = z * pz / d2
        return d, C

    def score(self, df):
        """이상 점수 (국소 Mahalanobis 거리)"""
        return self._local(self._prep(df))[0]

    def top3(self, df):
        """샷별 기여 상위 3개 변수 (이름(기여%, 편차))"""
        Xr = self._X(df)
        _, C = self._local(self._prep(df))
        out = []
        for i in range(len(C)):
            o = np.argsort(-C[i])[:3]
            out.append(', '.join(f'{FEAT_NAME.get(self.cols[j], self.cols[j])}({100 * C[i, j]:.0f}%, '
                                 f'{np.nan_to_num(Xr[i, j]):+.1f})' for j in o))
        return out

    def pct(self, s):
        """reference 분포 대비 백분위 (0.999 = 정상 샷의 0.1%만 이보다 높음)"""
        return np.searchsorted(self.s_ref, np.asarray(s, float), side='right') / (len(self.s_ref) + 1)

    def threshold(self, q):
        """pct ≥ q 에 해당하는 점수 임계값"""
        return float(np.quantile(self.s_ref, q))

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ====================================================================== 4. 평가 지표
def random_band(y, n=1000):
    """무작위 순위 n번의 AP 평균·97.5%"""
    aps = [average_precision_score(y, np.random.RandomState(s).rand(len(y))) for s in range(n)]
    return float(np.mean(aps)), float(np.percentile(aps, 97.5))


def info_ceiling(rows):
    """같은 샷(같은 입력)의 행은 같은 판정 → 달성 가능한 최대 F1"""
    g = rows.groupby('TimeStamp')['PassOrFail'].agg(p='sum', t='size')
    g['n'] = g['t'] - g['p']
    g = g[g['p'] > 0].assign(r=lambda x: x['p'] / x['t']).sort_values(['r', 'p'], ascending=False)
    P, tp, fp, best = rows['PassOrFail'].sum(), 0, 0, 0.0
    for _, r in g.iterrows():
        tp += r['p']; fp += r['n']
        best = max(best, 2 * tp / (2 * tp + fp + (P - tp)))
    return best


def _order(s):
    """점수 내림차순, 동점은 무작위 (시간 순서 덕에 유리해지지 않게)"""
    return np.lexsort((np.random.RandomState(0).rand(len(s)), -np.asarray(s, float)))


def budget_f1(y, s, q):
    """상위 q를 불량으로 판정했을 때 (TP, FP, FN, F1)"""
    y = np.asarray(y)
    k = int(round(len(s) * q))
    pred = np.zeros(len(s), int); pred[_order(s)[:k]] = 1
    tp = int((pred & y).sum()); fp = k - tp; fn = int(y.sum()) - tp
    return tp, fp, fn, 2 * tp / (2 * tp + fp + fn)


def ap_ci(y, s, n_boot=500, seed=SEED):
    """AP + 층화 부트스트랩 95% 구간"""
    y, s = np.asarray(y), np.asarray(s)
    ap = average_precision_score(y, s)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = [average_precision_score(y[i], s[i]) for i in
            (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n_boot))]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return ap, lo, hi


def recall_at_k_by_run(g, k, pos_mask):
    """Run마다 상위 k% 샷을 검사할 때 잡은 불량 / 전체 불량"""
    hit = tot = 0
    for _, r in g.groupby('run_id'):
        top = r.nlargest(max(1, int(np.ceil(k * len(r)))), 'score').index
        hit += int(pos_mask.loc[top].sum())
        tot += int(pos_mask.loc[r.index].sum())
    return hit, tot


def eval_frame1(p, rows, shot_score):
    """① 공식 2차 · 행 단위 — AP·무작위 대비·상위 k%·F1(최적·상한·예산)"""
    y = rows['PassOrFail'].values.astype(int)
    s = rows['TimeStamp'].map(shot_score).fillna(0).values
    ap = average_precision_score(y, s)
    rmean, rhi = random_band(y)
    ceil = info_ceiling(rows)
    pr, rc, _ = precision_recall_curve(y, s)
    with np.errstate(invalid='ignore', divide='ignore'):
        f1_best = float(np.nanmax(2 * pr * rc / (pr + rc)))
    r = {'제품': p, '행': len(y), '불량 행': int(y.sum()), 'AP': ap, '무작위 평균 대비': ap / rmean,
         '무작위 상단(97.5%)': rhi, '유의': 'O' if ap > rhi else '-'}
    o = _order(s)
    for k in (1, 2, 5, 10):
        r[f'상위{k}%'] = f'{int(y[o[:max(1, int(len(s) * k / 100))]].sum())}/{int(y.sum())}'
    for q in (0.05, 0.10):
        tp, fp, fn, f1 = budget_f1(y, s, q)
        r[f'F1@{int(q * 100)}%'] = f1
        r[f'TP@{int(q * 100)}%'] = tp
    r.update({'F1(최적,참고)': f1_best, '상한 F1': ceil, 'F1@5% / 상한': r['F1@5%'] / ceil,
              'F1(최적) / 상한': f1_best / ceil})
    return r


def eval_frame2(p, g):
    """② labeled 전체 · 샷 단위 · 초기허용 제외. g: run_id, y, Reason, score"""
    g = g.reset_index(drop=True).copy()
    g['score'] = g['score'].values + np.random.default_rng(SEED).random(len(g)) * 1e-9
    y = g['y'].values
    ap, lo, hi = ap_ci(y, g['score'].values)
    r = {'제품': p, '샷': len(g), '불량': int(y.sum()), '불량률': y.mean(),
         'AP': ap, 'AP_lo': lo, 'AP_hi': hi, 'AP/불량률': ap / y.mean()}
    pos = g['y'] == 1
    for k in (0.05, 0.10):
        h, t = recall_at_k_by_run(g, k, pos)
        r[f'Run별 상위{int(k * 100)}%'] = f'{h}/{t}'
    ga = g.assign(run_id='ALL')
    for k in (0.05, 0.10):
        h, t = recall_at_k_by_run(ga, k, pos)
        r[f'전체 상위{int(k * 100)}%'] = f'{h}/{t}'
        r[f'F1@{int(k * 100)}%'] = budget_f1(y, g['score'].values, k)[3]
    for name, reasons in TYPE_GROUPS.items():
        pm = g['Reason'].isin(reasons)
        if pm.sum() == 0:
            continue
        ev = pm | (g['y'] == 0)
        r[f'AP_{name}'] = average_precision_score(pm[ev].astype(int), g.loc[ev, 'score'])
        h, t = recall_at_k_by_run(g, 0.10, pm)
        r[f'Run별 상위10%_{name}'] = f'{h}/{t}'
    return r


def threshold_table(p, g, qs=QS):
    """q 임계값(라벨 미사용)으로 판정했을 때 혼동행렬 지표"""
    y, pc = g['y'].values, g['pct'].values
    rows = []
    for q in qs:
        pred = (pc >= q).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(pred.sum()) - tp; fn = int(y.sum()) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        rows.append({'제품': p, 'q': q, '정상 오경보율(가정)': f'{100 * (1 - q):g}%', '경보 수': int(pred.sum()),
                     '경보 비율': pred.mean(), 'TP': tp, 'FP': fp, 'FN': fn, 'TN': len(y) - tp - fp - fn,
                     'Precision': prec, 'Recall': rec, 'F1': 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                     '가스 탐지': int(((pred == 1) & (g['Reason'].values == '가스')).sum()),
                     '미성형계열 탐지': int(((pred == 1) & g['Reason'].isin(TYPE_GROUPS['미성형']).values).sum())})
    return pd.DataFrame(rows)


# ====================================================================== 5. 학습 / 테스트 (분리)
def train(base_dir=BASE_DIR, products=PRODUCTS, out_dir=OUT_DIR, verbose=True):
    """
    [학습] unlabeled만 사용 — 라벨 0개
      reference 생성 → 파생변수 → FinalModel.fit → {p}_final_model.pkl 저장
    반환: {'data_log': 처리 단계별 샷 수, 'params': 변수별 학습 통계, 'models': {p: FinalModel}}
    """
    os.makedirs(out_dir, exist_ok=True)
    LOG, PAR, MODELS = [], [], {}
    for p in products:
        shots = load_labeled_shots(p, base_dir)                   # labeled는 '시간대 제거'에만 사용 (라벨 X)
        period = ref_period(shots)                                # 평가 기간 이전 정상만 (미래 정보 없음)
        ref_shots, log = build_reference(p, shots, base_dir, period=period)
        steps, fr_step, src = load_scales(p, ref_shots, base_dir)
        steps_all = {v: resolution(ref_shots[raw(v)]) for v in ALL_RAW}
        steps_all.update(steps)
        ref_feat = make_features(ref_shots, steps, fr_step, steps_all)
        if len(ref_feat) > REF_MAX:
            ref_feat = ref_feat.sample(REF_MAX, random_state=SEED).reset_index(drop=True)

        model = FinalModel().fit(ref_feat)
        model.steps, model.fr_step, model.steps_all, model.product = steps, fr_step, steps_all, p
        model.ref_period = period
        model.save(f'{out_dir}/{p}_final_model.pkl')
        MODELS[p] = model

        log.update({'제품': p, '기준값 출처': src, '학습 기간': f'{period[0]} ~ {period[1]}', '학습 샷(최종)': model.n_ref,
                    '국소 이웃 수 k': model.k,
                    'Ledoit-Wolf 축소계수(전역, 참고)': float(model.lw.shrinkage_)})
        LOG.append(log)
        PAR.append(pd.DataFrame({'제품': p, '변수': model.cols, '중앙값': model.med, 'IQR(스케일)': model.iqr}))
        if verbose:
            print(f'[{p}] 학습 완료: reference {model.n_ref:,}샷({period[0]}~{period[1][:10]}), 변수 {len(model.cols)}개, k={model.k} '
                  f'→ {p}_final_model.pkl')
    T = {'data_log': pd.DataFrame(LOG), 'params': pd.concat(PAR, ignore_index=True), 'models': MODELS}
    T['data_log'].to_csv(f'{out_dir}/data_log.csv', index=False, encoding='utf-8-sig')
    T['params'].to_csv(f'{out_dir}/train_params.csv', index=False, encoding='utf-8-sig')
    return T


def test(base_dir=BASE_DIR, products=PRODUCTS, out_dir=OUT_DIR, verbose=True):
    """
    [테스트] 저장된 모델(.pkl)을 불러와 labeled에 점수 → 평가 (라벨은 채점에만 사용)
      파생변수 기준값(기록 단위 등)도 모델에 저장된 학습 때 값을 그대로 사용
    반환: frame1, frame2, threshold, check, scores
    """
    F1, F2, TH, CHK, SCORES = [], [], [], [], {}
    for p in products:
        path = f'{out_dir}/{p}_final_model.pkl'
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} 없음 → train()을 먼저 실행하세요')
        model = FinalModel.load(path)

        shots = load_labeled_shots(p, base_dir)
        lab_feat = make_features(shots, model.steps, model.fr_step, model.steps_all)   # 전체 흐름 그대로
        lab_feat['score'] = model.score(lab_feat)
        lab_feat['pct'] = model.pct(lab_feat['score'])
        lab_feat['top3_local'] = model.top3(lab_feat)
        out_cols = ['TimeStamp', 'run_id', 'split', 'PassOrFail', 'Reason', 'score', 'pct', 'top3_local'] + C17 + FEATURES
        lab_feat[[c for c in out_cols if c in lab_feat.columns]].to_csv(
            f'{out_dir}/{p}_scores.csv', index=False, encoding='utf-8-sig')
        SCORES[p] = lab_feat
        shot_score = pd.Series(lab_feat['score'].values, index=lab_feat['TimeStamp'].values)

        # ① 공식 2차 · 행
        F1.append(eval_frame1(p, load_official_rows(p, base_dir), shot_score))
        # ② 전체 · 샷 · 초기허용 제외
        f2 = lab_feat[(lab_feat['split'] != '제외') & (lab_feat['Reason'] != INITIAL)]
        g = pd.DataFrame({'run_id': f2['run_id'].values, 'y': f2['PassOrFail'].values.astype(int),
                          'Reason': f2['Reason'].values, 'score': f2['score'].values, 'pct': f2['pct'].values})
        F2.append(eval_frame2(p, g))
        TH.append(threshold_table(p, g))
        if verbose:
            print(f'[{p}] 테스트 완료: labeled {len(lab_feat):,}샷 점수')

    R = {'frame1': pd.DataFrame(F1), 'frame2': pd.DataFrame(F2),
         'threshold': pd.concat(TH, ignore_index=True), 'scores': SCORES}
    for p in products:                                                 # 재현 확인 (동결 기준값과 비교)
        for fr, tbl in (('①', R['frame1']), ('②', R['frame2'])):
            ap = float(tbl.loc[tbl['제품'] == p, 'AP'].iloc[0])
            exp = EXPECTED.get((p, fr))
            CHK.append({'제품': p, '틀': fr, 'AP(이번)': round(ap, 3), 'AP(기준)': exp,
                        '일치': 'O' if exp is not None and abs(ap - exp) < 0.0015 else '확인 필요'})
    R['check'] = pd.DataFrame(CHK)
    for k in ('frame1', 'frame2', 'threshold', 'check'):
        R[k].to_csv(f'{out_dir}/{k}.csv', index=False, encoding='utf-8-sig')
    return R


def run_all(base_dir=BASE_DIR, products=PRODUCTS, out_dir=OUT_DIR, verbose=True):
    """train() + test() 한 번에 (20_final_model.ipynb 호환)"""
    T = train(base_dir, products, out_dir, verbose)
    R = test(base_dir, products, out_dir, verbose)
    R.update({'data_log': T['data_log'], 'params': T['params'], 'models': T['models']})
    return R


# ====================================================================== 6. (참고) 저장된 모델로 새 샷 점수
def score_new_shots(shots, p, model_dir=OUT_DIR):
    """
    shots: run_id, TimeStamp, *_raw (같은 Run의 과거 샷 포함 — 기준선 계산에 직전 20샷 필요)
    반환: shots + 파생변수 + score, pct  (pct ≥ q 이면 검사 대상)
    """
    model = FinalModel.load(f'{model_dir}/{p}_final_model.pkl')
    f = make_features(shots, model.steps, model.fr_step, model.steps_all)   # 학습 때 저장한 기준값 그대로
    f['score'] = model.score(f)
    f['top3_local'] = model.top3(f)
    f['pct'] = model.pct(f['score'])
    return f
