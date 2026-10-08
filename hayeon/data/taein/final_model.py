"""
final_model.py — 최종 모델 (M_maha_free) 학습·테스트 단독 실행 파일

모델: 정상 기준(unlabeled) 비지도 이상탐지 — 라벨은 평가에만 사용
  1) 학습 데이터 : unlabeled 정상 생산 샷 (2020-09-01 ~ 11-30, 우진2호기)
                   labeled 샷·labeled Run 시간창(±60분)·끼어든 이상 기록·짧은 Run·0 모드(비가동) 제거
  2) 입력 변수   : 인과적 파생변수 17개 ('free') — 같은 Run의 '과거 샷'만 사용 (미래 정보 없음)
  3) 학습        : reference 중앙값·IQR로 스케일 → ±20 클리핑 → Ledoit-Wolf 공분산
  4) 점수        : Mahalanobis 거리 (클수록 불량 의심) → reference 분포 대비 백분위(pct)
  5) 임계값      : pct ≥ q  (q = 1 − 정상 오경보율, 라벨 없이 결정)

평가 (두 틀)
  ① 공식 2차 · 행 단위 : 팀원 official.py evaluate()와 같은 지표 (AP, 무작위 대비·유의, 상위 1/2/5/10%,
                          F1(최적,참고), 이론 상한 F1, 예산 F1@5/10%)
  ② labeled 전체 · 샷 단위 · 초기허용불량 제외 : AP(+95% CI), Run별/전체 상위 k%, 유형별(가스·미성형), q 임계값 성능

필요 파일 (notebooks 폴더 기준, 앞 단계 노트북 출력)
  ./linked/linked_unlabeled_{cn7,rg3}.csv   1차 unlabeled (TimeStamp, *_raw)          ← 01_link_primary_secondary
  ./linked/linked_labeled.csv               공식 2차 행 ↔ 1차 시각 연결             ← 01_link_primary_secondary
  ./base/{cn7,rg3}_shots.csv                labeled 샷 단위 (run_id, split, Reason)  ← 00_base_labeled
  ./base/reference_scales.json              변수별 기록 단위·흐름저항 하한          ← 06_loro_compare
                                            (없으면 reference로 계산 → 수치가 조금 다를 수 있음)
패키지: numpy, pandas, scikit-learn

사용
  from final_model import train, test
  T = train()        # 학습: unlabeled → 모델 저장 (outputs/final_model/{p}_final_model.pkl)
  R = test()         # 테스트: 저장된 모델 불러와 labeled 평가 → R['frame1'], R['frame2'], R['threshold'], R['check']
  (run_all() = train() + test())
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, precision_recall_curve, auc

# ====================================================================== 설정 (결과 보기 전에 고정한 값)
SEED = 42
PRODUCTS = ('cn7', 'rg3')
REF_PERIOD = ('2020-09-01', '2020-11-30')
RUN_GAP_MIN, STOP_GAP_MIN, MIN_RUN_SHOTS = 60, 5, 10    # Run 구분: 60분 공백 = 새 Run, 10샷 미만 Run 제거
MIN_CYCLE_SEC, CYCLE_DEV_SEC = 50, 2                    # 끼어든 이상 기록 판정
ZERO_MODE_BARREL = 100                                  # 배럴1 온도 < 100℃ = 비가동(0 모드)
REF_MAX = 30000                                         # reference 상한 (넘으면 무작위 추출)
W, MINP, REJECT, RESET_AFTER = 20, 5, 5.0, 5            # 인과적 기준선: 직전 20샷, 최소 5샷, |z|>5 제외, 5연속 제외 시 재설정
CLIP = 20
QS = (0.95, 0.98, 0.99, 0.995, 0.999)
INITIAL = '초기허용불량'
TYPE_GROUPS = {'가스': ['가스'], '미성형': ['미성형', '가스+미성형']}
# 평가에서 뺄 Run (팀 결정): RG3 R6(2020-11-04)은 공식 2차 범위 밖 → 제외 (공식 2차 기준 가스 17 / 미성형 8)
#   학습(reference)에서는 원래대로 이 시간대도 빼 둠 (누수 방지), 파생변수는 전체 시간 흐름으로 계산
EXCLUDE_RUNS = {'rg3': ['rg3_R6']}
# 18_final_compare 결과 (재현 확인용). RG3 ②는 R6 제외로 값이 바뀌어 비교 생략(None)
EXPECTED = {('cn7', '①'): 0.373, ('rg3', '①'): 0.024, ('cn7', '②'): 0.212, ('rg3', '②'): None}

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

# 최종 입력 변수 17개
FEATURES = ([f'dev_{v}' for v in EVENT_VARS]                                    # 이벤트 편차 8
            + ['fill_anomaly', 'ev_max_abs_dev']                                # 이벤트 요약 2
            + ['dev_Mold_Temperature_3', 'dev_Hopper_Temperature',              # 열 상태 6
               'mold_rise', 'hopper_rise', 'mold_slope20', 'hopper_slope20']
            + ['dev_flow_res'])                                                 # 흐름저항 1


# ====================================================================== 1. 데이터
def apply_exclusions(d, p):
    """EXCLUDE_RUNS의 Run을 split='제외'로 표시 → 평가·지도학습 대상에서 빠짐 (행은 남겨 파생변수 흐름 유지)"""
    ex = EXCLUDE_RUNS.get(p, [])
    if ex and 'run_id' in d.columns:
        d = d.copy()
        if 'split' not in d.columns:
            d['split'] = ''
        d.loc[d['run_id'].isin(ex), 'split'] = '제외'
    return d


def load_labeled_shots(p, base_dir='.'):
    s = pd.read_csv(f'{base_dir}/base/{p}_shots.csv', parse_dates=['TimeStamp'])
    s['Reason'] = s['Reason'].fillna('정상')
    return apply_exclusions(s.sort_values('TimeStamp').reset_index(drop=True), p)


def load_official_rows(p, base_dir='.'):
    """공식 2차 행 (LH/RH 행 단위 라벨)"""
    l = pd.read_csv(f'{base_dir}/linked/linked_labeled.csv', parse_dates=['TimeStamp'],
                    usecols=['product', 'in_secondary', 'TimeStamp', 'cavity', 'PassOrFail', 'Reason'])
    l = l[(l['product'] == p) & l['in_secondary']].copy()
    l['Reason'] = l['Reason'].fillna('정상')
    return l.reset_index(drop=True)


def build_reference(p, labeled_shots, base_dir='.', period=REF_PERIOD):
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

    # (4) 0 모드(비가동 기록) 제거 — Run 구분 뒤에 제거 (18번과 같은 순서)
    z = u[raw('Barrel_Temperature_1')] < ZERO_MODE_BARREL
    u = u[~z]
    log['0 모드 제거'] = int(z.sum())

    keep = ['run_id', 'TimeStamp'] + [raw(v) for v in ALL_RAW if raw(v) in u.columns]
    u = u[keep].reset_index(drop=True)
    log['reference 샷'] = len(u)
    log['reference Run'] = u['run_id'].nunique()
    return u, log


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


def make_features(df, steps, fr_step):
    """df: run_id, TimeStamp, *_raw (샷 단위). 반환: FEATURES가 추가된 df (Run → 시간 순)"""
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
    return d


def load_scales(p, ref_shots, base_dir='.'):
    """변수별 기록 단위·흐름저항 하한 — json 우선, 없으면 reference로 계산"""
    path = f'{base_dir}/base/reference_scales.json'
    if os.path.exists(path):
        s = json.load(open(path, encoding='utf-8'))[p]
        return s['steps'], s['fr_step'], 'reference_scales.json'
    steps = {v: resolution(ref_shots[raw(v)]) for v in EVENT_VARS + STATE_VARS}
    fr = (ref_shots[raw('Max_Injection_Pressure')] / ref_shots[raw('Max_Injection_Speed')]).replace([np.inf, -np.inf], np.nan)
    fr_step = float(fr.diff().abs().replace(0, np.nan).quantile(0.05))
    print(f'[{p}] ⚠ reference_scales.json 없음 → reference에서 계산 (18번 수치와 조금 다를 수 있음)')
    return steps, fr_step, 'reference에서 계산'


# ====================================================================== 3. 모델
class FinalModel:
    """정상 기준 Mahalanobis 이상탐지 (M_maha_free)"""

    def __init__(self, cols=FEATURES):
        self.cols = list(cols)

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
        X = self._prep(ref_feat)
        self.lw = LedoitWolf().fit(X)
        self.s_ref = np.sort(np.sqrt(self.lw.mahalanobis(X)))
        self.n_ref = len(X)
        return self

    def score(self, df):
        """이상 점수 (Mahalanobis 거리)"""
        return np.sqrt(self.lw.mahalanobis(self._prep(df)))

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
    """무작위 순위 n번의 AP 평균·97.5% (팀원 official.random_band와 같음)"""
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


def pr_auc_trapz(y, s):
    """
    PR-AUC (사다리꼴 적분) = PR 곡선 아래 면적을 점 사이 직선으로 이어 계산
    AP(average precision)도 PR-AUC의 한 계산법(계단식). 불량이 적으면 사다리꼴은 AP보다 다소 낙관적으로 나옴
    """
    pr, rc, _ = precision_recall_curve(y, s)
    return auc(rc, pr)


def eval_frame1(p, rows, shot_score):
    """① 공식 2차 · 행 단위 — 팀원 evaluate() 지표 + 예산 F1"""
    y = rows['PassOrFail'].values.astype(int)
    s = rows['TimeStamp'].map(shot_score).fillna(0).values
    ap = average_precision_score(y, s)
    rmean, rhi = random_band(y)
    ceil = info_ceiling(rows)
    pr, rc, _ = precision_recall_curve(y, s)
    with np.errstate(invalid='ignore', divide='ignore'):
        f1_best = float(np.nanmax(2 * pr * rc / (pr + rc)))
    r = {'제품': p, '행': len(y), '불량 행': int(y.sum()), 'AP': ap, 'PR-AUC(사다리꼴)': pr_auc_trapz(y, s),
         '무작위 PR-AUC(=불량률)': y.mean(), '무작위 평균 대비': ap / rmean,
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
         'AP': ap, 'AP_lo': lo, 'AP_hi': hi, 'AP/불량률': ap / y.mean(),
         'PR-AUC(사다리꼴)': pr_auc_trapz(y, g['score'].values)}
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
def train(base_dir='.', products=PRODUCTS, out_dir='./outputs/final_model', verbose=True):
    """
    [학습] unlabeled만 사용 — 라벨 0개
      reference 생성 → 파생변수 → FinalModel.fit → {p}_final_model.pkl 저장
      모델에 실제로 들어간 학습 데이터 → {p}_train_features.csv 저장
        (Run·시각 + 원값 21개 + 파생변수 17개 + 학습 데이터 자신의 점수·백분위)
    반환: {'data_log': 처리 단계별 샷 수, 'params': 변수별 학습 통계, 'models': {p: FinalModel},
           'train_features': {p: 학습 데이터 DataFrame}}
    """
    os.makedirs(out_dir, exist_ok=True)
    LOG, PAR, MODELS, TRAIN = [], [], {}, {}
    for p in products:
        shots = load_labeled_shots(p, base_dir)                   # labeled는 '시간대 제거'에만 사용 (라벨 X)
        ref_shots, log = build_reference(p, shots, base_dir)
        steps, fr_step, src = load_scales(p, ref_shots, base_dir)
        ref_feat = make_features(ref_shots, steps, fr_step)
        if len(ref_feat) > REF_MAX:
            ref_feat = ref_feat.sample(REF_MAX, random_state=SEED).reset_index(drop=True)

        model = FinalModel().fit(ref_feat)
        model.steps, model.fr_step, model.product = steps, fr_step, p
        model.save(f'{out_dir}/{p}_final_model.pkl')
        MODELS[p] = model

        # 모델에 실제로 들어간 학습 데이터 저장 (재현성·검토용)
        tf = ref_feat[['run_id', 'TimeStamp'] + [raw(v) for v in ALL_RAW if raw(v) in ref_feat.columns]
                      + FEATURES].copy()
        tf['score'] = model.score(ref_feat)
        tf['pct'] = model.pct(tf['score'])
        tf.sort_values(['run_id', 'TimeStamp']).to_csv(f'{out_dir}/{p}_train_features.csv', index=False,
                                                        encoding='utf-8-sig')
        TRAIN[p] = tf

        log.update({'제품': p, '기준값 출처': src, '학습 샷(최종)': model.n_ref,
                    'Ledoit-Wolf 축소계수': float(model.lw.shrinkage_)})
        LOG.append(log)
        PAR.append(pd.DataFrame({'제품': p, '변수': model.cols, '중앙값': model.med, 'IQR(스케일)': model.iqr}))
        if verbose:
            print(f'[{p}] 학습 완료: reference {model.n_ref:,}샷, 변수 {len(model.cols)}개 '
                  f'→ {out_dir}/{p}_final_model.pkl, {p}_train_features.csv')
    T = {'data_log': pd.DataFrame(LOG), 'params': pd.concat(PAR, ignore_index=True), 'models': MODELS,
         'train_features': TRAIN}
    T['data_log'].to_csv(f'{out_dir}/data_log.csv', index=False, encoding='utf-8-sig')
    T['params'].to_csv(f'{out_dir}/train_params.csv', index=False, encoding='utf-8-sig')
    return T


def test(base_dir='.', products=PRODUCTS, out_dir='./outputs/final_model', verbose=True):
    """
    [테스트] 저장된 모델(.pkl)을 불러와 labeled에 점수 → 평가 (라벨은 채점에만 사용)
      파생변수 기준값(기록 단위 등)도 모델에 저장된 학습 때 값을 그대로 사용
    반환: frame1, frame2, threshold, check, scores
    """
    F1, F2, TH, CHK, SCORES, OP = [], [], [], [], {}, []
    for p in products:
        path = f'{out_dir}/{p}_final_model.pkl'
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} 없음 → train()을 먼저 실행하세요')
        model = FinalModel.load(path)

        shots = load_labeled_shots(p, base_dir)
        lab_feat = make_features(shots, model.steps, model.fr_step)   # 전체 시간 흐름 그대로 (초기허용·제외 포함)
        lab_feat['score'] = model.score(lab_feat)
        lab_feat['pct'] = model.pct(lab_feat['score'])
        out_cols = ['TimeStamp', 'run_id', 'split', 'PassOrFail', 'Reason', 'score', 'pct'] + FEATURES
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
        # 운영 기준별 Precision·Recall·F1 + 유형별 몇 개 중 몇 개 (검사 상위 5·10%, 경보 q0.99·0.999)
        from op_metrics import op_table
        L = lab_feat.assign(eval=((lab_feat['split'] != '제외') if 'split' in lab_feat.columns else True)
                            & (lab_feat['Reason'] != INITIAL))
        OP.append(op_table(L, L['score'].values, L['pct'].values).assign(제품=p))
        if verbose:
            print(f'[{p}] 테스트 완료: labeled {len(lab_feat):,}샷 점수')

    R = {'frame1': pd.DataFrame(F1), 'frame2': pd.DataFrame(F2),
         'threshold': pd.concat(TH, ignore_index=True), 'scores': SCORES,
         'operating': pd.concat(OP, ignore_index=True)}
    for p in products:                                                 # 재현 확인 (18번 수치와 비교)
        for fr, tbl in (('①', R['frame1']), ('②', R['frame2'])):
            ap = float(tbl.loc[tbl['제품'] == p, 'AP'].iloc[0])
            exp = EXPECTED.get((p, fr))
            CHK.append({'제품': p, '틀': fr, 'AP(이번)': round(ap, 3), 'AP(18번)': exp,
                        '일치': ('-' if exp is None else 'O' if abs(ap - exp) < 0.0015 else '확인 필요')})
    R['check'] = pd.DataFrame(CHK)
    for k in ('frame1', 'frame2', 'threshold', 'check', 'operating'):
        R[k].to_csv(f'{out_dir}/{k}.csv', index=False, encoding='utf-8-sig')
    return R


def run_all(base_dir='.', products=PRODUCTS, out_dir='./outputs/final_model', verbose=True):
    """train() + test() 한 번에 (20_final_model.ipynb 호환)"""
    T = train(base_dir, products, out_dir, verbose)
    R = test(base_dir, products, out_dir, verbose)
    R.update({'data_log': T['data_log'], 'params': T['params'], 'models': T['models']})
    return R


# ====================================================================== 5-1. PR 곡선
def plot_pr(R, base_dir='.', products=PRODUCTS, qs=(0.99, 0.999), save_dir=None):
    """
    제품 × 평가 틀별 PR 곡선 (test() 결과 R 사용)
      회색 점선 = 무작위(불량률), 점 = 라벨 없이 정한 임계값 q의 운영점 (② 틀)
    """
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    fig, axes = plt.subplots(len(products), 2, figsize=(11, 4.2 * len(products)), squeeze=False)
    for i, p in enumerate(products):
        f = R['scores'][p]
        shot = pd.Series(f['score'].values, index=f['TimeStamp'].values)
        rows = load_official_rows(p, base_dir)
        y1 = rows['PassOrFail'].values.astype(int)
        s1 = rows['TimeStamp'].map(shot).fillna(0).values
        m2 = ((f['split'] != '제외') & (f['Reason'] != INITIAL)).values
        y2 = f['PassOrFail'].values[m2].astype(int)
        s2 = f['score'].values[m2]
        pc2 = f['pct'].values[m2]
        for j, (y, s, title) in enumerate([(y1, s1, '① 공식 2차 · 행'), (y2, s2, '② 전체 · 샷 (초기허용 제외)')]):
            ax = axes[i, j]
            pr, rc, _ = precision_recall_curve(y, s)
            ax.step(rc, pr, where='post', color='#1f4e79', lw=1.8,
                    label=f'최종 모델  AP {average_precision_score(y, s):.3f} · PR-AUC {auc(rc, pr):.3f}')
            ax.axhline(y.mean(), ls='--', color='#9a9a9a', lw=1, label=f'무작위 = 불량률 {y.mean():.3f}')
            if j == 1:
                for q, mk in zip(qs, ('o', 's')):
                    pred = pc2 >= q
                    tp = int((pred & (y == 1)).sum())
                    if pred.sum():
                        ax.plot(tp / y.sum(), tp / pred.sum(), mk, color='#c0392b', ms=7,
                                label=f'q={q} 운영점 (경보 {int(pred.sum())}, 적중 {tp})')
            ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
            ax.set_xlabel('재현율 (Recall)'); ax.set_ylabel('정밀도 (Precision)')
            ax.set_title(f'{p.upper()} — {title}', loc='left', fontsize=10)
            ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f'{save_dir}/pr_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


# ====================================================================== 6. (참고) 저장된 모델로 새 샷 점수
def score_new_shots(shots, p, model_dir='./outputs/final_model'):
    """
    shots: run_id, TimeStamp, *_raw (같은 Run의 과거 샷 포함 — 기준선 계산에 직전 20샷 필요)
    반환: shots + 파생변수 + score, pct  (pct ≥ q 이면 검사 대상)
    """
    model = FinalModel.load(f'{model_dir}/{p}_final_model.pkl')
    f = make_features(shots, model.steps, model.fr_step)          # 학습 때 저장한 기준값 그대로
    f['score'] = model.score(f)
    f['pct'] = model.pct(f['score'])
    return f


def compare_with_cache(R, p, base_dir='.'):
    """(선택) 이번에 계산한 labeled 파생변수와 기존 base/{p}_features.csv 비교 — 최대 절대차"""
    path = f'{base_dir}/base/{p}_features.csv'
    if not os.path.exists(path):
        return '캐시 파일 없음'
    c = pd.read_csv(path, parse_dates=['TimeStamp'])
    m = R['scores'][p][['TimeStamp'] + FEATURES].merge(c[['TimeStamp'] + FEATURES], on='TimeStamp',
                                                        suffixes=('', '_cache'))
    return pd.DataFrame({'변수': FEATURES,
                         '최대 절대차': [np.nanmax(np.abs(m[v] - m[f'{v}_cache'])) for v in FEATURES],
                         'NaN 불일치': [int((m[v].isna() != m[f'{v}_cache'].isna()).sum()) for v in FEATURES]})
