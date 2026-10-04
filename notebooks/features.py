"""
features.py — 파생변수 생성 모듈 (KAMP 사출성형 · 샷 단위)

A. make_causal_features(df, steps, fr_step)
   라벨 X, 같은 Run의 '과거 샷'만 사용 → 전처리 단계에서 전체 데이터에 한 번 적용 가능
   기준값(기록 단위·흐름 저항 하한)은 테스트 밖 데이터(unlabeled)에서 계산해 넘김
B. FoldFeaturizer
   학습 fold로 fit → 학습/테스트 fold에 transform (누수 방지)
C. FEATURE_GROUPS / COMBOS
   변수 조합 실험용 (C00~C8: 파생변수 효과, T0~T2: 시계열 효과)
D. screen_features
   단변량 AUC (탐색용, 최종 선택은 LORO 조합 비교로)
E. make_shuffled_features
   대조 실험: Run 안 순서를 섞고 시계열 피처 재계산

입력: base/{cn7,rg3}_shots.csv (00_base_labeled 출력, 샷 단위, split '제외' 제거 후 사용)
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

# ====================================================================== 설정
W = 20            # 인과적 기준선: 직전 20샷
MINP = 5          # 기준선 최소 샷 수 (Run 첫 5샷은 NaN)
REJECT = 5.0      # |dev| > 5인 샷은 기준선에서 제외 (연속 불량이 기준선 오염하는 것 방지)
RESET_AFTER = 5   # 5샷 연속 제외되면 '새 수준'으로 보고 기준선 재설정 (설정 변경 대응)
TRANSIENT = 5     # 재가동 후 과도구간 샷 수

raw = lambda c: f'{c}_raw'

# base 파일에 남아 있는 원본 공정변수 21개 (2차 24개 중 상수·기록오류 3개 제외)
ALL_RAW = ['Injection_Time', 'Filling_Time', 'Plasticizing_Time', 'Cycle_Time', 'Clamp_Close_Time',
           'Cushion_Position', 'Plasticizing_Position', 'Max_Injection_Speed', 'Max_Screw_RPM',
           'Max_Injection_Pressure', 'Max_Switch_Over_Pressure', 'Average_Back_Pressure',
           'Barrel_Temperature_1', 'Barrel_Temperature_2', 'Barrel_Temperature_3',
           'Barrel_Temperature_4', 'Barrel_Temperature_5', 'Barrel_Temperature_6',
           'Hopper_Temperature', 'Mold_Temperature_3', 'Mold_Temperature_4']

# 원본 핵심 변수 13개 (중복 제거: Injection_Time≈Filling_Time, Mold_4≈Mold_3 / RPM 거의 상수 / 배럴은 4구간만)
RAW_CORE = ['Max_Injection_Speed', 'Filling_Time', 'Max_Injection_Pressure', 'Max_Switch_Over_Pressure',
            'Average_Back_Pressure', 'Cushion_Position', 'Plasticizing_Time', 'Plasticizing_Position',
            'Cycle_Time', 'Clamp_Close_Time', 'Mold_Temperature_3', 'Hopper_Temperature',
            'Barrel_Temperature_4']

# 이벤트형: 사출 계열 (샷 단위로 튀는 변수)
EVENT_VARS = ['Max_Injection_Speed', 'Filling_Time', 'Max_Injection_Pressure',
              'Max_Switch_Over_Pressure', 'Average_Back_Pressure', 'Cushion_Position',
              'Plasticizing_Time', 'Cycle_Time']
# 상태형: 느리게 변하는 변수
STATE_VARS = ['Mold_Temperature_3', 'Hopper_Temperature']
# 흐름 저항(점도) 묶음 — A2·A10 근거
FLOW_VARS = ['Max_Injection_Speed', 'Max_Injection_Pressure', 'Max_Switch_Over_Pressure',
             'Cushion_Position', 'Average_Back_Pressure']

# 모델 입력 금지 열
LABEL_COLS = ['run_id', 'split', 'TimeStamp', 'PassOrFail', 'PassOrFail_process', 'Reason',
              'LH_fail', 'RH_fail']


# ====================================================================== 공통 함수
def resolution(x):
    """기록 단위(0.01 / 0.1 등) = 고유값 간 최소 간격"""
    v = np.unique(np.round(pd.Series(x).dropna().values.astype(float), 4))
    return float(np.round(np.min(np.diff(v)), 4)) if len(v) > 1 else 1.0


def causal_robust_dev(x, step, w=W, minp=MINP, reject=REJECT, reset_after=RESET_AFTER):
    """
    dev_t = (x_t − 기준선 중앙값) / max(1.4826·MAD, 기록 단위)
    기준선 = 같은 Run의 과거 샷 중 '정상으로 받아들인' 최근 w개 (라벨 미사용)
      - |dev| > reject 인 샷은 기준선에 넣지 않음
      - reset_after 샷 연속 제외되면 수준이 바뀐 것으로 보고 그 샷들로 기준선 교체
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
    """과거 w샷(현재 포함) 선형 기울기 (단위/샷)"""
    y = pd.Series(y, dtype=float)

    def _s(a):
        a = a[~np.isnan(a)]
        if len(a) < 3:
            return np.nan
        return np.polyfit(np.arange(len(a)), a, 1)[0]

    return y.rolling(w, min_periods=3).apply(_s, raw=True).values


def _by_run(d, col, func):
    """Run별로 func(1차원 배열) → 같은 길이 배열을 계산해서 Series로 반환"""
    out = pd.Series(np.nan, index=d.index)
    for _, idx in d.groupby('run_id', sort=False).groups.items():
        out.loc[idx] = func(d.loc[idx, col].values)
    return out


# ====================================================================== 기준값 (테스트 밖 데이터에서 계산)
def reference_scales(ref_df):
    """
    기록 단위·흐름 저항 하한을 기준 데이터에서 계산
    ref_df: labeled 평가에 안 쓰는 데이터 (예: unlabeled_data.csv 해당 제품), 열 이름은 *_raw
    """
    steps = {v: resolution(ref_df[raw(v)]) for v in EVENT_VARS + STATE_VARS}
    fr = ref_df[raw('Max_Injection_Pressure')] / ref_df[raw('Max_Injection_Speed')]
    fr_step = fr.diff().abs().replace(0, np.nan).quantile(0.05)
    return steps, fr_step


def load_reference(unlabeled_path, product):
    """
    unlabeled_data.csv(1차 원본)에서 해당 제품·설비 행을 읽어 *_raw 이름으로 변환
    product: 'cn7' 또는 'rg3'
    ※ Average_Screw_RPM 10배 기록 오류는 여기서 안 쓰므로 무시
    """
    ref = pd.read_csv(unlabeled_path, low_memory=False)
    ref = ref[ref['PART_NAME'].str.contains(product.upper(), na=False)
              & ref['EQUIP_NAME'].str.contains('우진2', na=False)]
    ref = ref.drop_duplicates(subset=['TimeStamp'])          # 2행 기록된 샷 → 1행
    cols = list(dict.fromkeys(EVENT_VARS + STATE_VARS + FLOW_VARS))
    return ref.rename(columns={v: raw(v) for v in cols})


# ====================================================================== A. 인과적 파생변수
def make_causal_features(df, steps=None, fr_step=None):
    """
    df      : base/{p}_shots.csv (한 제품, split '제외' 제거 후)
    steps   : {변수명: 기록 단위} — None이면 df에서 계산 (탐색용). 평가용은 reference_scales 결과
    fr_step : 흐름 저항 편차 하한 — None이면 df에서 계산 (탐색용). 평가용은 reference_scales 결과
    반환    : 파생변수가 추가된 df (Run → TimeStamp 순 정렬, Run 단위 계산)
    """
    d = df.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True).copy()

    # 기준값: 넘겨받은 값 우선, 없는 변수만 df에서 계산
    _steps = {v: resolution(d[raw(v)]) for v in EVENT_VARS + STATE_VARS}
    if steps:
        _steps.update({k: v for k, v in steps.items() if k in _steps})
    steps = _steps

    # --- 1) 맥락 (재가동·위치)
    d['is_run_start'] = (d.groupby('run_id').cumcount() == 0).astype(int)
    gap_min = d['prev_gap_sec'].fillna(0) / 60
    d['stop_len_log'] = np.where(gap_min >= 5, np.log1p(gap_min), 0.0)   # 5분 이상 정지만
    d['is_transient'] = (d['shots_since_stop'] <= TRANSIENT).astype(int)
    d['log_since_stop'] = np.log1p(d['shots_since_stop'])
    d['log_seq'] = np.log1p(d['shot_seq_in_run'])

    # --- 2) 이벤트형: 인과적 편차·변화량·변동성
    for v in EVENT_VARS:
        g = d.groupby('run_id', sort=False)[raw(v)]
        d[f'dev_{v}'] = _by_run(d, raw(v), lambda a, s=steps[v]: causal_robust_dev(a, s))
        d[f'delta_{v}'] = g.diff() / steps[v]                                    # 직전 샷 대비 (기록 단위 칸 수)
        d[f'vol_{v}'] = g.transform(lambda s: s.shift(1).rolling(W, min_periods=MINP).std()) / steps[v]

    dev_cols = [f'dev_{v}' for v in EVENT_VARS]
    A = d[dev_cols].abs()
    d['ev_max_abs_dev'] = A.max(axis=1, skipna=True)
    d['ev_n_over3'] = (A > 3).sum(axis=1)
    d['ev_sum_sq'] = (d[dev_cols].fillna(0) ** 2).sum(axis=1)
    # 충전 이상: 충전↑·속도↓ = 양수(램프업형), 충전↓·속도↑ = 음수(미성형형)
    d['fill_anomaly'] = d['dev_Filling_Time'] - d['dev_Max_Injection_Speed']

    # --- 3) 상태형: 금형·호퍼 열 상태
    for v, name in [('Mold_Temperature_3', 'mold'), ('Hopper_Temperature', 'hopper')]:
        g = d.groupby('run_id', sort=False)[raw(v)]
        # Run 첫 5샷 평균 (5샷 전까지는 지금까지의 평균 → 미래 정보 없음)
        d[f'{name}_start'] = g.transform(
            lambda s: s.expanding().mean().where(np.arange(len(s)) < 5).ffill())
        d[f'{name}_rise'] = d[raw(v)] - d[f'{name}_start']                       # Run 시작 대비 상승량
        d[f'{name}_roll20'] = g.transform(lambda s: s.rolling(W, min_periods=1).mean())
        d[f'{name}_slope20'] = _by_run(d, raw(v), causal_slope)
        d[f'dev_{v}'] = _by_run(d, raw(v), lambda a, s=steps[v]: causal_robust_dev(a, s))

    # --- 4) 흐름 저항 (정의식)
    d['flow_res'] = d[raw('Max_Injection_Pressure')] / d[raw('Max_Injection_Speed')]
    d['flow_res_roll20'] = d.groupby('run_id', sort=False)['flow_res'].transform(
        lambda s: s.rolling(W, min_periods=1).mean())
    if fr_step is None:
        fr_step = d['flow_res'].diff().abs().replace(0, np.nan).quantile(0.05)
    fr_step = 1e-4 if pd.isna(fr_step) else fr_step
    d['dev_flow_res'] = _by_run(d, 'flow_res', lambda a: causal_robust_dev(a, fr_step))

    return d


# ====================================================================== B. Fold 파생변수 (학습 fold로 fit)
class FoldFeaturizer:
    """
    fit(train): 학습 fold의 '양품 + 과도구간 아닌' 샷으로 기준 관계를 적합
    transform(df): 아래 변수 추가
      press_resid : 실제 사출압 − 예측(속도, 금형온도)        → 양품 관계가 깨진 정도 (비시계열)
      flow_pc1    : 흐름 저항 묶음 PCA 1성분                  → 점도 상태 요약 (비시계열)
      maha_dev    : 이벤트 편차 벡터의 Mahalanobis 거리         → 종합 이상 점수 (시계열, dev 기반)
    """
    def __init__(self, dev_cols=None):
        self.dev_cols = dev_cols or [f'dev_{v}' for v in EVENT_VARS]

    def fit(self, train):
        base = train[(train['PassOrFail'] == 0) & (train['shots_since_stop'] > TRANSIENT)]
        # 관계식 잔차
        self.reg_x = [raw('Max_Injection_Speed'), raw('Mold_Temperature_3')]
        self.reg = LinearRegression().fit(base[self.reg_x], base[raw('Max_Injection_Pressure')])
        # 흐름 저항 PCA
        self.flow_x = [raw(v) for v in FLOW_VARS]
        self.flow_sc = StandardScaler().fit(base[self.flow_x])
        self.flow_pca = PCA(n_components=1).fit(self.flow_sc.transform(base[self.flow_x]))
        # 부호 고정: 사출압과 같은 방향이 +
        if self.flow_pca.components_[0][self.flow_x.index(raw('Max_Injection_Pressure'))] < 0:
            self.flow_pca.components_ *= -1
        # Mahalanobis (Run 첫 5샷 NaN은 0 = '편차 없음'으로)
        Z = base[self.dev_cols].fillna(0).values
        self.lw = LedoitWolf().fit(Z)
        return self

    def transform(self, df):
        d = df.copy()
        d['press_resid'] = d[raw('Max_Injection_Pressure')] - self.reg.predict(d[self.reg_x])
        d['flow_pc1'] = self.flow_pca.transform(self.flow_sc.transform(d[self.flow_x]))[:, 0]
        Z = d[self.dev_cols].fillna(0).values
        d['maha_dev'] = np.sqrt(self.lw.mahalanobis(Z))     # sklearn은 제곱거리 반환 → sqrt
        return d


# ====================================================================== C. 변수 그룹 / 조합
FEATURE_GROUPS = {
    # --- 파생변수 효과 비교용
    'raw_all': [raw(v) for v in ALL_RAW],                                         # 21개 (2차 변수 전부에 해당)
    'raw':     [raw(v) for v in RAW_CORE],                                        # 13개 (정리본)
    'context': ['is_run_start', 'stop_len_log', 'is_transient', 'log_since_stop', 'log_seq'],
    'event':   [f'dev_{v}' for v in EVENT_VARS] + ['ev_max_abs_dev', 'ev_n_over3', 'fill_anomaly'],
    'delta':   [f'delta_{v}' for v in EVENT_VARS] + [f'vol_{v}' for v in EVENT_VARS],
    'state':   ['mold_start', 'mold_rise', 'mold_roll20', 'mold_slope20', 'dev_Mold_Temperature_3',
                'hopper_start', 'hopper_rise', 'hopper_slope20', 'dev_Hopper_Temperature'],
    'flow':    ['flow_res', 'flow_res_roll20', 'dev_flow_res'],
    'fold':    ['press_resid', 'flow_pc1', 'maha_dev'],                           # FoldFeaturizer 후에만 존재
}
# --- 시계열 vs 비시계열 비교용
FEATURE_GROUPS['static'] = FEATURE_GROUPS['raw'] + ['flow_res']                   # 그 샷 값만
FEATURE_GROUPS['static_fold'] = ['press_resid', 'flow_pc1']                       # FoldFeaturizer 필요
FEATURE_GROUPS['t_context'] = FEATURE_GROUPS['context']                           # 정지·순번 이력
FEATURE_GROUPS['t_event'] = FEATURE_GROUPS['event'] + FEATURE_GROUPS['delta']     # 과거 샷 대비 편차·변화
FEATURE_GROUPS['t_state'] = FEATURE_GROUPS['state'] + ['flow_res_roll20', 'dev_flow_res']  # 최근 추세
FEATURE_GROUPS['t_fold'] = ['maha_dev']                                           # dev 기반 → 시계열

COMBOS = {
    # --- 파생변수 효과
    'C00_raw_all':          ['raw_all'],                                          # 원본 21개 전부 (기준선)
    'C0_raw':               ['raw'],                                              # 원본 13개 정리본
    'C1_raw+context':       ['raw', 'context'],
    'C2_context+event':     ['context', 'event'],
    'C3_+delta':            ['context', 'event', 'delta'],
    'C4_context+state':     ['context', 'state', 'flow'],
    'C5_event+state':       ['context', 'event', 'state', 'flow'],
    'C6_event+state+fold':  ['context', 'event', 'state', 'flow', 'fold'],
    'C7_all':               ['raw', 'context', 'event', 'delta', 'state', 'flow', 'fold'],
    'C8_compact':           ['context', 'fold'],                                  # 요약 변수만 (과적합 비교용)
    # --- 시계열 효과 (T2_shuffled는 make_shuffled_features로 만든 데이터에 T2 조합을 적용)
    'T0_static':            ['static', 'static_fold'],
    'T1_temporal':          ['t_context', 't_event', 't_state', 't_fold'],
    'T2_static+temporal':   ['static', 'static_fold', 't_context', 't_event', 't_state', 't_fold'],
    'T0+context':           ['static', 'static_fold', 't_context'],
    'T0+event':             ['static', 'static_fold', 't_event', 't_fold'],
    'T0+state':             ['static', 'static_fold', 't_state'],
}

FOLD_GROUPS = ('fold', 'static_fold', 't_fold')


def combo_columns(combo_name):
    """조합 이름 → 열 목록 (중복 제거, 순서 유지)"""
    cols = []
    for gname in COMBOS[combo_name]:
        cols += FEATURE_GROUPS[gname]
    return list(dict.fromkeys(cols))


def needs_fold(combo_name):
    """FoldFeaturizer가 필요한 조합인지"""
    return any(g in FOLD_GROUPS for g in COMBOS[combo_name])


# ====================================================================== D. 단변량 스크리닝 (탐색용)
def screen_features(d, cols, label='PassOrFail', by_reason=True):
    """
    변수 하나로 불량을 얼마나 가르는지 AUC (0.5 = 무관, 방향 무관하게 |AUC−0.5|로 정렬)
    ※ 전체 데이터 사용 → 탐색용. 최종 선택은 LORO 조합 비교로 판단
    """
    rows = []
    groups = {'전체': d[label] == 1}
    if by_reason:
        for r in d['Reason'].dropna().unique():
            if r != '정상':
                groups[r] = d['Reason'] == r
    for c in cols:
        if c not in d.columns:
            continue
        x = d[c]
        ok = x.notna() & np.isfinite(x)
        for gname, pos in groups.items():
            m = ok & (pos | (d[label] == 0))
            y = pos[m].astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                continue
            auc = roc_auc_score(y, x[m])
            rows.append({'변수': c, '불량군': gname, 'n불량': int(y.sum()), 'AUC': auc,
                         '|AUC-0.5|': abs(auc - 0.5), 'NaN비율': 1 - ok.mean()})
    return pd.DataFrame(rows).sort_values(['불량군', '|AUC-0.5|'], ascending=[True, False])


# ====================================================================== E. 셔플 대조 실험
def make_shuffled_features(df, seed=0, steps=None, fr_step=None):
    """
    대조 실험: 각 Run 안에서 샷 순서를 무작위로 섞은 뒤 시계열 피처를 다시 계산
    - 샷의 공정값·라벨은 그대로 (행 단위로 같이 이동), '순서'만 망가뜨림
    - 순서에서 나오는 맥락(순번, 재가동 후 샷 수, 정지 시간)도 섞인 순서 기준으로 다시 만듦
      → 진짜 시간 정보가 새어 들어가지 않게
    - 같은 Run 안에서만 섞으므로 Run 구성(LORO 분할)은 동일
    - seed를 바꿔 5~10회 반복해 평균·범위로 볼 것
    """
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby('run_id', sort=False):
        g = g.sample(frac=1, random_state=int(rng.integers(1e9))).copy()
        n = len(g)
        first_gap = g['prev_gap_sec'].max()                       # Run 시작 정지 시간은 유지 (Run 단위 정보)
        g['shot_seq_in_run'] = np.arange(1, n + 1)
        g['shots_since_stop'] = np.arange(1, n + 1)               # 중간 정지 정보는 제거
        g['prev_gap_sec'] = [first_gap] + [0.0] * (n - 1)
        # TimeStamp를 섞인 순서대로 다시 매겨서 make_causal_features의 정렬이 이 순서를 따르게 함
        g['TimeStamp'] = pd.Timestamp('2000-01-01') + pd.to_timedelta(np.arange(n), unit='min')
        parts.append(g)
    sh = pd.concat(parts, ignore_index=True)
    return make_causal_features(sh, steps=steps, fr_step=fr_step)