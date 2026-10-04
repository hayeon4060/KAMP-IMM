"""
model_v2.py — 공정불량(초기허용불량 제외: 가스·미성형) 예측 v2

v1(06/07) 진단
  - CN7 불량 15개 중 14개가 R1 한 Run에 몰려 있음 → LORO에서 R1을 빼면 학습 불량 1개
    → 지도학습(logit/tree) AP ≈ 기본 불량률. 구조적으로 학습이 안 되는 상태
  - 지도학습 점수를 fold끼리 그대로 이어붙여 pooled AP를 계산 → fold마다 점수 척도가 달라 AP가 왜곡됨
  - 이상탐지(iforest)도 '다른 labeled Run 양품' 1,000샷 남짓으로 정상 기준을 학습

v2 변경
  1) 정상 기준 = unlabeled 생산 샷 (수만 샷). labeled 샷과 labeled Run 시간창(±60분)은 제거 → 누수 없음
     → 이상탐지 모델은 labeled를 전혀 안 보므로 fold 없이 한 번만 학습
  2) 지도학습은 CN7+RG3를 합쳐서 LORO (수준 무관 변수만 사용) → 학습 불량 15/32 → 최대 46
  3) 모든 점수를 '정상 생산 샷(reference) 대비' 척도로 변환
       score : reference 중앙값·IQR 기준 robust z (연속값, fold·제품 간 비교 가능 → pooled AP·앙상블 가능)
       pct   : reference ECDF 백분위 (0.995 = 정상 샷의 0.5%만 이보다 높음 = 오경보율 해석 가능)
  4) 평가: pooled AP(+CI), Run별 Recall@k, 유형별 hit, F1(임계값은 테스트 Run 안 보고 결정)
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, average_precision_score

from features import make_causal_features, EVENT_VARS, ALL_RAW, RAW_CORE, raw
from loro_eval import ap_ci, recall_at_k_by_run

SEED = 42
RUN_GAP_MIN, STOP_GAP_MIN, MIN_RUN_SHOTS = 60, 5, 10      # 00_base와 같은 기준
MIN_CYCLE_SEC, CYCLE_DEV_SEC = 50, 2
REF_MAX = 30000                                           # reference 샷 상한 (속도용 무작위 추출)
PCT_GRID = [0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999, 0.9995]
TYPE_GROUPS = {'가스': ['가스'], '미성형': ['미성형', '가스+미성형']}


# ====================================================================== 1. reference (unlabeled 정상 생산 샷)
def build_reference(unl_path, labeled_shots, product, period=None, pad_min=RUN_GAP_MIN):
    """
    unl_path      : linked/linked_unlabeled_{p}.csv (TimeStamp, *_raw, LH/RH 2행)
    labeled_shots : base/{p}_shots.csv (전체 split) — 이 샷들과 그 Run 시간창은 제거
    period        : ('2020-09-01', '2020-11-30')처럼 기간 제한 (None = 전체)
    반환          : make_causal_features 입력 형식 (run_id, TimeStamp, prev_gap_sec, shot_seq_in_run,
                    shots_since_stop, *_raw)
    """
    u = pd.read_csv(unl_path, parse_dates=['TimeStamp'], low_memory=False)
    if 'EQUIP_NAME' in u.columns:
        u = u[u['EQUIP_NAME'].astype(str).str.contains('우진2', na=False)]
    u = u.drop_duplicates('TimeStamp').sort_values('TimeStamp').reset_index(drop=True)   # LH/RH → 샷 1행
    n0 = len(u)

    # (1) labeled 샷 + labeled Run 시간창(±pad) 제거 → 평가 샷과 같은 시간대 정보가 정상 기준에 안 들어감
    lab_ts = pd.to_datetime(labeled_shots['TimeStamp'])
    u = u[~u['TimeStamp'].isin(lab_ts)]
    pad = pd.Timedelta(minutes=pad_min)
    win = labeled_shots.assign(TimeStamp=lab_ts).groupby('run_id')['TimeStamp'].agg(['min', 'max'])
    m = np.zeros(len(u), bool)
    for a, b in win.itertuples(index=False):
        m |= u['TimeStamp'].between(a - pad, b + pad).values
    u = u[~m]
    if period is not None:
        u = u[u['TimeStamp'].between(pd.Timestamp(period[0]), pd.Timestamp(period[1]))]
    u = u.reset_index(drop=True)

    # (2) 끼어든 이상 기록 제거 (00_base 6장과 같은 규칙)
    gp = u['TimeStamp'].diff().dt.total_seconds()
    gn = -u['TimeStamp'].diff(-1).dt.total_seconds()
    day_med = u.groupby(u['TimeStamp'].dt.date)[raw('Cycle_Time')].transform('median')
    odd = (np.fmin(gp, gn) < MIN_CYCLE_SEC) & ((u[raw('Cycle_Time')] - day_med).abs() > CYCLE_DEV_SEC)
    u = u[~odd].reset_index(drop=True)

    # (3) Run 구분
    gap = u['TimeStamp'].diff().dt.total_seconds()
    new = gap.isna() | (gap > RUN_GAP_MIN * 60)
    u['run_id'] = f'{product}_U' + new.cumsum().astype(int).astype(str).str.zfill(4)
    u['prev_gap_sec'] = gap.where(~new)
    u['shot_seq_in_run'] = u.groupby('run_id').cumcount() + 1
    after = u['prev_gap_sec'] > STOP_GAP_MIN * 60
    seg = (new | after).groupby(u['run_id']).cumsum()
    u['shots_since_stop'] = u.groupby([u['run_id'], seg]).cumcount() + 1
    u = u[u.groupby('run_id')['run_id'].transform('size') >= MIN_RUN_SHOTS]

    keep = ['run_id', 'TimeStamp', 'prev_gap_sec', 'shot_seq_in_run', 'shots_since_stop'] + \
           [raw(v) for v in ALL_RAW if raw(v) in u.columns]
    u = u[keep].reset_index(drop=True)
    print(f'[{product}] reference: unlabeled {n0:,}샷 → {len(u):,}샷, Run {u["run_id"].nunique()}개 '
          f'({u["TimeStamp"].min():%Y-%m-%d} ~ {u["TimeStamp"].max():%Y-%m-%d}), 이상기록 제거 {int(odd.sum())}')
    return u


def reference_features(ref_shots, steps, fr_step, max_n=REF_MAX):
    """reference에 labeled와 똑같은 인과적 파생변수 계산 (Run 단위라 계산 후 샘플링)"""
    f = make_causal_features(ref_shots, steps=steps, fr_step=fr_step)
    if len(f) > max_n:
        f = f.sample(max_n, random_state=SEED)
    return f.reset_index(drop=True)


# ====================================================================== 2. 변수 세트
EV_DEV = [f'dev_{v}' for v in EVENT_VARS] + ['fill_anomaly', 'ev_max_abs_dev']
EV_DELTA = [f'delta_{v}' for v in EVENT_VARS]
ST_FREE = ['dev_Mold_Temperature_3', 'dev_Hopper_Temperature', 'mold_rise', 'hopper_rise',
           'mold_slope20', 'hopper_slope20', 'dev_flow_res']
CONTEXT = ['log_since_stop', 'is_transient', 'stop_len_log']
LEVEL = [raw(v) for v in RAW_CORE] + ['flow_res']

FSETS = {
    # 수준 무관 (Run 내 과거 대비 편차) → 기간·제품이 달라도 같은 의미
    'free':        EV_DEV + ST_FREE,
    'free+delta':  EV_DEV + EV_DELTA + ST_FREE,
    # 절대 수준 (v1 CN7 최고였던 T0+state 계열) → 기간 드리프트에 민감
    'level':       LEVEL,
    'free+level':  EV_DEV + ST_FREE + LEVEL,
}
SUP_FSETS = {                                     # 통합 지도학습용 (수준 무관 + 맥락)
    'sup_free':       EV_DEV + ST_FREE + CONTEXT,
    'sup_free+delta': EV_DEV + EV_DELTA + ST_FREE + CONTEXT,
}


def _X(df, cols):
    return df[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)


# ====================================================================== 3. 점수 척도 변환 (reference 기준)
def to_ref_scale(s, s_ref):
    """반환 (score, pct): score = reference 기준 robust z, pct = reference ECDF"""
    s, s_ref = np.asarray(s, float), np.asarray(s_ref, float)
    med = np.median(s_ref)
    iqr = np.subtract(*np.percentile(s_ref, [75, 25]))
    iqr = iqr if iqr > 1e-12 else (np.std(s_ref) + 1e-12)
    srt = np.sort(s_ref)
    pct = np.searchsorted(srt, s, side='right') / (len(srt) + 1)
    return (s - med) / iqr, pct


def _squash(z):
    """앙상블 전에 꼬리 압축 (maha처럼 끝없이 커지는 점수가 평균을 지배하지 않게)"""
    return np.sign(z) * np.log1p(np.abs(z))


# ====================================================================== 4. 이상탐지 (reference로만 학습)
class RefDetector:
    """
    kind: 'iforest' | 'maha' | 'knn'
    전처리: reference 중앙값·IQR로 스케일 → ±20 클리핑 → NaN 0 (Run 첫 5샷 dev NaN = 편차 없음)
    """
    def __init__(self, kind, cols, k=10):
        self.kind, self.cols, self.k = kind, cols, k

    def _prep(self, df):
        X = _X(df, self.cols)
        X = (X - self.med) / self.iqr
        return np.nan_to_num(np.clip(X, -20, 20), nan=0.0)

    def fit(self, ref):
        X0 = _X(ref, self.cols)
        self.med = np.nanmedian(X0, axis=0)
        q75, q25 = np.nanpercentile(X0, [75, 25], axis=0)
        iqr = q75 - q25
        sd = np.nanstd(X0, axis=0)
        self.iqr = np.where(iqr > 1e-9, iqr, np.where(sd > 1e-9, sd, 1.0))
        X = self._prep(ref)
        if self.kind == 'iforest':
            self.m = IsolationForest(n_estimators=500, max_samples=min(1024, len(X)),
                                     random_state=SEED, n_jobs=-1).fit(X)
            self.s_ref = -self.m.score_samples(X)
        elif self.kind == 'maha':
            self.m = LedoitWolf().fit(X)
            self.s_ref = np.sqrt(self.m.mahalanobis(X))
        elif self.kind == 'knn':
            self.m = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
            d, _ = self.m.kneighbors(X)
            self.s_ref = d[:, 1:].mean(axis=1)            # 자기 자신 제외
        else:
            raise ValueError(self.kind)
        return self

    def raw_score(self, df):
        X = self._prep(df)
        if self.kind == 'iforest':
            return -self.m.score_samples(X)
        if self.kind == 'maha':
            return np.sqrt(self.m.mahalanobis(X))
        d, _ = self.m.kneighbors(X, n_neighbors=self.k)
        return d.mean(axis=1)


def _oof_frame(d, product, method, score, pct):
    return pd.DataFrame({'product': product, 'method': method, 'run_id': d['run_id'].values,
                         'shot_seq': d['shot_seq_in_run'].values, 'is_transient': d['is_transient'].values,
                         'y': d['PassOrFail'].values, 'Reason': d['Reason'].values,
                         'score': score, 'pct': pct})


def run_unsupervised(d, ref, product, fsets=FSETS, kinds=('iforest', 'maha', 'knn')):
    """
    labeled를 전혀 안 보는 모델 → fold 불필요 (모든 labeled 샷이 이미 out-of-sample)
    반환: (OOF, reference에서의 점수 dict — 앙상블용)
    """
    parts, ref_scores = [], {}
    for fname, cols in fsets.items():
        for kind in kinds:
            name = f'U_{kind}|{fname}'
            det = RefDetector(kind, cols).fit(ref)
            sc, pc = to_ref_scale(det.raw_score(d), det.s_ref)
            ref_scores[name] = to_ref_scale(det.s_ref, det.s_ref)[0]
            parts.append(_oof_frame(d, product, name, sc, pc))
            print(f'  {product} {name} 완료')
    return pd.concat(parts, ignore_index=True), ref_scores


# ====================================================================== 5. 통합 지도학습 (CN7+RG3, LORO)
def make_sup(name):
    imp = SimpleImputer(strategy='constant', fill_value=0)
    if name == 'logit':
        return make_pipeline(imp, StandardScaler(),
                             LogisticRegression(C=0.05, class_weight='balanced', max_iter=5000))
    if name == 'rf':
        return make_pipeline(imp, RandomForestClassifier(
            n_estimators=500, max_depth=4, min_samples_leaf=5, max_features='sqrt',
            class_weight='balanced_subsample', random_state=SEED, n_jobs=-1))
    raise ValueError(name)


def run_pooled_supervised(D, refs, fsets=SUP_FSETS, models=('logit', 'rf'), balance_products=True):
    """
    D    : {'cn7': feat_df, 'rg3': feat_df} (초기허용 제거 후)
    refs : {'cn7': ref_feat, 'rg3': ref_feat}  — fold 모델 점수를 테스트 제품 reference 기준으로 변환
    LORO : 테스트 Run 1개 제외, 나머지 두 제품 전체 Run으로 학습
    balance_products: 제품별 샘플 가중치 합을 같게 (CN7 샷이 3배 많아서 RG3가 묻히지 않게)
    반환: (OOF, sup_ref) — sup_ref[(method, test_run)] = 그 fold 모델의 reference z 점수 (앙상블용)
    """
    A = pd.concat([D[p].assign(product=p) for p in D], ignore_index=True)
    parts, sup_ref = [], {}
    for test_run in A['run_id'].unique():
        tr, te = A[A['run_id'] != test_run], A[A['run_id'] == test_run]
        p = te['product'].iloc[0]
        ytr = tr['PassOrFail'].values
        w = None
        if balance_products:
            cnt = tr['product'].map(tr['product'].value_counts())
            w = (len(tr) / tr['product'].nunique() / cnt).values
        for fname, cols in fsets.items():
            for mname in models:
                m = make_sup(mname)
                m.fit(_X(tr, cols), ytr, **{m.steps[-1][0] + '__sample_weight': w})
                s_te = m.predict_proba(_X(te, cols))[:, 1]
                s_ref = m.predict_proba(_X(refs[p], cols))[:, 1]
                sc, pc = to_ref_scale(s_te, s_ref)
                name = f'S_{mname}|{fname}'
                parts.append(_oof_frame(te, p, name, sc, pc))
                sup_ref[(name, test_run)] = to_ref_scale(s_ref, s_ref)[0]
        print(f'  pooled LORO {test_run} 완료 (학습 불량 {int(ytr.sum())})')
    return pd.concat(parts, ignore_index=True), sup_ref


# ====================================================================== 6. 앙상블
def ensemble(oof, members, name, ref_scores, sup_ref=None):
    """
    members    : 합칠 method 이름 리스트 (U_*, S_* 섞어도 됨)
    ref_scores : {product: {U_method: ref z 배열}} (run_unsupervised 반환값)
    sup_ref    : run_pooled_supervised 반환값 (S_* 멤버가 있을 때)
    점수 = 멤버 score를 꼬리 압축 후 평균. pct는 같은 연산을 reference 점수에 적용해 ECDF로 계산
    ※ 같은 제품의 reference 샷 순서가 모든 멤버에서 같아야 함 (같은 ref_feat를 썼으면 OK)
    """
    out = []
    for (p, r), g in oof[oof['method'].isin(members)].groupby(['product', 'run_id']):
        piv = g.pivot_table(index='shot_seq', columns='method', values='score')
        if piv.shape[1] < len(members):
            continue
        s = _squash(piv[members].values).mean(axis=1)
        refs = []
        for mth in members:
            if mth.startswith('U_'):
                refs.append(ref_scores[p][mth])
            else:
                refs.append(sup_ref[(mth, r)])
        s_ref = _squash(np.vstack(refs).T).mean(axis=1)
        _, pc = to_ref_scale(s, s_ref)
        base = g[g['method'] == members[0]].set_index('shot_seq').loc[piv.index]
        f = pd.DataFrame({'product': p, 'method': name, 'run_id': r, 'shot_seq': piv.index.values,
                          'is_transient': base['is_transient'].values, 'y': base['y'].values,
                          'Reason': base['Reason'].values, 'score': s, 'pct': pc})
        out.append(f)
    return pd.concat(out, ignore_index=True)


def mask_transient(oof, suffix='|과도제외'):
    """재가동 후 5샷은 규칙(초기허용 처리)으로 넘긴다고 보고 점수를 최하로 → 별도 method로 추가"""
    o = oof.copy()
    o['method'] = o['method'] + suffix
    o.loc[o['is_transient'] == 1, ['score', 'pct']] = [-1e9, 0.0]
    return o


# ====================================================================== 7. 평가
def evaluate_v2(oof, n_boot=500):
    """제품 × method: AP(+CI), Run별 R@5%/R@10%, 유형별 hit@10%"""
    rng = np.random.default_rng(SEED)
    rows = []
    for (p, mth), g in oof.groupby(['product', 'method'], sort=False):
        g = g.reset_index(drop=True).copy()
        g['score'] = g['score'].values + rng.random(len(g)) * 1e-9
        y = g['y'].values
        ap, lo, hi = ap_ci(y, g['score'].values, n_boot)
        row = {'product': p, 'method': mth, 'n_def': int(y.sum()), 'base': y.mean(),
               'AP': ap, 'AP_lo': lo, 'AP_hi': hi, 'AP/base': ap / y.mean()}
        for k in (0.05, 0.10):
            h, t = recall_at_k_by_run(g, k, g['y'] == 1)
            row[f'R@{int(k*100)}%'] = h / t if t else np.nan
            row[f'hit@{int(k*100)}%'] = f'{h}/{t}'
        for tname, rs in TYPE_GROUPS.items():
            pos = g['Reason'].isin(rs)
            if pos.sum() == 0:
                continue
            ev = pos | (g['y'] == 0)
            row[f'AP_{tname}'] = average_precision_score(pos[ev].astype(int), g.loc[ev, 'score'])
            h, t = recall_at_k_by_run(g, 0.10, pos)
            row[f'hit10_{tname}'] = f'{h}/{t}'
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['product', 'AP'], ascending=[True, False])


def _prf(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'F1': f, 'Precision': p, 'Recall': r, 'TP': tp, 'FP': fp, 'FN': fn}


def f1_v2(oof, methods=None):
    """
    pct 척도 임계값 (점수가 reference 기준이라 Run·fold 간 같은 의미)
      fixed_q  : pct ≥ q (q = 0.99/0.995/0.999) → 라벨 없이 '정상 샷 오경보율 1/0.5/0.1%'로 결정
      loro_thr : 테스트 Run을 뺀 같은 제품 Run들에서 F1 최대인 q를 골라 테스트 Run에 적용
    """
    rows, preds = [], []
    for (p, mth), g in oof.groupby(['product', 'method'], sort=False):
        if methods and mth not in methods:
            continue
        g = g.copy()
        for q in (0.99, 0.995, 0.999):
            rows.append({'product': p, 'method': mth, '임계값': f'pct≥{q}',
                         **_prf(g['y'].values, (g['pct'] >= q).astype(int).values)})
        pred = np.zeros(len(g), int)
        chosen = {}
        for r in g['run_id'].unique():
            oth, te = g[g['run_id'] != r], (g['run_id'] == r).values
            if oth['y'].sum() == 0:
                q_best = 0.999
            else:
                fs = [_prf(oth['y'].values, (oth['pct'] >= q).astype(int).values)['F1'] for q in PCT_GRID]
                best = max(fs)
                q_best = max(q for q, f in zip(PCT_GRID, fs) if f == best)   # 동점이면 보수적(높은 q)
            chosen[r] = q_best
            pred[te] = (g.loc[te, 'pct'] >= q_best).astype(int)
        g['pred'] = pred
        rows.append({'product': p, 'method': mth, '임계값': 'LORO 선택',
                     **_prf(g['y'].values, pred), '선택 q': chosen})
        preds.append(g)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
