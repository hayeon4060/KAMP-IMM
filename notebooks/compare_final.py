"""
compare_final.py — 최종 모델 주변 비교 (사후 점검용 ablation)

A. 모델 고정(Mahalanobis) → 변수 조합 바꾸기
B. 변수 고정(free 17개) → 비지도 모델 바꾸기

공통 조건 (final_model.py와 동일)
  - 학습: 같은 reference (unlabeled 정상 샷, 0 모드 제거) — 라벨 미사용
  - 테스트: 같은 labeled, 같은 평가 틀 ① 공식 2차·행 / ② 전체·샷·초기허용 제외
  - 전처리: reference 중앙값·IQR 스케일 → ±20 클리핑 → NaN 0
  - 하이퍼파라미터: 결과 보기 전에 고정 (아래 MODEL_PARAMS), 튜닝 없음
  - 기준 = free · maha (최종 모델). 각 방법과 기준의 AP 차이를 paired bootstrap 95% CI로 판정

⚠ 해석 주의: 같은 테스트셋에서 여러 조합·모델을 비교하고 1등을 고르면 '선택 과적합'.
   → 기준보다 CI상 뚜렷이 좋을 때만 교체를 검토하고, 아니면 '사전 고정한 최종 모델이 견고하다'는 근거로 사용
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import average_precision_score

import final_model as fm
from final_model import (raw, EVENT_VARS, STATE_VARS, ALL_RAW, FEATURES, SEED, CLIP, INITIAL, REF_MAX,
                         load_labeled_shots, load_official_rows, build_reference, load_scales, make_features,
                         eval_frame1, eval_frame2, threshold_table)

# random_band 캐시 (조합마다 1,000번 반복 방지 — 같은 y면 결과 동일)
_RB = {}
_rb_orig = fm.random_band
def _rb_cached(y, n=1000):
    k = (len(y), int(np.sum(y)), hash(np.asarray(y).tobytes()))
    if k not in _RB:
        _RB[k] = _rb_orig(y, n)
    return _RB[k]
fm.random_band = _rb_cached

# ====================================================================== 변수 그룹
G_EVENT = [f'dev_{v}' for v in EVENT_VARS]                                       # 8
G_SUMMARY = ['fill_anomaly', 'ev_max_abs_dev']                                   # 2
G_THERMAL = ['dev_Mold_Temperature_3', 'dev_Hopper_Temperature', 'mold_rise', 'hopper_rise',
             'mold_slope20', 'hopper_slope20']                                   # 6
G_FLOW = ['dev_flow_res']                                                        # 1
G_DELTA = [f'delta_{v}' for v in EVENT_VARS]                                     # 직전 샷 대비 변화 8
RAW_CORE = ['Max_Injection_Speed', 'Filling_Time', 'Max_Injection_Pressure', 'Max_Switch_Over_Pressure',
            'Average_Back_Pressure', 'Cushion_Position', 'Plasticizing_Time', 'Plasticizing_Position',
            'Cycle_Time', 'Clamp_Close_Time', 'Mold_Temperature_3', 'Hopper_Temperature', 'Barrel_Temperature_4']
G_LEVEL = [raw(v) for v in RAW_CORE] + ['flow_res']                              # 원값 13 + 흐름저항 = 14
G_RAW_ALL = [raw(v) for v in ALL_RAW]                                            # 원값 21

assert G_EVENT + G_SUMMARY + G_THERMAL + G_FLOW == FEATURES

COMBOS = {
    # 기준
    'free (17, 최종)':          FEATURES,
    # 원값 (시계열 X)
    'raw_all (원값 21)':         G_RAW_ALL,
    'level (원값 13+흐름저항)':  G_LEVEL,
    # 그룹 단독
    'event (8)':                 G_EVENT,
    'event+summary (10)':        G_EVENT + G_SUMMARY,
    'thermal (6)':               G_THERMAL,
    # 그룹 하나씩 빼기 (free − X)
    'free − event':              G_SUMMARY + G_THERMAL + G_FLOW,
    'free − summary':            G_EVENT + G_THERMAL + G_FLOW,
    'free − thermal':            G_EVENT + G_SUMMARY + G_FLOW,
    'free − flow':               G_EVENT + G_SUMMARY + G_THERMAL,
    # 더하기
    'free + delta (25)':         FEATURES + G_DELTA,
    'free + level (31)':         FEATURES + G_LEVEL,
}
BASE_COMBO, BASE_MODEL = 'free (17, 최종)', 'maha'

# ====================================================================== 모델 (하이퍼파라미터 사전 고정)
MODEL_PARAMS = {
    'maha':     'Ledoit-Wolf 공분산 + Mahalanobis 거리 (최종 모델)',
    'maha_emp': '표본 공분산(축소 없음) + Mahalanobis',
    'knn':      'k=10 최근접 이웃 평균 거리',
    'lof':      'LOF, 이웃 20',
    'iforest':  'Isolation Forest, 트리 500, max_samples 1024',
    'pca':      'PCA 재구성 오차, 분산 90%',
    'ocsvm':    'One-Class SVM, RBF, nu=0.01, gamma=scale',
    'gmm':      'Gaussian Mixture 3성분 full, −로그우도',
    'autoenc':  '오토인코더(MLP 12-6-12) 재구성 오차',
    'maxdev':   '(학습 없음) 변수별 |스케일값| 최대 — 단순 기준선',
}


class Detector:
    """final_model.FinalModel과 같은 전처리 + 여러 비지도 모델"""

    def __init__(self, kind, cols):
        self.kind, self.cols = kind, list(cols)

    def _X(self, df):
        return df[self.cols].replace([np.inf, -np.inf], np.nan).values.astype(float)

    def _prep(self, df):
        X = (self._X(df) - self.med) / self.iqr
        return np.nan_to_num(np.clip(X, -CLIP, CLIP), nan=0.0)

    def fit(self, ref):
        X0 = self._X(ref)
        self.med = np.nanmedian(X0, axis=0)
        q75, q25 = np.nanpercentile(X0, [75, 25], axis=0)
        sd = np.nanstd(X0, axis=0)
        self.iqr = np.where(q75 - q25 > 1e-9, q75 - q25, np.where(sd > 1e-9, sd, 1.0))
        X = self._prep(ref)
        k = self.kind
        if k == 'maha':
            self.m = LedoitWolf().fit(X)
        elif k == 'maha_emp':
            self.m = EmpiricalCovariance().fit(X)
        elif k == 'knn':
            self.m = NearestNeighbors(n_neighbors=11).fit(X)
        elif k == 'lof':
            self.m = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X)
        elif k == 'iforest':
            self.m = IsolationForest(n_estimators=500, max_samples=min(1024, len(X)), random_state=SEED,
                                     n_jobs=-1).fit(X)
        elif k == 'pca':
            self.sc = X.std(axis=0); self.sc[self.sc < 1e-9] = 1
            self.m = PCA(n_components=0.9, random_state=SEED).fit(X / self.sc)
        elif k == 'ocsvm':
            self.m = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale').fit(X)
        elif k == 'gmm':
            self.m = GaussianMixture(n_components=3, covariance_type='full', reg_covar=1e-4,
                                     random_state=SEED).fit(X)
        elif k == 'autoenc':
            self.sc = X.std(axis=0); self.sc[self.sc < 1e-9] = 1
            n_h = max(2, min(12, X.shape[1]))
            self.m = MLPRegressor(hidden_layer_sizes=(n_h, max(2, n_h // 2), n_h), max_iter=500,
                                  early_stopping=True, random_state=SEED).fit(X / self.sc, X / self.sc)
        elif k != 'maxdev':
            raise ValueError(k)
        # reference 점수 (knn·lof는 자기 자신 제외)
        if k == 'knn':
            d, _ = self.m.kneighbors(X)
            self.s_ref = d[:, 1:].mean(axis=1)
        elif k == 'lof':
            self.s_ref = -self.m.negative_outlier_factor_
        else:
            self.s_ref = self._score_X(X)
        self.s_ref_sorted = np.sort(self.s_ref)
        return self

    def _score_X(self, X):
        k = self.kind
        if k in ('maha', 'maha_emp'):
            return np.sqrt(self.m.mahalanobis(X))
        if k == 'knn':
            d, _ = self.m.kneighbors(X, n_neighbors=10)
            return d.mean(axis=1)
        if k in ('lof', 'iforest', 'gmm'):
            return -self.m.score_samples(X)
        if k == 'pca':
            Xs = X / self.sc
            return np.sqrt(((Xs - self.m.inverse_transform(self.m.transform(Xs))) ** 2).sum(axis=1))
        if k == 'ocsvm':
            return -self.m.decision_function(X)
        if k == 'autoenc':
            Xs = X / self.sc
            return np.sqrt(((Xs - self.m.predict(Xs)) ** 2).sum(axis=1))
        if k == 'maxdev':
            return np.abs(X).max(axis=1)

    def score(self, df):
        return self._score_X(self._prep(df))

    def pct(self, s):
        return np.searchsorted(self.s_ref_sorted, np.asarray(s, float), side='right') / (len(self.s_ref_sorted) + 1)


# ====================================================================== 데이터 준비 (제품당 한 번)
def add_delta(d, steps):
    """직전 샷 대비 변화량 (기록 단위 칸 수) — 같은 Run 안에서만"""
    for v in EVENT_VARS:
        d[f'delta_{v}'] = d.groupby('run_id', sort=False)[raw(v)].diff() / steps.get(v, 1.0)
    return d


def prepare(p, base_dir='.'):
    shots = load_labeled_shots(p, base_dir)
    ref_shots, log = build_reference(p, shots, base_dir)
    steps, fr_step, _ = load_scales(p, ref_shots, base_dir)
    ref = add_delta(make_features(ref_shots, steps, fr_step), steps)
    if len(ref) > REF_MAX:
        ref = ref.sample(REF_MAX, random_state=SEED).reset_index(drop=True)
    lab = add_delta(make_features(shots, steps, fr_step), steps)
    rows = load_official_rows(p, base_dir)
    m2 = (lab['split'] != '제외') & (lab['Reason'] != INITIAL)
    print(f'[{p}] reference {len(ref):,}샷 / labeled {len(lab):,}샷 준비 완료')
    return {'ref': ref, 'lab': lab, 'rows': rows, 'm2': m2.values}


# ====================================================================== paired bootstrap (기준 대비 ΔAP)
def _paired_delta(y, sa, sb, cluster=None, n_boot=1000, seed=SEED):
    y, sa, sb = map(np.asarray, (y, sa, sb))
    rng = np.random.default_rng(seed)
    cl = pd.Series(np.arange(len(y))).groupby(np.arange(len(y)) if cluster is None else np.asarray(cluster)).apply(list)
    has = np.array([y[ix].any() for ix in cl])
    pos_c, neg_c = cl[has].tolist(), cl[~has].tolist()
    obs = average_precision_score(y, sa) - average_precision_score(y, sb)
    d = []
    for _ in range(n_boot):
        ix = np.concatenate([pos_c[i] for i in rng.integers(0, len(pos_c), len(pos_c))] +
                            [neg_c[i] for i in rng.integers(0, len(neg_c), len(neg_c))])
        d.append(average_precision_score(y[ix], sa[ix]) - average_precision_score(y[ix], sb[ix]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return obs, lo, hi, ('기준보다 좋음' if lo > 0 else ('기준보다 나쁨' if hi < 0 else '차이 없음'))


# ====================================================================== 실행
def _evaluate(p, D, name, det, base_scores=None):
    lab, rows, m2 = D['lab'], D['rows'], D['m2']
    s = det.score(lab)
    pc = det.pct(s)
    shot = pd.Series(s, index=lab['TimeStamp'].values)
    r1 = eval_frame1(p, rows, shot)
    g = pd.DataFrame({'run_id': lab['run_id'].values[m2], 'y': lab['PassOrFail'].values[m2].astype(int),
                      'Reason': lab['Reason'].values[m2], 'score': s[m2], 'pct': pc[m2]})
    r2 = eval_frame2(p, g)
    th = threshold_table(p, g, qs=(0.99, 0.999)).set_index('q')
    out = {'제품': p, '방법': name, '변수 수': len(det.cols),
           '①AP': r1['AP'], '①유의': r1['유의'], '①상위5%': r1['상위5%'], '①상위10%': r1['상위10%'],
           '①F1@5%': r1['F1@5%'],
           '②AP': r2['AP'], '②AP_lo': r2['AP_lo'], '②AP_hi': r2['AP_hi'], '②AP/불량률': r2['AP/불량률'],
           '②Run별 상위10%': r2['Run별 상위10%'], '②전체 상위10%': r2['전체 상위10%'],
           '②AP_가스': r2.get('AP_가스', np.nan), '②AP_미성형': r2.get('AP_미성형', np.nan),
           'q0.99 경보비율': th.loc[0.99, '경보 비율'],
           'q0.999 경보': th.loc[0.999, '경보 수'], 'q0.999 TP': th.loc[0.999, 'TP'],
           'q0.999 정밀도': th.loc[0.999, 'Precision']}
    if base_scores is not None:
        bs = base_scores
        y1 = rows['PassOrFail'].values.astype(int)
        a1 = rows['TimeStamp'].map(shot).fillna(0).values
        b1 = rows['TimeStamp'].map(bs['shot']).fillna(0).values
        o, lo, hi, v = _paired_delta(y1, a1, b1, cluster=rows['TimeStamp'].values)
        out.update({'①ΔAP(vs 기준)': o, '①Δ CI': f'[{lo:.3f}, {hi:.3f}]', '①판정': v})
        o, lo, hi, v = _paired_delta(g['y'].values, s[m2], bs['s'][m2])
        out.update({'②ΔAP(vs 기준)': o, '②Δ CI': f'[{lo:.3f}, {hi:.3f}]', '②판정': v})
    return out, {'s': s, 'shot': shot}


def run_compare(DATA, combos=COMBOS, models=tuple(MODEL_PARAMS), verbose=True):
    """
    DATA: {p: prepare(p)}
    반환: (A 변수 조합 표, B 모델 표)
    """
    A, B = [], []
    for p, D in DATA.items():
        base_det = Detector(BASE_MODEL, combos[BASE_COMBO]).fit(D['ref'])
        base_row, base_sc = _evaluate(p, D, BASE_COMBO, base_det)
        # A. 모델 고정(maha) × 변수 조합
        for name, cols in combos.items():
            if name == BASE_COMBO:
                A.append({**base_row, '①판정': '(기준)', '②판정': '(기준)'})
                continue
            row, _ = _evaluate(p, D, name, Detector(BASE_MODEL, cols).fit(D['ref']), base_sc)
            A.append(row)
            if verbose:
                print(f'  A {p} {name}: ②AP {row["②AP"]:.3f}')
        # B. 변수 고정(free) × 모델
        for k in models:
            if k == BASE_MODEL:
                B.append({**base_row, '방법': k, '①판정': '(기준)', '②판정': '(기준)'})
                continue
            row, _ = _evaluate(p, D, k, Detector(k, combos[BASE_COMBO]).fit(D['ref']), base_sc)
            B.append(row)
            if verbose:
                print(f'  B {p} {k}: ②AP {row["②AP"]:.3f}')
    A = pd.DataFrame(A).rename(columns={'방법': '변수 조합'})
    B = pd.DataFrame(B).rename(columns={'방법': '모델'})
    B.insert(2, '설명', B['모델'].map(MODEL_PARAMS))
    return A, B


# ====================================================================== 그림
def plot_ap(tbl, key, product, title, save=None):
    """두 틀 AP 막대 (② CI 포함). 기준 = 진한 색"""
    t = tbl[tbl['제품'] == product].sort_values('②AP')
    y = np.arange(len(t))
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.38 * len(t) + 1.6), sharey=True)
    base = t['①판정'].eq('(기준)').values
    col = np.where(base, '#1f4e79', '#9db9d5')
    ax[0].barh(y, t['①AP'], color=col)
    ax[0].set_title('① 공식 2차 · 행  AP', loc='left', fontsize=10)
    ax[1].barh(y, t['②AP'], color=col)
    ax[1].hlines(y, t['②AP_lo'], t['②AP_hi'], color='k', lw=1)
    ax[1].set_title('② 전체 · 샷  AP (선 = 95% CI)', loc='left', fontsize=10)
    ax[0].set_yticks(y); ax[0].set_yticklabels(t[key], fontsize=9)
    for a, c in zip(ax, ['①AP', '②AP']):
        for yi, v in enumerate(t[c]):
            a.text(v, yi, f' {v:.3f}', va='center', fontsize=8)
    fig.suptitle(f'{product.upper()} — {title}', x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
