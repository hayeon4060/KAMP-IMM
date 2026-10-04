"""
unsup_eval.py — 순수 비지도: unlabeled로 '정상'을 학습하고 정상/불량을 가른 뒤, labeled로 테스트

규칙
  - 학습: unlabeled reference(9~10월, labeled 샷·Run 시간창 제거)만. 라벨 0개 사용
  - 임계값: reference 점수 분포의 q 분위수 = "unlabeled 중 (1−q)를 불량으로 보겠다"는 가정 (라벨 미사용)
  - 테스트: labeled 전체 샷 (fold 불필요 — 모델이 labeled를 본 적 없음)

변수 세트 (라벨 사용 여부 표시)
  raw   : 원본 핵심 13개 + 흐름저항 (수준값)              — 라벨 무관
  free  : Run 내 과거 대비 편차(dev_*) + 상태 변화         — 라벨 무관
  phys  : 물리 지표 (재료부족·쿠션·계량 계통)              — ⚠ 12번 결과(라벨)를 보고 고른 세트 → 순수 비지도 아님, 참고용

탐지기: iforest / maha / knn (model_v2) + lof / pca(재구성 오차)
기준선 (팀원 방식 재현, 학습 없음): 핵심 원본 변수의 robust z 중 최대값
  B_prev10    : 같은 Run 직전 10샷 기준 (인과적)
  B_expanding : 제품 전체 과거 샷 누적 기준 (인과적이지만 Run을 넘어 누적 → Run 수준 차이를 그대로 잡음)
  B_global    : labeled 전체 기준 (미래 포함, 테스트셋을 봄 → 참고용)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from features import RAW_CORE, raw
from model_v2 import RefDetector, to_ref_scale, FSETS, LEVEL, evaluate_v2, SEED
from model_phys import PHYS_FSETS
from loro_eval import recall_at_k_by_run
from plot_style import set_style, INK, SUB, MUTED, ACCENT, TYPE_COL

USETS = {'raw': LEVEL, 'free': FSETS['free'], 'phys(참고)': PHYS_FSETS['phys']}
KINDS = ('iforest', 'maha', 'knn', 'lof', 'pca')
QS = (0.95, 0.98, 0.99, 0.995, 0.999)
CORE = [raw(v) for v in RAW_CORE]


# ====================================================================== 탐지기
class UDetector(RefDetector):
    """RefDetector + lof / pca (같은 전처리: reference 중앙값·IQR 스케일, ±20 클리핑)"""
    def fit(self, ref):
        if self.kind in ('iforest', 'maha', 'knn'):
            return super().fit(ref)
        X0 = ref[self.cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
        self.med = np.nanmedian(X0, axis=0)
        q75, q25 = np.nanpercentile(X0, [75, 25], axis=0)
        sd = np.nanstd(X0, axis=0)
        self.iqr = np.where(q75 - q25 > 1e-9, q75 - q25, np.where(sd > 1e-9, sd, 1.0))
        X = self._prep(ref)
        if self.kind == 'lof':
            n = min(len(X), 20000)
            idx = np.random.default_rng(SEED).choice(len(X), n, replace=False)
            self.m = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X[idx])
            self.s_ref = -self.m.score_samples(X)
        elif self.kind == 'pca':
            self.sc = X.std(axis=0); self.sc[self.sc < 1e-9] = 1
            self.m = PCA(n_components=0.9, random_state=SEED).fit(X / self.sc)
            self.s_ref = self._recon(X)
        else:
            raise ValueError(self.kind)
        return self

    def _recon(self, X):
        Xs = X / self.sc
        return np.sqrt(((Xs - self.m.inverse_transform(self.m.transform(Xs))) ** 2).sum(axis=1))

    def raw_score(self, df):
        if self.kind in ('iforest', 'maha', 'knn'):
            return super().raw_score(df)
        X = self._prep(df)
        return -self.m.score_samples(X) if self.kind == 'lof' else self._recon(X)


# ====================================================================== 기준선 (팀원 방식)
def _robust_max_z(X, med, scale):
    z = np.abs((X - med) / scale)
    return np.nanmax(z, axis=1)


def _scale_floor(d, cols):
    from features import resolution
    return np.array([resolution(d[c]) for c in cols])


def baseline_prev(d, w=10):
    """같은 Run 직전 w샷의 중앙값·MAD 대비 최대 |z| (Run 첫 w샷은 가능한 만큼)"""
    floor = _scale_floor(d, CORE)
    out = np.full(len(d), np.nan)
    for _, idx in d.sort_values(['run_id', 'TimeStamp']).groupby('run_id').groups.items():
        X = d.loc[idx, CORE].values.astype(float)
        for i in range(1, len(X)):
            H = X[max(0, i - w):i]
            med = np.median(H, axis=0)
            mad = np.maximum(1.4826 * np.median(np.abs(H - med), axis=0), floor)
            out[d.index.get_loc(idx[i])] = np.nanmax(np.abs((X[i] - med) / mad))
    return out


def baseline_expanding(d, min_hist=10, update_every=50):
    """제품 전체 시간순 과거 샷 누적 기준 (Run 경계 무시). 속도 때문에 기준값은 update_every샷마다 갱신"""
    floor = _scale_floor(d, CORE)
    o = d.sort_values('TimeStamp')
    X = o[CORE].values.astype(float)
    out = np.full(len(X), np.nan)
    med = mad = None
    for i in range(min_hist, len(X)):
        if med is None or i < 200 or i % update_every == 0:
            H = X[:i]
            med = np.median(H, axis=0)
            mad = np.maximum(1.4826 * np.median(np.abs(H - med), axis=0), floor)
        out[i] = np.nanmax(np.abs((X[i] - med) / mad))
    return pd.Series(out, index=o.index).reindex(d.index).values


def baseline_global(d):
    floor = _scale_floor(d, CORE)
    X = d[CORE].values.astype(float)
    med = np.median(X, axis=0)
    mad = np.maximum(1.4826 * np.median(np.abs(X - med), axis=0), floor)
    return _robust_max_z(X, med, mad)


# ====================================================================== 실행
def _frame(d, product, method, score, pct):
    return pd.DataFrame({'product': product, 'method': method, 'run_id': d['run_id'].values,
                         'split': d['split'].values, 'shot_seq': d['shot_seq_in_run'].values,
                         'is_transient': d['is_transient'].values, 'y': d['PassOrFail'].values,
                         'Reason': d['Reason'].values, 'score': score, 'pct': pct})


def run_unsup(FEAT, REF, usets=USETS, kinds=KINDS, baselines=True):
    parts = []
    for p in FEAT:
        d, ref = FEAT[p], REF[p]
        for fname, cols in usets.items():
            for k in kinds:
                det = UDetector(k, cols).fit(ref)
                sc, pc = to_ref_scale(det.raw_score(d), det.s_ref)
                parts.append(_frame(d, p, f'{k}|{fname}', sc, pc))
                print(f'  {p} {k}|{fname}')
        if baselines:
            for name, fn in [('B_prev10', lambda x: baseline_prev(x, 10)),
                             ('B_expanding', baseline_expanding), ('B_global(미래포함)', baseline_global)]:
                s = np.nan_to_num(fn(d.reset_index(drop=True)), nan=0.0)
                s_ref = np.nan_to_num(fn(ref.reset_index(drop=True)), nan=0.0)
                sc, pc = to_ref_scale(s, s_ref)
                parts.append(_frame(d, p, name, sc, pc))
                print(f'  {p} {name}')
    return pd.concat(parts, ignore_index=True)


# ====================================================================== 평가
def global_recall(oof, ks=(0.05, 0.10)):
    """파일 전체에서 상위 k% (팀원 방식과 같은 기준) — Run별 R@k와 나란히 보기 위함"""
    rows = []
    for (p, m), g in oof.groupby(['product', 'method'], sort=False):
        g = g.assign(run_id='ALL').reset_index(drop=True)
        r = {'product': p, 'method': m}
        for k in ks:
            h, t = recall_at_k_by_run(g, k, g['y'] == 1)
            r[f'전체 상위{int(k * 100)}%'] = f'{h}/{t}'
        rows.append(r)
    return pd.DataFrame(rows)


def threshold_table(oof, qs=QS):
    """
    unlabeled 분위수 임계값으로 labeled 판정 → 혼동행렬 지표
    q = 0.99 → 'unlabeled의 1%를 불량으로 보는' 임계값. 라벨은 평가에만 사용
    """
    rows = []
    for (p, m), g in oof.groupby(['product', 'method'], sort=False):
        y = g['y'].values
        for q in qs:
            pred = (g['pct'].values >= q).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
            tp = int(((pred == 1) & (y == 1)).sum())
            rows.append({'product': p, 'method': m, 'q': q, '가정 불량률(unlabeled)': f'{100 * (1 - q):g}%',
                         '판정 불량(labeled)': int(pred.sum()), 'labeled 판정 비율': pred.mean(),
                         'TP': tp, 'FP': int(pred.sum()) - tp, 'FN': int(y.sum()) - tp,
                         'Precision': pr, 'Recall': rc, 'F1': f1,
                         '가스 탐지': int(((pred == 1) & (g['Reason'].values == '가스')).sum()),
                         '미성형계열 탐지': int(((pred == 1) & g['Reason'].isin(['미성형', '가스+미성형']).values).sum())})
    return pd.DataFrame(rows)


def best_per_method(tt):
    """방법별 F1 최고 q (※ 라벨로 고른 것이라 '사후 최적', 비교 참고용)"""
    return tt.loc[tt.groupby(['product', 'method'])['F1'].idxmax()].sort_values(['product', 'F1'], ascending=[True, False])


def plot_rank_compare(summ, product, top=15, save=None):
    """Run별 hit@10%와 AP를 방법별로 (회색 = 기준선 B_*)"""
    set_style()
    s = summ[summ['product'] == product].sort_values('AP', ascending=False).head(top).iloc[::-1]
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(s) + 1.6))
    col = [MUTED if m.startswith('B_') else (TYPE_COL['미성형'] if 'phys' in m else ACCENT) for m in s['method']]
    ax.barh(y, s['AP'], color=col, height=0.6)
    ax.hlines(y, s['AP_lo'], s['AP_hi'], color=INK, lw=1)
    for yi, (ap, h) in enumerate(zip(s['AP'], s['hit@10%'])):
        ax.text(s['AP_hi'].iloc[yi], yi, f'  {ap:.3f} · Run별 hit10 {h}', va='center', fontsize=8, color=SUB)
    ax.set_yticks(y); ax.set_yticklabels(s['method'], fontsize=8)
    ax.set_xlabel('AP (막대 끝 선 = 95% CI)')
    ax.set_title(f'{product.upper()} — 비지도 방법 비교 (파랑 = 라벨 무관, 보라 = phys(참고), 회색 = 기준선)',
                 loc='left', fontsize=10)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()
