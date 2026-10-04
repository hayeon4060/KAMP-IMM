"""
loro_eval.py — 제품별 Leave-One-Run-Out 비교 (변수 조합 × 모델)

- run_loro      : LORO 루프 → OOF(out-of-fold) 예측
- evaluate      : AP(부트스트랩 CI), Run별 검사 예산 Recall@k, 유형별 성능
- by_run_table  : Run별 결과 (어디서 성공/실패했나)
- run_shuffle / summarize_shuffle : 시계열 셔플 대조 실험 (scope_fn으로 평가 범위 필터 가능)
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import average_precision_score

from features import FoldFeaturizer, combo_columns, needs_fold, make_shuffled_features

SEED = 42
K_LIST = [0.05, 0.10]                     # 검사 예산: Run마다 상위 5%, 10%
N_BOOT = 500                              # AP 부트스트랩 반복
TYPE_GROUPS = {'가스': ['가스'],
               '미성형': ['미성형', '가스+미성형'],
               '초기허용불량': ['초기허용불량']}   # 범위에서 빠졌으면 자동으로 건너뜀
SUPERVISED = {'logit', 'tree'}
UNSUPERVISED = {'iforest'}
KEYS = ['product', 'variant', 'label_mode', 'combo', 'model']


# ====================================================================== 모델
def make_model(name):
    imp = SimpleImputer(strategy='constant', fill_value=0)   # Run 첫 5샷 dev NaN → 0 (편차 없음)
    if name == 'logit':
        return make_pipeline(imp, StandardScaler(),
                             LogisticRegression(C=0.1, class_weight='balanced', max_iter=5000))
    if name == 'tree':
        return make_pipeline(imp, RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=5,
            class_weight='balanced_subsample', random_state=SEED, n_jobs=-1))
    if name == 'iforest':
        return make_pipeline(imp, StandardScaler(),
                             IsolationForest(n_estimators=300, random_state=SEED))
    raise ValueError(name)


def fit_score(model_name, Xtr, ytr, Xte):
    """위험 점수 (클수록 불량 의심)"""
    m = make_model(model_name)
    if model_name in UNSUPERVISED:                     # 양품만으로 정상 기준 학습
        m.fit(Xtr[ytr == 0])
        return -m.score_samples(Xte)
    if len(np.unique(ytr)) < 2:                        # 학습 fold에 불량이 없으면 예측 불가
        return np.zeros(len(Xte))
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def train_mask(d, label_mode):
    """label_mode='process': 초기허용불량을 학습에서 제외 (평가에는 포함)"""
    if label_mode == 'process':
        return d['Reason'] != '초기허용불량'
    return pd.Series(True, index=d.index)


def _X(df, cols):
    return df[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)


# ====================================================================== LORO 루프
def run_loro(d, product, combos, models, label_mode='all', variant='original'):
    """
    d: make_causal_features 결과 (한 제품). 반환: OOF 예측 (모든 샷이 '자기 Run을 안 본 모델'로 예측됨)
    """
    parts = []
    use_fold = any(needs_fold(c) for c in combos)
    for test_run in d['run_id'].unique():
        tr = d[d['run_id'] != test_run]
        te = d[d['run_id'] == test_run]
        tr = tr[train_mask(tr, label_mode)]
        if use_fold:                                   # 회귀·PCA·공분산은 학습 Run으로만 fit
            ff = FoldFeaturizer().fit(tr)
            tr, te = ff.transform(tr), ff.transform(te)

        base = {'product': product, 'variant': variant, 'label_mode': label_mode, 'run_id': test_run,
                'shot_idx': te.index.values, 'shot_seq': te['shot_seq_in_run'].values,
                'y': te['PassOrFail'].values, 'Reason': te['Reason'].values}

        if 'rule' in models:                           # 학습 없는 기준선: 사출 계열 최대 |편차|
            parts.append(pd.DataFrame({**base, 'combo': '-', 'model': 'rule',
                                       'score': te['ev_max_abs_dev'].fillna(0).values}))
        ytr = tr['PassOrFail'].values
        for c in combos:
            cols = combo_columns(c)
            Xtr, Xte = _X(tr, cols), _X(te, cols)
            for mname in models:
                if mname == 'rule':
                    continue
                s = fit_score(mname, Xtr, ytr, Xte)
                parts.append(pd.DataFrame({**base, 'combo': c, 'model': mname, 'score': s}))
    return pd.concat(parts, ignore_index=True)


# ====================================================================== 지표
def _jitter(s, rng):
    """동점 점수(규칙·상수 예측)가 원래 순서 덕에 유리해지지 않게 아주 작은 난수 추가"""
    return s + rng.random(len(s)) * 1e-9


def ap_ci(y, s, n_boot=N_BOOT, seed=SEED):
    """AP + 층화 부트스트랩 95% 구간 (불량·양품 각각 복원추출)"""
    y = np.asarray(y); s = np.asarray(s)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.nan, np.nan
    ap = average_precision_score(y, s)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(average_precision_score(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return ap, lo, hi


def recall_at_k_by_run(g, k, pos_mask):
    """Run마다 상위 k% 샷을 검사한다고 할 때 잡은 불량 수 / 전체 불량 수"""
    hit = tot = 0
    for _, r in g.groupby('run_id'):
        n_top = max(1, int(np.ceil(k * len(r))))
        top = r.nlargest(n_top, 'score').index
        hit += int(pos_mask.loc[top].sum())
        tot += int(pos_mask.loc[r.index].sum())
    return hit, tot


def evaluate(oof, n_boot=N_BOOT):
    """반환: (요약표, 유형별 표)"""
    rng = np.random.default_rng(SEED)
    rows, type_rows = [], []
    for key, g in oof.groupby(KEYS, sort=False):
        g = g.copy()
        g['score'] = _jitter(g['score'].values, rng)
        info = dict(zip(KEYS, key))

        # --- 평가 세트: 전체 / (CN7) 가스 제외 = 학습 가능한 이벤트형 범위
        subsets = {'all': g}
        if info['product'] == 'cn7' and (g['Reason'] == '가스').any():
            subsets['event(가스 제외)'] = g[g['Reason'] != '가스']
        for sname, sg in subsets.items():
            y, s = sg['y'].values, sg['score'].values
            ap, lo, hi = ap_ci(y, s, n_boot)
            row = {**info, 'eval_set': sname, 'n_shot': len(sg), 'n_def': int(y.sum()),
                   'base_rate': y.mean(), 'AP': ap, 'AP_lo': lo, 'AP_hi': hi,
                   'AP/base': ap / y.mean() if y.mean() > 0 else np.nan}
            for k in K_LIST:
                h, t = recall_at_k_by_run(sg, k, sg['y'] == 1)
                row[f'R@{int(k * 100)}%'] = h / t if t else np.nan
                row[f'hit@{int(k * 100)}%'] = f'{h}/{t}'
            rows.append(row)

        # --- 불량 유형별 (검사 예산은 전체 샷 기준, 다른 불량도 자리를 차지하는 현실 그대로)
        for tname, reasons in TYPE_GROUPS.items():
            pos = g['Reason'].isin(reasons)
            if pos.sum() == 0:
                continue
            ev = g[pos | (g['y'] == 0)]                # AP: 해당 유형 vs 양품
            ap = average_precision_score(pos.loc[ev.index].astype(int), ev['score'])
            row = {**info, '유형': tname, 'n_def': int(pos.sum()), 'AP': ap,
                   'AP/base': ap / pos.loc[ev.index].mean()}
            for k in K_LIST:
                h, t = recall_at_k_by_run(g, k, pos)
                row[f'hit@{int(k * 100)}%'] = f'{h}/{t}'
                row[f'R@{int(k * 100)}%'] = h / t
            type_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(type_rows)


def by_run_table(oof, k=0.10):
    """Run별: 불량 수, 상위 k% 안에 든 불량 수, 불량 샷의 평균 위험 순위(백분위, 작을수록 상위)"""
    rows = []
    for key, g in oof.groupby(KEYS + ['run_id'], sort=False):
        g = g.copy()
        g['pct_rank'] = g['score'].rank(ascending=False, pct=True, method='average')
        n_top = max(1, int(np.ceil(k * len(g))))
        top = g.nlargest(n_top, 'score')
        nd = int(g['y'].sum())
        rows.append({**dict(zip(KEYS + ['run_id'], key)), 'n_shot': len(g), 'n_def': nd,
                     f'hit@{int(k * 100)}%': int(top['y'].sum()),
                     '불량 평균 순위(%)': g.loc[g['y'] == 1, 'pct_rank'].mean() * 100 if nd else np.nan,
                     '불량 유형': ', '.join(sorted(g.loc[g['y'] == 1, 'Reason'].unique()))})
    return pd.DataFrame(rows)


# ====================================================================== 셔플 대조
def run_shuffle(shots, product, steps, fr_step, combos=('T2_static+temporal',),
                models=('logit', 'tree'), n_rep=5, scope_fn=None):
    """
    Run 안 순서를 섞고 시계열 피처를 재계산 → 같은 LORO
    scope_fn: 피처 계산 '후'에 적용할 범위 필터 (예: 초기허용 제거). 원본 실험과 같은 범위로 맞추기 위함
    """
    outs = []
    for r in range(n_rep):
        d = make_shuffled_features(shots, seed=r, steps=steps, fr_step=fr_step)
        if scope_fn is not None:
            d = scope_fn(d)
        outs.append(run_loro(d, product, list(combos), list(models), variant=f'shuffle{r}'))
    return pd.concat(outs, ignore_index=True)


def summarize_shuffle(summary, combo='T2_static+temporal'):
    """원본 vs 셔플(n회 평균·범위) 비교표"""
    s = summary[summary['combo'] == combo].copy()
    s['group'] = np.where(s['variant'] == 'original', 'original', 'shuffled')
    agg = (s.groupby(['product', 'eval_set', 'model', 'group'])[['AP', 'R@5%', 'R@10%']]
             .agg(['mean', 'min', 'max']).round(3))
    return agg