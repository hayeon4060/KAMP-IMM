"""
pr_split_check.py — 같은 데이터·같은 모델이라도 '분할 방식 · 평가 단위 · PR-AUC 계산법'에 따라 점수가 얼마나 달라지나

데이터: 공식 2차 labeled 행 (초기허용불량 제외) + 1차 원값 21개 변수 (TimeStamp로 연결)
모델  : RF, 로지스틱 회귀 (지도학습, 흔히 쓰는 기본 설정) + 최종 모델(비지도, 학습 없음 → 점수 고정)

바꿔 보는 것
  단위  : row  = LH/RH 행 그대로 (2차 파일 그대로)
          shot = 같은 샷 두 행을 한 샷으로 (하나라도 불량이면 불량)
  분할  : random     = 층화 무작위 5-fold (행 단위면 LH/RH 쌍둥이가 학습·평가로 갈라짐 → 누수)
          group_shot = 같은 샷은 같은 fold (row 단위에서만 의미, shot 단위의 random과 같음)
          time_block = 시간순 연속 5구간
          loro       = Run 하나씩 빼기
  PR-AUC: AP          = 확률 점수로 계단식 (sklearn average_precision_score, 표준)
          trapz_prob  = 확률 점수로 사다리꼴 (auc(recall, precision))
          trapz_bin   = 0/1 판정값(확률 ≥ 0.5, 최종 모델은 pct ≥ 0.999)으로 사다리꼴 → 점이 2~3개라 과대 가능
  집계  : fold 평균 ± 표준편차 (보고서에 흔한 '0.69 ± 0.06' 형식) / 전체 OOF 한 번에 계산(pooled)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, precision_recall_curve, auc

from final_model import raw, ALL_RAW, INITIAL, load_labeled_shots, load_official_rows

FEATS = [raw(v) for v in ALL_RAW]
K = 5
SEEDS = (0, 1, 2, 3, 4)          # 무작위 분할은 시드 5개 반복
FINAL = '최종모델(비지도)'


# ====================================================================== 데이터
def load_frames(p, base_dir='.', model_dir='./outputs/final_model'):
    """반환: (row 단위 df, shot 단위 df, 기록)  — 공통 열: TimeStamp, run_id, y, Reason, FEATS, score, pct"""
    shots = load_labeled_shots(p, base_dir)
    rows = load_official_rows(p, base_dir)
    n0, d0 = len(rows), int(rows['PassOrFail'].sum())
    rows = rows[rows['Reason'] != INITIAL]
    rows = rows.merge(shots[['TimeStamp', 'run_id'] + FEATS], on='TimeStamp', how='left')
    sc = pd.read_csv(f'{model_dir}/{p}_scores.csv', parse_dates=['TimeStamp'], usecols=['TimeStamp', 'score', 'pct'])
    rows = rows.merge(sc, on='TimeStamp', how='left')
    miss = int(rows[FEATS + ['score']].isna().any(axis=1).sum())
    rows = rows.dropna(subset=FEATS + ['score']).sort_values(['TimeStamp', 'cavity']).reset_index(drop=True)
    rows['y'] = rows['PassOrFail'].astype(int)
    agg = {c: 'first' for c in ['run_id', 'Reason', 'score', 'pct'] + FEATS}
    agg['y'] = 'max'
    shot = rows.groupby('TimeStamp', as_index=False).agg(agg).sort_values('TimeStamp').reset_index(drop=True)
    log = {'제품': p, '2차 행': n0, '2차 불량 행': d0, '초기허용 제외 후 행': len(rows) + miss,
           '원값 연결 실패(제외)': miss, '행': len(rows), '불량 행': int(rows['y'].sum()),
           '샷': len(shot), '불량 샷': int(shot['y'].sum())}
    return rows, shot, log


# ====================================================================== 분할
def make_splits(df, kind, seed=0, k=K):
    y = df['y'].values
    if kind == 'random':
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(df, y))
    if kind == 'group_shot':
        g = pd.factorize(df['TimeStamp'])[0]
        return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(df, y, g))
    if kind == 'time_block':
        ts = np.sort(df['TimeStamp'].unique())
        blocks = np.array_split(ts, k)
        out = []
        for b in blocks:
            te = np.flatnonzero(df['TimeStamp'].isin(b).values)
            out.append((np.setdiff1d(np.arange(len(df)), te), te))
        return out
    if kind == 'loro':
        out = []
        for r in df['run_id'].unique():
            te = np.flatnonzero((df['run_id'] == r).values)
            out.append((np.setdiff1d(np.arange(len(df)), te), te))
        return out
    raise ValueError(kind)


def make_model(name, seed=0):
    if name == 'RF':
        return RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', n_jobs=-1,
                                      random_state=seed)
    if name == 'LR':
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=3000))
    raise ValueError(name)


# ====================================================================== 지표
def _trapz(y, s):
    pr, rc, _ = precision_recall_curve(y, s)
    return auc(rc, pr)


def fold_metrics(y, prob, binary):
    if y.sum() == 0:
        return None
    return {'AP': average_precision_score(y, prob), 'trapz_prob': _trapz(y, prob),
            'trapz_bin': _trapz(y, binary.astype(int))}


def twin_leak(df, tr, te):
    """테스트 불량 행 중 같은 샷의 다른 행(쌍둥이)이 학습에 있는 비율"""
    ts_tr = set(df['TimeStamp'].values[tr])
    d = df.iloc[te]
    d = d[d['y'] == 1]
    return np.nan if len(d) == 0 else d['TimeStamp'].isin(ts_tr).mean()


# ====================================================================== 실행
PLAN = [('row', 'random'), ('row', 'group_shot'), ('row', 'time_block'), ('row', 'loro'),
        ('shot', 'random'), ('shot', 'time_block'), ('shot', 'loro')]


def run(products=('cn7', 'rg3'), models=('RF', 'LR', FINAL), plan=PLAN, base_dir='.',
        model_dir='./outputs/final_model', verbose=True):
    folds_out, pooled_out, logs = [], [], []
    for p in products:
        rows, shot, log = load_frames(p, base_dir, model_dir)
        logs.append(log)
        data = {'row': rows, 'shot': shot}
        for unit, kind in plan:
            df = data[unit]
            X, y = df[FEATS].values, df['y'].values
            seeds = SEEDS if kind in ('random', 'group_shot') else (0,)
            for m in models:
                for sd in seeds:
                    oof = np.full(len(df), np.nan)
                    for f, (tr, te) in enumerate(make_splits(df, kind, sd)):
                        if m == FINAL:
                            prob = df['score'].values[te]
                            binary = df['pct'].values[te] >= 0.999
                        else:
                            if y[tr].sum() == 0:
                                continue
                            mdl = make_model(m, sd).fit(X[tr], y[tr])
                            prob = mdl.predict_proba(X[te])[:, 1]
                            binary = prob >= 0.5
                        oof[te] = prob
                        fm = fold_metrics(y[te], prob, binary)
                        if fm:
                            folds_out.append({'제품': p, '단위': unit, '분할': kind, '모델': m, 'seed': sd, 'fold': f,
                                              '테스트 불량': int(y[te].sum()),
                                              '쌍둥이 누수': twin_leak(df, tr, te) if unit == 'row' else 0.0, **fm})
                    ok = ~np.isnan(oof)
                    if y[ok].sum():
                        pooled_out.append({'제품': p, '단위': unit, '분할': kind, '모델': m, 'seed': sd,
                                           'AP_pooled': average_precision_score(y[ok], oof[ok])})
                if verbose:
                    print(f'  {p} {unit:4s} {kind:10s} {m}')
    F = pd.DataFrame(folds_out)
    P = pd.DataFrame(pooled_out)
    return F, P, pd.DataFrame(logs)


def summarize(F, P):
    """fold 평균 ± 표준편차 (보고서 형식) + pooled AP + 쌍둥이 누수율"""
    keys = ['제품', '단위', '분할', '모델']
    g = F.groupby(keys)
    s = pd.DataFrame({
        '평가 fold 수': g.size(),
        'AP (fold 평균±sd)': g['AP'].agg(lambda x: f'{x.mean():.3f} ± {x.std():.3f}'),
        'PR-AUC 사다리꼴·확률': g['trapz_prob'].agg(lambda x: f'{x.mean():.3f} ± {x.std():.3f}'),
        'PR-AUC 사다리꼴·0/1': g['trapz_bin'].agg(lambda x: f'{x.mean():.3f} ± {x.std():.3f}'),
        '쌍둥이 누수율': g['쌍둥이 누수'].mean(),
        '_AP평균': g['AP'].mean(),
    })
    pp = P.groupby(keys)['AP_pooled'].mean().rename('AP (전체 OOF)')
    s = s.join(pp).reset_index()
    order = {k: i for i, k in enumerate([f'{u}|{k}' for u, k in PLAN])}
    s['_o'] = (s['단위'] + '|' + s['분할']).map(order)
    return s.sort_values(['제품', '_o', '모델']).drop(columns=['_o'])
