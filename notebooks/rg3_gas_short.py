"""
rg3_gas_short.py — 39번: RG3 가스 모델 + 미성형 모델을 나눠 학습하고 검사 예산을 나눠 쓰기

가스 모델 (37번 반복 19에서 고정): 원값 21개 + 새 변수 3개(전환 위치·평균 스크류 회전·최대 배압)의 직전 10샷 std → RF (가중치)
미성형 모델 후보 (37번 반복 22 상위 3개 + 기준):
  under3_smote   : 정상을 불량의 3배로 무작위 언더샘플링 → SMOTE로 1:1
  under10_adasyn : 정상을 불량의 10배로 언더샘플링 → ADASYN으로 1:1
  balanced_rf    : BalancedRandomForest (트리마다 정상·불량을 같은 수로 뽑음)
  weight         : 샘플링 없이 클래스 가중치만 (기준)

정직한 평가를 위해 두 가지를 같이 계산
  ① 고정 평가 : 방식을 하나 정해 두고 Run 하나씩 빼고 검증 (LORO)
  ② 중첩 평가 : 바깥 LORO의 학습 Run 안에서만 다시 LORO를 돌려 '미성형 AP가 가장 좋은 방식'을 고르고,
                그 방식으로 학습해 빼 둔 Run을 평가 → "19가지 중 최고를 결과 보고 고른" 낙관 편향을 없앤 값

검사 예산 (평가 대상 샷의 10%) 나누기
  가스 몫 g, 미성형 몫 1−g. g는 결과를 보기 전에 정함: 0.68(= 불량 수 비율 17:8), 0.5(반반)
  가스 모델 상위 g×예산을 먼저 뽑고, 남은 예산을 미성형 모델 순위로 채움

데이터: outputs/final_model/rg3_scores.csv (20b 결과) + base/rg3_shots.csv + linked/linked_labeled.csv
       R6 제외 (final_model.EXCLUDE_RUNS) → 가스 17 / 미성형 8
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score

from final_model import (raw, ALL_RAW, INITIAL, TYPE_GROUPS, load_labeled_shots, apply_exclusions,
                         load_official_rows, eval_frame1)

P = 'rg3'
MD = './outputs/final_model'
OUT = './outputs/rg3_gas_short'
NEW = ['Switch_Over_Position_raw', 'Average_Screw_RPM_raw', 'Max_Back_Pressure_raw']
BUDGET = 0.10
GAS_SHARES = (1.0, 0.68, 0.5)
SHORT_METHODS = ('under3_smote', 'under10_adasyn', 'balanced_rf', 'weight')


# ====================================================================== 데이터
def load(base_dir='.'):
    f = f'{MD}/{P}_scores.csv'
    if not os.path.exists(f):
        raise FileNotFoundError(f'{f} 없음 → 20b_final_test 먼저 실행')
    lab = pd.read_csv(f, parse_dates=['TimeStamp'])
    lab['Reason'] = lab['Reason'].fillna('정상')
    lab = apply_exclusions(lab, P)
    rc = [raw(v) for v in ALL_RAW]
    shots = load_labeled_shots(P, base_dir)
    lab = lab.merge(shots[['TimeStamp'] + [c for c in rc if c in shots.columns]], on='TimeStamp', how='left')
    ll = pd.read_csv(f'{base_dir}/linked/linked_labeled.csv', parse_dates=['TimeStamp'],
                     usecols=['product', 'TimeStamp'] + NEW)
    ll = ll[ll['product'] == P].drop_duplicates('TimeStamp')[['TimeStamp'] + NEW]
    lab = lab.merge(ll, on='TimeStamp', how='left')
    lab = lab.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
    lab['seq'] = lab.groupby('run_id', sort=False).cumcount() + 1
    lab['eval'] = (lab['split'] != '제외') & (lab['Reason'] != INITIAL)
    lab['type'] = np.select([lab['Reason'].isin(TYPE_GROUPS['가스']), lab['Reason'].isin(TYPE_GROUPS['미성형'])],
                            ['가스', '미성형'], np.where(lab['PassOrFail'] == 1, '기타', '정상'))
    cols = [c for c in rc + NEW if c in lab.columns]
    X = pd.DataFrame({c.replace('_raw', '') + '|std10':
                      lab[c].groupby(lab['run_id'], sort=False).transform(lambda s: s.rolling(10, min_periods=3).std())
                      for c in cols})
    X = SimpleImputer(strategy='median').fit_transform(X.replace([np.inf, -np.inf], np.nan))
    return lab, X, cols


# ====================================================================== 모델
def _k(y):
    return int(max(1, min(5, y.sum() - 1)))


def fit_predict(method, Xtr, ytr, Xte, seed=0):
    """학습 데이터에만 샘플링 적용 → 평가 샷 점수"""
    if method == 'balanced_rf':
        from imblearn.ensemble import BalancedRandomForestClassifier
        m = BalancedRandomForestClassifier(n_estimators=500, max_depth=4, min_samples_leaf=3, sampling_strategy='all',
                                           replacement=True, bootstrap=False, n_jobs=-1, random_state=seed)
        return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    weight = None
    if method in ('under3_smote', 'under10_adasyn'):
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
        r = 3 if method == 'under3_smote' else 10
        n0 = int(min((ytr == 0).sum(), r * ytr.sum()))
        Xtr, ytr = RandomUnderSampler(sampling_strategy={0: n0}, random_state=seed).fit_resample(Xtr, ytr)
        try:
            over = SMOTE(k_neighbors=_k(ytr), random_state=seed) if method == 'under3_smote' \
                else ADASYN(n_neighbors=_k(ytr), random_state=seed)
            Xtr, ytr = over.fit_resample(Xtr, ytr)
        except (ValueError, RuntimeError):            # ADASYN이 생성할 이웃을 못 찾으면 무작위 복제로 대체
            Xtr, ytr = RandomOverSampler(random_state=seed).fit_resample(Xtr, ytr)
    else:                                            # 'weight' 또는 가스 모델
        weight = 'balanced_subsample'
    n_trees = 1000 if method == 'gas' else 500
    leaf = 5 if method == 'gas' else 3
    m = RandomForestClassifier(n_estimators=n_trees, max_depth=4, min_samples_leaf=leaf, class_weight=weight,
                               n_jobs=-1, random_state=seed)
    return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def loro(lab, X, target, method, rows=None, seed=0, min_pos=2):
    """Run 하나씩 빼고 검증. rows가 주어지면 그 샷들 안에서만 (중첩 평가의 안쪽 루프용)"""
    rows = np.ones(len(lab), bool) if rows is None else rows
    t, ev, runs = lab['type'].values, lab['eval'].values, lab['run_id'].values
    y = (t == target).astype(int)
    use = ev & ((y == 1) | (t == '정상')) & rows
    s = np.full(len(lab), np.nan)
    for r in pd.unique(runs[rows]):
        te = (runs == r) & rows
        tr = (~te) & use
        if y[tr].sum() < min_pos:
            continue
        s[te] = fit_predict(method, X[tr], y[tr], X[te], seed)
    return s


def nested_short(lab, X, methods=SHORT_METHODS, seed=0, verbose=True):
    """바깥 LORO마다: 학습 Run 안에서 안쪽 LORO로 방식별 미성형 AP → 최고 방식으로 학습 → 빼 둔 Run 점수"""
    t, ev, runs = lab['type'].values, lab['eval'].values, lab['run_id'].values
    y = (t == '미성형').astype(int)
    s = np.full(len(lab), np.nan)
    log = []
    for r in pd.unique(runs):
        te = runs == r
        inner_rows = ~te
        scores = {}
        for mth in methods:
            si = loro(lab, X, '미성형', mth, rows=inner_rows, seed=seed)
            mm = inner_rows & ev & ((y == 1) | (t == '정상')) & np.isfinite(si)
            scores[mth] = average_precision_score(y[mm], si[mm]) if 0 < y[mm].sum() < mm.sum() else np.nan
        best = max(scores, key=lambda k: -np.inf if np.isnan(scores[k]) else scores[k])
        tr = inner_rows & ev & ((y == 1) | (t == '정상'))
        if y[tr].sum() >= 2:
            s[te] = fit_predict(best, X[tr], y[tr], X[te], seed)
        log.append({'뺀 Run': r, '고른 방식': best, **{f'안쪽 AP {k}': v for k, v in scores.items()}})
        if verbose:
            print(f'  중첩: {r} 제외 → {best}', flush=True)
    return s, pd.DataFrame(log)


# ====================================================================== 평가
def _fill(s):
    s = np.asarray(s, float).copy()
    ok = np.isfinite(s)
    s[~ok] = (s[ok].min() - 1) if ok.any() else 0
    return s


def split_select(lab, gas_s, short_s, gas_share):
    """검사 예산 10% 중 gas_share만큼 가스 모델 상위, 나머지를 미성형 모델 순위로 → 선택 여부(평가 샷 기준)"""
    m = lab['eval'].values
    idx = np.flatnonzero(m)
    K = int(np.ceil(BUDGET * len(idx)))
    jit = np.random.default_rng(0).random(len(idx)) * 1e-9
    g = _fill(gas_s)[idx] + jit
    sh = _fill(short_s)[idx] + jit
    sel = np.zeros(len(idx), bool)
    sel[np.argsort(-g)[:int(round(gas_share * K))]] = True
    for i in np.argsort(-sh):
        if sel.sum() >= K:
            break
        sel[i] = True
    return idx, sel


def summarize(lab, name, gas_s, short_s, gas_share):
    idx, sel = split_select(lab, gas_s, short_s, gas_share)
    t = lab['type'].values[idx]
    y = (lab['PassOrFail'].values[idx] == 1).astype(int)
    tp = int((sel & (y == 1)).sum())
    P = tp / max(1, int(sel.sum()))
    Rc = tp / max(1, int(y.sum()))
    r = {'구성': name, '가스 몫': gas_share,
         '불량': f'{tp}/{int(y.sum())}',
         '가스': f'{int((sel & (t == "가스")).sum())}/{int((t == "가스").sum())}',
         '미성형': f'{int((sel & (t == "미성형")).sum())}/{int((t == "미성형").sum())}',
         '검사 샷 중 불량 비율(정밀도)': P, 'Recall': Rc, 'F1': 2 * P * Rc / (P + Rc) if P + Rc else 0.0}
    for nm, s in (('가스 모델', gas_s), ('미성형 모델', short_s)):
        ss = _fill(s)[idx]
        for ty in ('가스', '미성형'):
            mm = (t == ty) | (y == 0)
            r[f'{nm} AP_{ty}'] = average_precision_score((t[mm] == ty).astype(int), ss[mm])
    return r


def run(seeds=(0, 1, 2), base_dir='.', verbose=True):
    os.makedirs(OUT, exist_ok=True)
    lab, X, cols = load(base_dir)
    t = lab.loc[lab['eval'], 'type'].value_counts().to_dict()
    if verbose:
        print(f'RG3 평가 샷 {int(lab["eval"].sum())} | {t} | 변수 {len(cols)}개 std10')
    rows, logs, S = [], [], {}
    for sd in seeds:
        gas = loro(lab, X, '가스', 'gas', seed=sd)
        S[(sd, 'gas')] = gas
        shorts = {m: loro(lab, X, '미성형', m, seed=sd) for m in SHORT_METHODS}
        nest, log = nested_short(lab, X, seed=sd, verbose=verbose)
        log.insert(0, 'seed', sd)
        logs.append(log)
        shorts['중첩 선택 (정직한 추정)'] = nest
        for m, s in shorts.items():
            S[(sd, m)] = s
            for gsh in GAS_SHARES:
                r = summarize(lab, f'가스RF + 미성형[{m}]', gas, s, gsh)
                r['seed'] = sd
                rows.append(r)
        if verbose:
            print(f'seed {sd} 완료', flush=True)
    R = pd.DataFrame(rows)
    num = lambda c: R[c].str.split('/').str[0].astype(int)
    R['가스 수'], R['미성형 수'] = num('가스'), num('미성형')
    agg = R.groupby(['구성', '가스 몫'], sort=False).agg(
        가스_평균=('가스 수', 'mean'), 가스_최소=('가스 수', 'min'), 가스_최대=('가스 수', 'max'),
        미성형_평균=('미성형 수', 'mean'), 미성형_최소=('미성형 수', 'min'), 미성형_최대=('미성형 수', 'max'),
        Precision=('검사 샷 중 불량 비율(정밀도)', 'mean'), Recall=('Recall', 'mean'), F1=('F1', 'mean'),
        미성형모델_AP=('미성형 모델 AP_미성형', 'mean'),
        가스모델_AP=('가스 모델 AP_가스', 'mean')).reset_index()
    agg['목표(가스≥7 & 미성형≥4)'] = np.where((agg['가스_평균'] >= 7) & (agg['미성형_평균'] >= 4), 'O', '-')
    out = {'by_seed': R, 'summary': agg, 'nested_log': pd.concat(logs, ignore_index=True), 'scores': S, 'lab': lab}
    R.to_csv(f'{OUT}/by_seed.csv', index=False, encoding='utf-8-sig')
    agg.to_csv(f'{OUT}/summary.csv', index=False, encoding='utf-8-sig')
    out['nested_log'].to_csv(f'{OUT}/nested_log.csv', index=False, encoding='utf-8-sig')
    return out


# ====================================================================== (선택) 순열검정
def permutation_test(lab, X, method='중첩', gas_share=0.68, n=100, seed=11, verbose=True):
    """
    라벨을 Run 안에서 원형 이동(불량 몰림 구조 유지, 위치만 무작위) → 같은 학습·선택 절차를 처음부터 반복
    → 관측 (가스+미성형) 포착 수가 우연 분포의 몇 %인지. 시간이 오래 걸림 (n=100 기준 수십 분)
    """
    def stat(L):
        g = loro(L, X, '가스', 'gas')
        s = nested_short(L, X, verbose=False)[0] if method == '중첩' else loro(L, X, '미성형', method)
        idx, sel = split_select(L, g, s, gas_share)
        tt = L['type'].values[idx]
        return int((sel & (tt == '가스')).sum()), int((sel & (tt == '미성형')).sum())

    obs = stat(lab)
    rng = np.random.default_rng(seed)
    ev = np.flatnonzero(lab['eval'].values)
    runs = lab['run_id'].values
    null = []
    for b in range(n):
        L = lab.copy()
        tcol = L['type'].values.copy()
        for r in pd.unique(runs[ev]):
            ii = ev[runs[ev] == r]
            tcol[ii] = np.roll(lab['type'].values[ii], rng.integers(0, len(ii)))
        L['type'] = tcol
        null.append(stat(L))
        if verbose and b % 10 == 0:
            print('순열', b, flush=True)
    null = np.asarray(null)
    tot_obs = sum(obs)
    tot = null.sum(1)
    return {'관측 (가스, 미성형)': obs, '무작위 평균 (가스, 미성형)': tuple(np.round(null.mean(0), 2)),
            '무작위 95% 합계': float(np.quantile(tot, 0.95)), 'p (합계)': float((1 + (tot >= tot_obs).sum()) / (n + 1))}
