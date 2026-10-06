"""
semi_sup.py — 40번: unlabeled(정상 생산 수천 샷 + 일별 불량 수)를 활용한 준지도 학습

왜 다시 하나
  - labeled 불량은 CN7 15샷 / RG3 25샷뿐. unlabeled는 수천 샷 + '날짜·캐비티별 불량 수(ERR_FACT_QTY)'가 있음
  - 예전 14·16번(가이드북 방식)은 행 단위 분할·쌍둥이 행 누수가 있어 지금 평가 틀로는 못 씀 → 새로 설계

방법 (모두 사전 고정, 결과 보고 고르지 않음)
  M0 supervised      : 기준. labeled만 RF (가중치). 아래 방법이 'unlabeled 덕분에' 좋아졌는지 비교하는 대조군
  M1 self_training   : 자기 학습. labeled로 학습 → unlabeled 점수 → 확실한 정상(하위 50%)과 확실한 불량(상위 0.5%,
                       labeled 불량 수의 2배 이하, 가중치 0.5)을 의사 라벨로 추가 → 3회 반복
  M2 label_spreading : 그래프 기반. labeled + unlabeled(+평가 샷, 라벨 없이) kNN 그래프에서 라벨을 퍼뜨림 (PCA 10차원)
  M3 pretrain_dae    : 자기 지도 사전학습(VIME·SCARF 계열의 간단판). unlabeled로 잡음 제거 오토인코더 학습
                       → 은닉 표현 8차원 + 복원오차를 변수로 추가 → RF
  M4 weak_daily      : 약한 지도(다중 인스턴스 / 비율 학습). **labeled 라벨을 전혀 안 씀.**
                       unlabeled의 날짜별 불량 수만으로 '샷 불량 확률' 모델을 학습:
                       날짜 d의 기대 불량 수 = 2(캐비티) × 그날 샷 수 × 평균 p_i,  p_i = σ(b + w·x_i)  → 포아송 우도 최대화
                       → labeled 평가가 완전히 독립적인 시험 (CN7 가스처럼 Run 단위 검증이 불가능한 경우에도 정직하게 평가 가능)

신뢰성 장치
  - 평가: labeled를 Run 하나씩 빼고 검증(LORO). unlabeled는 labeled 시간대 ±60분을 이미 뺀 정상 기준 데이터(20a 학습 데이터)
  - 같은 시드·같은 fold에서 M0 대비 차이(Δ)를 보고, 시드 3개로 흔들림 확인
  - M4는 일별 수준에서 날짜 하나씩 빼고 예측한 상관(leave-one-day-out)을 따로 보고 → 날짜 수준에서 신호가 없으면 샷 결과도 믿지 않음
  - (선택) permutation_test: 라벨 원형 이동 후 같은 절차 반복 → 우연 대비 p
  - CN7: 가스 11샷이 모두 R1 → M0~M3는 R1을 뺄 때 가스 학습 불가(구조적). M4만 정직한 평가 가능

변수: 원값의 직전 10샷 std (+ RG3는 새 변수 3개 std) + 최종 모델 파생변수 17개.  스케일은 unlabeled 기준 robust z
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelSpreading
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize
from scipy.stats import spearmanr

from final_model import (raw, ALL_RAW, FEATURES, INITIAL, TYPE_GROUPS, CLIP, load_labeled_shots, apply_exclusions)

MD = './outputs/final_model'
OUT = './outputs/semi_sup'
NEW = ['Switch_Over_Position_raw', 'Average_Screw_RPM_raw', 'Max_Back_Pressure_raw']
BUDGET = 0.10
METHODS = ('M0_supervised', 'M1_self_training', 'M2_label_spreading', 'M3_pretrain_dae', 'M4_weak_daily')
TARGETS = ('가스', '미성형', '불량전체')


# ====================================================================== 데이터
def _std10(d, c):
    return d[c].groupby(d['run_id'], sort=False).transform(lambda s: s.rolling(10, min_periods=3).std())


def load(p, base_dir='.'):
    for f in (f'{MD}/{p}_scores.csv', f'{MD}/{p}_train_features.csv'):
        if not os.path.exists(f):
            raise FileNotFoundError(f'{f} 없음 → 20a·20b 먼저 실행')
    lab = pd.read_csv(f'{MD}/{p}_scores.csv', parse_dates=['TimeStamp'])
    lab['Reason'] = lab['Reason'].fillna('정상')
    lab = apply_exclusions(lab, p)
    rc = [raw(v) for v in ALL_RAW]
    sh = load_labeled_shots(p, base_dir)
    lab = lab.merge(sh[['TimeStamp'] + [c for c in rc if c in sh.columns]], on='TimeStamp', how='left')
    ll = pd.read_csv(f'{base_dir}/linked/linked_labeled.csv', parse_dates=['TimeStamp'], usecols=['product', 'TimeStamp'] + NEW)
    lab = lab.merge(ll[ll['product'] == p].drop_duplicates('TimeStamp')[['TimeStamp'] + NEW], on='TimeStamp', how='left')

    ref = pd.read_csv(f'{MD}/{p}_train_features.csv', parse_dates=['TimeStamp'])
    uu = pd.read_csv(f'{base_dir}/linked/linked_unlabeled_{p}.csv', parse_dates=['TimeStamp'], low_memory=False,
                     usecols=lambda c: c in ['TimeStamp', 'PART_FACT_PLAN_DATE', 'cavity', 'ERR_FACT_QTY'] + NEW)
    uu['plan_date'] = pd.to_datetime(uu['PART_FACT_PLAN_DATE'].astype(str).str[:10], errors='coerce')
    shot_info = uu.drop_duplicates('TimeStamp')[['TimeStamp', 'plan_date'] + [c for c in NEW if c in uu.columns]]
    ref = ref.merge(shot_info, on='TimeStamp', how='left')
    daily_err = uu.groupby(['plan_date', 'cavity'])['ERR_FACT_QTY'].first().groupby('plan_date').sum()

    for d in (lab, ref):
        d.sort_values(['run_id', 'TimeStamp'], inplace=True)
        d.reset_index(drop=True, inplace=True)
    lab['eval'] = (lab['split'] != '제외') & (lab['Reason'] != INITIAL)
    lab['type'] = np.select([lab['Reason'].isin(TYPE_GROUPS['가스']), lab['Reason'].isin(TYPE_GROUPS['미성형'])],
                            ['가스', '미성형'], np.where(lab['PassOrFail'] == 1, '기타', '정상'))

    # 변수: 두 데이터 모두에서 90% 이상 값이 있는 원값만 std10 + 파생 17개
    cols = [c for c in rc + NEW if c in lab.columns and c in ref.columns
            and lab[c].notna().mean() > 0.9 and ref[c].notna().mean() > 0.9 and ref[c].nunique() > 1]
    def feats(d):
        F = pd.DataFrame({c.replace('_raw', '') + '|std10': _std10(d, c) for c in cols})
        for f in FEATURES:
            F['F|' + f] = d[f].values
        return F.replace([np.inf, -np.inf], np.nan)
    Fl, Fr = feats(lab), feats(ref)
    med = Fr.median()
    q75, q25 = Fr.quantile(0.75), Fr.quantile(0.25)
    iqr = (q75 - q25).where((q75 - q25) > 1e-9, Fr.std().where(Fr.std() > 1e-9, 1.0))
    z = lambda F: np.clip(((F.fillna(med) - med) / iqr).values, -CLIP, CLIP)
    D = dict(p=p, lab=lab, ref=ref, Zl=z(Fl), Zr=z(Fr), names=list(Fl.columns), daily_err=daily_err, cols=cols)
    return D


def target_y(lab, target):
    t = lab['type'].values
    y = (lab['PassOrFail'].values == 1).astype(int) if target == '불량전체' else (t == target).astype(int)
    use = lab['eval'].values & ((y == 1) | (t == '정상'))
    return y, use


# ====================================================================== 방법
def _rf(seed, n=500):
    return RandomForestClassifier(n_estimators=n, max_depth=4, min_samples_leaf=3, class_weight='balanced_subsample',
                                  n_jobs=-1, random_state=seed)


def m0_supervised(Xtr, ytr, Xte, Zu, seed):
    return _rf(seed).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def m1_self_training(Xtr, ytr, Xte, Zu, seed, rounds=3, neg_q=0.5, pos_frac=0.005):
    m = _rf(seed).fit(Xtr, ytr)
    X, y, w = Xtr, ytr, np.ones(len(ytr))
    for _ in range(rounds):
        su = m.predict_proba(Zu)[:, 1]
        neg = np.flatnonzero(su <= np.quantile(su, neg_q))
        rng = np.random.default_rng(seed)
        if len(neg) > 2000:
            neg = rng.choice(neg, 2000, replace=False)
        kpos = int(min(len(Zu) * pos_frac, 2 * ytr.sum()))
        pos = np.argsort(-su)[:kpos]
        X = np.vstack([Xtr, Zu[neg], Zu[pos]])
        y = np.r_[ytr, np.zeros(len(neg), int), np.ones(len(pos), int)]
        w = np.r_[np.ones(len(ytr)), np.ones(len(neg)), 0.5 * np.ones(len(pos))]
        m = _rf(seed).fit(X, y, sample_weight=w)
    return m.predict_proba(Xte)[:, 1]


def m2_label_spreading(Xtr, ytr, Xte, Zu, seed, n_unl=3000):
    rng = np.random.default_rng(seed)
    Zs = Zu[rng.choice(len(Zu), min(n_unl, len(Zu)), replace=False)]
    allX = np.vstack([Xtr, Zs, Xte])
    pca = PCA(n_components=min(10, allX.shape[1]), random_state=seed).fit(np.vstack([Zs, Xtr]))
    P = pca.transform(allX)
    y = np.r_[ytr, -np.ones(len(Zs) + len(Xte), int)]
    ls = LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.2, max_iter=100).fit(P, y)
    dist = ls.label_distributions_[-len(Xte):]
    cls = list(ls.classes_)
    return dist[:, cls.index(1)] if 1 in cls else np.zeros(len(Xte))


class DAE:
    """잡음 제거 오토인코더 (unlabeled만으로 학습). 은닉 8차원 표현 + 복원오차 제공"""

    def __init__(self, seed=0, noise=0.15):
        self.seed, self.noise = seed, noise

    def fit(self, Zu):
        rng = np.random.default_rng(self.seed)
        X = Zu.copy()
        mask = rng.random(X.shape) < self.noise                          # 스왑 잡음: 다른 샷의 같은 변수 값으로 교체
        X[mask] = Zu[rng.integers(0, len(Zu), mask.sum()), np.nonzero(mask)[1]]
        self.m = MLPRegressor(hidden_layer_sizes=(32, 8, 32), activation='relu', alpha=1e-3, max_iter=300,
                              early_stopping=True, random_state=self.seed).fit(X, Zu)
        return self

    def embed(self, Z):
        h = Z
        for i in range(2):                                               # 입력 → 32 → 8 (가운데 층)
            h = np.maximum(0, h @ self.m.coefs_[i] + self.m.intercepts_[i])
        err = ((self.m.predict(Z) - Z) ** 2).mean(1, keepdims=True)
        return np.hstack([h, err])


def make_m3(D, seed):
    dae = DAE(seed).fit(D['Zr'])
    El, Er = dae.embed(D['Zl']), dae.embed(D['Zr'])
    return np.hstack([D['Zl'], El]), np.hstack([D['Zr'], Er])


def m4_weak_daily(D, l2=1.0, min_shots=100):
    """날짜별 불량 수(unlabeled)만으로 샷 불량 확률 모델 학습 → (labeled 점수, 일별 검증 정보)"""
    ref, Zr = D['ref'], D['Zr']
    ok = ref['plan_date'].notna().values
    days = ref.loc[ok, 'plan_date']
    cnt = days.value_counts()
    keep = cnt[cnt >= min_shots].index
    err = D['daily_err'].reindex(keep).dropna()
    keep = err.index
    idx = [np.flatnonzero(ok & (ref['plan_date'] == d).values) for d in keep]
    c = err.values.astype(float)
    n = np.array([len(i) for i in idx], float)
    if len(keep) < 5:
        return None, {'사용 날짜 수': len(keep), '비고': '날짜가 5개 미만 → 학습 불가'}

    def fit(sub):
        def nll(theta):
            b, w = theta[0], theta[1:]
            tot, g = 0.0, np.zeros_like(theta)
            for j in sub:
                X = Zr[idx[j]]
                pr = 1 / (1 + np.exp(-np.clip(b + X @ w, -30, 30)))
                lam = 2 * n[j] * pr.mean() + 1e-9
                tot += lam - c[j] * np.log(lam)
                dl = (1 - c[j] / lam) * 2 * n[j] / len(pr)
                gp = pr * (1 - pr)
                g[0] += dl * gp.sum()
                g[1:] += dl * (gp @ X)
            tot += l2 * (w ** 2).sum()
            g[1:] += 2 * l2 * w
            return tot, g
        rate = c[sub].sum() / (2 * n[sub].sum())
        th0 = np.r_[np.log(max(rate, 1e-6) / max(1 - rate, 1e-6)), np.zeros(Zr.shape[1])]
        return minimize(nll, th0, jac=True, method='L-BFGS-B', options={'maxiter': 500}).x

    th = fit(np.arange(len(keep)))
    # 날짜 하나씩 빼고 학습 → 빠진 날의 '샷당 불량률' 예측 vs 실제 (일별 신호가 있는지)
    pred, true = [], []
    for j in range(len(keep)):
        t2 = fit(np.array([k for k in range(len(keep)) if k != j]))
        pr = 1 / (1 + np.exp(-np.clip(t2[0] + Zr[idx[j]] @ t2[1:], -30, 30)))
        pred.append(pr.mean()); true.append(c[j] / (2 * n[j]))
    rho, pv = spearmanr(pred, true)
    info = {'사용 날짜 수': len(keep), '날짜별 샷 수 중앙값': float(np.median(n)), '날짜 빼고 예측 Spearman ρ': float(rho),
            'p': float(pv), '변수 가중치 상위': dict(pd.Series(th[1:], index=D['names']).abs().sort_values(ascending=False).head(5).round(3))}
    score = D['Zl'] @ th[1:]
    return score, info


# ====================================================================== 평가
def loro_scores(D, method, target, seed, Xl=None, Xr=None):
    lab = D['lab']
    Xl = D['Zl'] if Xl is None else Xl
    Xr = D['Zr'] if Xr is None else Xr
    y, use = target_y(lab, target)
    runs = lab['run_id'].values
    f = {'M0_supervised': m0_supervised, 'M1_self_training': m1_self_training,
         'M2_label_spreading': m2_label_spreading, 'M3_pretrain_dae': m0_supervised}[method]
    s = np.full(len(lab), np.nan)
    for r in pd.unique(runs):
        te = runs == r
        tr = (~te) & use
        if y[tr].sum() < 2:
            continue                                                     # 학습할 불량이 없으면 점수 없음(최하위)
        s[te] = f(Xl[tr], y[tr], Xl[te], Xr, seed)
    return s


def _fill(s):
    s = np.asarray(s, float).copy()
    ok = np.isfinite(s)
    s[~ok] = (s[ok].min() - 1) if ok.any() else 0
    return s


def counts(lab, s, by_run=False):
    m = lab['eval'].values
    idx = np.flatnonzero(m)
    sc = _fill(s)[idx] + np.random.default_rng(0).random(len(idx)) * 1e-9
    t = lab['type'].values[idx]
    sel = np.zeros(len(idx), bool)
    if by_run:
        runs = lab['run_id'].values[idx]
        for r in pd.unique(runs):
            ii = np.flatnonzero(runs == r)
            sel[ii[np.argsort(-sc[ii])[:int(np.ceil(BUDGET * len(ii)))]]] = True
    else:
        sel[np.argsort(-sc)[:int(np.ceil(BUDGET * len(idx)))]] = True
    return int((sel & (t == '가스')).sum()), int((sel & (t == '미성형')).sum())


def evaluate(D, s, method, target, seed):
    lab = D['lab']
    m = lab['eval'].values
    t = lab['type'].values[m]
    y = (lab['PassOrFail'].values[m] == 1).astype(int)
    sc = _fill(s)[m]
    r = {'제품': D['p'], '방법': method, '타깃': target, 'seed': seed, '② AP': average_precision_score(y, sc)}
    for ty in ('가스', '미성형'):
        mm = (t == ty) | (y == 0)
        r[f'AP_{ty}'] = average_precision_score((t[mm] == ty).astype(int), sc[mm])
    r['상위10% 가스'], r['상위10% 미성형'] = counts(lab, s)
    r['Run별10% 가스'], r['Run별10% 미성형'] = counts(lab, s, by_run=True)
    from op_metrics import op_flat
    r.update(op_flat(lab, s))                                            # 상위 5·10%의 Precision·Recall·F1·불량 a/b
    r['학습 못 한 Run 수'] = int(pd.Series(np.isnan(np.asarray(s, float))).groupby(lab['run_id'].values).all().sum())
    return r


def run(products=('cn7', 'rg3'), seeds=(0, 1, 2), base_dir='.', verbose=True):
    os.makedirs(OUT, exist_ok=True)
    rows, infos, S = [], [], {}
    for p in products:
        D = load(p, base_dir)
        if verbose:
            print(f'[{p}] labeled 평가 {int(D["lab"]["eval"].sum())}샷 {D["lab"].loc[D["lab"]["eval"], "type"].value_counts().to_dict()}'
                  f' | unlabeled {len(D["ref"]):,}샷 | 변수 {len(D["names"])}개', flush=True)
        # M4: labeled 라벨 미사용 → 한 번만
        s4, info = m4_weak_daily(D)
        info['제품'] = p
        infos.append(info)
        if verbose:
            print(f'  M4 일별 검증: {info}', flush=True)
        for sd in seeds:
            Xl3, Xr3 = make_m3(D, sd)
            for target in TARGETS:
                for mth in METHODS[:4]:
                    if mth == 'M3_pretrain_dae':
                        s = loro_scores(D, mth, target, sd, Xl3, Xr3)
                    else:
                        s = loro_scores(D, mth, target, sd)
                    S[(p, mth, target, sd)] = s
                    rows.append(evaluate(D, s, mth, target, sd))
                if s4 is not None:
                    rows.append(evaluate(D, s4, 'M4_weak_daily', target, sd))
                    S[(p, 'M4_weak_daily', target, sd)] = s4
            if verbose:
                print(f'  [{p}] seed {sd} 완료', flush=True)
    R = pd.DataFrame(rows)
    base = R[R['방법'] == 'M0_supervised'].set_index(['제품', '타깃', 'seed'])
    for c in ('② AP', 'AP_가스', 'AP_미성형', '상위10% 가스', '상위10% 미성형'):
        R[f'Δ{c} vs M0'] = R.apply(lambda r: r[c] - base.loc[(r['제품'], r['타깃'], r['seed']), c], axis=1)
    agg = R.groupby(['제품', '타깃', '방법'], sort=False).agg(
        **{'② AP': ('② AP', 'mean'), 'AP_가스': ('AP_가스', 'mean'), 'AP_미성형': ('AP_미성형', 'mean'),
           '가스 평균': ('상위10% 가스', 'mean'), '가스 최소': ('상위10% 가스', 'min'), '가스 최대': ('상위10% 가스', 'max'),
           '미성형 평균': ('상위10% 미성형', 'mean'), '미성형 최소': ('상위10% 미성형', 'min'), '미성형 최대': ('상위10% 미성형', 'max'),
           'Δ② AP vs M0 (평균)': ('Δ② AP vs M0', 'mean'), 'Δ② AP>0 시드 수': ('Δ② AP vs M0', lambda x: int((x > 0).sum())),
           '학습 못 한 Run': ('학습 못 한 Run 수', 'max')}).reset_index()
    out = {'by_seed': R, 'summary': agg, 'weak_info': pd.DataFrame(infos), 'scores': S}
    R.to_csv(f'{OUT}/by_seed.csv', index=False, encoding='utf-8-sig')
    agg.to_csv(f'{OUT}/summary.csv', index=False, encoding='utf-8-sig')
    out['weak_info'].to_csv(f'{OUT}/weak_daily_check.csv', index=False, encoding='utf-8-sig')
    return out


# ====================================================================== (선택) 순열검정
def permutation_test(p, method, target, n=50, seed=0, base_dir='.'):
    """라벨 원형 이동(Run 안, 불량 몰림 구조 유지) 후 같은 LORO 절차 반복 → 타깃 상위10% 포착·타깃 AP가 우연을 넘는지
       M4는 labeled 라벨을 안 쓰므로 점수 고정, 라벨만 이동"""
    D = load(p, base_dir)
    lab0 = D['lab']
    col = {'가스': 0, '미성형': 1}.get(target, None)

    def stat(Dx):
        if method == 'M4_weak_daily':
            s, _ = m4_weak_daily(Dx) if not hasattr(stat, 's4') else (stat.s4, None)
            stat.s4 = s
        elif method == 'M3_pretrain_dae':
            Xl3, Xr3 = make_m3(Dx, seed)
            s = loro_scores(Dx, method, target, seed, Xl3, Xr3)
        else:
            s = loro_scores(Dx, method, target, seed)
        c = counts(Dx['lab'], s)
        return c[col] if col is not None else sum(c)

    obs = stat(D)
    rng = np.random.default_rng(seed + 100)
    ev = np.flatnonzero(lab0['eval'].values)
    runs = lab0['run_id'].values
    null = []
    for b in range(n):
        L = lab0.copy()
        for c_ in ('type', 'PassOrFail'):
            v = L[c_].values.copy()
            for r in pd.unique(runs[ev]):
                ii = ev[runs[ev] == r]
                v[ii] = np.roll(lab0[c_].values[ii], rng.integers(0, len(ii))) if c_ == 'type' else v[ii]
            L[c_] = v
        L['PassOrFail'] = np.where(L['type'].isin(['가스', '미성형', '기타']), 1, 0)
        Dx = dict(D, lab=L)
        null.append(stat(Dx))
    null = np.asarray(null)
    return {'관측': obs, '무작위 평균': float(null.mean()), '무작위 95%': float(np.quantile(null, 0.95)),
            'p': float((1 + (null >= obs).sum()) / (n + 1)), 'n': n}
