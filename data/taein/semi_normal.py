"""
semi_normal.py — 41번: unlabeled는 '불량을 가져오기'보다 '정상 범위를 배우는 데' 집중 + 약한 지도(M4) 전체 기간 재시도

A. RG3 — 가스 RF와 같은 변수(원값 21 + 새 변수 3개의 직전 10샷 std = 24개)로 공정 비교
   unlabeled는 전부 '정상 쪽 정보'로만 사용 (의사 불량 라벨·라벨 전파 없음)
   M0  supervised           : 기준 = 37번 가스 RF와 같은 설정 (labeled만)
   N1  normal_augment       : unlabeled 3,000샷을 '정상'으로 학습에 추가 (가중치 0.3) → 정상 범위를 넓게 학습
   N2  normal_distance_feat : unlabeled로만 학습한 '정상에서 떨어진 정도'(Mahalanobis·IsolationForest) 2개를 변수로 추가
   N3  weighted_deviation   : 정상 기준 robust z의 제곱(= 변수별 정상 이탈 크기)을 입력으로 로지스틱 회귀
                              → 어떤 변수의 '정상 이탈'이 불량과 관련 있는지 가중치로 해석 가능 (Deep SAD의 선형판 개념)
   N4  pretrain_dae         : unlabeled(정상)로 잡음 제거 오토인코더 → 정상 표현 8차원 + 복원오차를 변수로 추가
   N5  self_train_negatives : 자기 학습이지만 '확실한 정상'(unlabeled 점수 하위 50%)만 추가, 불량 의사 라벨은 안 붙임
   평가: Run 하나씩 빼고 검증(LORO), 시드 3개, 같은 fold의 M0 대비 Δ

B. 약한 지도 M4 — unlabeled 전체 기간(3월~11월)으로 날짜 수 늘리기
   - 40번은 정상 기준 데이터(9~11월, 필터 후)만 써서 날짜가 CN7 6일 / RG3 4일 → 과적합(ρ −0.94)
   - 전체 기간 생산 샷(0 모드·끼어든 기록 제거, 최종 모델과 같은 규칙)으로 날짜를 늘리고,
     **labeled 날짜는 학습에서 제외**(평가 독립성), 변수는 미리 정한 1~2개로 제한(과적합 방지):
       w1 = 최종 모델 점수(날마다 평균 이상 정도), w2 = std4 흔들림 점수, w12 = 둘 다
   - 날짜 하나씩 빼고 예측한 '샷당 불량률' 상관(ρ) → 유의할 때만 labeled 샷 평가를 의미 있게 봄
   ⚠ ERR_FACT_QTY는 생산계획일 × 캐비티의 모든 불량 유형 합계, 기록된 샷 수는 실제 생산량보다 적은 날이 많음 → 비율에 잡음
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from scipy.stats import spearmanr

import semi_sup as SS
from final_model import raw, FinalModel, make_features, load_labeled_shots

OUT = './outputs/semi_normal'
MD = './outputs/final_model'
STD4 = ['Barrel_Temperature_2', 'Hopper_Temperature', 'Cushion_Position', 'Barrel_Temperature_5']
A_METHODS = ('M0_supervised', 'N1_normal_augment', 'N2_normal_distance_feat', 'N3_weighted_deviation',
             'N4_pretrain_dae', 'N5_self_train_negatives')


# ====================================================================== A. 정상 범위 중심 준지도 (RG3)
def load_std(p='rg3', base_dir='.'):
    D = SS.load(p, base_dir)
    keep = [i for i, n in enumerate(D['names']) if n.endswith('|std10')]
    D['Zl'], D['Zr'] = D['Zl'][:, keep], D['Zr'][:, keep]
    D['names'] = [D['names'][i] for i in keep]
    return D


def _rf(target, seed):
    gas = target == '가스'
    return RandomForestClassifier(n_estimators=1000 if gas else 500, max_depth=4, min_samples_leaf=5 if gas else 3,
                                  class_weight='balanced_subsample', n_jobs=-1, random_state=seed)


def fold_predict(method, Xtr, ytr, Xte, Zu, target, seed, extra=None):
    """extra: (labeled 추가 변수, unlabeled 추가 변수) — N2·N4용, unlabeled만으로 미리 계산"""
    rng = np.random.default_rng(seed)
    if method == 'M0_supervised':
        return _rf(target, seed).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    if method == 'N1_normal_augment':
        u = Zu[rng.choice(len(Zu), min(3000, len(Zu)), replace=False)]
        X = np.vstack([Xtr, u])
        y = np.r_[ytr, np.zeros(len(u), int)]
        w = np.r_[np.ones(len(ytr)), 0.3 * np.ones(len(u))]
        return _rf(target, seed).fit(X, y, sample_weight=w).predict_proba(Xte)[:, 1]
    if method == 'N3_weighted_deviation':
        m = LogisticRegression(C=0.1, class_weight='balanced', max_iter=5000)
        return m.fit(Xtr ** 2, ytr).predict_proba(Xte ** 2)[:, 1]
    if method == 'N5_self_train_negatives':
        m = _rf(target, seed).fit(Xtr, ytr)
        for _ in range(2):
            su = m.predict_proba(Zu)[:, 1]
            neg = np.flatnonzero(su <= np.quantile(su, 0.5))
            if len(neg) > 2000:
                neg = rng.choice(neg, 2000, replace=False)
            X = np.vstack([Xtr, Zu[neg]])
            y = np.r_[ytr, np.zeros(len(neg), int)]
            w = np.r_[np.ones(len(ytr)), 0.3 * np.ones(len(neg))]
            m = _rf(target, seed).fit(X, y, sample_weight=w)
        return m.predict_proba(Xte)[:, 1]
    raise ValueError(method)


def normal_distance_features(D, seed=0):
    """unlabeled(정상)만으로 학습: Mahalanobis 거리, IsolationForest 이상 점수 → (labeled, unlabeled) 2열"""
    lw = LedoitWolf().fit(D['Zr'])
    ifo = IsolationForest(n_estimators=300, random_state=seed).fit(D['Zr'])
    f = lambda Z: np.column_stack([np.sqrt(lw.mahalanobis(Z)), -ifo.score_samples(Z)])
    return f(D['Zl']), f(D['Zr'])


def loro(D, method, target, seed, Xl=None, Xr=None):
    lab = D['lab']
    Xl = D['Zl'] if Xl is None else Xl
    Xr = D['Zr'] if Xr is None else Xr
    y, use = SS.target_y(lab, target)
    runs = lab['run_id'].values
    s = np.full(len(lab), np.nan)
    base = 'M0_supervised' if method in ('N2_normal_distance_feat', 'N4_pretrain_dae') else method
    for r in pd.unique(runs):
        te = runs == r
        tr = (~te) & use
        if y[tr].sum() < 2:
            continue
        s[te] = fold_predict(base, Xl[tr], y[tr], Xl[te], Xr, target, seed)
    return s


def run_A(p='rg3', seeds=(0, 1, 2), targets=('가스', '미성형'), base_dir='.', verbose=True):
    os.makedirs(OUT, exist_ok=True)
    D = load_std(p, base_dir)
    if verbose:
        print(f'[{p}] 변수 {len(D["names"])}개 (std10만) | unlabeled {len(D["Zr"]):,}샷', flush=True)
    rows, coefs = [], []
    for sd in seeds:
        dl, du = normal_distance_features(D, sd)
        X2l, X2r = np.hstack([D['Zl'], dl]), np.hstack([D['Zr'], du])
        X4l, X4r = SS.make_m3(dict(D), sd)
        for target in targets:
            for mth in A_METHODS:
                if mth == 'N2_normal_distance_feat':
                    s = loro(D, mth, target, sd, X2l, X2r)
                elif mth == 'N4_pretrain_dae':
                    s = loro(D, mth, target, sd, X4l, X4r)
                else:
                    s = loro(D, mth, target, sd)
                rows.append(SS.evaluate(D, s, mth, target, sd))
            # N3 해석: 전체 labeled로 학습한 가중치 (기술용, 성능 근거 아님)
            y, use = SS.target_y(D['lab'], target)
            m = LogisticRegression(C=0.1, class_weight='balanced', max_iter=5000).fit(D['Zl'][use] ** 2, y[use])
            coefs.append(pd.Series(m.coef_[0], index=D['names']).rename(f'{target}|seed{sd}'))
        if verbose:
            print(f'  seed {sd} 완료', flush=True)
    R = pd.DataFrame(rows)
    base = R[R['방법'] == 'M0_supervised'].set_index(['타깃', 'seed'])
    for c in ('② AP', 'AP_가스', 'AP_미성형', '상위10% 가스', '상위10% 미성형'):
        R[f'Δ{c}'] = R.apply(lambda r: r[c] - base.loc[(r['타깃'], r['seed']), c], axis=1)
    agg = R.groupby(['타깃', '방법'], sort=False).agg(
        **{'② AP': ('② AP', 'mean'), 'AP_가스': ('AP_가스', 'mean'), 'AP_미성형': ('AP_미성형', 'mean'),
           '가스 평균': ('상위10% 가스', 'mean'), '가스 범위': ('상위10% 가스', lambda x: f'{x.min()}~{x.max()}'),
           '미성형 평균': ('상위10% 미성형', 'mean'), '미성형 범위': ('상위10% 미성형', lambda x: f'{x.min()}~{x.max()}'),
           'Δ② AP 평균': ('Δ② AP', 'mean'), 'Δ② AP>0 시드': ('Δ② AP', lambda x: int((x > 0).sum())),
           'Δ가스 평균': ('Δ상위10% 가스', 'mean'), 'Δ미성형 평균': ('Δ상위10% 미성형', 'mean')}).reset_index()
    C = pd.concat(coefs, axis=1)
    R.to_csv(f'{OUT}/A_by_seed.csv', index=False, encoding='utf-8-sig')
    agg.to_csv(f'{OUT}/A_summary.csv', index=False, encoding='utf-8-sig')
    C.to_csv(f'{OUT}/A_N3_coefs.csv', encoding='utf-8-sig')
    return {'by_seed': R, 'summary': agg, 'n3_coefs': C, 'D': D}


# ====================================================================== B. 약한 지도 M4 — 전체 기간
def daily_table(p, base_dir='.', min_shots=100):
    """unlabeled 전체 기간 생산 샷 → 샷별 점수 2개(최종 모델, std4) + 날짜별 불량 수. labeled 날짜는 표시"""
    from daily_check import load_daily_errors, production_shots
    model = FinalModel.load(f'{MD}/{p}_final_model.pkl')
    u, err = load_daily_errors(p, base_dir)
    s, _ = production_shots(u)
    f = make_features(s, model.steps, model.fr_step)
    f['final'] = model.score(f)
    ref = pd.read_csv(f'{MD}/{p}_train_features.csv', parse_dates=['TimeStamp']).sort_values(['run_id', 'TimeStamp'])
    std = lambda d, v: d[raw(v)].groupby(d['run_id'], sort=False).transform(lambda x: x.rolling(10, min_periods=3).std())
    Xr = np.column_stack([std(ref, v) for v in STD4])
    med = np.nanmedian(Xr, 0)
    q75, q25 = np.nanpercentile(Xr, [75, 25], 0)
    iqr = np.where(q75 - q25 > 1e-9, q75 - q25, 1.0)
    zf = lambda X: np.nan_to_num(np.clip((X - med) / iqr, -20, 20))
    lw = LedoitWolf().fit(zf(Xr))
    std4_fn = lambda d: np.sqrt(lw.mahalanobis(zf(np.column_stack([std(d, v) for v in STD4]))))
    f['std4'] = std4_fn(f)
    lab_days = set(pd.to_datetime(load_labeled_shots(p, base_dir)['TimeStamp']).dt.normalize())
    day = f.groupby('plan_date').agg(n=('TimeStamp', 'size'))
    day['err'] = err['불량_합계']
    day['labeled_day'] = day.index.isin(lab_days)
    day = day[(day['n'] >= min_shots) & day['err'].notna()]
    return f, day, std4_fn


def fit_poisson(Xs_by_day, n, c, l2=0.5):
    """날짜별: 기대 불량 = 2·n·mean σ(b + w·x).  x는 미리 표준화된 1~2개 변수"""
    k = Xs_by_day[0].shape[1]

    def nll(th):
        b, w = th[0], th[1:]
        tot, g = 0.0, np.zeros_like(th)
        for X, nj, cj in zip(Xs_by_day, n, c):
            pr = 1 / (1 + np.exp(-np.clip(b + X @ w, -30, 30)))
            lam = 2 * nj * pr.mean() + 1e-9
            tot += lam - cj * np.log(lam)
            dl = (1 - cj / lam) * 2 * nj / len(pr)
            gp = pr * (1 - pr)
            g[0] += dl * gp.sum()
            g[1:] += dl * (gp @ X)
        tot += l2 * (w ** 2).sum()
        g[1:] += 2 * l2 * w
        return tot, g
    rate = c.sum() / (2 * n.sum())
    th0 = np.r_[np.log(max(rate, 1e-6) / max(1 - rate, 1e-6)), np.zeros(k)]
    return minimize(nll, th0, jac=True, method='L-BFGS-B', options={'maxiter': 500}).x


def run_B(products=('cn7', 'rg3'), base_dir='.', verbose=True):
    os.makedirs(OUT, exist_ok=True)
    rows, evals, days_all = [], [], {}
    for p in products:
        f, day, std4_fn = daily_table(p, base_dir)
        tr_days = day[~day['labeled_day']]
        days_all[p] = day
        if verbose:
            print(f'[{p}] 날짜 {len(day)}일 (labeled 날짜 {int(day["labeled_day"].sum())}일 제외 → 학습 {len(tr_days)}일)', flush=True)
        # 학습 데이터 기준 표준화 (학습 날짜의 샷만)
        trf = f[f['plan_date'].isin(tr_days.index)]
        mu = trf[['final', 'std4']].median()
        sc = (trf[['final', 'std4']].quantile(0.75) - trf[['final', 'std4']].quantile(0.25)).replace(0, 1)
        Zs = lambda d, cols: np.clip(((d[cols] - mu[cols]) / sc[cols]).fillna(0).values, -20, 20)
        # 학습 없이 보는 단순 상관: 날짜별 평균 점수 vs 샷당 불량률
        g = trf.groupby('plan_date')[['final', 'std4']].mean().reindex(tr_days.index)
        rate = tr_days['err'] / (2 * tr_days['n'])
        simple = {c: spearmanr(g[c], rate) for c in ('final', 'std4')}
        D = SS.load(p, base_dir)
        lab = D['lab']
        lab_std4 = std4_fn(lab)
        for name, cols in (('w1_final', ['final']), ('w2_std4', ['std4']), ('w12_both', ['final', 'std4'])):
            Xd = [Zs(trf[trf['plan_date'] == d], cols) for d in tr_days.index]
            n, c = tr_days['n'].values.astype(float), tr_days['err'].values.astype(float)
            if len(Xd) < 5:
                rows.append({'제품': p, '모형': name, '학습 날짜': len(Xd), '비고': '날짜 5개 미만'})
                continue
            th = fit_poisson(Xd, n, c)
            pred = []
            for j in range(len(Xd)):
                idx = [k for k in range(len(Xd)) if k != j]
                t2 = fit_poisson([Xd[k] for k in idx], n[idx], c[idx])
                pred.append((1 / (1 + np.exp(-np.clip(t2[0] + Xd[j] @ t2[1:], -30, 30)))).mean())
            rho, pv = spearmanr(pred, c / (2 * n))
            rows.append({'제품': p, '모형': name, '학습 날짜': len(Xd), '가중치': np.round(th[1:], 3).tolist(),
                         '날짜 빼고 예측 ρ': rho, 'p': pv,
                         '단순 상관 ρ(final)': simple['final'].correlation, 'p(final)': simple['final'].pvalue,
                         '단순 상관 ρ(std4)': simple['std4'].correlation, 'p(std4)': simple['std4'].pvalue})
            L = lab.assign(final=lab['score'].values, std4=lab_std4)
            s = Zs(L, cols) @ th[1:]
            e = SS.evaluate(D, s, f'M4_full|{name}', '불량전체', 0)
            evals.append(e)
        if verbose:
            print(f'  [{p}] 완료', flush=True)
    R = pd.DataFrame(rows)
    E = pd.DataFrame(evals)
    R.to_csv(f'{OUT}/B_daily_check.csv', index=False, encoding='utf-8-sig')
    E.to_csv(f'{OUT}/B_labeled_eval.csv', index=False, encoding='utf-8-sig')
    return {'daily': R, 'labeled_eval': E, 'days': days_all}
