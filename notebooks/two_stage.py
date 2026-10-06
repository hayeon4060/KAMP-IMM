"""
two_stage.py — 45번: 2단계 판정 (1단계 검사 대상 고르기 → 2단계 그 안에서 불량/정상 판정)

모델 3개 (최종 모델 단독은 제외)
  A. CN7 경보 + std4      : 1단계 = 최종 모델 경보 샷 + 나머지를 std4 흔들림 점수 순으로 채워 상위 10%
                             2단계 = 최종 pct ≥ 0.999  또는  std4 pct ≥ q  → 불량
  B. RG3 가스 전용 RF      : 1단계 = 가스 RF 점수 상위 10%
                             2단계 = 가스 RF pct ≥ q → 불량
  C. RG3 가스 RF + 미성형 N1: 1단계 = 검사 샷 중 68%는 가스 RF 상위, 나머지는 미성형 N1 순위 (43번 규칙)
                             2단계 = 가스 RF pct ≥ q  또는  미성형 N1 pct ≥ q → 불량

pct (정상 기준 백분위) 만드는 법
  - 비지도 점수(std4): 정상 기준 데이터(unlabeled, 20a 학습 데이터)의 같은 점수 분포 대비 백분위
  - 지도 점수(RF, N1): Run 하나씩 빼고 학습한 '그 fold의 모델'로 정상 기준 데이터에도 점수를 매기고, 그 분포 대비 백분위
    (N1이 학습에 쓴 unlabeled 샷은 백분위 기준에서 제외 → 학습에 본 샷으로 기준이 낮아지는 것 방지)
  → "정상 생산 샷 중 상위 1%(q=0.99)보다 더 이상하면 불량"이라는 같은 의미의 기준
q: 0.99 / 0.995 / 0.999 (미리 정함, 결과 보고 고르지 않음)

출력 (모델 × q): 검사 대상 샷과 그 안의 불량·가스·미성형 / 불량 판정 수 / TP·FP·FN·TN /
                Precision(판정) / Recall(검사 대상 안) / Recall(전체 불량 기준) / F1 / 판정된 가스·미성형
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import LedoitWolf

import cn7_alarm_plus_gas as C
import semi_normal as SN
import semi_sup as SS
from final_model import raw, CLIP

OUT = './outputs/two_stage'
QS = (0.99, 0.995, 0.999)
BUDGET = 0.10
GAS_SHARE = 0.68


def to_pct(ref_scores, s):
    r = np.sort(np.asarray(ref_scores, float))
    r = r[np.isfinite(r)]
    s = np.asarray(s, float)
    return np.searchsorted(r, np.where(np.isfinite(s), s, -np.inf), side='right') / (len(r) + 1)


# ====================================================================== 공통 평가
def judge(lab, stage1, defect_pred, name, q):
    """stage1: 검사 대상(bool, 전체 길이) / defect_pred: 불량 판정(bool, 전체 길이). 평가 대상 샷만 집계"""
    ev = lab['eval'].values
    t = lab['type'].values
    y = (lab['PassOrFail'].values == 1) & ev
    s1 = stage1 & ev
    pr = defect_pred & s1
    tp, fp = int((pr & y).sum()), int((pr & ~y).sum())
    fn, tn = int((s1 & ~pr & y).sum()), int((s1 & ~pr & ~y).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R_in = tp / (tp + fn) if tp + fn else 0.0
    R_all = tp / int(y.sum()) if y.sum() else 0.0
    out = {'모델': name, 'q': q, '검사 대상 샷': int(s1.sum()), '그 안의 불량': f'{int((s1 & y).sum())}/{int(y.sum())}',
           '불량 판정 수': tp + fp, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
           'Precision(판정)': P, 'Recall(검사 대상 안)': R_in, 'Recall(전체 불량)': R_all,
           'F1(전체 기준)': 2 * P * R_all / (P + R_all) if P + R_all else 0.0}
    for ty in ('가스', '미성형'):
        n = int((ev & (t == ty)).sum())
        if n:
            out[f'1단계 {ty}'] = f'{int((s1 & (t == ty)).sum())}/{n}'
            out[f'판정 {ty}'] = f'{int((pr & (t == ty)).sum())}/{n}'
    return out


def topk(lab, s, budget=BUDGET):
    ev = np.flatnonzero(lab['eval'].values)
    v = np.asarray(s, float)[ev]
    v = np.where(np.isfinite(v), v, -1e18) + np.random.default_rng(0).random(len(ev)) * 1e-9
    m = np.zeros(len(lab), bool)
    m[ev[np.argsort(-v)[:int(np.ceil(budget * len(ev)))]]] = True
    return m


def split_top(lab, gas_s, short_s, share=GAS_SHARE, budget=BUDGET):
    ev = np.flatnonzero(lab['eval'].values)
    K = int(np.ceil(budget * len(ev)))
    jit = np.random.default_rng(0).random(len(ev)) * 1e-9
    g = np.where(np.isfinite(gas_s[ev]), gas_s[ev], -1e18) + jit
    sh = np.where(np.isfinite(short_s[ev]), short_s[ev], -1e18) + jit
    sel = np.zeros(len(ev), bool)
    sel[np.argsort(-g)[:int(round(share * K))]] = True
    for i in np.argsort(-sh):
        if sel.sum() >= K:
            break
        sel[i] = True
    m = np.zeros(len(lab), bool)
    m[ev[sel]] = True
    return m


# ====================================================================== A. CN7 경보 + std4
def std4_scores(ref, lab, k=10):
    std = lambda d, v: d[raw(v)].groupby(d['run_id'], sort=False).transform(lambda s: s.rolling(k, min_periods=3).std())
    Xr = np.column_stack([std(ref, v) for v in C.STD4])
    Xl = np.column_stack([std(lab, v) for v in C.STD4])
    med = np.nanmedian(Xr, 0)
    q75, q25 = np.nanpercentile(Xr, [75, 25], 0)
    iqr = np.where(q75 - q25 > 1e-9, q75 - q25, 1.0)
    z = lambda X: np.nan_to_num(np.clip((X - med) / iqr, -CLIP, CLIP))
    lw = LedoitWolf().fit(z(Xr))
    return np.sqrt(lw.mahalanobis(z(Xl))), np.sqrt(lw.mahalanobis(z(Xr)))


def run_cn7(qs=QS, k=10):
    ref, lab = C.load('cn7')
    sl, sr = std4_scores(ref, lab, k)
    p_std4 = to_pct(sr, sl)
    alarm = lab['pct'].values >= C.ALARM_Q
    idx, sel = C.select(lab, sl)                                     # 경보 먼저 + std4로 10% 채움 (38번과 같은 규칙)
    stage1 = np.zeros(len(lab), bool)
    stage1[idx[sel]] = True
    rows = []
    for q in qs:
        rows.append(judge(lab, stage1, alarm | (p_std4 >= q), 'CN7 경보 + std4', q))
    rows.append(judge(lab, stage1, alarm, 'CN7 경보 + std4 (2단계 = 최종 경보만)', 0.999))
    return pd.DataFrame(rows)


# ====================================================================== B·C. RG3
def _rf(gas, seed):
    return RandomForestClassifier(n_estimators=1000 if gas else 500, max_depth=4, min_samples_leaf=5 if gas else 3,
                                  class_weight='balanced_subsample', n_jobs=-1, random_state=seed)


def rg3_scores_with_pct(D, seed=0, n_unl=3000):
    """Run 하나씩 빼고: 가스 RF / 미성형 N1 점수 + 그 fold 모델로 매긴 정상 기준 데이터 대비 백분위"""
    lab, Zl, Zr = D['lab'], D['Zl'], D['Zr']
    runs = lab['run_id'].values
    out = {k: np.full(len(lab), np.nan) for k in ('gas', 'gas_pct', 'short', 'short_pct')}
    rng = np.random.default_rng(seed)
    for target, key in (('가스', 'gas'), ('미성형', 'short')):
        y, use = SS.target_y(lab, target)
        for r in pd.unique(runs):
            te = runs == r
            tr = (~te) & use
            if y[tr].sum() < 2:
                continue
            if key == 'gas':
                m = _rf(True, seed).fit(Zl[tr], y[tr])
                ref_mask = np.ones(len(Zr), bool)
            else:                                                      # N1: unlabeled 3,000샷을 정상으로 추가(가중치 0.3)
                ui = rng.choice(len(Zr), min(n_unl, len(Zr)), replace=False)
                X = np.vstack([Zl[tr], Zr[ui]])
                yy = np.r_[y[tr], np.zeros(len(ui), int)]
                w = np.r_[np.ones(tr.sum()), 0.3 * np.ones(len(ui))]
                m = _rf(False, seed).fit(X, yy, sample_weight=w)
                ref_mask = np.ones(len(Zr), bool)
                ref_mask[ui] = False                                   # 학습에 쓴 unlabeled 샷은 백분위 기준에서 제외
            st = m.predict_proba(Zl[te])[:, 1]
            sr = m.predict_proba(Zr[ref_mask])[:, 1]
            out[key][te] = st
            out[key + '_pct'][te] = to_pct(sr, st)
    return out


def run_rg3(seeds=(0, 1, 2), qs=QS, base_dir='.'):
    D = SN.load_std('rg3', base_dir)
    lab = D['lab']
    rows = []
    for sd in seeds:
        S = rg3_scores_with_pct(D, sd)
        st_gas = topk(lab, S['gas'])
        st_combo = split_top(lab, S['gas'], S['short'])
        for q in qs:
            gp = np.nan_to_num(S['gas_pct']) >= q
            sp = np.nan_to_num(S['short_pct']) >= q
            r1 = judge(lab, st_gas, gp, 'RG3 가스 전용 RF', q); r1['seed'] = sd
            r2 = judge(lab, st_combo, gp | sp, 'RG3 가스 RF + 미성형 N1', q); r2['seed'] = sd
            rows += [r1, r2]
    return pd.DataFrame(rows)


def summarize(R):
    """시드 평균 (RG3)"""
    num = ['검사 대상 샷', '불량 판정 수', 'TP', 'FP', 'FN', 'TN', 'Precision(판정)', 'Recall(검사 대상 안)',
           'Recall(전체 불량)', 'F1(전체 기준)']
    cnt = [c for c in R.columns if c.startswith(('1단계', '판정 ', '그 안의'))]
    R = R.copy()
    for c in cnt:
        R[c + ' 수'] = R[c].str.split('/').str[0].astype(int)
    agg = R.groupby(['모델', 'q'], sort=False)[num + [c + ' 수' for c in cnt]].mean().reset_index()
    tot = {c: R[c].iloc[0].split('/')[1] for c in cnt}
    agg = agg.rename(columns={c + ' 수': f'{c} (/{tot[c]}, 평균)' for c in cnt})
    return agg


def run(seeds=(0, 1, 2), base_dir='.'):
    os.makedirs(OUT, exist_ok=True)
    A = run_cn7()
    B = run_rg3(seeds, base_dir=base_dir)
    Bs = summarize(B)
    A.to_csv(f'{OUT}/cn7.csv', index=False, encoding='utf-8-sig')
    B.to_csv(f'{OUT}/rg3_by_seed.csv', index=False, encoding='utf-8-sig')
    Bs.to_csv(f'{OUT}/rg3_summary.csv', index=False, encoding='utf-8-sig')
    return {'cn7': A, 'rg3_by_seed': B, 'rg3': Bs}
