"""
contrib.py — 대표 설정 4개의 변수별 기여도 (CN7 공식 기간만). FEATURE_SUMMARY.md [2]용.

기여도 = 쇼트 이상 점수의 제곱 거리를 변수별 항으로 나눈 비율 (합 = 1)
  Mahalanobis 계열: d² = zᵀPz = Σ_j z_j·(Pz)_j → 항 j = z_j·(Pz)_j  (z = 전처리값 − 평균, P = 정밀도 행렬; 음수 가능)
  GMM: 그 쇼트의 사후확률 최대 성분의 평균·정밀도로 같은 분해, 시드 0~4 비율 평균
  AE: 재구성 오차² = Σ_j (x_j − x̂_j)² → 항 j = (x_j − x̂_j)², 시드 0~4 비율 평균
출력: outputs/contrib_vars.csv (쇼트 묶음 × 변수), outputs/contrib_groups.csv (쇼트 묶음 × 변수 묶음), outputs/contrib_meta.csv
"""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

import lib as L

M = L.M
SEEDS = range(5)
EVENT = ['2020-10-16 05:55:21', '2020-10-16 05:56:22', '2020-10-16 05:57:20']
PRE_GAS = ['2020-10-16 05:45:26', '2020-10-16 05:49:24', '2020-10-16 05:53:23', '2020-10-16 05:54:21']

SPEED_TIME = ['Injection_Time', 'Filling_Time', 'Plasticizing_Time', 'Cycle_Time', 'Clamp_Close_Time',
              'Max_Injection_Speed', 'Max_Screw_RPM']
PRESSURE = ['Max_Injection_Pressure', 'Max_Switch_Over_Pressure', 'Average_Back_Pressure']
POSITION = ['Cushion_Position', 'Plasticizing_Position']
BARREL = [f'Barrel_Temperature_{i}' for i in range(1, 7)]
HOPPER_MOLD = ['Hopper_Temperature', 'Mold_Temperature_3', 'Mold_Temperature_4', 'mold', 'hopper']
REL = ['rel_fill_speed', 'rel_plast', 'rel_hold', 'rel_sw_ratio', 'rel_rest', 'flow_res', 'fill_anomaly']
NEW_FAM = ('rz5_', 'rz50_', 'rz200_', 'mx5_', 'drift0_', 'd1_', 'd2_', 'ew_', 'vol10_', 'lag1_', 'lag2_', 'rmed_')


def base_var(c):
    for pre in ('mx5_dv_', 'lag1_dv_', 'lag2_dv_', 'rz5_', 'rz50_', 'rz200_', 'drift0_', 'rmed_', 'd1_', 'd2_',
                'ew_', 'vol10_', 'dv_', 'dev_', 'raw_'):
        if c.startswith(pre):
            return c[len(pre):]
    return c


def phys_group(c):
    b = base_var(c)
    if b in SPEED_TIME:
        return '속도·시간'
    if b in PRESSURE:
        return '압력'
    if b in POSITION:
        return '위치'
    if b in BARREL:
        return '배럴 온도'
    if b in HOPPER_MOLD or b.split('_')[0] in ('mold', 'hopper'):
        return '호퍼·금형'
    if b in REL:
        return '관계(R1~R5·c17 관계)'
    if b == 'ev_max_abs_dev':
        return '요약(c17 ev_max_abs_dev)'
    return '기타'


def group_of(c):
    """변수 묶음: 새 계열(rz5·mx5·drift0 등) 열은 '새 계열', 나머지는 물리 묶음"""
    return '새 계열(rz5·mx5·drift0)' if c.startswith(NEW_FAM) else phys_group(c)


def maha_terms(Z, P):
    return Z * (Z @ P)


# ====================================================================== 설정별 (라벨 쇼트 점수, 항 행렬)
def cfg_a(U, lab, SM):
    """현 제출: cur_0901 · c17 · LW-Mahalanobis (FinalModel 그대로)"""
    cols = L.C.FEAT_C17
    m = L.C.fm.FinalModel(cols).fit(U[SM['cur_0901']])
    Z = m._prep(lab) - m.lw.location_
    return m.score(lab), maha_terms(Z, m.lw.precision_), cols


def _prep(ref, lab, cols, drop_const=False):
    Xr = ref[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
    Xl = lab[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
    pr = M.Prep().fit(Xr)
    A, B = pr(Xr), pr(Xl)
    keep = A.std(0) > 0 if drop_const else np.ones(A.shape[1], bool)
    return A[:, keep], B[:, keep], [c for c, k in zip(cols, keep) if k]


def cfg_b(U, lab, SM):
    """w2_pre · dev26 · GMM 2성분 (unsup_search2 R1 W1과 같은 설정), 시드 0~4"""
    cols = L.C.FEAT_DEV + L.C.FEAT_REL
    A, B, cols = _prep(U[SM['w2_pre']], lab, cols)
    sc, T = [], []
    for sd in SEEDS:
        g = M.GMM(2, seed=sd).fit(A).m
        sc.append(-g.score_samples(B))
        k = g.predict(B)
        t = np.empty_like(B)
        for j in range(g.n_components):
            i = k == j
            t[i] = maha_terms(B[i] - g.means_[j], g.precisions_[j])
        T.append(t / t.sum(1, keepdims=True))
    return np.mean(sc, 0), np.mean(T, 0), cols


def cfg_c(U, lab, SM, k=300):
    """w2_pre · dev26 · 국소 Mahalanobis k=300 (unsup_search2 C2와 같음)"""
    cols = L.C.FEAT_DEV + L.C.FEAT_REL
    A, B, cols = _prep(U[SM['w2_pre']], lab, cols)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(B, return_distance=False)[:, :k]
    sc, T = np.empty(len(B)), np.empty_like(B)
    for i, q in enumerate(B):
        lw = LedoitWolf().fit(A[idx[i]])
        T[i] = maha_terms((q - lw.location_)[None], lw.precision_)[0]
        sc[i] = np.sqrt(T[i].sum())
    return sc, T, cols


def cfg_d(U, lab, SM, fam):
    """unsup_search3 판정 통과 1위: w2_all · dv26+rz5+mx5+drift0 · AE, 시드 0~4 (lib.fit_score_all과 같은 전처리)"""
    torch = M._torch()
    cols = L.cols_of(fam, ['dv26', 'rz5', 'mx5', 'drift0'])
    ref = U[SM['w2_all']]
    ref = ref.sample(L.REF_MAX, random_state=42) if len(ref) > L.REF_MAX else ref
    A, B, cols = _prep(ref.astype({c: np.float32 for c in cols}), lab, cols, drop_const=True)
    sc, T = [], []
    for sd in SEEDS:
        ae = M.AE(seed=sd).fit(A)
        with torch.no_grad():
            x = torch.tensor(B, dtype=torch.float32)
            e2 = ((ae.dec(ae.enc(x)) - x) ** 2).numpy().astype(float)
        sc.append(np.sqrt(e2.sum(1)))
        T.append(e2 / e2.sum(1, keepdims=True))
    return np.mean(sc, 0), np.mean(T, 0), cols


# ====================================================================== 쇼트 묶음
def shot_groups(ev, lab, s):
    e = ev[ev['period'] == 'official'].reset_index(drop=True)
    ts = pd.DatetimeIndex(lab['TimeStamp'])
    pos = {t: i for i, t in enumerate(ts)}
    rs = L._tb(np.asarray(s, float)[e['pos'].values])
    o = np.argsort(-rs, kind='stable')
    k5 = int(round(len(e) * 0.05))
    bad_shot = set(e.loc[e['y'] == 1, 'pos'])
    fa = sorted({int(e.loc[i, 'pos']) for i in o[:k5]} - bad_shot)
    return {'① 사건 05:55~05:57': [pos[pd.Timestamp(t)] for t in EVENT],
            '② 사건 전 가스 05:45·49·53·54': [pos[pd.Timestamp(t)] for t in PRE_GAS],
            '③ 오경보(정상, 상위 5%)': fa}, e


def main():
    LU, LL, fam = L.load()
    p = 'cn7'
    U, lab = L.for_product(LU, p), L.for_product(LL[p], p)
    ev = L.eval_table(p, lab['TimeStamp'])
    SM = L.slice_masks(U, p)
    CFG = {'(a) 현 제출 cur_0901·c17·LW-Maha': lambda: cfg_a(U, lab, SM),
           '(b) w2_pre·dev26·GMM2': lambda: cfg_b(U, lab, SM),
           '(c) w2_pre·dev26·국소 Maha k=300': lambda: cfg_c(U, lab, SM),
           '(d) w2_all·dv26+rz5+mx5+drift0·AE': lambda: cfg_d(U, lab, SM, fam)}
    rv, rg, meta = [], [], []
    for name, f in CFG.items():
        L.log(name)
        s, T, cols = f()
        T = T / T.sum(1, keepdims=True) if not name.startswith(('(b)', '(d)')) else T
        G, e = shot_groups(ev, lab, s)
        meta.append({'config': name, **{k: v for k, v in L.evaluate(ev, s).items() if k.startswith('off')},
                     'ext_AP(참고)': L.evaluate(ev, s)['ext_AP'], 'n_false_alarm_shots': len(G['③ 오경보(정상, 상위 5%)'])})
        for gname, idx in G.items():
            r = pd.Series(T[idx].mean(0), index=cols)
            for c, v in r.items():
                rv.append({'config': name, 'shots': gname, 'n_shots': len(idx), 'var': c, 'group': group_of(c),
                           'phys': phys_group(c), 'ratio': v})
    V = pd.DataFrame(rv)
    V.to_csv(L.OUT / 'contrib_vars.csv', index=False, encoding='utf-8-sig')
    Gs = V.groupby(['config', 'shots', 'group'])['ratio'].sum().reset_index()
    Gs.to_csv(L.OUT / 'contrib_groups.csv', index=False, encoding='utf-8-sig')
    V.groupby(['config', 'shots', 'phys'])['ratio'].sum().reset_index().to_csv(
        L.OUT / 'contrib_phys.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(meta).to_csv(L.OUT / 'contrib_meta.csv', index=False, encoding='utf-8-sig')
    print(pd.DataFrame(meta).round(3).to_string(index=False))
    for (c, sh), g in V.groupby(['config', 'shots'], sort=False):
        print(f'\n{c} | {sh} (n={g.n_shots.iloc[0]})')
        print(g.nlargest(5, 'ratio')[['var', 'ratio']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
