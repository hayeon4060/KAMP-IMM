"""
feats4.py — 대규모 파생변수 라이브러리 (PREREG 표): 기준 신호 약 200 × 변환 22(쌍 신호는 6) + ctx
범위: 우진2 CN7·RG3 + 같은 제품 다른 설비 (비라벨), 라벨 쇼트. 모든 변환은 같은 Run의 과거(+현재)만.
열 이름: '{변환}|{신호 묶음}|{신호}'  → 블록 = (변환, 신호 묶음)
출력: outputs/cache/lib4_unlabeled.pkl (메타 + float32 행렬), lib4_labeled.pkl, columns.json
"""
import os
import sys
import json
import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'unsup_search'))
sys.path.insert(0, str(ROOT.parent / 'unsup_search3'))
import common as C  # noqa: E402
import feats as F3  # noqa: E402

OUT = ROOT / 'outputs'
CACHE = OUT / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
raw = C.raw

G_RAW = {'시간': ['Injection_Time', 'Filling_Time', 'Plasticizing_Time', 'Cycle_Time', 'Clamp_Close_Time'],
         '속도': ['Max_Injection_Speed', 'Max_Screw_RPM'],
         '압력': ['Max_Injection_Pressure', 'Max_Switch_Over_Pressure', 'Average_Back_Pressure'],
         '위치': ['Cushion_Position', 'Plasticizing_Position'],
         '배럴': [f'Barrel_Temperature_{i}' for i in range(1, 7)],
         '호퍼금형': ['Hopper_Temperature', 'Mold_Temperature_3', 'Mold_Temperature_4']}
TEMPS = G_RAW['배럴'] + G_RAW['호퍼금형']
TR_ALL = ['raw', 'dv', 'rz5', 'rz50', 'rz200', 'd1', 'd2', 'ew2', 'ew5', 'ew20', 'vol10', 'mx3', 'mx5', 'mx10',
          'lag1', 'lag2', 'drift0', 'rmed', 'cup', 'cum', 'slope20']
TR_PAIR = ['raw', 'dv', 'rz50', 'd1', 'ew5', 'mx5']


def signals(d):
    """d: *_raw (+ x3 원값). 반환: {신호 묶음: DataFrame}, {신호: step}"""
    x = {v: d[raw(v)].astype(float) for v in C.ALL_RAW}
    x.update({v: d[f'{v}_raw'].astype(float) for v in F3.X3})
    pos = lambda s: s.where(s > 0)                                          # noqa: E731
    lg = lambda s: np.log(pos(s))                                           # noqa: E731
    S, step = {}, {}
    for g, vs in G_RAW.items():
        S[g] = pd.DataFrame({v: x[v] for v in vs})
        step.update({v: C.step_of(v) for v in vs})
    hold = x['Injection_Time'] - x['Filling_Time']
    S['관계R'] = pd.DataFrame({
        'R1': lg(x['Filling_Time']) + lg(x['Max_Injection_Speed']),
        'R2': lg(x['Plasticizing_Time']) + lg(x['Max_Screw_RPM']) - lg(x['Plasticizing_Position']),
        'R3': hold, 'R4': x['Max_Switch_Over_Pressure'] / pos(x['Max_Injection_Pressure']),
        'R5': x['Cycle_Time'] - x['Injection_Time'] - x['Plasticizing_Time'] - x['Clamp_Close_Time']})
    step.update({'R1': 1e-3, 'R2': 1e-3, 'R3': 0.01, 'R4': 1e-3, 'R5': 0.01})
    n = {'n1': lg(x['Max_Injection_Pressure']) + lg(x['Filling_Time']),
         'n2': lg(x['Max_Switch_Over_Pressure']) + lg(hold),
         'n3': lg(x['Max_Injection_Pressure']) - lg(x['Max_Injection_Speed']),
         'n4': lg(x['Max_Screw_RPM']) + lg(x['Plasticizing_Time']),
         'n5': lg(x['Average_Back_Pressure']) + lg(x['Plasticizing_Time'])}
    for i, (a, b) in enumerate([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], start=6):
        n[f'n{i}'] = x[f'Barrel_Temperature_{a}'] - x[f'Barrel_Temperature_{b}']
    n['n11'] = x['Mold_Temperature_3'] - x['Mold_Temperature_4']
    n['n12'] = x['Hopper_Temperature'] - x['Barrel_Temperature_6']
    S['관계n'] = pd.DataFrame(n)
    step.update({k: (1e-3 if i < 5 else 0.1) for i, k in enumerate(n)})
    S['x3'] = pd.DataFrame({v: x[v] for v in F3.X3})
    step.update(F3.X3_STEP)
    shot = x['Plasticizing_Position'] - x['Cushion_Position']
    bar = pd.concat([x[f'Barrel_Temperature_{i}'] for i in range(1, 7)], axis=1)
    ph = {'plast_speed': x['Plasticizing_Position'] / pos(x['Plasticizing_Time']),
          'shot_vol': shot,
          'inj_mean_speed': shot / pos(x['Filling_Time']),
          'shear_proxy': x['Max_Screw_RPM'] * x['Plasticizing_Time'],
          'back_work': x['Average_Back_Pressure'] * x['Plasticizing_Time'],
          'press_ratio': x['Average_Back_Pressure'] / pos(x['Max_Injection_Pressure']),
          'hold_ratio': hold / pos(x['Injection_Time']),
          'fill_press_grad': x['Max_Injection_Pressure'] / pos(x['Filling_Time']),
          'cushion_ratio': x['Cushion_Position'] / pos(x['Plasticizing_Position']),
          'barrel_mean': bar.mean(axis=1), 'barrel_slope': x['Barrel_Temperature_1'] - x['Barrel_Temperature_6']}
    for k in range(1, 5):
        ph[f'barrel_curv{k}'] = (x[f'Barrel_Temperature_{k}'] - 2 * x[f'Barrel_Temperature_{k + 1}']
                                 + x[f'Barrel_Temperature_{k + 2}'])
    ph['mold_mean'] = (x['Mold_Temperature_3'] + x['Mold_Temperature_4']) / 2
    ph['mold_minus_barrel'] = ph['mold_mean'] - ph['barrel_mean']
    ph['hopper_minus_b1'] = x['Hopper_Temperature'] - x['Barrel_Temperature_1']
    S['물리'] = pd.DataFrame(ph)
    for k in ph:
        step[k] = 0.1 if k.startswith(('barrel', 'mold', 'hopper')) else None   # None → 상대 단위(아래)
    proc = G_RAW['시간'] + G_RAW['속도'] + G_RAW['압력'] + G_RAW['위치']
    pv = {v: x[v] for v in proc}
    pv.update({k: ph[k] for k in ('plast_speed', 'shot_vol', 'shear_proxy')})
    S['쌍공정'] = pd.DataFrame({f'{a}/{b}': lg(pv[a]) - lg(pv[b]) for a, b in itertools.combinations(pv, 2)})
    step.update({c: 1e-3 for c in S['쌍공정']})
    S['쌍온도'] = pd.DataFrame({f'{a}-{b}': x[a] - x[b] for a, b in itertools.combinations(TEMPS, 2)})
    step.update({c: 0.1 for c in S['쌍온도']})
    for g in S:
        S[g] = S[g].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    return S, step


# ====================================================================== 변환
def _dv_cols(vals, starts, steps):
    """인과 강건 편차, 여러 신호 (열마다 기존 함수, Run 경계 = starts)"""
    out = np.full(vals.shape, np.nan)
    bounds = list(starts) + [len(vals)]
    for j in range(vals.shape[1]):
        for a, b in zip(bounds[:-1], bounds[1:]):
            out[a:b, j] = C.fm.causal_robust_dev(vals[a:b, j], steps[j])
    return out


def cusum(dv, starts, k=0.5):
    up, dn = np.zeros_like(dv), np.zeros_like(dv)
    sp, sn = np.zeros(dv.shape[1]), np.zeros(dv.shape[1])
    st = np.zeros(len(dv), bool)
    st[list(starts)] = True
    z = np.nan_to_num(dv, nan=0.0)
    for i in range(len(dv)):
        if st[i]:
            sp[:], sn[:] = 0.0, 0.0
        sp = np.maximum(0.0, sp + z[i] - k)
        sn = np.maximum(0.0, sn - z[i] - k)
        up[i], dn[i] = sp, sn
    return up, dn


def transform(X, gid, starts, steps, trs, n_jobs):
    """X: 한 신호 묶음 (행 = Run→시간 정렬). 반환 {변환: DataFrame}"""
    g = X.groupby(gid, sort=False)
    out = {}
    stepv = np.array(steps, float)

    def roll(frame, w, mp, fn, shift=False):
        f = frame.groupby(gid, sort=False).shift(1) if shift else frame
        r = getattr(f.groupby(gid, sort=False).rolling(w, min_periods=mp), fn[0])(*fn[1:])
        return r.reset_index(level=0, drop=True).sort_index()

    if 'raw' in trs:
        out['raw'] = X
    # dv (반복문, 열 묶음 병렬)
    chunks = np.array_split(np.arange(X.shape[1]), min(n_jobs, X.shape[1]))
    res = Parallel(n_jobs=n_jobs)(delayed(_dv_cols)(X.values[:, c], starts, stepv[c]) for c in chunks if len(c))
    dv = pd.DataFrame(np.hstack(res), columns=X.columns)
    dv = dv[X.columns] if list(dv.columns) == list(X.columns) else dv
    if 'dv' in trs:
        out['dv'] = dv
    for w, mp in ((5, 5), (50, 10), (200, 20)):
        if f'rz{w}' in trs:
            med = roll(X, w, mp, ('median',), shift=True)
            iqr = roll(X, w, mp, ('quantile', 0.75), shift=True) - roll(X, w, mp, ('quantile', 0.25), shift=True)
            out[f'rz{w}'] = (X - med) / np.maximum(iqr.values / 1.349, stepv)
    d1 = X - g.shift(1)
    if 'd1' in trs:
        out['d1'] = d1
    if 'd2' in trs:
        out['d2'] = d1 - d1.groupby(gid, sort=False).shift(1)
    for hl in (2, 5, 20):
        if f'ew{hl}' in trs:
            e = X.groupby(gid, sort=False).ewm(halflife=hl, min_periods=3).mean().reset_index(level=0, drop=True)
            out[f'ew{hl}'] = X - e.sort_index().groupby(gid, sort=False).shift(1)
    if 'vol10' in trs:
        out['vol10'] = roll(d1, 10, 5, ('std',))
    A = dv.abs()
    for k in (3, 5, 10):
        if f'mx{k}' in trs:
            out[f'mx{k}'] = roll(A, k, 1, ('max',))
    L = dv.groupby(gid, sort=False)
    for k in (1, 2):
        if f'lag{k}' in trs:
            out[f'lag{k}'] = L.shift(k)
    if 'drift0' in trs:
        em = g.expanding().mean().reset_index(level=0, drop=True).sort_index()
        seq = X.groupby(gid, sort=False).cumcount().values + 1
        first5 = em.where(np.repeat((seq <= 5)[:, None], X.shape[1], 1)).groupby(gid, sort=False).ffill()
        out['drift0'] = X - first5
    if 'rmed' in trs:
        xm = X.groupby(gid, sort=False).shift(1).groupby(gid, sort=False).expanding(min_periods=5).median()
        out['rmed'] = X - xm.reset_index(level=0, drop=True).sort_index()
    if 'cup' in trs or 'cum' in trs:
        up, dn = cusum(dv.values, starts)
        out['cup'], out['cum'] = pd.DataFrame(up, columns=X.columns), pd.DataFrame(dn, columns=X.columns)
    if 'slope20' in trs:                                   # 최근 20쇼트(현재 포함) 최소제곱 기울기, 최소 5
        xf = X.groupby(gid, sort=False).ffill().groupby(gid, sort=False).bfill()
        t = pd.Series(X.groupby(gid, sort=False).cumcount().values.astype(float), index=X.index)
        T = pd.DataFrame(np.repeat(t.values[:, None], X.shape[1], 1), columns=X.columns)
        n = roll(xf.notna().astype(float), 20, 5, ('sum',))
        st, stt = roll(T, 20, 5, ('sum',)), roll(T * T, 20, 5, ('sum',))
        sx, stx = roll(xf, 20, 5, ('sum',)), roll(T * xf, 20, 5, ('sum',))
        den = n * stt - st * st
        out['slope20'] = ((n * stx - st * sx) / den.where(den > 0))
    return out


def build(d, n_jobs=15):
    """d: 쇼트 표 (run_id, TimeStamp, seq_in_run, *_raw, x3 원값, dv26 열 등). 반환 (meta, 행렬, 열 이름)"""
    d = d.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
    gid = d['run_id']
    starts = np.flatnonzero(gid.ne(gid.shift()).values)
    S, step = signals(d)
    cols, mats = [], []
    for grp, X in S.items():
        trs = TR_PAIR if grp in ('쌍공정', '쌍온도') else TR_ALL
        stp = []
        for c in X.columns:
            s = step.get(c)
            if s is None:                                    # 상대 기록 단위: 중앙값 절대값의 0.1%
                s = max(1e-6, 1e-3 * float(np.nanmedian(np.abs(X[c].values))))
            stp.append(s)
        t0 = time.time()
        out = transform(X, gid, starts, stp, trs, n_jobs)
        for tr in trs:
            M = out[tr]
            mats.append(M.values.astype(np.float32))
            cols += [f'{tr}|{grp}|{c}' for c in X.columns]
        C.log(f'  {grp}: 신호 {X.shape[1]} × 변환 {len(trs)} ({time.time() - t0:.0f}s)')
    # ctx (unsup_search3과 같은 정의)
    t = d['TimeStamp']
    gap = t.groupby(gid, sort=False).diff().dt.total_seconds()
    big = (d[F3.DV26].abs().max(axis=1) > 5).astype(float)
    ctx = pd.DataFrame({
        'logseq': np.log1p(d['seq_in_run'].astype(float)), 'loggap': np.log1p(gap.clip(lower=0)),
        'gap_minus_cycle': gap - d[raw('Cycle_Time')],
        'min_since_start': (t - t.groupby(gid, sort=False).transform('min')).dt.total_seconds() / 60,
        'frac_big': big.groupby(gid, sort=False).cumsum() / d.groupby(gid, sort=False).cumcount().add(1)})
    mats.append(ctx.values.astype(np.float32))
    cols += [f'ctx|ctx|{c}' for c in ctx.columns]
    M = np.hstack(mats)
    M[~np.isfinite(M)] = np.nan
    meta = d[[c for c in F3.META if c in d.columns]].reset_index(drop=True)
    return meta, M, cols


def main():
    C.log('비라벨 캐시 (CN7·RG3 흐름) + 추가 원값 3개')
    U = C.build_cache()
    U = U[U['prod'].isin(['CN7', 'RG3'])].copy()
    U = U.merge(F3._x3_table(C.DATA / 'unlabeled_data.csv'), on=['equip', 'prod', 'TimeStamp'], how='left')
    C.log(f'비라벨 {len(U):,}쇼트 계산')
    meta, M, cols = build(U)
    pd.to_pickle({'meta': meta, 'X': M}, CACHE / 'lib4_unlabeled.pkl')
    json.dump(cols, open(CACHE / 'columns.json', 'w', encoding='utf-8'), ensure_ascii=False)
    C.log(f'비라벨 라이브러리 {M.shape} ({M.nbytes / 1e9:.2f} GB)')
    LAB = C.build_labeled_feats()
    xl = F3._x3_table(C.DATA / 'labeled_data.csv', C.EQUIP_W2)
    out = {}
    for p in C.PRODUCTS:
        d = LAB[p].copy()
        d['equip'], d['prod'] = C.EQUIP_W2, p.upper()
        d = d.merge(xl, on=['equip', 'prod', 'TimeStamp'], how='left')
        m, X, c2 = build(d)
        assert c2 == cols
        out[p] = {'meta': m, 'X': X}
    pd.to_pickle(out, CACHE / 'lib4_labeled.pkl')
    C.log('완료')


if __name__ == '__main__':
    main()
