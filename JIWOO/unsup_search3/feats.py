"""
feats.py — 파생변수 계열 17개 라이브러리 (PREREG 표). 비라벨 캐시와 라벨 샷에 같은 함수 적용.

모든 계열은 같은 Run의 과거(현재 포함) 샷만 사용. 라벨 미사용.
출력: outputs/cache/lib_unlabeled.pkl, outputs/cache/lib_labeled.pkl, outputs/cache/families.json
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'unsup_search'))
sys.path.insert(0, str(ROOT.parent / 'unsup_search2'))
import common as C  # noqa: E402

OUT = ROOT / 'outputs'
CACHE = OUT / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
raw = C.raw

SIG = C.FEAT_RAW + C.REL                     # 신호 26개 (원변수 21 + 관계 5, 원값)
SIG_STEP = {**{raw(v): C.step_of(v) for v in C.ALL_RAW},
            'rel_fill_speed': 1e-3, 'rel_plast': 1e-3, 'rel_hold': 0.01, 'rel_sw_ratio': 1e-3, 'rel_rest': 0.01}
DV26 = C.FEAT_DEV + C.FEAT_REL
X3 = ['Clamp_Open_Position', 'Average_Screw_RPM', 'Max_Back_Pressure']
X3_STEP = {'Clamp_Open_Position': 0.01, 'Average_Screw_RPM': 0.1, 'Max_Back_Pressure': 0.1}


META = ['equip', 'prod', 'run_id', 'TimeStamp', 'seq_in_run', 'PART_NAME', 'lot', 'lot_err', 'split', 'PassOrFail',
        'Reason', 'LH_fail', 'RH_fail', 'dev_flow_res_rg3']


def short(c):
    return c.replace('_raw', '')


# ====================================================================== 추가 원값 3개 (형개위치·평균 RPM·최대 배압)
def _x3_table(path, equip_filter=None):
    U = pd.read_csv(path, usecols=['TimeStamp', 'EQUIP_NAME', 'PART_NAME'] + X3, low_memory=False)
    if equip_filter is not None:
        U = U[U['EQUIP_NAME'] == equip_filter]
    U['prod'] = U['PART_NAME'].astype(str).str[:3].str.strip()
    U['TimeStamp'] = pd.to_datetime(U['TimeStamp'])
    U = U.sort_values('TimeStamp', kind='stable').drop_duplicates(['EQUIP_NAME', 'prod', 'TimeStamp'])
    x10 = U['Average_Screw_RPM'] > 100                     # prep_base.py와 같은 10배 기록 보정
    U.loc[x10, 'Average_Screw_RPM'] = U.loc[x10, 'Average_Screw_RPM'] / 10
    return U.rename(columns={'EQUIP_NAME': 'equip', **{v: f'{v}_raw' for v in X3}})[
        ['equip', 'prod', 'TimeStamp'] + [f'{v}_raw' for v in X3]]


# ====================================================================== 계열 계산
def _dev_loop(d, col, step):
    return C.fm._by_run(d, col, lambda a: C.fm.causal_robust_dev(a, step)).values


def build(d, n_jobs=15):
    """d: run_id, TimeStamp, seq_in_run, *_raw, rel_*, dv26, c17, x3 원값 (Run → 시간 순 정렬)"""
    d = d.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
    gid = d['run_id']
    X = d[SIG].astype(float).replace([np.inf, -np.inf], np.nan)
    g = X.groupby(gid, sort=False)
    out = {}

    def roll(frame, w, mp, fn, shift=False):
        f = frame.groupby(gid, sort=False).shift(1) if shift else frame
        r = getattr(f.groupby(gid, sort=False).rolling(w, min_periods=mp), fn[0])(*fn[1:])
        return r.reset_index(level=0, drop=True).sort_index()

    step = pd.Series(SIG_STEP)[SIG].values
    for w, mp in ((5, 5), (50, 10), (200, 20)):            # rz_w: 직전 w샷 중앙값·IQR z (현재 제외)
        med = roll(X, w, mp, ('median',), shift=True)
        q75 = roll(X, w, mp, ('quantile', 0.75), shift=True)
        q25 = roll(X, w, mp, ('quantile', 0.25), shift=True)
        sc = np.maximum((q75 - q25).values / 1.349, step)
        out[f'rz{w}'] = pd.DataFrame((X.values - med.values) / sc, columns=[f'rz{w}_{short(c)}' for c in SIG])
    d1 = X - g.shift(1)
    out['d1'] = d1.set_axis([f'd1_{short(c)}' for c in SIG], axis=1)
    out['d2'] = (d1 - d1.groupby(gid, sort=False).shift(1)).set_axis([f'd2_{short(c)}' for c in SIG], axis=1)
    ewm = X.groupby(gid, sort=False).ewm(halflife=5, min_periods=3).mean().reset_index(level=0, drop=True).sort_index()
    out['ew'] = (X - ewm.groupby(gid, sort=False).shift(1)).set_axis([f'ew_{short(c)}' for c in SIG], axis=1)
    out['vol10'] = roll(d1, 10, 5, ('std',)).set_axis([f'vol10_{short(c)}' for c in SIG], axis=1)
    A = d[DV26].abs()
    out['mx5'] = roll(A, 5, 1, ('max',)).set_axis([f'mx5_{c}' for c in DV26], axis=1)
    L = d[DV26].groupby(gid, sort=False)
    out['lag'] = pd.concat([L.shift(1).set_axis([f'lag1_{c}' for c in DV26], axis=1),
                            L.shift(2).set_axis([f'lag2_{c}' for c in DV26], axis=1)], axis=1)
    # ctx: Run 맥락
    t = d['TimeStamp']
    gap = t.groupby(gid, sort=False).diff().dt.total_seconds()
    big = (A.max(axis=1) > 5).astype(float)
    out['ctx'] = pd.DataFrame({
        'ctx_logseq': np.log1p(d['seq_in_run'].astype(float)),
        'ctx_loggap': np.log1p(gap.clip(lower=0)),
        'ctx_gap_minus_cycle': gap - d[raw('Cycle_Time')],
        'ctx_min_since_start': (t - t.groupby(gid, sort=False).transform('min')).dt.total_seconds() / 60,
        'ctx_frac_big': big.groupby(gid, sort=False).cumsum() / d.groupby(gid, sort=False).cumcount().add(1),
    })
    # drift0: Run 첫 5샷 평균 대비 (5샷 전까지는 지금까지 평균)
    em = g.expanding().mean().reset_index(level=0, drop=True).sort_index()
    early = np.repeat((d['seq_in_run'].values <= 5)[:, None], em.shape[1], 1)
    first5 = em.where(early).groupby(gid, sort=False).ffill()
    out['drift0'] = (X - first5).set_axis([f'drift0_{short(c)}' for c in SIG], axis=1)
    # rmed: Run 과거 전체(현재 제외) 중앙값 대비
    xm = X.groupby(gid, sort=False).shift(1).groupby(gid, sort=False).expanding(min_periods=5).median()
    xm = xm.reset_index(level=0, drop=True).sort_index()
    out['rmed'] = (X - xm).set_axis([f'rmed_{short(c)}' for c in SIG], axis=1)
    # x3 + n12 (인과 강건 편차 — 기존 반복문 함수, 변수별 병렬)
    import run as U2                                       # unsup_search2: 새 관계 12개 정의
    d = U2.add_new_relations(d)                            # 같은 정렬 → 행 순서 유지
    jobs = [(f'{v}_raw', X3_STEP[v]) for v in X3]
    res = Parallel(n_jobs=n_jobs)(delayed(_dev_loop)(d[['run_id', c]], c, s) for c, s in jobs)
    x3 = {f'x3raw_{v}': d[f'{v}_raw'].values for v in X3}
    x3.update({f'x3dv_{v}': r for v, r in zip(X3, res)})
    out['x3'] = pd.DataFrame(x3)
    out['n12'] = d[U2.NEW].reset_index(drop=True)
    out['c17'] = d[C.FEAT_C17].reset_index(drop=True)
    out['dv26'] = d[DV26].reset_index(drop=True)
    out['raw26'] = X.reset_index(drop=True).set_axis([f'raw_{short(c)}' for c in SIG], axis=1)
    fam = {k: list(v.columns) for k, v in out.items()}
    lib = pd.concat([v.reset_index(drop=True) for v in out.values()], axis=1)
    lib = lib.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    meta = d[[c for c in META if c in d.columns]]
    return pd.concat([meta.reset_index(drop=True), lib], axis=1), fam


def main():
    C.log('비라벨 캐시 + 추가 원값 3개')
    F = C.build_cache()
    xu = _x3_table(C.DATA / 'unlabeled_data.csv')
    F = F.merge(xu, on=['equip', 'prod', 'TimeStamp'], how='left')
    C.log(f'추가 원값 결측률: {F[[f"{v}_raw" for v in X3]].isna().mean().round(4).to_dict()}')
    C.log('비라벨 계열 계산')
    LU, fam = build(F)
    LU.to_pickle(CACHE / 'lib_unlabeled.pkl')
    json.dump(fam, open(CACHE / 'families.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    C.log(f'비라벨 라이브러리: {LU.shape}, 계열 {len(fam)}개, 열 {sum(map(len, fam.values()))}개')
    LAB = C.build_labeled_feats()
    xl = _x3_table(C.DATA / 'labeled_data.csv', C.EQUIP_W2)
    out = {}
    for p in C.PRODUCTS:
        d = LAB[p].copy()
        d['equip'], d['prod'] = C.EQUIP_W2, p.upper()
        d = d.merge(xl, on=['equip', 'prod', 'TimeStamp'], how='left')
        C.log(f'{p} 라벨 추가 원값 결측률: {d[[f"{v}_raw" for v in X3]].isna().mean().round(4).to_dict()}')
        out[p], _ = build(d)
    pd.to_pickle(out, CACHE / 'lib_labeled.pkl')
    C.log({p: v.shape for p, v in out.items()})


if __name__ == '__main__':
    main()
