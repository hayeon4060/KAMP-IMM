"""
4단계: 최종 후보 (PREREG)
  후보 = 제품별 2단계 공식 AP 상위 5 ∪ 두 기간 lift 기하평균 상위 5 ∪ 3단계 전진 선택 최종 묶음
  각 후보: 원 모델(시드 5 평균) + 같은 묶음·구간 AE·Deep SVDD(시드 5)
  학습 창: main = 후보 구간 / W1 = 후보 구간 ∩ 라벨 시작 이전 / W2 = 후보 구간 − 평가일(두 기간 라벨 날짜)
  비교: R0 = 제출 방식(FinalModel, c17, cur_0901), R1 = w2_all·dev26·gmm2(시드 5) — 같은 창으로 다시 학습
  paired 부트스트랩 ΔAP (층화 행 재표집 500회)
출력: outputs/s4_finalists.csv, outputs/scores/s4_scores.pkl
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import lib as L

SEEDS = range(5)
DEEP = ['ae', 'dsvdd']
N_BOOT = 500
RNG = np.random.default_rng(42)


def families_of(name, FS):
    return FS[name] if name in FS else name.split('+')


def candidates(s2, s3p, FS):
    out = []
    for p in L.C.PRODUCTS:
        a = s2[(s2['product'] == p) & (s2['err'].fillna('') == '')].copy()
        a['gm'] = np.sqrt((a['off_AP'] / a['off_prev']).clip(lower=1e-6) * (a['ext_AP'] / a['ext_prev']).clip(lower=1e-6))
        pick = pd.concat([a.nlargest(5, 'off_AP').assign(why='공식 AP 상위 5'),
                          a.nlargest(5, 'gm').assign(why='lift 기하평균 상위 5')])
        for r in pick.itertuples():
            out.append((p, r.slice, tuple(families_of(r.featset, FS)), r.model, r.why))
        if s3p is not None:
            q = s3p[s3p['product'] == p]
            last = q.loc[q.groupby(['model', 'criterion'])['step'].idxmax()]
            for r in last.itertuples():
                out.append((p, 'w2_all', tuple(r.set.split('+')), r.model, f'3단계 전진 선택({r.criterion})'))
    # 중복 제거 (이유는 합침)
    d = {}
    for p, sl, fl, m, why in out:
        k = (p, sl, tuple(sorted(fl, key=L.FAMILIES.index)), m)
        d[k] = d.get(k, '') + ('; ' if k in d else '') + why
    return d


def windows(U, p, mask):
    t = U['TimeStamp']
    ev_days = set(L.C.eval_rows(p)['TimeStamp'].dt.date)
    pre = (t < L.SL.LAB_START[p]).values
    noday = ~t.dt.date.isin(ev_days).values
    return {'main': mask, 'W1': mask & pre, 'W2': mask & noday}


def run_one(key, Xr, Xl, model, seeds):
    t = time.time()
    if model == 'final_c17':           # 제출 방식 그대로 (FinalModel: c17, LW-Mahalanobis)
        return key, None, time.time() - t
    ss = []
    for sd in seeds:
        res, _ = L.fit_score_all(Xr, Xl, [model], seed=sd)
        ss.append(res[model][0])
    return key, np.nanmean(np.vstack(ss), 0).astype(np.float32), time.time() - t


def ap_vec(y, s):
    o = np.argsort(-s, kind='stable')
    ys = y[o]
    return float((ys * np.cumsum(ys) / np.arange(1, len(y) + 1)).sum() / y.sum())


def boot_delta(ev, per, a, b):
    e = ev[ev['period'] == per]
    y = e['y'].values.astype(float)
    sa, sb = L._tb(a[e['pos'].values]), L._tb(b[e['pos'].values])
    pi, ni = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = []
    for _ in range(N_BOOT):
        i = np.concatenate([RNG.choice(pi, len(pi)), RNG.choice(ni, len(ni))])
        d.append(ap_vec(y[i], sa[i]) - ap_vec(y[i], sb[i]))
    return np.percentile(d, [2.5, 97.5])


def main(n_jobs=15):
    LU, LL, fam = L.load()
    FS = L.featsets(fam)
    s2 = pd.read_csv(L.OUT / 's2_results.csv')
    for per in ('off', 'ext'):
        s2[f'{per}_prev'] = s2['product'].map(
            {p: L.C.eval_rows(p).query(f'period == "{"official" if per == "off" else "extra"}"')['y'].mean()
             for p in L.C.PRODUCTS})
    s3p = pd.read_csv(L.OUT / 's3_path.csv') if (L.OUT / 's3_path.csv').exists() else None
    cands = candidates(s2, s3p, FS)
    L.log(f'후보 {len(cands)}개')
    refs = {}
    tasks, EV, LAB = [], {}, {}
    for p in L.C.PRODUCTS:
        U = L.for_product(LU, p)
        lab = L.for_product(LL[p], p)
        LAB[p], EV[p] = lab, L.eval_table(p, lab['TimeStamp'])
        SM = L.slice_masks(U, p)
        todo = [(p, sl, fl, m, why) for (p2, sl, fl, m), why in cands.items() if p2 == p]
        todo += [(p, 'cur_0901', ('c17',), 'final_c17', 'R0 제출 방식'),
                 (p, 'w2_all', ('dv26',), 'gmm2', 'R1 이전 탐색 최고')]
        for _, sl, fl, m, why in todo:
            cols = L.cols_of(fam, list(fl))
            for wn, wm in windows(U, p, SM[sl]).items():
                ref = U[wm]
                if len(ref) < L.MIN_REF:
                    continue
                if len(ref) > L.REF_MAX:
                    ref = ref.sample(L.REF_MAX, random_state=42)
                models = [m] + ([d for d in DEEP] if wn in ('main', 'W1') and m != 'final_c17' else [])
                for mm in models:
                    key = (p, sl, '+'.join(fl), mm, wn)
                    refs[key] = (why if mm == m else f'{why} → 같은 묶음 {mm}', len(ref))
                    if mm == 'final_c17':
                        mod = L.C.fm.FinalModel(L.C.FEAT_C17).fit(ref)
                        refs[key] += (mod.score(lab).astype(np.float32),)
                        continue
                    seeds = SEEDS if mm in ('gmm2', 'gmm4', 'ae', 'dsvdd') else [0]
                    tasks.append((key, ref[cols].values.astype(np.float32), lab[cols].values.astype(np.float32),
                                  mm, list(seeds)))
    tasks.sort(key=lambda t: t[3] not in DEEP)
    L.log(f'4단계 작업 {len(tasks)}개')
    S = {k: v[2] for k, v in refs.items() if len(v) == 3}
    for i, (key, s, sec) in enumerate(Parallel(n_jobs=n_jobs, return_as='generator_unordered', max_nbytes='1M')(
            delayed(run_one)(*t) for t in tasks), 1):
        S[key] = s
        if i % 20 == 0:
            L.log(f'{i}/{len(tasks)} 최근 {key} ({sec:.0f}s)')
    pd.to_pickle(S, L.SCORES / 's4_scores.pkl')
    rows = []
    for key, s in S.items():
        p, sl, fs, mm, wn = key
        r = {'product': p, 'slice': sl, 'featset': fs, 'model': mm, 'window': wn, 'why': refs[key][0],
             'n_ref': refs[key][1], **L.evaluate(EV[p], s)}
        rows.append(r)
    R = pd.DataFrame(rows)
    # 대조 (같은 창): R0, R1
    for p in L.C.PRODUCTS:
        for wn in ('main', 'W1', 'W2'):
            for tag, rk in (('R0', (p, 'cur_0901', 'c17', 'final_c17', wn)), ('R1', (p, 'w2_all', 'dv26', 'gmm2', wn))):
                if rk not in S:
                    continue
                m = (R['product'] == p) & (R['window'] == wn)
                R.loc[m, f'{tag}_off_AP'] = R.loc[(R['product'] == p) & (R['window'] == wn) &
                                                  (R['slice'] == rk[1]) & (R['featset'] == rk[2]) &
                                                  (R['model'] == rk[3]), 'off_AP'].iloc[0]
                if wn != 'main':
                    continue
                for i in R.index[m]:
                    k = tuple(R.loc[i, ['product', 'slice', 'featset', 'model', 'window']])
                    for per, pk in (('official', 'off'), ('extra', 'ext')):
                        lo, hi = boot_delta(EV[p], per, S[k], S[rk])
                        R.loc[i, f'd{pk}_{tag}_lo'], R.loc[i, f'd{pk}_{tag}_hi'] = lo, hi
    R.to_csv(L.OUT / 's4_finalists.csv', index=False, encoding='utf-8-sig')
    L.log('완료')


if __name__ == '__main__':
    main()
