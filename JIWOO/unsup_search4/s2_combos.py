"""
2단계: 조합 K1~K6 × 모델 4 × 제품 2, 학습 창 W1 (PREREG). K7(앙상블)은 K3 백분위로 계산.
출력: outputs/s2_results.csv, outputs/s2_combos.json, outputs/scores/s2_{p}.npz (점수·백분위)
"""
import sys
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import lib4 as L

SEED = 2026
SC = L.OUT / 'scores'
SC.mkdir(exist_ok=True)


def make_combos(B, U, p):
    rng = np.random.default_rng(SEED)
    names = sorted(B)
    combos = []
    add = lambda kind, bl: combos.append({'kind': kind, 'blocks': sorted(bl)})   # noqa: E731
    for b in names:
        add('K1', [b])
    seen = set()
    while len(seen) < 2000:
        pair = tuple(sorted(rng.choice(names, 2, replace=False)))
        if pair not in seen:
            seen.add(pair)
            add('K2', list(pair))
    for k in (3, 5, 8, 12):
        seen = set()
        while len(seen) < 500:
            s = tuple(sorted(rng.choice(names, k, replace=False)))
            if s not in seen:
                seen.add(s)
                add(f'K3_{k}', list(s))
    trs = sorted({b.split('|')[0] for b in names})
    grps = sorted({b.split('|')[1] for b in names})
    for t in trs:
        add('K4_tr', [b for b in names if b.split('|')[0] == t])
    for g in grps:
        add('K4_grp', [b for b in names if b.split('|')[1] == g])
    for b in names:
        if b not in L.DV26:
            add('K5', L.DV26 + [b])
    return combos


def k6_columns(p, cols):
    """라벨 미사용 선택 (W1 학습 데이터만)"""
    X = np.load(L.mm_path(p, 'W1'), mmap_mode='r')
    X = np.asarray(X, float)
    n = len(X)
    med_all = np.nanmedian(X, 0)
    q75, q25 = np.nanpercentile(X, [75, 25], 0)
    iqr = q75 - q25
    ok = (iqr > 1e-9) & (np.isfinite(X).mean(0) > 0.5)
    h1, h2 = np.nanmedian(X[: n // 2], 0), np.nanmedian(X[n // 2:], 0)
    stab = np.where(ok, np.abs(h1 - h2) / np.where(iqr > 1e-9, iqr, 1), np.inf)
    order = [i for i in np.argsort(stab, kind='stable') if np.isfinite(stab[i])]
    sel = {}
    for k in (20, 50, 100):
        sel[f'K6_stab{k}'] = order[:k]
    Z = np.nan_to_num((X - med_all) / np.where(iqr > 1e-9, iqr, 1), nan=0.0)
    Z = np.clip(Z, -20, 20)
    chosen = []
    for i in order:
        if len(chosen) >= 100:
            break
        if not chosen or np.max(np.abs(np.corrcoef(Z[:, [i] + chosen].T)[0, 1:])) < 0.8:
            chosen.append(i)
    for k in (20, 50, 100):
        sel[f'K6_nored{k}'] = chosen[:k]
    sel['K6_pca_all'] = [i for i in range(len(cols)) if ok[i]]
    return sel


def job(p, cid, idx):
    t = time.time()
    res, nc = L.fit_score(p, 'W1', idx)
    return p, cid, res, nc, time.time() - t


def main(n_jobs=15):
    U, LBL, cols = L.load()
    B = L.blocks(cols)
    EV, allc, tasks = {}, {}, []
    for p in L.C.PRODUCTS:
        L.prepare_memmaps(U, LBL, p, ('W1',))
        EV[p] = L.eval_table(p, LBL)
        combos = make_combos(B, U, p)
        for k, idx in k6_columns(p, cols).items():
            combos.append({'kind': k, 'blocks': [], 'cols': [int(i) for i in idx]})
        for i, c in enumerate(combos):
            c['id'] = i
            idx = c.get('cols') or sorted(j for b in c['blocks'] for j in B[b])
            c['n_cols'] = len(idx)
            tasks.append((p, i, idx))
        allc[p] = combos
    json.dump(allc, open(L.OUT / 's2_combos.json', 'w', encoding='utf-8'), ensure_ascii=False)
    L.log(f'2단계 작업 {len(tasks)}개 × 모델 {len(L.MODELS)}')
    tasks.sort(key=lambda t: -len(t[2]))
    S = {p: {} for p in L.C.PRODUCTS}
    rows = []
    t0 = time.time()
    gen = Parallel(n_jobs=n_jobs, return_as='generator_unordered')(delayed(job)(*t) for t in tasks)
    for i, (p, cid, res, nc, sec) in enumerate(gen, 1):
        c = allc[p][cid]
        for m, (s, pct, err) in res.items():
            r = {'product': p, 'id': cid, 'kind': c['kind'], 'blocks': '+'.join(c['blocks']), 'n_cols': c['n_cols'],
                 'n_used': nc, 'model': m, 'err': err, 'sec': round(sec, 1)}
            if s is not None and np.isfinite(s).any():
                r.update(L.evaluate(EV[p], s))
                S[p][(cid, m)] = (s, pct)
            rows.append(r)
        if i % 200 == 0 or i == len(tasks):
            el = time.time() - t0
            pd.DataFrame(rows).to_csv(L.OUT / 's2_results.csv', index=False, encoding='utf-8-sig')
            L.log(f'{i}/{len(tasks)} ({el / 60:.0f}분, 남은 예상 {el / i * (len(tasks) - i) / 60:.0f}분)')
    for p in L.C.PRODUCTS:
        keys = sorted(S[p])
        np.savez(SC / f's2_{p}.npz', keys=np.array([f'{c}|{m}' for c, m in keys]),
                 s=np.vstack([S[p][k][0] for k in keys]),
                 pct=np.vstack([S[p][k][1] if S[p][k][1] is not None else np.full_like(S[p][k][0], np.nan)
                                for k in keys]))
    # K7: 무작위 하위공간 앙상블 (K3 조합 50개 백분위 평균, 10세트, 모델별)
    rng = np.random.default_rng(SEED + 7)
    for p in L.C.PRODUCTS:
        k3 = [c['id'] for c in allc[p] if c['kind'].startswith('K3_')]
        sets = rng.permutation(k3)[:500].reshape(10, 50)
        for j, st in enumerate(sets):
            for m in L.MODELS:
                P = [S[p][(int(c), m)][1] for c in st if (int(c), m) in S[p]]
                if not P:
                    continue
                s = np.mean(P, 0)
                r = {'product': p, 'id': f'K7_{j}', 'kind': 'K7', 'blocks': '|'.join(map(str, st)), 'model': m,
                     'n_members': len(P), **L.evaluate(EV[p], s)}
                rows.append(r)
                S[p][(f'K7_{j}', m)] = (s.astype(np.float32), None)
        np.savez(SC / f's2k7_{p}.npz', keys=np.array([f'{c}|{m}' for c, m in S[p] if str(c).startswith('K7')]),
                 s=np.vstack([v[0] for k, v in S[p].items() if str(k[0]).startswith('K7')]))
    pd.DataFrame(rows).to_csv(L.OUT / 's2_results.csv', index=False, encoding='utf-8-sig')
    L.log('완료')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
