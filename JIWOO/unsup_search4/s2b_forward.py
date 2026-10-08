"""
K8: 블록 전진 선택 (PREREG, 명시적 과적합 대조군) — W1 학습, dv26 기준에서 시작, 공식 AP 기준, 최대 8단계, GMM2·LW
출력: outputs/s2b_steps.csv, outputs/s2b_path.csv, outputs/scores/s2b_{p}.npz
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import lib4 as L

MODELS8 = ['gmm2', 'maha_lw']


def job(p, key, idx, m):
    res, _ = L.fit_score(p, 'W1', idx, [m])
    return key, res[m][0]


def main(n_jobs=15):
    U, LBL, cols = L.load()
    B = L.blocks(cols)
    names = sorted(B)
    tries, path, store = [], [], {}
    for p in L.C.PRODUCTS:
        ev = L.eval_table(p, LBL)
        for m in MODELS8:
            cur = list(L.DV26)

            def run(sets):
                todo = [s for s in sets if (p, m, tuple(s)) not in store]
                out = Parallel(n_jobs=n_jobs)(delayed(job)(p, (p, m, tuple(s)), sorted(j for b in s for j in B[b]), m)
                                              for s in todo)
                for k, s in out:
                    store[k] = s
                return {tuple(s): L.evaluate(ev, store[(p, m, tuple(s))]) for s in sets}

            base = run([cur])[tuple(cur)]
            path.append({'product': p, 'model': m, 'step': 0, 'added': 'dv26', 'set': '+'.join(cur), **base})
            for step in range(1, 9):
                cands = [b for b in names if b not in cur]
                sets = [sorted(cur + [b]) for b in cands]
                ev_s = run(sets)
                for b, s in zip(cands, sets):
                    tries.append({'product': p, 'model': m, 'step': step, 'added': b, **ev_s[tuple(s)]})
                bb, bs = max(zip(cands, sets), key=lambda x: ev_s[tuple(x[1])]['off_AP'])
                if ev_s[tuple(bs)]['off_AP'] <= base['off_AP'] + 1e-9:
                    break
                cur, base = bs, ev_s[tuple(bs)]
                path.append({'product': p, 'model': m, 'step': step, 'added': bb, 'set': '+'.join(cur), **base})
            L.log(f'{p} {m}: {path[-1]["set"]} → off {base["off_AP"]:.3f} ext {base["ext_AP"]:.3f}')
            pd.DataFrame(tries).to_csv(L.OUT / 's2b_steps.csv', index=False, encoding='utf-8-sig')
            pd.DataFrame(path).to_csv(L.OUT / 's2b_path.csv', index=False, encoding='utf-8-sig')
    for p in L.C.PRODUCTS:
        ks = [k for k in store if k[0] == p]
        np.savez(L.OUT / 'scores' / f's2b_{p}.npz', keys=np.array([f'K8|{k[1]}|{"+".join(k[2])}' for k in ks]),
                 s=np.vstack([store[k] for k in ks]))
    L.log('완료')


if __name__ == '__main__':
    main()
