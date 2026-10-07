"""
3단계: 계열 전진 선택 (PREREG) — 구간 w2_all, 모델 gmm2·maha_lw·knn5
  dv26에서 시작해 계열을 하나씩 더함. 기준 A = 공식 AP로 고름 / 기준 B = 추가 AP로 고름. 오르지 않으면 정지.
출력: outputs/s3_steps.csv (모든 시도), outputs/s3_path.csv (선택 경로), outputs/scores/s3_scores.pkl
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import lib as L

MODELS3 = ['gmm2', 'maha_lw', 'knn5']
SLICE3 = 'w2_all'


def job(key, Xr, Xl, model):
    res, _ = L.fit_score_all(Xr, Xl, [model])
    return key, res[model][0]


def main(n_jobs=15):
    LU, LL, fam = L.load()
    cache, tries, path = {}, [], []
    for p in L.C.PRODUCTS:
        U = L.for_product(LU, p)
        lab = L.for_product(LL[p], p)
        ev = L.eval_table(p, lab['TimeStamp'])
        ref = U[L.slice_masks(U, p)[SLICE3]]
        if len(ref) > L.REF_MAX:
            ref = ref.sample(L.REF_MAX, random_state=42)

        def score_sets(sets, model):
            todo = [s for s in sets if (p, model, s) not in cache]
            out = Parallel(n_jobs=n_jobs)(
                delayed(job)((p, model, s), ref[L.cols_of(fam, list(s))].values.astype(np.float32),
                             lab[L.cols_of(fam, list(s))].values.astype(np.float32), model) for s in todo)
            for k, s in out:
                cache[k] = s
            return {s: L.evaluate(ev, cache[(p, model, s)]) for s in sets}

        for model in MODELS3:
            for crit, other in (('off_AP', 'ext_AP'), ('ext_AP', 'off_AP')):
                cur = ('dv26',)
                base = score_sets([cur], model)[cur]
                path.append({'product': p, 'model': model, 'criterion': crit, 'step': 0, 'added': 'dv26',
                             'set': '+'.join(cur), 'off_AP': base['off_AP'], 'ext_AP': base['ext_AP']})
                step = 0
                while True:
                    step += 1
                    cands = [f for f in L.FAMILIES if f not in cur]
                    if not cands:
                        break
                    sets = [tuple(sorted(cur + (f,), key=L.FAMILIES.index)) for f in cands]
                    ev_sets = score_sets(sets, model)
                    for f, s in zip(cands, sets):
                        tries.append({'product': p, 'model': model, 'criterion': crit, 'step': step, 'added': f,
                                      'set': '+'.join(s), **ev_sets[s]})
                    bf, bs = max(zip(cands, sets), key=lambda fs: ev_sets[fs[1]][crit])
                    if ev_sets[bs][crit] <= base[crit] + 1e-9:
                        break
                    cur, base = bs, ev_sets[bs]
                    path.append({'product': p, 'model': model, 'criterion': crit, 'step': step, 'added': bf,
                                 'set': '+'.join(cur), 'off_AP': base['off_AP'], 'ext_AP': base['ext_AP']})
                L.log(f'{p} {model} 기준 {crit}: {"+".join(cur)} → off {base["off_AP"]:.3f} ext {base["ext_AP"]:.3f}')
                pd.DataFrame(tries).to_csv(L.OUT / 's3_steps.csv', index=False, encoding='utf-8-sig')
                pd.DataFrame(path).to_csv(L.OUT / 's3_path.csv', index=False, encoding='utf-8-sig')
    pd.to_pickle({(p, SLICE3, '+'.join(s), m): v for (p, m, s), v in cache.items()}, L.SCORES / 's3_scores.pkl')
    L.log('완료')


if __name__ == '__main__':
    main()
