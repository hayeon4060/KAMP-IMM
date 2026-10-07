"""
4b: 통과 후보 중 신경망(AE) 2개의 W2(전체 − 평가일) 재학습 — PREREG W2 확인 누락분 보완 (시드 5)
출력: outputs/s4b_w2_deep.csv
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import lib as L
import s4_finalists as S4

CANDS = [('w2_all', ['dv26', 'rz5', 'mx5', 'drift0']), ('eq_all', ['dv26', 'mx5'])]


def main():
    LU, LL, fam = L.load()
    p = 'cn7'
    U, lab = L.for_product(LU, p), L.for_product(LL[p], p)
    ev = L.eval_table(p, lab['TimeStamp'])
    SM = L.slice_masks(U, p)
    tasks = []
    for sl, fl in CANDS:
        cols = L.cols_of(fam, fl)
        ref = U[S4.windows(U, p, SM[sl])['W2']]
        ref = ref.sample(L.REF_MAX, random_state=42) if len(ref) > L.REF_MAX else ref
        for sd in S4.SEEDS:
            tasks.append(((sl, '+'.join(fl), sd, len(ref)), ref[cols].values.astype(np.float32),
                          lab[cols].values.astype(np.float32)))
    out = Parallel(n_jobs=10)(delayed(lambda k, a, b: (k, L.fit_score_all(a, b, ['ae'], seed=k[2])[0]['ae'][0]))(*t)
                              for t in tasks)
    rows = []
    for sl, fs in {(k[0], k[1]) for k, _ in out}:
        s = np.mean([v for k, v in out if (k[0], k[1]) == (sl, fs)], 0)
        n = [k[3] for k, _ in out if k[0] == sl][0]
        rows.append({'product': p, 'slice': sl, 'featset': fs, 'model': 'ae', 'window': 'W2', 'n_ref': n,
                     **L.evaluate(ev, s)})
    R = pd.DataFrame(rows)
    R.to_csv(L.OUT / 's4b_w2_deep.csv', index=False, encoding='utf-8-sig')
    print(R.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
