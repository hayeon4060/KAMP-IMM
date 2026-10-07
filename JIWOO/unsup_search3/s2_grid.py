"""
2단계: 변수 묶음 35 × 학습 구간 15 × 모델 10 × 제품 2 = 10,500 설정 (PREREG)
작업 단위 = (제품, 구간, 묶음) → 모델 10종을 한 번에. 중간 저장 → 재실행 시 이어서.
출력: outputs/s2_results.csv, outputs/scores/s2_scores.pkl
"""
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import lib as L

OUTP = L.OUT / 's2_results.csv'
SCP = L.SCORES / 's2_scores.pkl'


def job(p, sname, fs, Xr, Xl, n_ref):
    t = time.time()
    res, n_col = L.fit_score_all(Xr, Xl)
    return (p, sname, fs), res, n_ref, n_col, time.time() - t


def main(n_jobs=15):
    LU, LL, fam = L.load()
    FS = L.featsets(fam)
    done = pd.read_pickle(SCP) if SCP.exists() else {}
    rows = pd.read_csv(OUTP).to_dict('records') if OUTP.exists() else []
    tasks, EV = [], {}
    for p in L.C.PRODUCTS:
        U = L.for_product(LU, p)
        lab = L.for_product(LL[p], p)
        EV[p] = L.eval_table(p, lab['TimeStamp'])
        for sname, m in L.slice_masks(U, p).items():
            n = int(m.sum())
            if n < L.MIN_REF:
                L.log(f'{p} {sname}: {n}샷 < {L.MIN_REF} → 제외')
                continue
            ref = U[m]
            if len(ref) > L.REF_MAX:
                ref = ref.sample(L.REF_MAX, random_state=42)
            for fs, fl in FS.items():
                if (p, sname, fs, L.MODELS[-1]) in done:
                    continue
                cols = L.cols_of(fam, fl)
                tasks.append((p, sname, fs, ref[cols].values.astype(np.float32),
                              lab[cols].values.astype(np.float32), n))
    tasks.sort(key=lambda t: -(t[3].shape[0] * t[3].shape[1] ** 1.5))      # 큰 작업 먼저
    L.log(f'2단계 작업 {len(tasks)}개 × 모델 {len(L.MODELS)} (완료 {len(done)}개 설정 건너뜀)')
    gen = Parallel(n_jobs=n_jobs, return_as='generator_unordered', max_nbytes='1M')(
        delayed(job)(*t) for t in tasks)
    t0 = time.time()
    for i, ((p, sname, fs), res, n_ref, n_col, sec) in enumerate(gen, 1):
        for m, (s, err) in res.items():
            done[(p, sname, fs, m)] = s
            r = {'product': p, 'slice': sname, 'featset': fs, 'model': m, 'n_ref': n_ref, 'n_col': n_col,
                 'job_sec': round(sec, 1), 'err': err}
            if not err and np.isfinite(s).any():
                r.update(L.evaluate(EV[p], s))
            rows.append(r)
        if i % 20 == 0 or i == len(tasks):
            pd.DataFrame(rows).to_csv(OUTP, index=False, encoding='utf-8-sig')
            pd.to_pickle(done, SCP)
            el = time.time() - t0
            best = max((r for r in rows[-len(res):]), key=lambda r: r.get('off_AP', -1))
            L.log(f'{i}/{len(tasks)} ({el / 60:.0f}분, 남은 예상 {el / i * (len(tasks) - i) / 60:.0f}분) '
                  f'최근 {p} {sname} {fs} 최고 {best["model"]} off={best.get("off_AP", np.nan):.3f} '
                  f'ext={best.get("ext_AP", np.nan):.3f} ({sec:.0f}s)')
    L.log('완료')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
