"""
2단계: 모델 × 변수 세트 탐색 — 1단계 상위 구간에서
  구간: 성격이 다른 9개 (현 제출·전체·이전·직전8주·하반기·전 설비·CN7+RG3·저불량 로트·전 부품)
  변수: c17 / dev26 / raw21 / c17+dev26
  모델: models.REGISTRY 23종 (신경망은 시드 1개, 4단계에서 최종 후보만 시드 3개)
출력: outputs/s2_results.csv, outputs/scores/s2_scores.pkl  (중간 저장 → 재실행 시 이어서)
"""
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import common as C
import slices as SL
import models as M
from s1_slices import for_product

FEATSETS = {'c17': C.FEAT_C17, 'dev26': C.FEAT_DEV + C.FEAT_REL, 'raw21': C.FEAT_RAW,
            'c17dev26': C.FEAT_C17 + [c for c in C.FEAT_DEV + C.FEAT_REL if c not in C.FEAT_C17]}
REF_MAX = 30000
N_TOP = 6
FIXED = ('cur_0901', 'w2_all', 'pr_everything')
OUTP = C.OUT / 's2_results.csv'
SCP = C.OUT / 'scores' / 's2_scores.pkl'


def pick_slices(p):
    """1단계 결과: 같은 제품·우진2 구간끼리는 거의 같은 값 → 성격이 다른 구간 9개로 고정"""
    return ['cur_0901', 'w2_all', 'w2_pre', 'w2_last8w', 'w2_H2', 'eq_all', 'pr_cn7rg3_w2',
            'cl_lowerr_lot', 'pr_everything']


def job(p, sname, fs, model, Xref, Xlab):
    t = time.time()
    try:
        s = M.fit_score(model, Xref, Xlab, seed=0)
        err = ''
    except Exception as e:      # noqa
        s, err = np.full(len(Xlab), np.nan), repr(e)[:200]
    return (p, sname, fs, model), s, time.time() - t, err


def main(n_jobs=14):
    F = C.build_cache()
    LAB = C.build_labeled_feats()
    done = pd.read_pickle(SCP) if SCP.exists() else {}
    rows = pd.read_csv(OUTP).to_dict('records') if OUTP.exists() else []
    tasks, EV, LABP = [], {}, {}
    for p in C.PRODUCTS:
        Fp = for_product(F, p)
        lab = for_product(LAB[p], p)
        LABP[p], EV[p] = lab, C.eval_rows(p)
        S = {n: m for n, _, _, m in SL.define(F, p)}
        for sname in pick_slices(p):
            ref = Fp[S[sname]]
            if len(ref) > REF_MAX:
                ref = ref.sample(REF_MAX, random_state=42)
            for fs, cols in FEATSETS.items():
                Xr = ref[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
                Xl = lab[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
                for model in M.REGISTRY:
                    if (p, sname, fs, model) not in done:
                        tasks.append((p, sname, fs, model, Xr, Xl))
    # 신경망·느린 모델 먼저 (병렬 꼬리 줄이기)
    slow = M.DEEP | {'gmm8', 'gmm4', 'maha_mcd', 'ocsvm'}
    tasks.sort(key=lambda t: t[3] not in slow)
    C.log(f'2단계 작업 {len(tasks)}개 (완료 {len(done)}개 건너뜀)')
    gen = Parallel(n_jobs=n_jobs, return_as='generator_unordered')(delayed(job)(*t) for t in tasks)
    for i, (key, s, sec, err) in enumerate(gen, 1):
        p, sname, fs, model = key
        done[key] = s
        r = {'product': p, 'slice': sname, 'featset': fs, 'model': model, 'sec': round(sec, 1), 'err': err}
        if not err:
            r.update(C.evaluate(p, LABP[p]['TimeStamp'], s, EV[p]))
        rows.append(r)
        if i % 25 == 0 or i == len(tasks):
            pd.DataFrame(rows).to_csv(OUTP, index=False, encoding='utf-8-sig')
            pd.to_pickle(done, SCP)
            C.log(f'{i}/{len(tasks)}  최근: {p} {sname} {fs} {model} off={r.get("off_AP", np.nan):.3f} '
                  f'ext={r.get("ext_AP", np.nan):.3f} ({sec:.0f}s)')
    C.log('완료')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
