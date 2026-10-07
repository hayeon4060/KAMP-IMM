"""
1단계: 데이터 구간 탐색 — 모델 고정(Ledoit-Wolf Mahalanobis), 구간 × 변수 세트 × 트리밍
출력: outputs/s1_results.csv, outputs/scores/s1_scores.pkl (라벨 샷 점수, 4단계 탐색 보정용)
"""
import numpy as np
import pandas as pd

import common as C
import slices as SL

FEATSETS = {'c17': C.FEAT_C17, 'dev26': C.FEAT_DEV + C.FEAT_REL, 'raw21': C.FEAT_RAW}
REF_MAX = 30000
MIN_REF = 300


def for_product(df, p):
    """RG3는 흐름저항 편차를 RG3 기록 단위 기준 열로 교체"""
    if p == 'rg3':
        df = df.copy()
        df['dev_flow_res'] = df['dev_flow_res_rg3']
    return df


def fit_score(ref, lab, cols, trim=None, seed=42):
    if len(ref) > REF_MAX:
        ref = ref.sample(REF_MAX, random_state=seed)
    m = C.fm.FinalModel(cols).fit(ref)
    if trim:                      # 반복 트리밍: 학습 구간 상위 trim 비율 제거 후 재적합
        s = m.score(ref)
        ref = ref[s <= np.quantile(s, 1 - trim)]
        m = C.fm.FinalModel(cols).fit(ref)
    return m.score(lab), len(ref)


def main():
    F = C.build_cache()
    LAB = C.build_labeled_feats()
    rows, SCORES = [], {}
    for p in C.PRODUCTS:
        Fp = for_product(F, p)
        lab = for_product(LAB[p], p)
        ev = C.eval_rows(p)
        S = SL.define(F, p)
        C.log(f'[{p}] 구간 {len(S)}개')

        def run(name, axis, desc, mask, fs, trim=None, seed=42, tag=''):
            ref = Fp[mask]
            if len(ref) < MIN_REF:
                return
            sc, n = fit_score(ref, lab, FEATSETS[fs], trim, seed)
            key = (p, name + tag, fs, f'trim{trim}' if trim else 'none')
            SCORES[key] = sc
            r = {'product': p, 'slice': name + tag, 'axis': axis, 'desc': desc, 'featset': fs,
                 'variant': key[3], 'n_ref': n, 'n_runs': ref['run_id'].nunique(),
                 't0': ref['TimeStamp'].min(), 't1': ref['TimeStamp'].max()}
            r.update(C.evaluate(p, lab['TimeStamp'], sc, ev))
            rows.append(r)

        for name, axis, desc, mask in S:
            for fs in FEATSETS:
                run(name, axis, desc, mask, fs)
            for tr in (0.02, 0.05):
                run(name, axis, desc, mask, 'c17', trim=tr)
            got = [r for r in rows if r['product'] == p and r['slice'] == name and r['variant'] == 'none']
            C.log(f'[{p}] {name:22s} n={mask.sum():6d}  ' +
                  '  '.join(f'{r["featset"]} off={r["off_AP"]:.3f}/ext={r["ext_AP"]:.3f}' for r in got))
        # 표본 크기 학습 곡선
        base = {n: m for n, _, _, m in S}
        for bname in ('w2_all', 'pr_everything'):
            for n in (500, 1000, 2000, 5000, 20000):
                for seed in range(3):
                    m = SL.subsample_mask(base[bname], n, seed)
                    run(bname, '크기', f'{bname} 무작위 {n}샷', m, 'c17', seed=seed, tag=f'_n{n}_s{seed}')
        pd.DataFrame(rows).to_csv(C.OUT / 's1_results.csv', index=False, encoding='utf-8-sig')

    R = pd.DataFrame(rows)
    R.to_csv(C.OUT / 's1_results.csv', index=False, encoding='utf-8-sig')
    (C.OUT / 'scores').mkdir(exist_ok=True)
    pd.to_pickle({'scores': SCORES, 'lab_ts': {p: LAB[p]['TimeStamp'].values for p in C.PRODUCTS}},
                 C.OUT / 'scores' / 's1_scores.pkl')
    C.log(f'완료: {len(R)}개 설정')


if __name__ == '__main__':
    main()
