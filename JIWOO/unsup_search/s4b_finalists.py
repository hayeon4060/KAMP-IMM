"""
4b단계: 최종 후보 검증 (CN7) — 시드 안정성 · 현 제출 대비 paired 부트스트랩 · 라벨 없는 운용점

후보: 4단계 공식 AP 상위 계열 대표 + 두 기간 기준 상위 + 현 제출(기준)
  - 시드 5개 (결정적 모델은 시드 영향 없음)
  - ΔAP = 후보 − 현 제출: 층화 부트스트랩 2,000회 (같은 재표집을 두 모델에 공유)
  - 운용점: 학습 구간 점수의 99.9% 분위 이상 = 경보 (라벨 미사용) → 경보 수·잡은 불량·F1 (행 단위)
출력: outputs/s4b_finalists.csv
"""
import numpy as np
import pandas as pd

import common as C
import slices as SL
import models as M
from s1_slices import for_product
from s2_models import FEATSETS

P = 'cn7'
CANDS = [  # (구간, 변수, 모델)
    ('cur_0901', 'c17', 'maha_lw'),          # 현 제출 (기준)
    ('cur_0901', 'dev26', 'maha_lw'),        # 변수만 교체
    ('w2_pre', 'dev26', 'dsvdd'),            # 공식 AP 1위
    ('w2_H2', 'dev26', 'gmm2'),
    ('cur_0901', 'dev26', 'gmm4'),
    ('cur_0901', 'dev26', 'ocsvm'),          # 두 기간 1위
    ('pr_cn7rg3_w2', 'dev26', 'gmm4'),       # 두 기간 2위
    ('w2_last8w', 'dev26', 'ae'),
    ('w2_all', 'dev26', 'gmm2'),
]
SEEDS = range(5)
Q = 0.999
REF_MAX = 30000


def main():
    F = C.build_cache()
    lab = for_product(C.build_labeled_feats()[P], P)
    Fp = for_product(F, P)
    S = {n: m for n, _, _, m in SL.define(F, P)}
    ev = C.eval_rows(P)
    pos = pd.Series(np.arange(len(lab)), index=pd.DatetimeIndex(lab['TimeStamp']))
    pos = pos[~pos.index.duplicated()]
    E = {per: ev[ev['period'] == per].reset_index(drop=True) for per in ('official', 'extra')}
    IDX = {per: e['TimeStamp'].map(pos).values for per, e in E.items()}
    rng = np.random.default_rng(7)
    BOOT = {}
    for per, e in E.items():
        y = e['y'].values
        pi, ni = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        BOOT[per] = [np.concatenate([rng.choice(pi, len(pi)), rng.choice(ni, len(ni))]) for _ in range(2000)]

    def ap(y, s):
        o = np.argsort(-s, kind='stable'); ys = y[o]
        return (ys * np.cumsum(ys) / np.arange(1, len(ys) + 1)).sum() / ys.sum()

    shot_scores, rows = {}, []
    for sl, fs, mdl in CANDS:
        ref = Fp[S[sl]]
        if len(ref) > REF_MAX:
            ref = ref.sample(REF_MAX, random_state=42)
        cols = FEATSETS[fs]
        Xr = ref[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
        Xl = lab[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
        pr = M.Prep().fit(Xr)
        per_seed = []
        for sd in SEEDS:
            m = M.REGISTRY[mdl]()
            if hasattr(m, 'seed'):
                m.seed = sd
            m.fit(pr(Xr))
            sl_s, ref_s = m.score(pr(Xl)), m.score(pr(Xr))
            thr = np.quantile(ref_s, Q)
            per_seed.append((sl_s, thr))
        name = f'{sl}|{fs}|{mdl}'
        avg = np.mean([s for s, _ in per_seed], 0)
        shot_scores[name] = avg
        r = {'config': name}
        for per, e in E.items():
            k = per[:3]
            y = e['y'].values
            aps = [ap(y, s[IDX[per]]) for s, _ in per_seed]
            r[f'{k}_AP_mean'], r[f'{k}_AP_sd'] = np.mean(aps), np.std(aps)
            al = [(s[IDX[per]] >= t) for s, t in per_seed]          # 라벨 없는 운용점
            tp = [int((a & (y == 1)).sum()) for a in al]
            na = [int(a.sum()) for a in al]
            r[f'{k}_alarms'], r[f'{k}_alarm_TP'] = np.mean(na), np.mean(tp)
            r[f'{k}_F1_alarm'] = np.mean([2 * t / (n + y.sum()) for t, n in zip(tp, na)])
        rows.append(r)
        C.log(f'{name:32s} off {r["off_AP_mean"]:.3f}±{r["off_AP_sd"]:.3f}  ext {r["ext_AP_mean"]:.3f}±{r["ext_AP_sd"]:.3f}  '
              f'경보 off {r["off_alarm_TP"]:.1f}/{r["off_alarms"]:.1f} ext {r["ext_alarm_TP"]:.1f}/{r["ext_alarms"]:.1f}')
    # 현 제출 대비 paired 부트스트랩 (시드 평균 점수 기준)
    base = shot_scores['cur_0901|c17|maha_lw']
    for r in rows:
        s = shot_scores[r['config']]
        for per, e in E.items():
            y = e['y'].values
            a, b = s[IDX[per]], base[IDX[per]]
            d = np.array([ap(y[i], a[i]) - ap(y[i], b[i]) for i in BOOT[per]])
            k = per[:3]
            r[f'{k}_dAP'] = ap(y, a) - ap(y, b)
            r[f'{k}_dAP_lo'], r[f'{k}_dAP_hi'] = np.percentile(d, [2.5, 97.5])
            r[f'{k}_P(d>0)'] = float((d > 0).mean())
    R = pd.DataFrame(rows)
    R.to_csv(C.OUT / 's4b_finalists.csv', index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print(R.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
