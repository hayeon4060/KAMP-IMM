"""
5단계: 탐색 보정 · 요약 · 판정 (PREREG)
  - 제품·기간별로 2~4단계 모든 설정의 최고 AP vs 라벨 순환 이동 1,000회 '전 설정 최고 AP' 분포 → p
  - 계열·구간·모델별 요약
  - CN7 판정(대 R0, 대 R1), RG3 신호 판정
출력: outputs/s5_summary.json, outputs/s5_family.csv, outputs/s5_slice.csv, outputs/s5_model.csv, outputs/s5_judge.csv
"""
import json

import numpy as np
import pandas as pd

import lib as L

N_NULL = 1000
RNG = np.random.default_rng(7)


def all_scores():
    out = {}
    for name, f in (('s2', 's2_scores.pkl'), ('s3', 's3_scores.pkl'), ('s4', 's4_scores.pkl')):
        path = L.SCORES / f
        if not path.exists():
            continue
        for k, s in pd.read_pickle(path).items():
            if s is not None and np.isfinite(s).all():
                out[(name,) + tuple(k)] = s
    return out


def null_test(ev, per, keys, S):
    """순위 행렬로 빠르게: 양성 위치의 순위만 보면 AP 계산 가능"""
    e = ev[ev['period'] == per].reset_index(drop=True)                # 시간순
    y = e['y'].values.astype(int)
    M = np.vstack([L._tb(S[k][e['pos'].values]) for k in keys])
    R = np.empty_like(M, dtype=np.int32)
    order = np.argsort(-M, axis=1, kind='stable')
    R[np.arange(len(M))[:, None], order] = np.arange(1, M.shape[1] + 1)

    def aps(pos):
        r = np.sort(R[:, pos], axis=1)
        return (np.arange(1, len(pos) + 1) / r).mean(1)

    P = np.flatnonzero(y == 1)
    ap = aps(P)
    n = len(y)
    mx = np.array([aps((P + RNG.integers(n // 20, n - n // 20)) % n).max() for _ in range(N_NULL)])
    return ap, mx


def main():
    LU, LL, fam = L.load()
    S = all_scores()
    summary, judge, APS = {}, [], {}
    s2 = pd.read_csv(L.OUT / 's2_results.csv')
    s4 = pd.read_csv(L.OUT / 's4_finalists.csv') if (L.OUT / 's4_finalists.csv').exists() else None
    for p in L.C.PRODUCTS:
        lab = L.for_product(LL[p], p)
        ev = L.eval_table(p, lab['TimeStamp'])
        keys = [k for k in S if k[1] == p]
        for per in ('official', 'extra'):
            ap, mx = null_test(ev, per, keys, S)
            APS[(p, per)] = dict(zip(keys, ap))
            i = int(ap.argmax())
            prev = float(ev.loc[ev['period'] == per, 'y'].mean())
            summary[f'{p}_{per}'] = {
                'n_cfg': len(keys), 'prevalence': prev, 'best_AP': float(ap[i]), 'best_cfg': '|'.join(map(str, keys[i])),
                'median_AP': float(np.median(ap)), 'null_max_mean': float(mx.mean()),
                'null_max_95': float(np.quantile(mx, 0.95)),
                'p_best': float((1 + (mx >= ap[i]).sum()) / (N_NULL + 1))}
            L.log(f'{p} {per}: 설정 {len(keys)}, 최고 {ap[i]:.3f} {keys[i]} | 귀무 최고 평균 {mx.mean():.3f} '
                  f'95% {np.quantile(mx, .95):.3f} → p={summary[f"{p}_{per}"]["p_best"]:.3f}')
    json.dump(summary, open(L.OUT / 's5_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    # ---------------- 요약표 (2단계)
    a = s2[s2['err'].fillna('') == ''].copy()
    a['family'] = a['featset'].where(~a['featset'].str.startswith('dv26+'), a['featset'].str[5:])
    a['kind'] = np.where(a['featset'].str.startswith('dv26+'), 'dv26+계열', np.where(
        a['featset'].isin(['devall', 'all']), '큰 묶음', '단독'))
    agg = dict(off_best=('off_AP', 'max'), off_med=('off_AP', 'median'), ext_best=('ext_AP', 'max'),
               ext_med=('ext_AP', 'median'), n=('off_AP', 'size'))
    a.groupby(['product', 'kind', 'family']).agg(**agg).round(3).reset_index().to_csv(
        L.OUT / 's5_family.csv', index=False, encoding='utf-8-sig')
    a.groupby(['product', 'slice']).agg(**agg).round(3).reset_index().to_csv(
        L.OUT / 's5_slice.csv', index=False, encoding='utf-8-sig')
    a.groupby(['product', 'model']).agg(**agg).round(3).reset_index().to_csv(
        L.OUT / 's5_model.csv', index=False, encoding='utf-8-sig')

    # ---------------- 판정
    if s4 is not None:
        for p in L.C.PRODUCTS:
            m = s4[(s4['product'] == p) & (s4['window'] == 'main')].copy()
            w1 = s4[(s4['product'] == p) & (s4['window'] == 'W1')].set_index(['slice', 'featset', 'model'])
            r0 = m[m['model'] == 'final_c17'].iloc[0]
            r1 = m[(m['slice'] == 'w2_all') & (m['featset'] == 'dv26') & (m['model'] == 'gmm2')].iloc[0]
            prev_o = summary[f'{p}_official']['prevalence']
            prev_e = summary[f'{p}_extra']['prevalence']
            for r in m.itertuples():
                k = (r.slice, r.featset, r.model)
                w1_off = w1.loc[k, 'off_AP'] if k in w1.index else np.nan
                w1_r0 = w1.loc[('cur_0901', 'c17', 'final_c17'), 'off_AP'] if ('cur_0901', 'c17', 'final_c17') in w1.index else np.nan
                w1_r1 = w1.loc[('w2_all', 'dv26', 'gmm2'), 'off_AP'] if ('w2_all', 'dv26', 'gmm2') in w1.index else np.nan
                row = {'product': p, 'slice': r.slice, 'featset': r.featset, 'model': r.model, 'why': r.why,
                       'off_AP': r.off_AP, 'ext_AP': r.ext_AP,
                       'W1_off_AP': w1_off,
                       'gm_lift': np.sqrt(max(r.off_AP / prev_o, 1e-6) * max(r.ext_AP / prev_e, 1e-6))}
                for tag, ref, w1ref in (('R0', r0, w1_r0), ('R1', r1, w1_r1)):
                    c1 = getattr(r, f'doff_{tag}_lo', np.nan) > 0
                    c2 = r.ext_AP >= ref['ext_AP'] - 0.05
                    c3 = w1_off > w1ref
                    c4 = summary[f'{p}_official']['p_best'] < 0.05
                    row.update({f'{tag}_c1': c1, f'{tag}_c2': c2, f'{tag}_c3': c3, f'{tag}_c4': c4,
                                f'{tag}_pass': bool(c1 and c2 and c3 and c4),
                                f'doff_{tag}': f'[{getattr(r, f"doff_{tag}_lo", np.nan):.3f}, '
                                               f'{getattr(r, f"doff_{tag}_hi", np.nan):.3f}]',
                                f'dext_{tag}': f'[{getattr(r, f"dext_{tag}_lo", np.nan):.3f}, '
                                               f'{getattr(r, f"dext_{tag}_hi", np.nan):.3f}]'})
                judge.append(row)
        J = pd.DataFrame(judge).sort_values(['product', 'gm_lift'], ascending=[True, False])
        J.to_csv(L.OUT / 's5_judge.csv', index=False, encoding='utf-8-sig')
        for p in L.C.PRODUCTS:
            j = J[J['product'] == p]
            print(f'\n===== {p} 최종 후보 (lift 기하평균 순)')
            print(j[['slice', 'featset', 'model', 'off_AP', 'ext_AP', 'W1_off_AP', 'doff_R0', 'doff_R1',
                     'R0_pass', 'R1_pass', 'gm_lift']].round(3).to_string(index=False))
    rg = summary['rg3_official']
    bk = max(APS[('rg3', 'official')], key=APS[('rg3', 'official')].get)
    ext = APS[('rg3', 'extra')][bk]
    rg['best_cfg_ext_AP'] = float(ext)
    rg['signal'] = bool(rg['p_best'] < 0.05 and ext > 2 * summary['rg3_extra']['prevalence'])
    json.dump(summary, open(L.OUT / 's5_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    L.log(f'RG3 신호 판정: 공식 p={rg["p_best"]:.3f} (기준 < 0.05), 최고 설정 {bk}, 그 추가 AP {ext:.3f} '
          f'(기준 > {2 * summary["rg3_extra"]["prevalence"]:.3f}) → {"신호 있음" if rg["signal"] else "신호 없음"}')


if __name__ == '__main__':
    main()
