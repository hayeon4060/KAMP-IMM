"""
4단계: 탐색 보정 · 블록 효과 · 판정 (PREREG)
  - 제품·기간별 전 설정(2단계 K1~K7, K8, 3단계 확인) 최고 AP vs 순환 이동 1,000회 '전 설정 최고' → p
  - 블록 효과: K3 무작위 조합에서 블록 포함 vs 미포함 공식 AP 평균 차이 (모델별, 같은 크기 조합끼리)
  - 변환·신호 묶음별 요약 (K1 단독, K5 dv26+블록)
  - CN7 판정(대 R0·R1·S3), RG3 신호 판정
출력: outputs/s4_summary.json, s4_block_effect.csv, s4_tr_group.csv, s4_judge.csv
"""
import json

import numpy as np
import pandas as pd

import lib4 as L

N_NULL = 1000
RNG = np.random.default_rng(7)


def all_scores(p):
    out = {}
    for f, tag in ((f's2_{p}.npz', 's2'), (f's2k7_{p}.npz', 's2'), (f's2b_{p}.npz', 's2b'), (f's3_{p}.npz', 's3')):
        path = L.OUT / 'scores' / f
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=True)
        for k, s in zip(z['keys'], z['s']):
            if np.isfinite(s).all():
                out[f'{tag}|{k}'] = s
    return out


def null_test(ev, per, S):
    e = ev[ev['period'] == per].reset_index(drop=True)
    y = e['y'].values.astype(int)
    keys = list(S)
    R = np.empty((len(keys), len(e)), np.int32)
    for i, k in enumerate(keys):
        o = np.argsort(-L.L3._tb(S[k][e['pos'].values]), kind='stable')
        R[i, o] = np.arange(1, len(e) + 1)

    def aps(pos):
        r = np.sort(R[:, pos], axis=1)
        return (np.arange(1, len(pos) + 1) / r).mean(1)

    P = np.flatnonzero(y == 1)
    ap = aps(P)
    n = len(y)
    mx = np.array([aps((P + RNG.integers(n // 20, n - n // 20)) % n).max() for _ in range(N_NULL)])
    return keys, ap, mx


def main():
    U, LBL, cols = L.load()
    summary, APS = {}, {}
    for p in L.C.PRODUCTS:
        ev = L.eval_table(p, LBL)
        S = all_scores(p)
        for per in ('official', 'extra'):
            keys, ap, mx = null_test(ev, per, S)
            APS[(p, per)] = dict(zip(keys, ap))
            i = int(ap.argmax())
            summary[f'{p}_{per}'] = {'n_cfg': len(keys), 'prevalence': float(ev.loc[ev['period'] == per, 'y'].mean()),
                                     'best_AP': float(ap[i]), 'best_cfg': keys[i], 'median_AP': float(np.median(ap)),
                                     'null_max_mean': float(mx.mean()), 'null_max_95': float(np.quantile(mx, .95)),
                                     'p_best': float((1 + (mx >= ap[i]).sum()) / (N_NULL + 1))}
            L.log(f'{p} {per}: 설정 {len(keys)}, 최고 {ap[i]:.3f} ({keys[i]}) | 귀무 최고 평균 {mx.mean():.3f} '
                  f'95% {np.quantile(mx, .95):.3f} → p={summary[f"{p}_{per}"]["p_best"]:.3f}')
    rg = summary['rg3_official']
    bk = max(APS[('rg3', 'official')], key=APS[('rg3', 'official')].get)
    rg['best_cfg_ext_AP'] = float(APS[('rg3', 'extra')][bk])
    rg['signal'] = bool(rg['p_best'] < 0.05 and rg['best_cfg_ext_AP'] > 2 * summary['rg3_extra']['prevalence'])
    json.dump(summary, open(L.OUT / 's4_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    L.log(f'RG3 신호: {rg["signal"]} (p={rg["p_best"]:.3f}, 최고 설정 추가 AP {rg["best_cfg_ext_AP"]:.3f})')

    # ---------------- 블록 효과 (K3 무작위 조합)
    s2 = pd.read_csv(L.OUT / 's2_results.csv')
    s2 = s2[s2['off_AP'].notna()]
    B = sorted(L.blocks(cols))
    rows = []
    for p in L.C.PRODUCTS:
        k3 = s2[(s2['product'] == p) & s2['kind'].str.startswith('K3_')].copy()
        k3['bl'] = k3['blocks'].str.split('+')
        for m, g in k3.groupby('model'):
            for b in B:
                has = g['bl'].apply(lambda x: b in x)
                if has.sum() < 5:
                    continue
                # 같은 크기 조합끼리 비교 후 평균
                d = [(g.loc[has & (g['kind'] == k), 'off_AP'].mean() - g.loc[~has & (g['kind'] == k), 'off_AP'].mean())
                     for k in g['kind'].unique() if (has & (g['kind'] == k)).any()]
                de = [(g.loc[has & (g['kind'] == k), 'ext_AP'].mean() - g.loc[~has & (g['kind'] == k), 'ext_AP'].mean())
                      for k in g['kind'].unique() if (has & (g['kind'] == k)).any()]
                rows.append({'product': p, 'model': m, 'block': b, 'transform': b.split('|')[0],
                             'group': b.split('|')[1], 'n_with': int(has.sum()),
                             'off_effect': float(np.nanmean(d)), 'ext_effect(참고)': float(np.nanmean(de))})
    BE = pd.DataFrame(rows)
    BE.to_csv(L.OUT / 's4_block_effect.csv', index=False, encoding='utf-8-sig')
    # ---------------- 변환·묶음 요약 (K1 단독, K5 dv26+블록)
    k = s2[s2['kind'].isin(['K1', 'K5'])].copy()
    k['block'] = np.where(k['kind'] == 'K1', k['blocks'],
                          k['blocks'].str.split('+').apply(lambda x: [b for b in x if b not in L.DV26]).str[0])
    k['transform'] = k['block'].str.split('|').str[0]
    k['group'] = k['block'].str.split('|').str[1]
    k.groupby(['product', 'kind', 'transform']).agg(off_med=('off_AP', 'median'), off_max=('off_AP', 'max'),
                                                     ext_med=('ext_AP', 'median')).round(3).reset_index().to_csv(
        L.OUT / 's4_tr.csv', index=False, encoding='utf-8-sig')
    k.groupby(['product', 'kind', 'group']).agg(off_med=('off_AP', 'median'), off_max=('off_AP', 'max'),
                                                 ext_med=('ext_AP', 'median')).round(3).reset_index().to_csv(
        L.OUT / 's4_group.csv', index=False, encoding='utf-8-sig')

    # ---------------- 판정
    cf = L.OUT / 's3_confirm.csv'
    if not cf.exists():
        return
    R = pd.read_csv(cf)
    J = []
    for p in L.C.PRODUCTS:
        r = R[R['product'] == p]
        piv = r.pivot_table(index=['config', 'model'], columns='window', values=['off_AP', 'ext_AP'])
        ref = lambda tag, w, col='off_AP': r.loc[(r['model'] == tag) & (r['window'] == w), col]   # noqa: E731
        w1 = r[(r['window'] == 'W1') & ~r['model'].isin(['R0', 'R1', 'S3'])]
        prev_o, prev_e = summary[f'{p}_official']['prevalence'], summary[f'{p}_extra']['prevalence']
        for x in w1.itertuples():
            row = {'product': p, 'config': x.config, 'model': x.model, 'why': x.why,
                   'W1_off': x.off_AP, 'W1_ext(참고)': x.ext_AP,
                   'w2_all_off': piv.loc[(x.config, x.model), ('off_AP', 'w2_all')] if ('off_AP', 'w2_all') in piv.columns else np.nan,
                   'eq_all_off': piv.loc[(x.config, x.model), ('off_AP', 'eq_all')] if ('off_AP', 'eq_all') in piv.columns else np.nan,
                   'W2_off': piv.loc[(x.config, x.model), ('off_AP', 'W2')] if ('off_AP', 'W2') in piv.columns else np.nan,
                   'gm_lift': np.sqrt(max(x.off_AP / prev_o, 1e-6) * max(x.ext_AP / prev_e, 1e-6))}
            for tag in ('R0', 'R1', 'S3'):
                lo = getattr(x, f'doff_{tag}_lo', np.nan)
                hi = getattr(x, f'doff_{tag}_hi', np.nan)
                ext_ref = ref(tag, 'w2_all', 'ext_AP')
                w2_ref = ref(tag, 'W2')
                c1 = bool(lo > 0)
                c2 = bool(len(ext_ref) and x.ext_AP >= ext_ref.iloc[0] - 0.05)
                c3 = bool(len(w2_ref) and row['W2_off'] > w2_ref.iloc[0]) if len(w2_ref) else False
                c4 = summary[f'{p}_official']['p_best'] < 0.05
                row.update({f'd_{tag}': f'[{lo:.3f}, {hi:.3f}]', f'{tag}_c1': c1, f'{tag}_c2': c2,
                            f'{tag}_c3': c3, f'{tag}_c4': c4, f'{tag}_pass': c1 and c2 and c3 and c4})
            J.append(row)
    J = pd.DataFrame(J).sort_values(['product', 'gm_lift'], ascending=[True, False])
    J.to_csv(L.OUT / 's4_judge.csv', index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    for p in L.C.PRODUCTS:
        print(f'\n===== {p}')
        print(J[J['product'] == p][['config', 'model', 'W1_off', 'W1_ext(참고)', 'w2_all_off', 'eq_all_off', 'W2_off',
                                    'd_R0', 'd_R1', 'd_S3', 'R0_pass', 'R1_pass', 'S3_pass']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
