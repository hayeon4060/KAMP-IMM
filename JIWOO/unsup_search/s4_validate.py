"""
4단계: 검증·집계 — 1~3단계 모든 비지도 설정(라벨 0개 학습)

 1) 통합 표: 설정마다 공식 기간 AP(선택용) · 추가 기간 AP(사후 확인용, 비라벨 데이터가 끝난 뒤 기간)
 2) 탐색 보정: 라벨을 시간순 순환 이동(사건 군집 보존) 1,000회 → '전체 설정 중 최고 AP'의 귀무분포
    → 실제 최고 AP가 우연으로 나올 확률 (탐색 횟수만큼 부풀려진 최고값 보정)
 3) 상위 후보 층화 부트스트랩 95% 구간
출력: outputs/s4_all.csv, outputs/s4_summary.json
"""
import json
import numpy as np
import pandas as pd

import common as C

N_NULL = 1000
RNG = np.random.default_rng(2026)


def load_all():
    LAB = C.build_labeled_feats()
    out = []   # (product, family, config, scores)
    s1 = pd.read_pickle(C.OUT / 'scores' / 's1_scores.pkl')['scores']
    for (p, sl, fs, var), s in s1.items():
        out.append((p, 'S1 구간·Maha', f'{sl}|{fs}|maha_lw|{var}', s))
    f2 = C.OUT / 'scores' / 's2_scores.pkl'
    if f2.exists():
        for (p, sl, fs, m), s in pd.read_pickle(f2).items():
            if (fs, m) == ('c17', 'maha_lw') or (fs in ('dev26', 'raw21') and m == 'maha_lw'):
                continue                                     # 1단계와 중복
            out.append((p, 'S2 모델', f'{sl}|{fs}|{m}', s))
    f3 = C.OUT / 'scores' / 's3_scores.pkl'
    if f3.exists():
        for (p, part, meth, pool, head), s in pd.read_pickle(f3).items():
            out.append((p, f'S3{part} {meth}', f'{pool}|{head}', s))
    return LAB, out


def ap_matrix(Y, S):
    """Y: (n,) 0/1, S: (k, n) 점수 → (k,) AP (동점은 미리 작은 잡음으로 깸)"""
    order = np.argsort(-S, axis=1, kind='stable')
    Ys = Y[order]
    tp = np.cumsum(Ys, 1)
    rank = np.arange(1, S.shape[1] + 1)
    return (Ys * tp / rank).sum(1) / Y.sum()


def main():
    LAB, cfgs = load_all()
    summary, tables = {}, []
    for p in C.PRODUCTS:
        ev = C.eval_rows(p).sort_values('TimeStamp', kind='stable').reset_index(drop=True)
        ts = pd.DatetimeIndex(LAB[p]['TimeStamp'])
        pos = pd.Series(np.arange(len(ts)), index=ts)
        pos = pos[~pos.index.duplicated()]
        mine = [(fam, name, s) for q, fam, name, s in cfgs if q == p and np.isfinite(s).all()]
        Sshot = np.vstack([s for _, _, s in mine])
        Sshot = Sshot + RNG.random(Sshot.shape) * 1e-9 * (np.nanstd(Sshot, 1, keepdims=True) + 1e-12)
        res = pd.DataFrame({'product': p, 'family': [f for f, _, _ in mine], 'config': [n for _, n, _ in mine]})
        for per in ('official', 'extra'):
            e = ev[ev['period'] == per].reset_index(drop=True)
            idx = e['TimeStamp'].map(pos).values
            S = Sshot[:, idx]
            Y = e['y'].values.astype(float)
            ap = ap_matrix(Y, S)
            key = per[:3]
            res[f'{key}_AP'] = ap
            res[f'{key}_lift'] = ap / Y.mean()
            k5 = int(round(len(Y) * 0.05))
            res[f'{key}_TP5'] = Y[np.argsort(-S, 1)[:, :k5]].sum(1).astype(int)
            # 탐색 보정: 순환 이동 귀무분포의 '설정 전체 최고 AP'
            order = np.argsort(-S, axis=1, kind='stable')
            rank = np.arange(1, len(Y) + 1)
            mx = np.empty(N_NULL)
            for b in range(N_NULL):
                sh = RNG.integers(len(Y) // 20, len(Y) - len(Y) // 20)
                Yb = np.roll(Y, sh)
                Ys = Yb[order]
                mx[b] = ((Ys * np.cumsum(Ys, 1) / rank).sum(1) / Yb.sum()).max()
            best = float(ap.max())
            summary[f'{p}_{key}'] = {
                'n_cfg': len(mine), 'rows': len(Y), 'P': int(Y.sum()), 'prevalence': float(Y.mean()),
                'best_AP': best, 'best_cfg': res.loc[int(ap.argmax()), 'config'],
                'best_family': res.loc[int(ap.argmax()), 'family'],
                'null_max_mean': float(mx.mean()), 'null_max_95': float(np.quantile(mx, 0.95)),
                'p_best_vs_null': float((1 + (mx >= best).sum()) / (N_NULL + 1)),
                'median_cfg_AP': float(np.median(ap))}
            C.log(f'{p} {per}: 설정 {len(mine)}개, 최고 AP {best:.3f} ({summary[f"{p}_{key}"]["best_family"]} '
                  f'{summary[f"{p}_{key}"]["best_cfg"]}) | 귀무 최고 평균 {mx.mean():.3f}, 95% {np.quantile(mx, .95):.3f}, '
                  f'p={summary[f"{p}_{key}"]["p_best_vs_null"]:.3f}')
        # 두 기간 모두 고려한 순위: lift의 기하평균
        res['both_gmean_lift'] = np.sqrt(res['off_lift'].clip(lower=1e-6) * res['ext_lift'].clip(lower=1e-6))
        # 상위 후보 부트스트랩 (공식 기준 상위 10 + 두 기간 기준 상위 10)
        cand = set(res.nlargest(10, 'off_AP').index) | set(res.nlargest(10, 'both_gmean_lift').index)
        e = ev[ev['period'] == 'official'].reset_index(drop=True)
        idx, Y = e['TimeStamp'].map(pos).values, e['y'].values
        pi, ni = np.flatnonzero(Y == 1), np.flatnonzero(Y == 0)
        B = [np.concatenate([RNG.choice(pi, len(pi)), RNG.choice(ni, len(ni))]) for _ in range(500)]
        for i in cand:
            s = Sshot[i, idx]
            v = [ap_matrix(Y[b].astype(float), s[b][None])[0] for b in B]
            res.loc[i, 'off_AP_lo'], res.loc[i, 'off_AP_hi'] = np.percentile(v, [2.5, 97.5])
        tables.append(res)
    A = pd.concat(tables, ignore_index=True)
    A.to_csv(C.OUT / 's4_all.csv', index=False, encoding='utf-8-sig')
    json.dump(summary, open(C.OUT / 's4_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for p in C.PRODUCTS:
        a = A[A['product'] == p]
        print(f'\n===== {p} 공식 AP 상위 15')
        print(a.nlargest(15, 'off_AP')[['family', 'config', 'off_AP', 'off_AP_lo', 'off_AP_hi', 'off_TP5',
                                          'ext_AP', 'ext_TP5', 'both_gmean_lift']].round(3).to_string(index=False))
        print(f'===== {p} 두 기간 lift 기하평균 상위 15')
        print(a.nlargest(15, 'both_gmean_lift')[['family', 'config', 'off_AP', 'ext_AP', 'off_TP5', 'ext_TP5',
                                                   'both_gmean_lift']].round(3).to_string(index=False))
        print(f'===== {p} 계열별 최고 (공식 AP)')
        print(a.loc[a.groupby('family')['off_AP'].idxmax(), ['family', 'config', 'off_AP', 'ext_AP']]
              .sort_values('off_AP', ascending=False).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
