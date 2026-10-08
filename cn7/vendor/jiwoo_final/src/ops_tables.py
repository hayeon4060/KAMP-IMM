"""운영 표: 초품검사 그리드, 비용표, Brier 기여 분해, 경보 덩어리

firstarticle_grid_{제품}_{기간}.csv : 정지 후 재가동(T분 초과 공백) 첫 N쇼트 검사 규칙 × AI 경보의 중복 분해
cost_table_beta.csv  : 정책 4종(p≥p*, 경보 pct0.999, 예산5%, 무검사) × 비용비(1:10, 1:30, 1:100)의 기대비용
                       (비용 = 검사행 + 비용비 × 놓친 불량행, 검사비 단위)
brier_decomp_cn7_extra.csv : CN7 전향 Brier 개선(기준 = 공식 기저율 상수)의 그룹별 기여
alarm_clusters.csv   : 경보 쇼트의 덩어리 수·최대 연속·양품 쇼트 경보율
"""
import numpy as np
import pandas as pd

from paths import LEDGER

TS = {'5분': 5, '30분': 30, '4시간': 240}
NS = [5, 10, 30]


def firstarticle_grid():
    L = pd.read_csv(LEDGER / 'ledger_v1.csv', parse_dates=['TimeStamp'])
    for (prod, per), g in L.groupby(['product', 'period']):
        g = g.sort_values(['TimeStamp', 'cavity']).reset_index(drop=True)
        sh = g.drop_duplicates('TimeStamp')[['TimeStamp']].reset_index(drop=True)
        sh['gapmin'] = sh.TimeStamp.diff().dt.total_seconds().fillna(1e9) / 60
        rows = []
        P = int(g.y.sum())
        ai_def = set(g.index[(g.alarm == 1) & (g.y == 1)])
        for tn, T in TS.items():
            seg = (sh.gapmin > T).cumsum()
            k = sh.groupby(seg).cumcount()
            for N in NS:
                cover_ts = set(sh.TimeStamp[k < N])
                rule_mask = g.TimeStamp.isin(cover_ts)
                rule_def = set(g.index[rule_mask & (g.y == 1)])
                both = len(rule_def & ai_def); ai_only = len(ai_def - rule_def)
                rule_only = len(rule_def - ai_def); neither = P - both - ai_only - rule_only
                union_rows = int((rule_mask | (g.alarm == 1)).sum())
                rows.append(dict(T=tn, N=N, 규칙검사행=int(rule_mask.sum()),
                                 합계검사행=union_rows, 합계검사율=round(union_rows / len(g) * 100, 1),
                                 둘다=both, AI만=ai_only, 규칙만=rule_only, 둘다놓침=neither))
        pd.DataFrame(rows).to_csv(LEDGER / f'firstarticle_grid_{prod}_{per}.csv', index=False)


def cost_table():
    L = pd.read_csv(LEDGER / 'ledger_v1_plus.csv', parse_dates=['TimeStamp'])
    out = []
    for (prod, per), g in L.groupby(['product', 'period']):
        y = g.y.to_numpy(); n = len(g)
        for ratio in [10, 30, 100]:
            pstar = 1 / ratio
            rows = []
            for nm, mask in [("p>=p*", g.p >= pstar),
                             ("경보 pct0.999", g.alarm == 1),
                             ("예산5%", g.p.rank(ascending=False, method='first') <= np.ceil(0.05 * n)),
                             ("무검사", pd.Series(False, index=g.index))]:
                k = int(mask.sum()); tp = int(g.y[mask].sum()); miss = int(y.sum()) - tp
                rows.append((nm, k, tp, k + ratio * miss))
            best = min(rows, key=lambda r: r[3])
            out.append(dict(제품=prod, 기간=per, 비용비=f"1:{ratio}", **{nm: c for nm, k, tp, c in rows}, 최적=best[0]))
    pd.DataFrame(out).to_csv(LEDGER / 'cost_table_beta.csv', index=False, encoding='utf-8-sig')


def brier_decomp():
    L = pd.read_csv(LEDGER / 'ledger_v1_plus.csv', parse_dates=['TimeStamp'])
    ce = L[(L['product'] == 'CN7') & (L.period == 'extra')].copy()
    pref = float(L[(L['product'] == 'CN7') & (L.period == 'official')].y.mean())
    gi = (ce.y - pref) ** 2 - (ce.y - ce.p.clip(1e-4, 1 - 1e-4)) ** 2      # 양수 = 기준 대비 개선
    grp = np.where(ce.Reason.astype(str).str.contains('초기허용'), '초기허용',
                   np.where(ce.Reason.astype(str).str.contains('미성형'), '미성형', '정상'))
    D = pd.DataFrame({'그룹': grp, 'g': gi}).groupby('그룹').agg(행수=('g', 'size'), 기여합=('g', 'sum'))
    D['기여합'] = D.기여합.round(4)
    D.to_csv(LEDGER / 'brier_decomp_cn7_extra.csv', encoding='utf-8-sig')


def alarm_clusters():
    L = pd.read_csv(LEDGER / 'ledger_v1.csv', parse_dates=['TimeStamp'])
    rows = []
    for (prod, per), g in L.groupby(['product', 'period']):
        sh = g.drop_duplicates('TimeStamp').sort_values('TimeStamp')
        arr = sh.alarm.to_numpy()
        runs, cur = [], 0
        for v in arr:
            cur = cur + 1 if v else 0
            if cur: runs.append(cur)
        nb = int(((arr[1:] == 1) & (arr[:-1] == 0)).sum() + (arr[0] == 1)) if len(arr) else 0
        good = sh[sh.TimeStamp.isin(g[g.y == 0].TimeStamp)]
        rows.append(dict(제품=prod, 기간=per, 경보쇼트=int(arr.sum()), 덩어리=nb, 최대연속=max(runs) if runs else 0,
                         양품쇼트경보율_pct=round(good.alarm.mean() * 100, 2)))
    pd.DataFrame(rows).to_csv(LEDGER / 'alarm_clusters.csv', index=False, encoding='utf-8-sig')


def main():
    firstarticle_grid()
    cost_table()
    brier_decomp()
    alarm_clusters()


if __name__ == '__main__':
    main()
