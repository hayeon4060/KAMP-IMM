"""[탐색, 선택 실행] 검사 정책 비교 — docs/PREREG_POLICY_1006.md 고정 규칙. 약 8분(2코어)."""
import sys, numpy as np, pandas as pd, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LEDGER, FINAL, INTER, EXP, DATA
OUTX = EXP / 'explore'; OUTX.mkdir(parents=True, exist_ok=True)
L = pd.read_csv(LEDGER / 'ledger_v1.csv')
NS = 200
def run(prod, per):
    g = L[(L['product'] == prod) & (L.period == per)].copy(); g['date'] = g.TimeStamp.str[:10]
    g = g.sort_values(['TimeStamp', 'cavity']).reset_index(drop=True)
    sh = g.drop_duplicates('TimeStamp')[['TimeStamp', 'date']].reset_index(drop=True); sh['rk'] = sh.groupby('date').cumcount()
    g = g.merge(sh[['TimeStamp', 'rk']], on='TimeStamp'); g['rid'] = np.arange(len(g))
    D = int(g.y.sum()); bad_ids = g.index[g.y == 1].tolist()
    budget = {d: int(round(0.36 * len(x))) for d, x in g.groupby('date')}
    out = {}
    for samp in ('S1', 'S2'):
        for meth in ('R', 'D', 'DS', 'T', 'TS', 'M', 'C'):
            found, miss = [], np.zeros(len(g))
            for seed in range(NS):
                rng = np.random.default_rng(seed)
                ref = set()
                for d, s in sh.groupby('date'):
                    ts = s.TimeStamp.values
                    if samp == 'S1':
                        for b in range(0, len(ts), 5): ref.add(rng.choice(ts[b:b + 5]))
                    else:
                        ref.update(ts[int(rng.integers(0, 5))::5])
                isref = g.TimeStamp.isin(ref).values
                insp = isref.copy()
                for d, gd in g.groupby('date'):
                    idx = gd.index.values; rf = gd[isref[idx]]; un = gd[~isref[idx]]
                    nb = budget[d] - len(rf)
                    if nb <= 0 or len(un) == 0: continue
                    bad = rf[rf.y == 1]; tie = rng.random(len(un)) * 1e-9
                    if meth == 'C':
                        order = []
                        a = un[un.pct >= 0.999]; order += list(a.sort_values('score', ascending=False).index)
                        if len(bad):
                            dist = np.array([np.abs(bad.rk.values - r).min() for r in un.rk.values])
                            nn = un[(dist <= 5)].assign(dd=dist[dist <= 5]).sort_values('dd')
                            order += [i for i in nn.index if i not in order]
                        rest = un.drop(index=[i for i in order if i in un.index]).sort_values('score', ascending=False)
                        order += list(rest.index)
                        insp[order[:nb]] = True
                        continue
                    if meth == 'R': s = tie
                    elif meth == 'M': s = un.score.values + tie
                    elif meth == 'D': s = np.full(len(un), rf.y.mean()) + tie
                    elif meth == 'DS': s = un.cavity.map(rf.groupby('cavity').y.mean()).fillna(0).values + tie
                    else:
                        s = np.zeros(len(un))
                        for j, (r, c) in enumerate(zip(un.rk.values, un.cavity.values)):
                            b2 = bad[bad.cavity == c] if meth == 'TS' else bad
                            if len(b2): s[j] = np.exp(-np.abs(b2.rk.values - r) / 3).sum()
                        s = s + tie
                    insp[un.index.values[np.argsort(-s)[:nb]]] = True
                found.append(int(g.y.values[insp].sum())); miss += (~insp) & (g.y.values == 1)
            f = np.array(found)
            mb = g.loc[bad_ids].assign(miss=miss[bad_ids] / NS)
            out[(samp, meth)] = dict(mean=round(f.mean(), 2), min=int(f.min()), p5=float(np.percentile(f, 5)),
                                     miss={f'{d} {r}': v for (d, r), v in mb.groupby(['date', 'Reason']).miss.mean().round(2).items()})
    return D, len(g), sum(budget.values()), out
ALL = {}
for prod, per in (('CN7', 'official'), ('CN7', 'extra'), ('RG3', 'official'), ('RG3', 'extra')):
    D, n, b, o = run(prod, per)
    print(f'== {prod} {per}: 행 {n}, 불량 {D}, 검사 {b}행 ({b/n:.1%}), 무작위 기대 {D*b/n:.2f}')
    for (sp, m), v in o.items():
        print(f'  {sp} {m:2s} 평균 {v["mean"]:5.2f} 최저 {v["min"]:2d} 5% {v["p5"]:4.0f} | 놓침률 {v["miss"]}')
    ALL[f'{prod}_{per}'] = {f'{sp}_{m}': v for (sp, m), v in o.items()}
json.dump(ALL, open(OUTX / 'policy_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)

rows = []
for k, d in ALL.items():
    for m, v in d.items():
        rows.append(dict(기간=k, 방법=m, 평균=v['mean'], 최저=v['min'], p5=v['p5'], 놓침률=json.dumps(v['miss'], ensure_ascii=False)))
pd.DataFrame(rows).to_csv(OUTX / 'policy_result.csv', index=False, encoding='utf-8-sig')
print('saved policy_result.json / policy_result.csv')
