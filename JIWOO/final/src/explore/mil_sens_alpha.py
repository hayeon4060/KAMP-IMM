"""[탐색, 선택 실행] 사후 민감도(사전 기록 밖): L2 강도 α만 바꿔 전체 적합 → 라벨 AP. LOBO·부트스트랩 생략."""
import sys, json, warnings; warnings.filterwarnings('ignore')
sys.argv = ['x']
from pathlib import Path
src = open(Path(__file__).resolve().parent / 'mil_run.py', encoding='utf-8').read().split('RES = {}')[0]
__file__ = str(Path(__file__).resolve().parent / 'mil_run.py')
exec(src)
out = {}
for p in ('cn7', 'rg3'):
    I, B, cols = D[p]['I'], D[p]['B'], D[p]['cols']
    Bk = B[B.keep].sort_index()
    Xs = [I.loc[I.lot == l, cols].values for l in Bk.index]
    sides, ks, ns = list(Bk.side), list(Bk.k.astype(float)), list(Bk.n.astype(float))
    Xl, _ = labeled_X(p)
    for a in (0.0, 0.001, 0.1):
        pr = fit(Xs, sides, ks, ns, True, len(set(sides)) > 1, alpha=a)
        f = lambda r, w=pr[2]: float(Xl[r.TimeStamp] @ w) if r.TimeStamp in Xl.index else np.nan
        out[f'{p} α={a}'] = evaluate(p, f)
        print(p, a, out[f'{p} α={a}'], flush=True)
json.dump(out, open(OUTX / 'mil_sens_alpha.json', 'w'), indent=1)
