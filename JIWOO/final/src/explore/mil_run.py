"""[탐색, 선택 실행] MIL 학습·평가 — docs/PREREG_MIL_1006.md 고정 설정. 먼저 mil_build_bags.py 실행."""
import sys, pickle, json
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LEDGER, FINAL, INTER, EXP, DATA
OUTX = EXP / 'explore'; OUTX.mkdir(parents=True, exist_ok=True)
import final_model as FM
torch.set_default_dtype(torch.float64)
D = pickle.load(open(OUTX / 'mil_bags.pkl', 'rb'))
L = pd.read_csv(LEDGER / 'ledger_v1.csv')
ALPHA = 0.01


def fit(Xs, sides, ks, ns, use_x=True, use_side=True, alpha=ALPHA):
    """Xs: list of (n_i, d) arrays per bag. 반환 (b, b_rh, w)"""
    d = Xs[0].shape[1]
    b = torch.zeros(1, requires_grad=True); brh = torch.zeros(1, requires_grad=True)
    w = torch.zeros(d, requires_grad=True)
    Xt = [torch.tensor(x) for x in Xs]; rh = torch.tensor([1.0 if s == 'RH' else 0.0 for s in sides])
    k = torch.tensor(ks, dtype=torch.float64); n = torch.tensor(ns, dtype=torch.float64); N = n.sum()
    # 초기값: 전체 비율
    with torch.no_grad(): b.fill_(float(np.log(k.sum() / (N - k.sum()))))
    params = [b] + ([brh] if use_side else []) + ([w] if use_x else [])
    opt = torch.optim.LBFGS(params, max_iter=500, tolerance_grad=1e-9, line_search_fn='strong_wolfe')

    def loss_fn():
        opt.zero_grad()
        pbar = torch.stack([torch.sigmoid(b + (brh * rh[i] if use_side else 0) + (x @ w if use_x else 0)).mean()
                            for i, x in enumerate(Xt)]).squeeze(-1).clamp(1e-9, 1 - 1e-9)
        ll = (k * torch.log(pbar) + (n - k) * torch.log(1 - pbar)).sum() / N
        loss = -ll + (alpha * (w ** 2).sum() if use_x else 0)
        loss.backward(); return loss
    opt.step(loss_fn)
    return float(b), float(brh) if use_side else 0.0, w.detach().numpy().copy()


def bag_rate(x, side, prm, use_x=True):
    b, brh, w = prm
    z = b + (brh if side == 'RH' else 0) + (x @ w if use_x else 0)
    return float((1 / (1 + np.exp(-z))).mean())


def binom_dev(k, n, p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    ll = k * np.log(p) + (n - k) * np.log(1 - p)
    ph = k / n
    sat = (k * np.log(ph) if k > 0 else 0) + ((n - k) * np.log(1 - ph) if k < n else 0)
    return 2 * (sat - ll)


def labeled_X(p):
    m = FM.FinalModel.load(str(FINAL / f'{p}_final_model.pkl'))
    shots = FM.load_labeled_shots(p, str(INTER))
    F = FM.make_features(shots, m.steps, m.fr_step)
    return pd.Series(list(m._prep(F)), index=F.TimeStamp.dt.strftime('%Y-%m-%d %H:%M:%S').values), \
        pd.Series(m.score(F), index=F.TimeStamp.dt.strftime('%Y-%m-%d %H:%M:%S').values)


def evaluate(p, score_of_row):
    g = L[L['product'] == p.upper()].copy()
    out = {}
    for per, h in g.groupby('period'):
        s = h.apply(score_of_row, axis=1).values.astype(float)
        y = h.y.values
        ok = ~np.isnan(s)
        y, s = y[ok], s[ok]
        k = int(np.ceil(0.05 * len(y)))
        o = np.lexsort((np.random.RandomState(0).rand(len(s)), -s))
        out[per] = dict(AP=round(float(average_precision_score(y, s)), 3), TP5=int(y[o[:k]].sum()), D=int(y.sum()), n=int(len(y)))
    return out


RES = {}
for p in ('cn7', 'rg3'):
    I, B, cols = D[p]['I'], D[p]['B'], D[p]['cols']
    Bk = B[B.keep].sort_index()
    Xs = [I.loc[I.lot == l, cols].values for l in Bk.index]
    Ms = [np.log(I.loc[I.lot == l, 'maha'].values.reshape(-1, 1) + 1e-6) for l in Bk.index]
    sides, ks, ns = list(Bk.side), list(Bk.k.astype(float)), list(Bk.n.astype(float))
    use_side = len(set(sides)) > 1
    R = dict(bags=len(Bk), sides=dict(pd.Series(sides).value_counts()), k=int(sum(ks)), n=int(sum(ns)), use_side=use_side)
    # 전체 적합
    pA = fit(Xs, sides, ks, ns, True, use_side)
    pS = fit(Xs, sides, ks, ns, False, use_side)
    pB = fit(Ms, sides, ks, ns, True, use_side, alpha=0.0)
    R['MIL_A_w'] = dict(zip(cols, np.round(pA[2], 3)))
    R['MIL_A_bRH'] = pA[1]; R['MIL_B_w'] = float(pB[2][0])
    # 적합 내 이탈도
    R['dev_in'] = dict(A=sum(binom_dev(k, n, bag_rate(x, s, pA)) for x, s, k, n in zip(Xs, sides, ks, ns)),
                       S=sum(binom_dev(k, n, bag_rate(x, s, pS, False)) for x, s, k, n in zip(Xs, sides, ks, ns)))
    # LOBO
    dA = dS = 0.0; per_bag = []
    for j in range(len(Bk)):
        tr = [i for i in range(len(Bk)) if i != j]
        sub = lambda a: [a[i] for i in tr]
        us = use_side and len(set(sub(sides))) > 1
        qA = fit(sub(Xs), sub(sides), sub(ks), sub(ns), True, us)
        qS = fit(sub(Xs), sub(sides), sub(ks), sub(ns), False, us)
        rA, rS = bag_rate(Xs[j], sides[j], qA), bag_rate(Xs[j], sides[j], qS, False)
        a, s_ = binom_dev(ks[j], ns[j], rA), binom_dev(ks[j], ns[j], rS)
        dA += a; dS += s_
        per_bag.append(dict(lot=Bk.index[j], k=ks[j], n=ns[j], obs=round(ks[j] / ns[j], 4), predA=round(rA, 4), predS=round(rS, 4)))
    R['dev_lobo'] = dict(A=round(dA, 1), S=round(dS, 1)); R['lobo_bags'] = per_bag
    # MIL-B 부트스트랩
    rng = np.random.default_rng(0); wb = []
    for _ in range(1000):
        ii = rng.integers(0, len(Bk), len(Bk))
        ss = [sides[i] for i in ii]
        q = fit([Ms[i] for i in ii], ss, [ks[i] for i in ii], [ns[i] for i in ii], True, use_side and len(set(ss)) > 1, alpha=0.0)
        wb.append(q[2][0])
    R['MIL_B_ci'] = [round(float(np.percentile(wb, 2.5)), 3), round(float(np.percentile(wb, 97.5)), 3)]
    # 라벨 평가
    Xl, Ml = labeled_X(p)
    wA, bA, brhA = pA[2], pA[0], pA[1]
    main = lambda r: float(Xl[r.TimeStamp] @ wA) if r.TimeStamp in Xl.index else np.nan
    full = lambda r: (bA + (brhA if r.cavity == 'RH' else 0) + float(Xl[r.TimeStamp] @ wA)) if r.TimeStamp in Xl.index else np.nan
    mahaB = lambda r: float(np.sign(pB[2][0]) * Ml[r.TimeStamp]) if r.TimeStamp in Ml.index else np.nan
    R['eval_A_main'] = evaluate(p, main); R['eval_A_full'] = evaluate(p, full); R['eval_B'] = evaluate(p, mahaB)
    RES[p] = R
    print(p, json.dumps({k: v for k, v in R.items() if k != 'lobo_bags'}, ensure_ascii=False, default=float), flush=True)
    print(pd.DataFrame(per_bag).to_string(index=False), flush=True)
json.dump(RES, open(OUTX / 'mil_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
