"""[탐색, 선택 실행] 물리 제약 신경망(PINN-lite) — docs/PREREG_PINN_1006.md 고정 설정 그대로.
실행: python run_all.py 후, pip install torch, python src/explore/pinn_lite.py → outputs/experiments/explore/pinn_result.json
정상 reference(라벨 0개)에서 조건 → 공정 응답 회귀 + 물리 페널티, 잔차를 LW Mahalanobis 점수로."""
import pickle, json, sys
import numpy as np, pandas as pd, torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score

torch.set_num_threads(2)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LEDGER, FINAL, INTER, EXP
import final_model as FM
R = lambda c: f'{c}_raw'
IN = ['v', 'Tmelt', 'Tback', 'Mold', 'Hopper', 'Cushion', 'PlastPos', 'BackP']
OUT = ['Filling_Time', 'Max_Injection_Pressure', 'Max_Switch_Over_Pressure', 'Plasticizing_Time']


def frame(d):
    X = pd.DataFrame({
        'v': d[R('Max_Injection_Speed')], 'Tmelt': d[[R(f'Barrel_Temperature_{i}') for i in (1, 2, 3)]].mean(1),
        'Tback': d[[R(f'Barrel_Temperature_{i}') for i in (4, 5, 6)]].mean(1), 'Mold': d[R('Mold_Temperature_3')],
        'Hopper': d[R('Hopper_Temperature')], 'Cushion': d[R('Cushion_Position')],
        'PlastPos': d[R('Plasticizing_Position')], 'BackP': d[R('Average_Back_Pressure')]})
    Y = pd.DataFrame({c: d[R(c)] for c in OUT})
    return X.astype(float), Y.astype(float)


class Scale:
    def fit(s, A):
        s.m = np.nanmedian(A, 0); s.s = np.nanpercentile(A, 75, 0) - np.nanpercentile(A, 25, 0)
        s.s = np.where(s.s > 1e-9, s.s, np.nanstd(A, 0) + 1e-9); return s
    def t(s, A): return np.clip((A - s.m) / s.s, -20, 20)


def train(Xs, Ys, xsc, lam, seed=0, epochs=300):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(len(IN), 32), torch.nn.Tanh(), torch.nn.Linear(32, 32),
                              torch.nn.Tanh(), torch.nn.Linear(32, len(OUT)))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    X = torch.tensor(Xs, dtype=torch.float32); Y = torch.tensor(Ys, dtype=torch.float32)
    iv, it, ipp, ibp = IN.index('v'), IN.index('Tmelt'), IN.index('PlastPos'), IN.index('BackP')
    # 체적 보존 탄성: d log t_f / d log v = -1  →  표준화 공간 변환 계수
    #   t_f = m_t + s_t*y0, v = m_v + s_v*x  →  dlog t_f/dlog v = (s_t*dy0/dx / t_f) * (v / s_v)
    m_v, s_v = xsc.m[iv], xsc.s[iv]
    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=gen)
        for b in range(0, len(X), 512):
            idx = perm[b:b + 512]; xb = X[idx].clone().requires_grad_(True)
            pred = net(xb); loss = ((pred - Y[idx]) ** 2).mean()
            if lam > 0:
                g = [torch.autograd.grad(pred[:, k].sum(), xb, create_graph=True)[0] for k in range(len(OUT))]
                tf = ysc_m[0] + ysc_s[0] * pred[:, 0]; vv = m_v + s_v * xb[:, iv]
                elast = ysc_s[0] * g[0][:, iv] / tf.clamp(min=1e-3) * vv / s_v
                pen = ((elast + 1) ** 2).mean()                              # ① 체적 보존
                pen = pen + torch.relu(-g[1][:, iv]).pow(2).mean()           # ② ∂P/∂v ≥ 0
                pen = pen + torch.relu(g[1][:, it]).pow(2).mean()            #    ∂P/∂T ≤ 0
                pen = pen + torch.relu(-g[3][:, ipp]).pow(2).mean()          # ③ ∂t_p/∂계량위치 ≥ 0
                pen = pen + torch.relu(-g[3][:, ibp]).pow(2).mean()          #    ∂t_p/∂배압 ≥ 0
                loss = loss + lam * pen
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def causal(res, run_ids, w=20, minp=5):
    out = np.full_like(res, np.nan)
    for r in pd.unique(run_ids):
        ii = np.flatnonzero(run_ids == r)
        for k, i in enumerate(ii):
            prev = ii[max(0, k - w):k]
            if len(prev) >= minp: out[i] = res[i] - np.nanmedian(res[prev], 0)
    return out


def physics_check(net, Xs, xsc):
    X = torch.tensor(Xs[:2000], dtype=torch.float32).requires_grad_(True); p = net(X)
    gP = torch.autograd.grad(p[:, 1].sum(), X)[0]
    return dict(dPdv_pos=float((gP[:, IN.index('v')] >= 0).float().mean()),
                dPdT_neg=float((gP[:, IN.index('Tmelt')] <= 0).float().mean()))


L = pd.read_csv(LEDGER / 'ledger_v1.csv')
C = {}
for _p in ('cn7', 'rg3'):   # 제출 모델과 같은 정상 기준(reference)·라벨 쇼트 파생변수
    _shots = FM.load_labeled_shots(_p, str(INTER))
    _ref, _ = FM.build_reference(_p, _shots, str(INTER))
    _m = FM.FinalModel.load(str(FINAL / f'{_p}_final_model.pkl'))
    C[_p] = dict(ref=FM.make_features(_ref, _m.steps, _m.fr_step), lab=FM.make_features(_shots, _m.steps, _m.fr_step))
res_all = {}
for p in ('cn7', 'rg3'):
    ref, lab = C[p]['ref'].reset_index(drop=True), C[p]['lab'].reset_index(drop=True)
    Xr, Yr = frame(ref); Xl, Yl = frame(lab)
    ok = Xr.notna().all(1) & Yr.notna().all(1)
    xsc, ysc = Scale().fit(Xr[ok].values), Scale().fit(Yr[ok].values)
    ysc_m = torch.tensor(ysc.m, dtype=torch.float32); ysc_s = torch.tensor(ysc.s, dtype=torch.float32)
    for lam in (0.0, 0.1):
        net = train(xsc.t(Xr[ok].values), ysc.t(Yr[ok].values), xsc, lam)
        with torch.no_grad():
            f = lambda X, Y: ysc.t(Y.values) - net(torch.tensor(xsc.t(X.values), dtype=torch.float32)).numpy()
            rr, rl = f(Xr, Yr), f(Xl, Yl)
        chk = physics_check(net, xsc.t(Xr[ok].values), xsc)
        for var, (A, B) in {'A_static': (rr, rl),
                            'B_causal': (causal(rr, ref.run_id.values), causal(rl, lab.run_id.values))}.items():
            okr = ~np.isnan(A).any(1)
            lw = LedoitWolf().fit(A[okr]); sref = np.sort(np.sqrt(lw.mahalanobis(A[okr])))
            sl = np.full(len(B), np.nan); okl = ~np.isnan(B).any(1)
            sl[okl] = np.sqrt(lw.mahalanobis(B[okl]))
            pct = np.searchsorted(sref, sl) / (len(sref) + 1)
            m = dict(zip(pd.to_datetime(lab.TimeStamp).dt.strftime('%Y-%m-%d %H:%M:%S'), zip(sl, pct)))
            g = L[L['product'] == p.upper()].copy()
            g['s'] = g.TimeStamp.map(lambda t: m.get(t, (np.nan, np.nan))[0])
            g['pc'] = g.TimeStamp.map(lambda t: m.get(t, (np.nan, np.nan))[1])
            out = {}
            for per, h in g.groupby('period'):
                okh = h.s.notna(); y = h.y[okh].values; s = h.s[okh].values
                k = int(np.ceil(0.05 * len(y))); o = np.argsort(-s, kind='stable')
                al = h.pc[okh].values >= 0.999
                out[per] = dict(AP=round(float(average_precision_score(y, s)), 3), TP5=int(y[o[:k]].sum()),
                                n=int(okh.sum()), D=int(y.sum()), alarm_pct=round(float(al.mean() * 100), 2),
                                TP_alarm=int(y[al].sum()), missing=int((~okh).sum()))
            key = f'{p} λ={lam} {var}'
            res_all[key] = dict(out, phys=chk)
            print(key, json.dumps(out, ensure_ascii=False), chk, flush=True)
(EXP / 'explore').mkdir(parents=True, exist_ok=True)
json.dump(res_all, open(EXP / 'explore' / 'pinn_result.json', 'w'), ensure_ascii=False, indent=1)
