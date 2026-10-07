"""
3단계: 자기지도 사전학습 · 순서 모델 · 정상 세분화 앙상블 (모두 라벨 0개로 학습)

 A. 사전학습 (DINOv2식 '큰 비라벨로 표현 학습 → 대상 정상에 head') — 표형 버전
    풀(pool) 3종: 같은제품·우진2(w2_all) / CN7+RG3 전 설비 / 전 설비·전 부품
    방법: MFM(변수 가리고 복원, VIME/MCM식) · SCARF(대조학습)
    점수: (i) MFM 복원 오차(가린 위치, 마스크 16개 평균) — head 없음
          (ii) 임베딩 + head(Mahalanobis, kNN5) — head는 대상 정상 구간(1단계 최고 구간·현 제출 구간)
 B. 순서 모델 (직전 10샷 → 현재 샷): GRU 예측 오차 · 1D-CNN 윈도우 AE 마지막 샷 복원 오차
 C. Dual-model: 넓은 정상 풀을 원값 GMM으로 운전 모드 k개로 나눔 → 모드별 VAE + WGAN-GP
    → 모드 안 백분위로 바꾼 뒤 소속 확률로 가중합 (VAE·WGAN 동일 가중)
출력: outputs/s3_results.csv, outputs/scores/s3_scores.pkl
"""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

import common as C
import slices as SL
import models as M
from s1_slices import for_product

import torch
from torch import nn

torch.set_num_threads(14)
FS_REL = C.FEAT_C17 + [c for c in C.FEAT_DEV + C.FEAT_REL if c not in C.FEAT_C17]   # 상대(편차) 변수 — 설비 간 이전 가능
FS_RAW = C.FEAT_RAW
POOLS = ('w2_all', 'pr_cn7rg3_all', 'pr_everything')
POOL_MAX = 150000
OUTP = C.OUT / 's3_results.csv'
SCP = C.OUT / 'scores' / 's3_scores.pkl'


def X_of(df, cols):
    return df[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)


def best_s1_slice(p):
    r = pd.read_csv(C.OUT / 's1_results.csv')
    r = r[(r['product'] == p) & (r['variant'] == 'none') & (r['axis'] != '크기') & (r['featset'] == 'c17')]
    return r.sort_values('off_AP', ascending=False)['slice'].iloc[0]


# ====================================================================== A. 사전학습
class Enc(nn.Module):
    def __init__(self, d, z=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 128), nn.LeakyReLU(0.1), nn.Linear(128, 128), nn.LeakyReLU(0.1),
                               nn.Linear(128, z))

    def forward(self, x):
        return self.f(x)


def corrupt(x, p, gen):
    """마스크 위치를 같은 열의 무작위 다른 샷 값으로 교체 (VIME/SCARF 방식)"""
    m = (torch.rand(x.shape, generator=gen) < p).float()
    perm = torch.stack([torch.randperm(len(x), generator=gen) for _ in range(x.shape[1])], 1)
    xs = torch.gather(x, 0, perm)
    return x * (1 - m) + xs * m, m


def pretrain_mfm(X, epochs=30, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    d = X.shape[1]
    enc = Enc(d); dec = nn.Sequential(nn.Linear(32, 128), nn.LeakyReLU(0.1), nn.Linear(128, d))
    mh = nn.Linear(32, d)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters(), *mh.parameters()], 1e-3)
    X = torch.tensor(X, dtype=torch.float32)
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=g).split(512):
            xb = X[idx]
            xc, m = corrupt(xb, 0.3, g)
            z = enc(xc)
            loss = ((dec(z) - xb) ** 2).mean() + nn.functional.binary_cross_entropy_with_logits(mh(z), m)
            opt.zero_grad(); loss.backward(); opt.step()
    return enc, dec


def mfm_score(enc, dec, X, n_mask=16, seed=7):
    g = torch.Generator().manual_seed(seed)
    X = torch.tensor(X, dtype=torch.float32)
    tot = torch.zeros(len(X)); cnt = torch.zeros(len(X))
    with torch.no_grad():
        for _ in range(n_mask):
            m = (torch.rand(X.shape, generator=g) < 0.3).float()
            xc = X * (1 - m)                  # 평가 시에는 0(=중앙값)으로 가림 → 결정적
            err = ((dec(enc(xc)) - X) ** 2) * m
            tot += err.sum(1); cnt += m.sum(1)
    return (tot / cnt.clamp(min=1)).sqrt().numpy()


def pretrain_scarf(X, epochs=30, seed=0, tau=0.5):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    d = X.shape[1]
    enc = Enc(d); proj = nn.Sequential(nn.Linear(32, 64), nn.LeakyReLU(0.1), nn.Linear(64, 32))
    opt = torch.optim.Adam([*enc.parameters(), *proj.parameters()], 1e-3)
    X = torch.tensor(X, dtype=torch.float32)
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=g).split(512):
            xb = X[idx]
            xa, _ = corrupt(xb, 0.6, g)
            za = nn.functional.normalize(proj(enc(xa)), dim=1)
            zb = nn.functional.normalize(proj(enc(xb)), dim=1)
            logits = za @ zb.T / tau
            loss = nn.functional.cross_entropy(logits, torch.arange(len(xb)))
            opt.zero_grad(); loss.backward(); opt.step()
    return enc


def embed(enc, X):
    with torch.no_grad():
        return enc(torch.tensor(X, dtype=torch.float32)).numpy()


def heads(Zref, Zlab):
    pr = M.Prep().fit(Zref)
    A, B = pr(Zref), pr(Zlab)
    lw = LedoitWolf().fit(A)
    nnb = NearestNeighbors(n_neighbors=5).fit(A[np.random.default_rng(0).permutation(len(A))[:20000]])
    return {'maha': np.sqrt(lw.mahalanobis(B)), 'knn5': nnb.kneighbors(B)[0].mean(1)}


# ====================================================================== B. 순서 모델
W = 10


def windows(df, X):
    """Run 안에서 직전 W샷 윈도우 (부족하면 Run 첫 샷 반복으로 채움 — 미래 정보 없음)"""
    d = df.reset_index(drop=True)
    out = np.zeros((len(d), W + 1, X.shape[1]), np.float32)
    for _, idx in d.groupby('run_id', sort=False).indices.items():
        idx = np.sort(idx) if not isinstance(idx, np.ndarray) else idx
        xs = X[idx]
        pad = np.concatenate([np.repeat(xs[:1], W, 0), xs], 0)
        for j in range(len(idx)):
            out[idx[j]] = pad[j:j + W + 1]
    return out


class GRUF(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.GRU(d, 64, batch_first=True); self.o = nn.Linear(64, d)

    def forward(self, w):        # w: (B, W, d) → 다음 샷 예측
        h, _ = self.g(w)
        return self.o(h[:, -1])


class CNNAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.e = nn.Sequential(nn.Conv1d(d, 32, 3, padding=1), nn.LeakyReLU(0.1), nn.Conv1d(32, 8, 3, padding=1))
        self.dd = nn.Sequential(nn.Conv1d(8, 32, 3, padding=1), nn.LeakyReLU(0.1), nn.Conv1d(32, d, 3, padding=1))

    def forward(self, w):        # w: (B, W+1, d) → 복원
        return self.dd(self.e(w.transpose(1, 2))).transpose(1, 2)


def train_seq(kind, Wref, epochs=25, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    d = Wref.shape[2]
    net = GRUF(d) if kind == 'gru' else CNNAE(d)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Wt = torch.tensor(Wref)
    for _ in range(epochs):
        for idx in torch.randperm(len(Wt), generator=g).split(256):
            wb = Wt[idx]
            loss = (((net(wb[:, :-1]) - wb[:, -1]) ** 2).mean() if kind == 'gru'
                    else ((net(wb) - wb) ** 2).mean())
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def seq_score(kind, net, Wl):
    with torch.no_grad():
        w = torch.tensor(Wl)
        if kind == 'gru':
            return ((net(w[:, :-1]) - w[:, -1]) ** 2).sum(1).sqrt().numpy()
        return ((net(w)[:, -1] - w[:, -1]) ** 2).sum(1).sqrt().numpy()


# ====================================================================== C. Dual-model (정상 세분화 앙상블)
def pct_of(ref_scores, s):
    r = np.sort(ref_scores)
    return np.searchsorted(r, s, side='right') / (len(r) + 1)


def dual_model(ref, lab, k, seed=0):
    Rr, Rl = X_of(ref, FS_RAW), X_of(lab, FS_RAW)
    pr_raw = M.Prep().fit(Rr)
    gm = GaussianMixture(k, covariance_type='diag', random_state=seed, n_init=3).fit(pr_raw(Rr))
    cr, resp = gm.predict(pr_raw(Rr)), gm.predict_proba(pr_raw(Rl))
    Xr, Xl = X_of(ref, FS_REL), X_of(lab, FS_REL)
    out = {'vae': np.zeros(len(lab)), 'wgan': np.zeros(len(lab))}
    sizes = [int((cr == c).sum()) for c in range(k)]
    todo = [(c, name) for c in range(k) if sizes[c] >= 200 for name in ('vae', 'wgan')]
    res = Parallel(n_jobs=min(14, len(todo)))(delayed(_fit_mode)(Xr[cr == c], Xl, name, seed) for c, name in todo)
    for (c, name), pc in zip(todo, res):
        out[name] += resp[:, c] * pc
    out['ens'] = 0.5 * (out['vae'] + out['wgan'])
    return out, sizes


def _fit_mode(A0, Xl, name, seed):
    """한 운전 모드의 생성 모델 → 라벨 샷의 모드 내 백분위"""
    pr = M.Prep().fit(A0)
    A, B = pr(A0), pr(Xl)
    mdl = M.VAE(epochs=40, seed=seed) if name == 'vae' else M.WGANGP(iters=2500, seed=seed)
    mdl.fit(A)
    return pct_of(mdl.score(A), mdl.score(B))


# ====================================================================== 실행
def main():
    F = C.build_cache()
    LAB = C.build_labeled_feats()
    SC = pd.read_pickle(SCP) if SCP.exists() else {}
    rows = pd.read_csv(OUTP).to_dict('records') if OUTP.exists() else []

    def rec(p, part, method, pool, head, s, lab, ev, sec, extra=None):
        key = (p, part, method, pool, head)
        SC[key] = s
        r = {'product': p, 'part': part, 'method': method, 'pool': pool, 'head': head, 'sec': round(sec, 1)}
        r.update(C.evaluate(p, lab['TimeStamp'], s, ev))
        if extra:
            r.update(extra)
        rows.append(r)
        pd.DataFrame(rows).to_csv(OUTP, index=False, encoding='utf-8-sig')
        pd.to_pickle(SC, SCP)
        C.log(f'{p} {part} {method:6s} {pool:16s} {head:10s} off={r["off_AP"]:.3f} ext={r["ext_AP"]:.3f} ({sec:.0f}s)')

    have = {(r['product'], r['part'], r['method'], r['pool'], r['head']) for r in rows}
    for p in C.PRODUCTS:
        Fp, lab, ev = for_product(F, p), for_product(LAB[p], p), C.eval_rows(p)
        S = {n: m for n, _, _, m in SL.define(F, p)}
        head_slices = list(dict.fromkeys([best_s1_slice(p), 'cur_0901']))
        Xl = X_of(lab, FS_REL)
        # ---- A. 사전학습
        for pool in POOLS:
            ref = Fp[S[pool]]
            if len(ref) > POOL_MAX:
                ref = ref.sample(POOL_MAX, random_state=0)
            prep = M.Prep().fit(X_of(ref, FS_REL))      # 풀 기준 스케일 (head 단계에서 다시 스케일)
            Xp = prep(X_of(ref, FS_REL))
            for method in ('mfm', 'scarf'):
                if all((p, 'A', method, pool, h) in have for h in
                       ['score'] * (method == 'mfm') + [f'{hs}:{hh}' for hs in head_slices for hh in ('maha', 'knn5')]):
                    continue
                t = time.time()
                if method == 'mfm':
                    enc, dec = pretrain_mfm(Xp)
                    rec(p, 'A', method, pool, 'score', mfm_score(enc, dec, prep(Xl)), lab, ev, time.time() - t,
                        {'n_pool': len(ref)})
                else:
                    enc = pretrain_scarf(Xp)
                for hs in head_slices:
                    t = time.time()
                    Zr = embed(enc, prep(X_of(Fp[S[hs]], FS_REL)))
                    for hh, s in heads(Zr, embed(enc, prep(Xl))).items():
                        rec(p, 'A', method, pool, f'{hs}:{hh}', s, lab, ev, time.time() - t, {'n_pool': len(ref)})
        # ---- B. 순서 모델
        for hs in head_slices:
            ref = Fp[S[hs]]
            for fsn, cols in (('rel', FS_REL), ('raw', FS_RAW)):
                pr = M.Prep().fit(X_of(ref, cols))
                Wr = windows(ref, pr(X_of(ref, cols)))
                if len(Wr) > 30000:
                    Wr = Wr[np.random.default_rng(0).choice(len(Wr), 30000, replace=False)]
                Wl = windows(lab, pr(X_of(lab, cols)))
                for kind in ('gru', 'cnnae'):
                    if (p, 'B', kind, hs, fsn) in have:
                        continue
                    t = time.time()
                    net = train_seq(kind, Wr)
                    rec(p, 'B', kind, hs, fsn, seq_score(kind, net, Wl), lab, ev, time.time() - t)
        # ---- C. Dual-model
        for pool in ('eq_all', 'pr_cn7rg3_all', 'w2_all'):
            ref = Fp[S[pool]]
            if len(ref) > 60000:
                ref = ref.sample(60000, random_state=0)
            for k in (3, 6):
                if (p, 'C', 'dual', pool, f'k{k}:ens') in have:
                    continue
                t = time.time()
                out, sizes = dual_model(ref, lab, k)
                for nm, s in out.items():
                    rec(p, 'C', 'dual', pool, f'k{k}:{nm}', s, lab, ev, time.time() - t, {'sizes': str(sizes)})
    C.log('완료')


if __name__ == '__main__':
    main()
