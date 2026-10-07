"""
models.py — 비지도 이상탐지 모델 모음 (정상 구간으로만 학습, 점수 클수록 불량 의심)

공통 전처리: 학습 구간 중앙값·IQR 스케일 → ±20 클리핑 → NaN=0 (제출 모델과 같음)
make(name) → 객체.fit(X) / .score(X)   (X는 전처리된 numpy 배열)
"""
import numpy as np
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

CLIP = 20.0


class Prep:
    def fit(self, X):
        self.med = np.nanmedian(X, 0)
        q75, q25 = np.nanpercentile(X, [75, 25], 0)
        sd = np.nanstd(X, 0)
        self.iqr = np.where(q75 - q25 > 1e-9, q75 - q25, np.where(sd > 1e-9, sd, 1.0))
        return self

    def __call__(self, X):
        return np.nan_to_num(np.clip((X - self.med) / self.iqr, -CLIP, CLIP), nan=0.0)


def _sub(X, n, seed=0):
    if len(X) <= n:
        return X
    return X[np.random.default_rng(seed).choice(len(X), n, replace=False)]


# ---------------------------------------------------------------- 고전 모델
class MahaLW:
    def fit(self, X):
        self.m = LedoitWolf().fit(X); return self

    def score(self, X):
        return np.sqrt(self.m.mahalanobis(X))


class MahaMCD:
    def fit(self, X):
        self.m = MinCovDet(random_state=0, support_fraction=0.9).fit(_sub(X, 5000)); return self

    def score(self, X):
        return np.sqrt(self.m.mahalanobis(X))


class GMM:
    def __init__(self, k, seed=0):
        self.k, self.seed = k, seed

    def fit(self, X):
        self.m = GaussianMixture(self.k, covariance_type='full', reg_covar=1e-3, random_state=self.seed,
                                 n_init=2).fit(_sub(X, 20000)); return self

    def score(self, X):
        return -self.m.score_samples(X)


class IF:
    def fit(self, X):
        self.m = IsolationForest(n_estimators=300, random_state=0, n_jobs=1).fit(X); return self

    def score(self, X):
        return -self.m.score_samples(X)


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.m = NearestNeighbors(n_neighbors=self.k).fit(_sub(X, 20000)); return self

    def score(self, X):
        return self.m.kneighbors(X)[0].mean(1)


class LOF:
    def fit(self, X):
        self.m = LocalOutlierFactor(n_neighbors=35, novelty=True).fit(_sub(X, 10000)); return self

    def score(self, X):
        return -self.m.score_samples(X)


class OCSVM:
    def fit(self, X):
        self.m = OneClassSVM(nu=0.01, gamma='scale').fit(_sub(X, 5000)); return self

    def score(self, X):
        return -self.m.score_samples(X)


class PCARecon:
    def fit(self, X):
        self.m = PCA(0.9, random_state=0).fit(X); return self

    def score(self, X):
        Z = self.m.inverse_transform(self.m.transform(X))
        return np.sqrt(((X - Z) ** 2).sum(1))


class ECOD:
    """경험적 누적분포 꼬리 확률 (Li 외 2022) — 변수별 양쪽 꼬리 −log p 합, 왜도로 방향 선택"""
    def fit(self, X):
        self.S = np.sort(X, 0)
        n = len(X)
        self.skew = ((X - X.mean(0)) ** 3).mean(0) / (X.std(0) ** 3 + 1e-12)
        self.n = n; return self

    def score(self, X):
        n = self.n
        L = np.stack([np.searchsorted(self.S[:, j], X[:, j], side='right') for j in range(X.shape[1])], 1)
        R = np.stack([n - np.searchsorted(self.S[:, j], X[:, j], side='left') for j in range(X.shape[1])], 1)
        pl, pr = -np.log((L + 1) / (n + 1)), -np.log((R + 1) / (n + 1))
        auto = np.where(self.skew < 0, pl, pr)
        return np.maximum.reduce([pl.sum(1), pr.sum(1), auto.sum(1)])


class HBOS:
    def fit(self, X, bins=30):
        self.edges, self.dens = [], []
        for j in range(X.shape[1]):
            h, e = np.histogram(X[:, j], bins=bins)
            d = (h + 1) / (h.sum() + bins)
            self.edges.append(e); self.dens.append(d)
        return self

    def score(self, X):
        s = np.zeros(len(X))
        for j, (e, d) in enumerate(zip(self.edges, self.dens)):
            i = np.clip(np.searchsorted(e, X[:, j]) - 1, 0, len(d) - 1)
            out = (X[:, j] < e[0]) | (X[:, j] > e[-1])
            s += np.where(out, -np.log(1e-6), -np.log(d[i]))
        return s


# ---------------------------------------------------------------- 신경망 모델 (torch, CPU)
def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


def _mlp(nn, sizes, last_act=False):
    L = []
    for i in range(len(sizes) - 1):
        L.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            L.append(nn.LeakyReLU(0.1))
    return nn.Sequential(*L)


class AE:
    """오토인코더 / Denoising AE(noise>0) — 재구성 오차"""
    def __init__(self, noise=0.0, epochs=60, latent=None, seed=0):
        self.noise, self.epochs, self.latent, self.seed = noise, epochs, latent, seed

    def fit(self, X):
        torch = _torch(); nn = torch.nn
        torch.manual_seed(self.seed)
        d = X.shape[1]; h = max(8, d); z = self.latent or max(3, d // 4)
        self.enc = _mlp(nn, [d, 2 * h, h, z]); self.dec = _mlp(nn, [z, h, 2 * h, d])
        X = torch.tensor(_sub(X, 30000, self.seed), dtype=torch.float32)
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), 1e-3, weight_decay=1e-5)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            for idx in torch.randperm(len(X), generator=g).split(256):
                xb = X[idx]
                xin = xb + self.noise * torch.randn_like(xb) if self.noise else xb
                loss = ((self.dec(self.enc(xin)) - xb) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def embed(self, X):
        torch = _torch()
        with torch.no_grad():
            return self.enc(torch.tensor(X, dtype=torch.float32)).numpy()

    def score(self, X):
        torch = _torch()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            return ((self.dec(self.enc(x)) - x) ** 2).sum(1).sqrt().numpy()


class VAE:
    """VAE — 음의 ELBO (재구성 + KL), 평가 시 표본 10개 평균"""
    def __init__(self, epochs=60, seed=0):
        self.epochs, self.seed = epochs, seed

    def fit(self, X):
        torch = _torch(); nn = torch.nn
        torch.manual_seed(self.seed)
        d = X.shape[1]; h = max(8, d); self.z = z = max(3, d // 4)
        self.enc = _mlp(nn, [d, 2 * h, h], last_act=True); self.mu = nn.Linear(h, z); self.lv = nn.Linear(h, z)
        self.dec = _mlp(nn, [z, h, 2 * h, d])
        ps = [p for m in (self.enc, self.mu, self.lv, self.dec) for p in m.parameters()]
        opt = torch.optim.Adam(ps, 1e-3)
        X = torch.tensor(_sub(X, 30000, self.seed), dtype=torch.float32)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            for idx in torch.randperm(len(X), generator=g).split(256):
                loss = self._nelbo(X[idx]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def _nelbo(self, x):
        torch = _torch()
        hh = self.enc(x); mu, lv = self.mu(hh), self.lv(hh).clamp(-8, 8)
        zz = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        rec = ((self.dec(zz) - x) ** 2).sum(1)
        kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(1)
        return rec + kl

    def score(self, X):
        torch = _torch()
        torch.manual_seed(123)
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            return torch.stack([self._nelbo(x) for _ in range(10)]).mean(0).numpy()


class WGANGP:
    """WGAN-GP — 학습 후 critic 점수의 부호 반대(정상일수록 critic 높음)"""
    def __init__(self, iters=4000, seed=0):
        self.iters, self.seed = iters, seed

    def fit(self, X):
        torch = _torch(); nn = torch.nn
        torch.manual_seed(self.seed)
        d = X.shape[1]; h = max(32, 2 * d); self.zd = zd = max(4, d // 2)
        G = _mlp(nn, [zd, h, h, d]); D = _mlp(nn, [d, h, h, 1])
        oG = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.9))
        oD = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.5, 0.9))
        X = torch.tensor(_sub(X, 30000, self.seed), dtype=torch.float32)
        g = torch.Generator().manual_seed(self.seed)
        for it in range(self.iters):
            for _ in range(5):
                xb = X[torch.randint(len(X), (256,), generator=g)]
                fake = G(torch.randn(256, zd)).detach()
                a = torch.rand(256, 1)
                xi = (a * xb + (1 - a) * fake).requires_grad_(True)
                gr = torch.autograd.grad(D(xi).sum(), xi, create_graph=True)[0]
                lD = D(fake).mean() - D(xb).mean() + 10 * ((gr.norm(2, 1) - 1) ** 2).mean()
                oD.zero_grad(); lD.backward(); oD.step()
            lG = -D(G(torch.randn(256, zd))).mean()
            oG.zero_grad(); lG.backward(); oG.step()
        self.D = D
        return self

    def score(self, X):
        torch = _torch()
        with torch.no_grad():
            return -self.D(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()


class GAAL:
    """SO-GAAL 간이판 (Liu 외 2019): 생성기가 경계 근처 가짜를 만들고, 판별기가 정상/가짜 경계 학습.
    생성기를 일정 반복 후 고정하고 판별기만 추가 학습 → 점수 = 1 − D(x)"""
    def __init__(self, iters=3000, stop_g=1500, seed=0):
        self.iters, self.stop_g, self.seed = iters, stop_g, seed

    def fit(self, X):
        torch = _torch(); nn = torch.nn
        torch.manual_seed(self.seed)
        d = X.shape[1]; h = max(32, 2 * d)
        G = _mlp(nn, [d, h, h, d]); D = nn.Sequential(_mlp(nn, [d, h, h, 1]), nn.Sigmoid())
        oG = torch.optim.Adam(G.parameters(), 1e-4); oD = torch.optim.Adam(D.parameters(), 1e-4)
        bce = nn.BCELoss()
        X = torch.tensor(_sub(X, 30000, self.seed), dtype=torch.float32)
        g = torch.Generator().manual_seed(self.seed)
        sd = X.std(0)
        for it in range(self.iters):
            xb = X[torch.randint(len(X), (256,), generator=g)]
            fake = G(torch.randn(256, d) * sd)
            lD = bce(D(xb), torch.ones(256, 1)) + bce(D(fake.detach()), torch.zeros(256, 1))
            oD.zero_grad(); lD.backward(); oD.step()
            if it < self.stop_g:
                lG = bce(D(fake), torch.ones(256, 1))
                oG.zero_grad(); lG.backward(); oG.step()
        self.D = D
        return self

    def score(self, X):
        torch = _torch()
        with torch.no_grad():
            return 1 - self.D(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()


class DeepSVDD:
    """Deep SVDD (Ruff 외 2018, one-class) — 편향 없는 망, 중심까지 거리"""
    def __init__(self, epochs=60, seed=0):
        self.epochs, self.seed = epochs, seed

    def fit(self, X):
        torch = _torch(); nn = torch.nn
        torch.manual_seed(self.seed)
        d = X.shape[1]; h = max(16, d)
        self.net = nn.Sequential(nn.Linear(d, h, bias=False), nn.LeakyReLU(0.1), nn.Linear(h, 8, bias=False))
        X = torch.tensor(_sub(X, 30000, self.seed), dtype=torch.float32)
        with torch.no_grad():
            c = self.net(X).mean(0)
            c[(c.abs() < 0.01)] = 0.01
        self.c = c
        opt = torch.optim.Adam(self.net.parameters(), 1e-3, weight_decay=1e-5)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            for idx in torch.randperm(len(X), generator=g).split(256):
                loss = ((self.net(X[idx]) - c) ** 2).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def score(self, X):
        torch = _torch()
        with torch.no_grad():
            return ((self.net(torch.tensor(X, dtype=torch.float32)) - self.c) ** 2).sum(1).numpy()


class LatentHead:
    """AE 잠재공간 위 고전 모델 (IF / LOF / kNN / Mahalanobis)"""
    def __init__(self, head):
        self.head = head

    def fit(self, X):
        self.ae = AE(epochs=60).fit(X)
        Z = self.ae.embed(X)
        self.zp = Prep().fit(Z)
        self.h = make(self.head).fit(self.zp(Z))
        return self

    def score(self, X):
        return self.h.score(self.zp(self.ae.embed(X)))


REGISTRY = {
    'maha_lw': MahaLW, 'maha_mcd': MahaMCD,
    'gmm2': lambda: GMM(2), 'gmm4': lambda: GMM(4), 'gmm8': lambda: GMM(8),
    'iforest': IF, 'knn5': lambda: KNN(5), 'knn20': lambda: KNN(20), 'lof': LOF, 'ocsvm': OCSVM,
    'pca': PCARecon, 'ecod': ECOD, 'hbos': HBOS,
    'ae': AE, 'dae': lambda: AE(noise=0.3), 'vae': VAE, 'wgan': WGANGP, 'gaal': GAAL, 'dsvdd': DeepSVDD,
    'lat_if': lambda: LatentHead('iforest'), 'lat_knn': lambda: LatentHead('knn5'),
    'lat_lof': lambda: LatentHead('lof'), 'lat_maha': lambda: LatentHead('maha_lw'),
}
DEEP = {'ae', 'dae', 'vae', 'wgan', 'gaal', 'dsvdd', 'lat_if', 'lat_knn', 'lat_lof', 'lat_maha'}


def make(name):
    return REGISTRY[name]()


def fit_score(name, Xref, Xlab, seed=0):
    """원 변수 행렬 → 전처리 → 학습 → 라벨 점수"""
    pr = Prep().fit(Xref)
    m = REGISTRY[name]()
    if hasattr(m, 'seed'):
        m.seed = seed
    m.fit(pr(Xref))
    return m.score(pr(Xlab))
