"""
5단계 (별도 트랙, 라벨 사용): 적응 임계값 준지도 — FixMatch / FreeMatch / ATSSL식(불균형 적응)
 + 비교: 같은 분할 지도학습(MLP·RF)

누설 차단: 기간 블록 교차 — 공식 기간 라벨로 학습 → 추가 기간 채점, 추가 → 공식
  (같은 기간 안 CV는 같은 사건의 이웃 샷을 외우는 누설이 생기므로 쓰지 않음)
입력: 상대 변수(c17+dev26), 비라벨 = 1단계 최고 구간(같은 제품) 정상 샷 3만
증강: 약 = 가우시안 0.05 / 강 = 변수 30% 교체(SCARF식) + 가우시안 0.2
ATSSL식: 클래스별 '확신 예측 수' 비율로 소수 클래스 임계값을 낮춤 (FlexMatch 사상, 논문 코드 아님·설명 기반 구현)
출력: outputs/s5_results.csv, outputs/scores/s5_scores.pkl
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier

import common as C
import slices as SL
import models as M
from s1_slices import for_product
from s3_selfsup import FS_REL, X_of, best_s1_slice

torch.set_num_threads(14)
SEEDS = (0, 1, 2)
ITERS = 3000
OUTP = C.OUT / 's5_results.csv'
SCP = C.OUT / 'scores' / 's5_scores.pkl'


def net(d):
    return nn.Sequential(nn.Linear(d, 64), nn.LeakyReLU(0.1), nn.Dropout(0.2), nn.Linear(64, 32),
                         nn.LeakyReLU(0.1), nn.Linear(32, 2))


def weak(x, g):
    return x + 0.05 * torch.randn(x.shape, generator=g)


def strong(x, g):
    m = (torch.rand(x.shape, generator=g) < 0.3).float()
    perm = torch.randint(len(x), x.shape, generator=g)
    xs = torch.gather(x, 0, perm)
    return x * (1 - m) + xs * m + 0.2 * torch.randn(x.shape, generator=g)


def train(method, Xl, yl, Xu, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    f = net(Xl.shape[1])
    opt = torch.optim.Adam(f.parameters(), 1e-3, weight_decay=1e-4)
    Xl, Xu = torch.tensor(Xl, dtype=torch.float32), torch.tensor(Xu, dtype=torch.float32)
    yl = torch.tensor(yl, dtype=torch.long)
    pos, neg = torch.where(yl == 1)[0], torch.where(yl == 0)[0]
    tau, ema_conf, ema_p = 0.95, torch.tensor(0.5), torch.tensor([0.5, 0.5])   # FreeMatch 상태
    sigma = torch.zeros(2)                                                      # ATSSL식 상태
    ce = nn.CrossEntropyLoss(reduction='none')
    for it in range(ITERS):
        bi = torch.cat([pos[torch.randint(len(pos), (32,), generator=g)],
                        neg[torch.randint(len(neg), (32,), generator=g)]])      # 균형 표집
        loss = ce(f(weak(Xl[bi], g)), yl[bi]).mean()
        if method != 'sup_mlp' and it > 300:                                    # 300회 예열 뒤 비라벨 사용
            ub = Xu[torch.randint(len(Xu), (448,), generator=g)]
            with torch.no_grad():
                pw = torch.softmax(f(weak(ub, g)), 1)
            conf, pl = pw.max(1)
            if method == 'fixmatch':
                thr = torch.full((2,), tau)
            elif method == 'freematch':
                ema_conf = 0.999 * ema_conf + 0.001 * conf.mean()
                ema_p = 0.999 * ema_p + 0.001 * pw.mean(0)
                thr = ema_p / ema_p.max() * ema_conf                             # 자기적응 전역 × 클래스별 국소
            else:   # atssl: 클래스별 확신 예측 수 → 불균형 적응 임계값 (소수 클래스일수록 낮춤)
                sigma = 0.99 * sigma + 0.01 * torch.bincount(pl[conf > tau], minlength=2).float()
                beta = sigma / sigma.max().clamp(min=1e-6)
                thr = (beta / (2 - beta)).clamp(min=0.5) * tau
            mask = (conf >= thr[pl]).float()
            lu = (ce(f(strong(ub, g)), pl) * mask).mean()
            loss = loss + lu
        opt.zero_grad(); loss.backward(); opt.step()
    f.eval()
    return f


def predict(f, X):
    with torch.no_grad():
        return torch.softmax(f(torch.tensor(X, dtype=torch.float32)), 1)[:, 1].numpy()


def main():
    F = C.build_cache()
    LAB = C.build_labeled_feats()
    rows, SC = [], {}
    for p in C.PRODUCTS:
        Fp, lab, ev = for_product(F, p), for_product(LAB[p], p), C.eval_rows(p)
        S = {n: m for n, _, _, m in SL.define(F, p)}
        U = Fp[S[best_s1_slice(p)]]
        U = U.sample(min(30000, len(U)), random_state=0)
        pr = M.Prep().fit(X_of(U, FS_REL))
        Xu = pr(X_of(U, FS_REL))
        Xall = pr(X_of(lab, FS_REL))
        y = lab['PassOrFail'].values.astype(int)
        usable = (lab['split'] != '제외').values & (lab['Reason'] != '초기허용불량').values
        for tr, te in (('2차', '추가'), ('추가', '2차')):
            mtr = usable & (lab['split'] == tr).values
            if y[mtr].sum() < 3:
                continue
            ev_te = ev[ev['period'] == ('extra' if te == '추가' else 'official')]
            for method in ('sup_rf', 'sup_mlp', 'fixmatch', 'freematch', 'atssl'):
                ss = []
                for seed in SEEDS:
                    if method == 'sup_rf':
                        rf = RandomForestClassifier(500, min_samples_leaf=3, class_weight='balanced_subsample',
                                                    random_state=seed, n_jobs=14).fit(Xall[mtr], y[mtr])
                        ss.append(rf.predict_proba(Xall)[:, 1])
                    else:
                        ss.append(predict(train(method, Xall[mtr], y[mtr], Xu, seed), Xall))
                s = np.mean(ss, 0)
                SC[(p, tr, method)] = s
                r = C.evaluate(p, lab['TimeStamp'], s, ev_te)
                key = 'ext' if te == '추가' else 'off'
                rr = {'product': p, 'train': tr, 'test': te, 'method': method, 'n_train_pos': int(y[mtr].sum()),
                      'AP': r[f'{key}_AP'], 'TP5': r[f'{key}_TP5'], 'P': r[f'{key}_P'], 'n': r[f'{key}_n']}
                rows.append(rr)
                C.log(f'{p} {tr}→{te} {method:9s} AP={rr["AP"]:.3f} TP5={rr["TP5"]}/{rr["P"]}')
                pd.DataFrame(rows).to_csv(OUTP, index=False, encoding='utf-8-sig')
                pd.to_pickle(SC, SCP)


if __name__ == '__main__':
    main()
