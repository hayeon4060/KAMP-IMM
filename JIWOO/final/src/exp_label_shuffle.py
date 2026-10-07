"""검증 실험 A. 라벨 섞기 검증 (y-randomization)

같은 평가 절차를 '진짜 라벨'과 '무작위로 섞은 라벨'에 똑같이 돌려,
점수 차이가 라벨 정보에서 오는지 확인한다. 섞어도 같은 점수가 나오면 그 점수는 신호가 아니다.

1) 가이드북식: SMOTE(분할 전) → RF(200) → 무작위 층화 5-fold, F1(기본 임계 0.5, 합성 포함 데이터)
2) 지도학습 정직 검증: 쇼트 단위 층화 5-fold, 시험 행 ±10행 학습 제외(간격 두기), fold 안 SMOTE → RF(200), 검사예산 5% F1
3) 과거기준선 이탈도(K=10, 라벨 없음), 검사예산 5% F1
출력: outputs/experiments/label_shuffle.json
"""
import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict

from paths import EXP
from score_z import past_baseline_z, rms, to_rows, eval_budget
from shots_data import load_all, official, stream, PRODS

N_SHUF_RF, N_SHUF_SCORE = 20, 1000


def guidebook(X, y, seed):
    Xs, ys = SMOTE(random_state=seed).fit_resample(X, y)
    rf = RandomForestClassifier(200, random_state=seed, n_jobs=1)
    p = cross_val_predict(rf, Xs, ys, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return f1_score(ys, p)


def honest_rf(X, y, groups, seed, gap=10):
    n = len(y); oof = np.zeros(n)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        near = np.zeros(n, bool)
        for i in te:
            near[max(0, i - gap):i + gap + 1] = True
        tr = tr[~near[tr]]
        Xt, yt = X[tr], y[tr]
        if yt.sum() >= 2:
            Xt, yt = SMOTE(random_state=seed, k_neighbors=min(5, int(yt.sum()) - 1)).fit_resample(Xt, yt)
        rf = RandomForestClassifier(200, random_state=seed, n_jobs=1).fit(Xt, yt)
        oof[te] = rf.predict_proba(X[te])[:, 1] if yt.sum() else 0
    return eval_budget(y, oof)["F1"]


def main():
    d = load_all(); L, S, U = d["L"], d["S"], d["U"]
    rows = []
    for p in PRODS:
        O = official(p).sort_values("Unnamed: 0").reset_index(drop=True)
        X = O.drop(columns=["Unnamed: 0", "PassOrFail"]).to_numpy(float)
        y = O.PassOrFail.to_numpy(int)
        Lo = L[(L["prod"] == p) & (L.period == "official")].set_index("oid").loc[O["Unnamed: 0"]]
        assert (Lo.y.to_numpy() == y).all()
        groups = pd.factorize(Lo.TimeStamp)[0]
        st = stream(S, U, p); st["s"] = rms(past_baseline_z(st, 10))
        s = to_rows(Lo, st, "s")
        rng = np.random.default_rng(0)
        perms = [rng.permutation(y) for _ in range(max(N_SHUF_RF, N_SHUF_SCORE))]

        real = Parallel(-1)(delayed(guidebook)(X, y, sd) for sd in range(5))
        shuf = Parallel(-1)(delayed(guidebook)(X, perms[i], i) for i in range(N_SHUF_RF))
        rows.append(dict(prod=p, method="guidebook", metric="F1 (SMOTE 후 무작위 5-fold, 임계 0.5)",
                         real=np.mean(real), real_all=real, shuf=shuf))
        real = Parallel(-1)(delayed(honest_rf)(X, y, groups, sd) for sd in range(5))
        shuf = Parallel(-1)(delayed(honest_rf)(X, perms[i], groups, i) for i in range(N_SHUF_RF))
        rows.append(dict(prod=p, method="honest_rf", metric="F1@검사 5%",
                         real=np.mean(real), real_all=real, shuf=shuf))
        r = eval_budget(y, s)["F1"]
        shuf = [eval_budget(perms[i], s)["F1"] for i in range(N_SHUF_SCORE)]
        rows.append(dict(prod=p, method="deviation_K10", metric="F1@검사 5%", real=r, real_all=[r], shuf=shuf))
        for rr in rows[-3:]:
            sh = np.array(rr["shuf"])
            print(f"  {p} {rr['method']:14s} real {rr['real']:.4f} | shuf mean {sh.mean():.4f} "
                  f"[{sh.min():.3f}, {sh.max():.3f}] p={(np.sum(sh >= rr['real']) + 1) / (len(sh) + 1):.4f}", flush=True)
    json.dump(rows, open(EXP / "label_shuffle.json", "w"), ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
