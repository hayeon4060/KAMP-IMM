"""공식 문제 데이터(4개 파일) 공용 함수.

이 모듈 자체는 외부데이터를 읽지 않는다. 외부데이터가 필요한 함수
(attach_side)는 이름과 docstring에 명시해 두었다.

노트북에서:
    import sys; sys.path.append("..")
    from src.official import *
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = {"CN7": ("moldset_labeled_cn7.csv", "moldset_unlabeled_cn7.csv"),
         "RG3": ("moldset_labeled_rg3.csv", "moldset_unlabeled_rg3.csv")}
VARS = [c for c in pd.read_csv(RAW / FILES["CN7"][0], nrows=1).columns
        if c not in ("Unnamed: 0", "PassOrFail")]
K = 5


# ── 0. 불러오기와 쌍 구조 ──────────────────────────────────────────
def add_pair(df, window=5):
    """24개 값이 완전히 같은 두 행을 같은 쇼트(pair)로 묶는다. 두 행이 window 행 이내여야 한다.

    같은 쇼트의 LH/RH가 늘 바로 옆에 붙어 있지는 않다(LH,LH,RH,RH 처럼 2~5행 떨어진 경우가
    CN7 35쇼트, RG3 17쇼트). 1차 시각 기준 쇼트표(팀원 cn7_shots/rg3_shots)와 대조해 확인:
    라벨 파일은 CN7 605쌍 + 1행(RH만 있는 쇼트), RG3 591쌍으로 시각 기준 쇼트 수와 정확히 같다.
    window 를 두는 이유: 비라벨에는 50행·233행 떨어진 서로 다른 쇼트가 우연히 값이 같은 경우가 있다."""
    k = pd.util.hash_pandas_object(df[VARS], index=False).values
    pid = np.empty(len(df), dtype=int)
    open_row = {}              # key -> 아직 짝이 없는 행 위치
    p = 0
    for i in range(len(df)):
        j = open_row.get(k[i])
        if j is not None and i - j <= window:
            pid[i] = pid[j]
            del open_row[k[i]]
        else:
            pid[i] = p
            p += 1
            open_row[k[i]] = i
    df = df.copy()
    df["pair"] = pid
    df["key"] = k
    return df


def load(prod):
    """공식 라벨·비라벨을 읽어 order(시간 순), pair, key 를 붙여 돌려준다."""
    lab = (pd.read_csv(RAW / FILES[prod][0])
             .rename(columns={"Unnamed: 0": "order", "PassOrFail": "y"}))
    unl = pd.read_csv(RAW / FILES[prod][1]).rename(columns={"Unnamed: 0": "order"})
    lab = add_pair(lab.sort_values("order").reset_index(drop=True))
    unl = add_pair(unl.sort_values("order").reset_index(drop=True))
    return lab, unl


# ── 1. 이론 상한 F1 ────────────────────────────────────────────────
def f1_ceiling(lab, cols=None):
    """입력(cols)이 같은 행은 같은 판정을 받는다는 제약에서 달성 가능한 최대 F1."""
    cols = VARS if cols is None else cols
    key = pd.util.hash_pandas_object(lab[cols], index=False)
    g = lab.assign(_k=key.values).groupby("_k")["y"].agg(p="sum", t="size")
    g["n"] = g["t"] - g["p"]
    g = g[g["p"] > 0].copy()
    P = lab["y"].sum()
    g["ratio"] = g["p"] / g["t"]
    g = g.sort_values(["ratio", "p"], ascending=False)
    best = dict(F1=0.0)
    tp = fp = 0
    for _, r in g.iterrows():
        tp += r["p"]; fp += r["n"]
        f1 = 2 * tp / (2 * tp + fp + (P - tp))
        if f1 > best["F1"]:
            best = dict(F1=f1, TP=int(tp), FP=int(fp), FN=int(P - tp),
                        정밀도=tp / (tp + fp), 재현율=tp / P)
    return best


# ── 2. 평가 틀 ────────────────────────────────────────────────────
def make_folds(lab, kind, seed=42):
    """A_쌍층화: 쌍을 안 깨고 불량을 고르게 / B_시간블록: 시간 순 연속 5구간."""
    fold = np.full(len(lab), -1)
    if kind == "A_쌍층화":
        # scikit-learn 버전에 따라 StratifiedGroupKFold 결과가 달라져 직접 구현한다.
        # 쌍을 무작위로 섞은 뒤, 불량 있는 쌍부터 '불량이 가장 적은 fold'에,
        # 나머지 쌍은 '행이 가장 적은 fold'에 하나씩 넣는다. 어느 환경에서도 같은 결과.
        rng = np.random.RandomState(seed)
        g = lab.groupby("pair")["y"].agg(bad="sum", n="size")
        idx = g.index.values.copy()
        rng.shuffle(idx)
        g = g.loc[idx].sort_values("bad", ascending=False, kind="stable")
        fold_bad, fold_n, assign = np.zeros(K), np.zeros(K), {}
        for pid, bad, n in zip(g.index, g["bad"].values, g["n"].values):
            f = np.lexsort((fold_n, fold_bad))[0] if bad > 0 else int(np.argmin(fold_n))
            assign[pid] = f
            fold_bad[f] += bad
            fold_n[f] += n
        fold = lab["pair"].map(assign).values.astype(int)
    elif kind == "B_시간블록":
        pairs = lab["pair"].unique()
        for f, chunk in enumerate(np.array_split(pairs, K)):
            fold[lab["pair"].isin(chunk).values] = f
    else:
        raise ValueError(kind)
    return fold


def gap_train_mask(lab, te_mask, gap):
    """시험(test) 행과 order 차이가 gap 이내인 행을 학습에서 뺀 train 마스크.

    A_쌍층화는 바로 옆 쇼트가 학습에 들어가 '같은 시간대'를 외워 맞힐 수 있다.
    gap 행만큼 떼어 놓으면 그 지름길이 막힌다. (1행 ≈ 쇼트 반 개, 쇼트 1개 ≈ 1분 남짓)
    gap=0 이면 일반 교차검증과 같다."""
    o = lab["order"].values
    tr = ~np.asarray(te_mask)
    if gap > 0:
        near = np.zeros(len(lab), bool)
        for t in o[te_mask]:
            near |= np.abs(o - t) <= gap
        tr &= ~near
    return tr


_BAND_CACHE = {}


def random_band(y, n=1000):
    """무작위 순위를 n번 뽑았을 때 AP의 평균과 95% 범위.
    불량이 적으면 AP가 우연히 크게 흔들리므로, 모델 AP가 이 범위를 벗어나야 '유의'로 본다."""
    y = np.asarray(y)
    key = (len(y), int(y.sum()))
    if key not in _BAND_CACHE:
        aps = [average_precision_score(y, np.random.RandomState(s).rand(len(y)))
               for s in range(n)]
        _BAND_CACHE[key] = (float(np.mean(aps)), *np.percentile(aps, [2.5, 97.5]))
    return _BAND_CACHE[key]


def evaluate(lab, score, name, prod, ceiling, split="-"):
    """점수(높을수록 불량 의심)로 지표 계산. ceiling 은 그 조건의 이론 상한 F1."""
    y = lab["y"].values
    s = np.asarray(score, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    ap = average_precision_score(y, s)
    rmean, rlo, rhi = random_band(y)
    p, r, _ = precision_recall_curve(y, s)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.nanmax(2 * p * r / (p + r))
    out = {"모델": name, "제품": prod, "분할": split, "채점 행": int(ok.sum()),
           "AP": ap, "무작위 평균 대비": ap / rmean,
           "무작위 상단(97.5%)": rhi, "유의": "O" if ap > rhi else "-",
           "F1(최적,참고)": f1, "상한": ceiling, "상한 대비": f1 / ceiling}
    # 동점은 무작위로 섞는다. 행 순서(=시간 순서)대로 두면 불량이 몰린 시간대가
    # 이유 없이 위로 올라와 검출률이 부풀려진다.
    tie = np.random.RandomState(0).rand(len(s))
    order = np.lexsort((tie, -s))
    for k in [1, 2, 5, 10]:
        top = order[: max(1, int(len(s) * k / 100))]
        out[f"상위{k}%"] = y[top].sum() / y.sum()
    return out


# ── 외부데이터 사용 (B층) ──────────────────────────────────────────
def attach_side(lab):
    """[외부데이터] moldset_labeled.csv(KAMP 공개본)의 PART_NAME 으로 좌우(LH/RH)를 붙인다.
    공식 파일의 order 와 원본의 Unnamed: 0 이 1:1 대응함을 확인했다(라벨 불일치 0)."""
    ml = pd.read_csv(RAW / "moldset_labeled.csv",
                     usecols=["Unnamed: 0", "PART_NAME", "PassOrFail"])
    m = ml.drop_duplicates("Unnamed: 0").set_index("Unnamed: 0")
    lab = lab.copy()
    lab["side"] = lab["order"].map(m["PART_NAME"].str.extract(r"(LH|RH)")[0])
    y_ml = lab["order"].map(m["PassOrFail"])
    if not pd.api.types.is_numeric_dtype(y_ml):          # "Y"/"N" 문자열이면 숫자로
        y_ml = (y_ml.astype(str).str.strip() == "N").astype(float)
    bad = int((lab["y"] != y_ml).sum()) + int(lab["side"].isna().sum())
    if bad:
        raise ValueError(f"좌우 연결 불일치 {bad}행 — 원본 파일을 확인하세요")
    lab["is_RH"] = (lab["side"] == "RH").astype(int)
    return lab