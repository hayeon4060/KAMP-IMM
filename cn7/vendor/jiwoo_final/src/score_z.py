"""비교 기준선: 라벨 없는 과거기준선 이탈도 z (미래를 보지 않음)

각 쇼트에 대해 바로 앞 K개 쇼트(검사 여부와 무관, 비라벨 정상 모드 포함)를 기준선으로
변수별 강건 z = |x - 중앙값| / max(1.4826*MAD, 표준편차, 바닥값) 을 구해 20에서 자르고 rms 로 합친다.
"""
import warnings
import numpy as np
from sklearn.metrics import average_precision_score

from shots_data import VARS

warnings.filterwarnings('ignore', category=RuntimeWarning)

ZCAP = 20.0
# 바닥값 = 변수별 기록 분해능(한 눈금). 시간·위치 0.01, 속도·압력·온도·RPM 0.1, 형개위치 1.0
FLOOR = {v: 0.01 for v in ["Injection_Time", "Filling_Time", "Plasticizing_Time", "Cycle_Time",
                           "Clamp_Close_Time", "Cushion_Position", "Plasticizing_Position"]}
FLOOR["Clamp_Open_Position"] = 1.0
FLOOR_ARR = np.array([FLOOR.get(v, 0.1) for v in VARS])


def past_baseline_z(st, K=10, floor=FLOOR_ARR):
    X = st[VARS].to_numpy(float)
    n, p = X.shape
    Z = np.full((n, p), np.nan)
    for i in range(K, n):
        B = X[i - K:i]
        med = np.median(B, 0)
        mad = 1.4826 * np.median(np.abs(B - med), 0)
        sd = B.std(0, ddof=1)
        den = np.maximum(np.maximum(mad, sd), floor)
        Z[i] = np.minimum(np.abs(X[i] - med) / den, ZCAP)
    return Z


def rms(Z):
    with np.errstate(all='ignore'):
        return np.sqrt(np.nanmean(Z ** 2, 1))


def to_rows(L, st, col):
    """쇼트 점수를 라벨 행(LH/RH)에 붙임."""
    m = st.set_index("TimeStamp")[col]
    return L.TimeStamp.map(m).to_numpy()


def eval_budget(y, s, budget=0.05):
    """점수 상위 budget 비율 행을 불량 판정. TP, F1, AP 반환."""
    y = np.asarray(y); s = np.asarray(s, float)
    n = len(y); k = int(np.ceil(budget * n))
    order = np.argsort(-s, kind="stable")
    pred = np.zeros(n, int); pred[order[:k]] = 1
    tp = int((pred & y).sum())
    f1 = 2 * tp / (k + y.sum()) if (k + y.sum()) else 0.0
    return dict(TP=tp, F1=f1, AP=average_precision_score(y, s), k=k, P=int(y.sum()))
