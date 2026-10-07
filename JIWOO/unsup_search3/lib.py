"""
lib.py — unsup_search3 공용: 라이브러리 읽기, 변수 묶음 35개, 학습 구간 15개, 모델 10종, 평가
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'unsup_search'))
import common as C  # noqa: E402
import slices as SL  # noqa: E402
import models as M  # noqa: E402

OUT = ROOT / 'outputs'
CACHE = OUT / 'cache'
SCORES = OUT / 'scores'
SCORES.mkdir(parents=True, exist_ok=True)
REF_MAX = 30000
MIN_REF = 300
log = C.log


# ====================================================================== 데이터
def load():
    LU = pd.read_pickle(CACHE / 'lib_unlabeled.pkl')
    LL = pd.read_pickle(CACHE / 'lib_labeled.pkl')
    fam = json.load(open(CACHE / 'families.json', encoding='utf-8'))
    return LU, LL, fam


def for_product(df, p):
    """RG3는 c17의 흐름저항 편차를 RG3 기록 단위 기준으로 교체 (기존과 같음)"""
    if p == 'rg3' and 'dev_flow_res_rg3' in df.columns:
        df = df.copy()
        df['dev_flow_res'] = df['dev_flow_res_rg3']
    return df


# ====================================================================== 변수 묶음 35개 (PREREG)
FAMILIES = ['c17', 'dv26', 'raw26', 'rz5', 'rz50', 'rz200', 'd1', 'd2', 'ew', 'vol10', 'mx5', 'lag', 'ctx',
            'drift0', 'rmed', 'x3', 'n12']


def featsets(fam):
    fs = {f: [f] for f in FAMILIES}
    for f in FAMILIES:
        if f != 'dv26':
            fs[f'dv26+{f}'] = ['dv26', f]
    fs['devall'] = ['dv26', 'rz5', 'rz50', 'rz200', 'ew', 'd1']
    fs['all'] = list(FAMILIES)
    return fs


def cols_of(fam, families):
    out = []
    for f in families:
        out += [c for c in fam[f] if c not in out]
    return out


# ====================================================================== 학습 구간 15개 (PREREG)
SLICES = ['cur_0901', 'w2_all', 'w2_pre', 'w2_last4w', 'w2_last8w', 'w2_H1', 'cl_skipstart20', 'cl_lowerr_lot',
          'cl_longrun', 'eq_other', 'eq_all', 'pr_other', 'pr_cn7rg3_w2', 'pr_w2_allparts', 'pr_everything']


def slice_masks(LU, p):
    S = {n: m for n, _, _, m in SL.define(LU, p)}
    ws = ((LU['equip'] == C.EQUIP_W2) & (LU['prod'] == p.upper())).values
    runlen = LU.groupby('run_id')['run_id'].transform('size').values
    S['cl_longrun'] = ws & (runlen >= 300)
    return {k: S[k] for k in SLICES}


# ====================================================================== 모델 10종 (PREREG)
MODELS = ['maha_lw', 'gmm2', 'gmm4', 'knn5', 'lof', 'ocsvm', 'iforest', 'pca', 'ecod', 'lmaha300']


def local_maha(A, B, k=300):
    """국소 Mahalanobis: 학습 구간 유클리드 최근접 k개로 Ledoit-Wolf → 거리 (unsup_search2 C2와 같음)"""
    k = min(k, len(A) - 1)
    idx = NearestNeighbors(n_neighbors=k).fit(A).kneighbors(B, return_distance=False)
    out = np.empty(len(B))
    for i, q in enumerate(B):
        out[i] = np.sqrt(LedoitWolf().fit(A[idx[i]]).mahalanobis(q[None])[0])
    return out


def fit_score_all(Xr, Xl, models=MODELS, seed=0):
    """원 변수 행렬 → Prep(학습 구간 중앙값·IQR) → 모델별 라벨 점수. 실패는 NaN + 오류문"""
    pr = M.Prep().fit(Xr.astype(float))
    A, B = pr(Xr.astype(float)), pr(Xl.astype(float))
    keep = A.std(0) > 0                                    # 학습 구간에서 상수인 열 제거
    A, B = A[:, keep], B[:, keep]
    res = {}
    for m in models:
        try:
            if m == 'lmaha300':
                s = local_maha(A, B, 300)
            else:
                mod = M.REGISTRY[m]()
                if hasattr(mod, 'seed'):
                    mod.seed = seed
                s = mod.fit(A).score(B)
            res[m] = (np.asarray(s, np.float32), '')
        except Exception as e:  # noqa
            res[m] = (np.full(len(B), np.nan, np.float32), repr(e)[:200])
    return res, int(keep.sum())


# ====================================================================== 평가
def eval_table(p, lab_ts):
    """행 단위 평가표 + 각 행이 가리키는 라벨 샷 위치"""
    ev = C.eval_rows(p).sort_values('TimeStamp', kind='stable').reset_index(drop=True)
    pos = pd.Series(np.arange(len(lab_ts)), index=pd.DatetimeIndex(lab_ts))
    pos = pos[~pos.index.duplicated()]
    ev['pos'] = ev['TimeStamp'].map(pos)
    assert ev['pos'].notna().all(), '평가 행과 라벨 샷 연결 실패'
    ev['pos'] = ev['pos'].astype(int)
    ev['hour'] = ev['TimeStamp'].dt.floor('h')
    return ev


def _tb(s):
    s = np.asarray(s, float)
    s = np.where(np.isfinite(s), s, np.nanmin(s[np.isfinite(s)]) if np.isfinite(s).any() else 0.0)
    return s + np.random.RandomState(0).rand(len(s)) * 1e-9 * (np.std(s) + 1e-12)


def evaluate(ev, shot_score):
    r = {}
    for per in ('official', 'extra'):
        e = ev[ev['period'] == per]
        y = e['y'].values.astype(int)
        s = _tb(np.asarray(shot_score, float)[e['pos'].values])
        key = per[:3]
        r[f'{key}_AP'] = average_precision_score(y, s)
        o = np.argsort(-s, kind='stable')
        for b in (0.008, 0.0165, 0.05):
            k = max(1, int(round(len(y) * b)))
            r[f'{key}_TP{b * 100:g}'] = int(y[o[:k]].sum())
        k5 = int(round(len(y) * 0.05))
        hit = o[:k5][y[o[:k5]] == 1]
        r[f'{key}_hours5'] = len(set(e['hour'].values[hit]))   # 상위 5%에 든 불량의 서로 다른 시간대 수
    return r
