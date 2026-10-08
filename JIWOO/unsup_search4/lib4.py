"""
lib4.py — unsup_search4 공용: 라이브러리 읽기, 학습 창, 블록, 학습·점수, 평가
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'unsup_search'))
sys.path.insert(0, str(ROOT.parent / 'unsup_search3'))
import common as C  # noqa: E402
import slices as SL  # noqa: E402
import models as M  # noqa: E402
import lib as L3  # noqa: E402

OUT = ROOT / 'outputs'
CACHE = OUT / 'cache'
MM = CACHE / 'mm'
MM.mkdir(parents=True, exist_ok=True)
MODELS = ['gmm2', 'maha_lw', 'knn5', 'pca']
MAX_COLS = 400
REF_MAX = 30000
log = C.log


def load():
    U = pd.read_pickle(CACHE / 'lib4_unlabeled.pkl')
    LBL = pd.read_pickle(CACHE / 'lib4_labeled.pkl')
    cols = json.load(open(CACHE / 'columns.json', encoding='utf-8'))
    return U, LBL, cols


def blocks(cols):
    b = {}
    for i, c in enumerate(cols):
        tr, grp, _ = c.split('|', 2)
        b.setdefault(f'{tr}|{grp}', []).append(i)
    return b


DV26 = ['dv|시간', 'dv|속도', 'dv|압력', 'dv|위치', 'dv|배럴', 'dv|호퍼금형', 'dv|관계R']


def windows(meta, p):
    t = meta['TimeStamp']
    P = p.upper()
    ws = ((meta['equip'] == C.EQUIP_W2) & (meta['prod'] == P)).values
    ev_days = set(C.eval_rows(p)['TimeStamp'].dt.date)
    return {'W1': ws & (t < SL.LAB_START[p]).values,
            'w2_all': ws,
            'eq_all': (meta['prod'] == P).values,
            'W2': ws & ~t.dt.date.isin(ev_days).values}


def mm_path(p, win):
    return MM / f'{p}_{win}.npy'


def prepare_memmaps(U, LBL, p, wins=('W1',)):
    """창별 학습 행렬과 라벨 행렬을 .npy로 저장 (작업자가 mmap으로 읽음)"""
    W = windows(U['meta'], p)
    for w in wins:
        idx = np.flatnonzero(W[w])
        if len(idx) > REF_MAX:
            idx = np.sort(np.random.default_rng(42).choice(idx, REF_MAX, replace=False))
        np.save(mm_path(p, w), U['X'][idx])
    np.save(mm_path(p, 'lab'), LBL[p]['X'])


_CACHE = {}


def _mm(p, w):
    k = (p, w)
    if k not in _CACHE:
        _CACHE[k] = np.load(mm_path(p, w), mmap_mode='r')
    return _CACHE[k]


def fit_score(p, win, idx, models=MODELS, seed=0):
    """열 번호 idx → 학습(창 win) → 라벨 점수, 라벨 점수의 학습 구간 백분위"""
    Xr = np.asarray(_mm(p, win)[:, idx], float)
    Xl = np.asarray(_mm(p, 'lab')[:, idx], float)
    pr = M.Prep().fit(Xr)
    A, B = pr(Xr), pr(Xl)
    keep = A.std(0) > 0
    A, B = A[:, keep], B[:, keep]
    if A.shape[1] == 0:
        return {m: (None, None, 'no_cols') for m in models}, 0
    if A.shape[1] > MAX_COLS:
        pc = PCA(0.95, random_state=0).fit(A)
        A, B = pc.transform(A), pc.transform(B)
    out = {}
    for m in models:
        try:
            if m == 'lmaha300':
                s, sr = L3.local_maha(A, B, 300), None
            else:
                mod = M.REGISTRY[m]()
                if hasattr(mod, 'seed'):
                    mod.seed = seed
                mod.fit(A)
                s, sr = mod.score(B), mod.score(A)
            pct = None if sr is None else np.searchsorted(np.sort(sr), s, side='right') / (len(sr) + 1)
            out[m] = (np.asarray(s, np.float32), None if pct is None else pct.astype(np.float32), '')
        except Exception as e:  # noqa
            out[m] = (None, None, repr(e)[:150])
    return out, int(A.shape[1])


def eval_table(p, LBL):
    return L3.eval_table(p, LBL[p]['meta']['TimeStamp'])


evaluate = L3.evaluate
