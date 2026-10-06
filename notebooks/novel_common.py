"""
novel_common.py — 46~49번(아직 안 해본 기법) 공통 도구

  load(p)              : 정상 기준(ref = 20a 학습 데이터) / labeled(평가 표시·유형·seq·since_restart) / 최종 모델
  segments(d)          : Run 안 '재가동 구간'(5분 넘게 끊기면 새 구간) 인덱스 목록 → 순차 감시는 구간마다 0에서 시작
  probit(pct)          : 정상 기준 백분위 → 정상에서 N(0,1)이 되도록 변환 (z ≈ 0 정상, 3 = 상위 0.13%)
  std4_pct / block_pct : 흔들림(std4) · 열 상태 블록 Mahalanobis 점수의 정상 기준 백분위 (라벨 미사용)
  threshold(ref, rate) : 정상 기준 데이터에서 rate 비율만 경보가 나도록 임계값 → 라벨 없이 정해지는 운영 기준
  evaluate(lab, flag)  : 경보(검사) 샷 기준 Precision·Recall·F1 + 불량/가스/미성형 몇 개 중 몇 개
                         + '불량 없는 Run의 경보율'(정상 기준에서 정한 경보율이 실제 labeled 기간에서도 유지되나)
  circ_null(lab, flag) : 유형 라벨을 Run 안에서 원형 이동(몰린 구조 유지, 위치만 무작위) → 우연히 잡히는 수와 비교한 p
  topk_flag            : 같은 검사 샷 수(K)로 다른 점수 상위 K를 뽑아 공정 비교
평가 대상: 초기허용불량·RG3 R6 제외 (lab['eval']), 가스+미성형은 미성형으로만 셈
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

import cn7_alarm_plus_gas as C
from final_model import FinalModel, FEATURES

MD = './outputs/final_model'
TYPES = ('가스', '미성형')
THERMAL = ['dev_Mold_Temperature_3', 'dev_Hopper_Temperature', 'mold_rise', 'hopper_rise', 'mold_slope20',
           'hopper_slope20']


# ====================================================================== 데이터
def load(p):
    ref, lab = C.load(p)
    model = FinalModel.load(f'{MD}/{p}_final_model.pkl')
    return ref, lab, model


def segments(d):
    """Run → 재가동 구간별 행 위치 목록 (d는 run_id·TimeStamp 순 정렬, since_restart 포함)"""
    seg = d['run_id'].astype(str) + '#' + (d['since_restart'] == 1).groupby(d['run_id']).cumsum().astype(str)
    return [np.asarray(ix) for ix in d.groupby(seg, sort=False).indices.values()]


def to_pct(ref_scores, s):
    r = np.sort(np.asarray(ref_scores, float))
    r = r[np.isfinite(r)]
    s = np.asarray(s, float)
    return np.searchsorted(r, np.where(np.isfinite(s), s, -np.inf), side='right') / (len(r) + 1)


def probit(pct, eps=1e-5):
    return norm.ppf(np.clip(np.nan_to_num(np.asarray(pct, float), nan=0.5), eps, 1 - eps))


def std4_pct(ref, lab, k=10):
    """흔들림 점수(배럴2·호퍼·쿠션·배럴5 직전 k샷 std → 정상 기준 Mahalanobis)의 백분위 (lab, ref)"""
    from two_stage import std4_scores
    sl, sr = std4_scores(ref, lab, k)
    return to_pct(sr, sl), to_pct(sr, sr)


def block_pct(ref, lab, cols):
    """변수 묶음(예: 열 상태 6개)만으로 정상 기준 Mahalanobis → 백분위 (lab, ref)"""
    Xr = ref[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
    Xl = lab[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
    z = C._robust(Xr)
    lw = LedoitWolf().fit(z(Xr))
    sr, sl = np.sqrt(lw.mahalanobis(z(Xr))), np.sqrt(lw.mahalanobis(z(Xl)))
    return to_pct(sr, sl), to_pct(sr, sr)


def base_signals(ref, lab):
    """순차 감시에 넣을 1차원 신호 3개 (모두 라벨 미사용, 정상에서 ≈ N(0,1))"""
    s4l, s4r = std4_pct(ref, lab)
    thl, thr = block_pct(ref, lab, THERMAL)
    return {'final': (probit(ref['pct'].values), probit(lab['pct'].values)),
            'std4': (probit(s4r), probit(s4l)),
            'thermal': (probit(thr), probit(thl))}


# ====================================================================== 기준·평가
def threshold(ref_stat, rate):
    v = np.asarray(ref_stat, float)
    return float(np.nanquantile(v[np.isfinite(v)], 1 - rate))


def type_mask(lab, ty):
    """유형 마스크. lab에 'gas_any' 열이 있으면 가스 = 가스 + 가스+미성형 (혼합은 가스·미성형 양쪽에 셈)"""
    if ty == '가스' and 'gas_any' in lab.columns:
        return lab['gas_any'].values.astype(bool)
    return lab['type'].values == ty


def evaluate(lab, flag, name='', **extra):
    ev = lab['eval'].values
    t = lab['type'].values
    y = (lab['PassOrFail'].values == 1) & ev
    f = np.asarray(flag, bool) & ev
    tp, fp, D, K = int((f & y).sum()), int((f & ~y).sum()), int(y.sum()), int(f.sum())
    P = tp / K if K else 0.0
    R = tp / D if D else 0.0
    base = D / ev.sum()
    clean_runs = [r for r in pd.unique(lab['run_id'][ev]) if not y[(lab['run_id'] == r).values].any()]
    cm = ev & lab['run_id'].isin(clean_runs).values
    row = {'방법': name, **extra, '경보(검사) 샷': K, '경보 비율': K / ev.sum(), '불량': f'{tp}/{D}', 'TP': tp, 'FP': fp,
           'Precision': P, 'Recall': R, 'F1': 2 * P * R / (P + R) if P + R else 0.0,
           '정규화 Precision': tp / min(K, D) if K and D else np.nan,
           '무작위 대비(배)': P / base if base else np.nan,
           '불량 없는 Run 경보율': f[cm].mean() if cm.any() else np.nan}
    for ty in TYPES:
        tm = type_mask(lab, ty)
        n = int((ev & tm).sum())
        if n:
            row[ty] = f'{int((f & tm).sum())}/{n}'
    return row


def circ_null(lab, flag, n=1000, seed=7):
    """반환: {'p_불량', 'p_가스', 'p_미성형'} — 같은 경보 위치에 원형 이동한 라벨이 같거나 더 많이 들어올 확률"""
    ev = np.flatnonzero(lab['eval'].values)
    t = lab['type'].values[ev]
    f = np.asarray(flag, bool)[ev]
    runs = lab['run_id'].values[ev]
    groups = [np.flatnonzero(runs == r) for r in pd.unique(runs)]
    cnt = lambda tt: (int((f & (tt != '정상')).sum()), int((f & (tt == '가스')).sum()), int((f & (tt == '미성형')).sum()))
    obs = cnt(t)
    rng = np.random.default_rng(seed)
    null = np.empty((n, 3))
    for b in range(n):
        tt = t.copy()
        for g in groups:
            tt[g] = np.roll(t[g], rng.integers(0, len(g)))
        null[b] = cnt(tt)
    out = {}
    for i, k in enumerate(('불량', '가스', '미성형')):
        out[f'p_{k}'] = float((1 + (null[:, i] >= obs[i]).sum()) / (n + 1)) if obs[i] > 0 else np.nan
        out[f'무작위 평균_{k}'] = round(float(null[:, i].mean()), 2)
    return out


def topk_flag(lab, score, K):
    ev = np.flatnonzero(lab['eval'].values)
    v = np.asarray(score, float)[ev]
    v = np.where(np.isfinite(v), v, -1e18) + np.random.default_rng(0).random(len(ev)) * 1e-9
    m = np.zeros(len(lab), bool)
    m[ev[np.argsort(-v)[:K]]] = True
    return m


def holm(pvals):
    """Holm 보정 (여러 규칙을 한꺼번에 검정할 때)"""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full(len(p), np.nan)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    m, run = len(order), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        out[j] = run
    return out


def style():
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
