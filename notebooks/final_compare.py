"""
final_compare.py — 미래 정보 완전 제거 + 팀원·내 방식 합산(사전 고정) + 두 방법 차이의 부트스트랩 신뢰구간

미래 정보 규칙
  - 모든 점수는 그 샷 '이전' 정보 또는 labeled와 겹치지 않는 reference(9~10월 unlabeled, 0 모드 제거)만 사용
  - T_global(파일 전체 기준) 제거
  - 합산 점수도 '평가 파일 안 순위'가 아니라 reference 분포 대비 백분위로 변환 → 다른 테스트 샷을 보지 않음

방법 (결과 보기 전에 고정)
  팀원: T_prev10, T_past10, T_phys300
  나  : M_knn_free, M_maha_free, M_rule_phys
  합산: C_maha+phys300 = 평균( −log10(1−pct_maha), −log10(1−pct_phys300) )
        C_maha+past10  = 평균( −log10(1−pct_maha), −log10(1−pct_past10) )
        pct = 같은 점수를 reference 샷에 계산한 분포 대비 백분위 (reference 샷 점수도 그 샷 이전 정보만 사용)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from compare_frames import VARS, robust_z, _rms, physics_relations, _past_base
from model_v2 import to_ref_scale

SEED = 42


# ====================================================================== 과거 기준 점수 (대상 샷을 따로 지정)
def _past_scores(targets, base, K, rel=False, minp=3):
    bt = base['TimeStamp'].values
    if rel:
        BX = physics_relations(base).values
        TX = physics_relations(targets).values
    else:
        BX = base[VARS].values.astype(float)
        TX = targets[VARS].values.astype(float)
    pos = np.searchsorted(bt, targets['TimeStamp'].values, side='left')       # 엄격히 과거
    sc = np.zeros(len(TX))
    for i, k in enumerate(pos):
        P = BX[max(0, k - K):k]
        if rel:
            P = P[~np.isnan(P).any(axis=1)]
            if np.isnan(TX[i]).any():
                continue
        if len(P) >= minp:
            sc[i] = _rms(robust_z(TX[i], P))
    return sc


def teammate_scores_with_ref(shots, unl, ref_targets):
    """반환 {method: (labeled 샷 점수, reference 샷 점수)} — T_past10, T_phys300"""
    base = _past_base(shots, unl)
    out = {}
    out['T_past10'] = (_past_scores(shots, base, 10), _past_scores(ref_targets, base, 10))
    out['T_phys300'] = (_past_scores(shots, base, 300, rel=True, minp=10),
                        _past_scores(ref_targets, base, 300, rel=True, minp=10))
    return out


def surprise_from_ref(s, s_ref):
    _, pct = to_ref_scale(np.asarray(s, float), np.asarray(s_ref, float))
    return -np.log10(1 - np.clip(pct, 0, 1 - 1e-9))


def combine(parts):
    """parts: [(score, ref_score), ...] → 합산 점수 (labeled), 합산 ref 점수"""
    s = np.mean([surprise_from_ref(a, b) for a, b in parts], axis=0)
    r = np.mean([surprise_from_ref(b, b) for _, b in parts], axis=0)
    return s, r


# ====================================================================== 부트스트랩 (두 방법 차이)
def _top_recall(y, s, k=0.10):
    order = np.lexsort((np.random.RandomState(0).rand(len(s)), -s))
    n = max(1, int(len(s) * k))
    return y[order[:n]].sum() / max(1, y.sum())


def paired_bootstrap(y, sa, sb, cluster=None, n_boot=1000, seed=SEED):
    """
    같은 샘플을 다시 뽑아 (A − B)의 AP·상위10% 재현율 차이 분포
    cluster: 같은 샷의 행(LH/RH)을 함께 뽑기 위한 묶음 id (① 행 단위 틀). None이면 행 하나씩
    층화: 불량 있는 묶음 / 없는 묶음을 따로 복원추출
    """
    y, sa, sb = map(np.asarray, (y, sa, sb))
    rng = np.random.default_rng(seed)
    if cluster is None:
        cluster = np.arange(len(y))
    cl = pd.Series(np.arange(len(y))).groupby(np.asarray(cluster)).apply(list)
    has = np.array([y[ix].any() for ix in cl])
    pos_c, neg_c = cl[has].tolist(), cl[~has].tolist()
    base = {'ΔAP': average_precision_score(y, sa) - average_precision_score(y, sb),
            'Δ상위10%': _top_recall(y, sa) - _top_recall(y, sb)}
    d = {k: [] for k in base}
    for _ in range(n_boot):
        pick = [pos_c[i] for i in rng.integers(0, len(pos_c), len(pos_c))] + \
               [neg_c[i] for i in rng.integers(0, len(neg_c), len(neg_c))]
        ix = np.concatenate(pick)
        yy, a, b = y[ix], sa[ix], sb[ix]
        d['ΔAP'].append(average_precision_score(yy, a) - average_precision_score(yy, b))
        d['Δ상위10%'].append(_top_recall(yy, a) - _top_recall(yy, b))
    rows = []
    for k, v in d.items():
        v = np.array(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append({'지표': k, '관측 차이(A−B)': base[k], 'CI 하한': lo, 'CI 상한': hi,
                     'P(A>B)': (v > 0).mean(), '판정': 'A 우세' if lo > 0 else ('B 우세' if hi < 0 else '차이 없음')})
    return pd.DataFrame(rows)
