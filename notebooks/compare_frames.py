"""
compare_frames.py — 팀원(JIWOO) 방식 vs 내 방식, 같은 두 평가 틀에서 맞대결

평가 틀
  ① 팀원 틀  : 공식 2차 데이터만, '행' 단위(LH/RH 두 행에 같은 샷 점수), 파일 전체 상위 k%,
               예산 F1(상위 5%/10%), 정보 상한 대비, 무작위 AP 범위
  ② 내 틀    : labeled 전체(2차+추가, 초기허용 제외), '샷' 단위, Run별 hit@k, AP, 유형별 + 예산 F1

방법 (라벨을 학습에 쓰지 않는 것 위주)
  팀원 방식 재현 (JIWOO notebooks 07·08·10 로직)
    T_prev10      : 직전 10샷(파일 시간순, Run 경계 무시) 대비 robust z 의 rms
    T_global      : labeled 파일 전체 기준 robust z rms (미래 포함 — 참고)
    T_past10      : 과거 기준선 K=10 — 비라벨 정상 모드 + 라벨 샷 '값'을 합친 시간순 과거 10샷 기준 (08번 '(나)')
    T_phys300     : 물리 관계 5개(R1~R5)의 과거 300샷 기준 robust z rms (10번 물리 잔차)
       robust z = |x − 중앙값| / max(1.4826·MAD, 표준편차), 최대 20, 퍼짐 0이면 0 (팀원 robust_z 그대로)
  내 방식
    M_knn_free / M_maha_free : 정상 기준(reference, 9~10월 unlabeled, 0 모드 제거)으로 학습한 이상탐지, 편차 변수
    M_rule_phys              : 학습 없는 물리 규칙 max(재료부족, 쿠션부족, 계량지연)
    (선택) M_E2_phys         : 13번 지도+비지도 앙상블 (LORO, 라벨 사용 — 참고)

변수 차이 메모: 팀원은 2차 표준화 24변수, 여기는 원값 21변수(상수·기록오류 3개 제외).
robust z 는 변수별 선형변환에 불변이라 같은 변수끼리는 같은 값 → 차이는 빠진 3개 변수뿐
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from features import ALL_RAW, raw
from model_v2 import build_reference, reference_features, to_ref_scale, evaluate_v2
from model_phys import add_phys, rule_score
from unsup_eval import UDetector
from model_v2 import FSETS
from loro_eval import recall_at_k_by_run

VARS = [raw(v) for v in ALL_RAW]
SEED = 42


# ====================================================================== 데이터
def load_all_shots(p):
    """labeled 전체 샷 (점수 계산용 — 초기허용·제외 포함해서 시간 흐름 그대로)"""
    s = pd.read_csv(f'./base/{p}_shots.csv', parse_dates=['TimeStamp'])
    s['Reason'] = s['Reason'].fillna('정상')
    return s.sort_values('TimeStamp').reset_index(drop=True)


def load_unlabeled_normal(p):
    """비라벨 원값 — 0 모드(배럴온도 0, 사이클 약 36초) 제거, 시각당 한 샷"""
    u = pd.read_csv(f'./linked/linked_unlabeled_{p}.csv', parse_dates=['TimeStamp'], low_memory=False)
    u = u.drop_duplicates('TimeStamp')
    zero = u[raw('Barrel_Temperature_1')] < 100
    print(f'[{p}] 비라벨 {len(u)}샷 중 0 모드 {int(zero.sum())} ({zero.mean():.1%}) 제거')
    return u[~zero].sort_values('TimeStamp').reset_index(drop=True)


def official_rows(p):
    """공식 2차 행 (LH/RH 행 단위 라벨)"""
    l = pd.read_csv('./linked/linked_labeled.csv', parse_dates=['TimeStamp'],
                    usecols=['product', 'in_secondary', 'TimeStamp', 'cavity', 'PassOrFail', 'Reason'])
    l = l[(l['product'] == p) & l['in_secondary']].copy()
    l['Reason'] = l['Reason'].fillna('정상')
    return l.reset_index(drop=True)


# ====================================================================== 팀원 방식 점수
def robust_z(x, P):
    med = np.median(P, axis=0)
    sd = np.maximum(np.median(np.abs(P - med), axis=0) * 1.4826, P.std(axis=0))
    ok = sd > 1e-6
    return np.where(ok, np.minimum(np.abs(x - med) / np.where(ok, sd, 1), 20), 0)


def _rms(z):
    return float(np.sqrt((z ** 2).mean()))


def t_prev(shots, N=10, by_run=False):
    """직전 N샷 이탈도. by_run=False면 팀원과 같이 파일 시간순(Run 경계 무시)"""
    X = shots[VARS].values.astype(float)
    sc = np.zeros(len(X))
    groups = shots.groupby('run_id').indices.values() if by_run else [np.arange(len(X))]
    for idx in groups:
        for j, i in enumerate(idx):
            P = X[idx[max(0, j - N):j]]
            if len(P) >= 3:
                sc[i] = _rms(robust_z(X[i], P))
    return sc


def t_global(shots):
    X = shots[VARS].values.astype(float)
    return np.array([_rms(robust_z(x, X)) for x in X])


def _past_base(shots, unl):
    """비라벨 정상 + 라벨 샷 '값' (08번 (나)) — 시간순"""
    b = pd.concat([unl[['TimeStamp'] + VARS], shots[['TimeStamp'] + VARS]], ignore_index=True)
    return b.drop_duplicates('TimeStamp').sort_values('TimeStamp').reset_index(drop=True)


def t_past(shots, unl, K=10):
    base = _past_base(shots, unl)
    bt, BX = base['TimeStamp'].values, base[VARS].values.astype(float)
    X, t = shots[VARS].values.astype(float), shots['TimeStamp'].values
    pos = np.searchsorted(bt, t, side='left')          # 엄격히 과거
    sc = np.zeros(len(X))
    for i, k in enumerate(pos):
        P = BX[max(0, k - K):k]
        if len(P) >= 3:
            sc[i] = _rms(robust_z(X[i], P))
    return sc


def physics_relations(d):
    """팀원 10번 물리 관계 R1~R5"""
    r = lambda c: d[raw(c)].astype(float)
    return pd.DataFrame({
        'R1': np.log(r('Filling_Time')) + np.log(r('Max_Injection_Speed')),
        'R2': np.log(r('Plasticizing_Time')) + np.log(r('Max_Screw_RPM')) - np.log(r('Plasticizing_Position')),
        'R3': r('Injection_Time') - r('Filling_Time'),
        'R4': r('Max_Switch_Over_Pressure') / r('Max_Injection_Pressure'),
        'R5': r('Cycle_Time') - r('Injection_Time') - r('Plasticizing_Time') - r('Clamp_Close_Time'),
    }, index=d.index).replace([np.inf, -np.inf], np.nan)


def t_phys(shots, unl, K=300):
    base = _past_base(shots, unl)
    BR = physics_relations(base).values
    bt = base['TimeStamp'].values
    XR = physics_relations(shots).values
    pos = np.searchsorted(bt, shots['TimeStamp'].values, side='left')
    sc = np.zeros(len(XR))
    for i, k in enumerate(pos):
        P = BR[max(0, k - K):k]
        P = P[~np.isnan(P).any(axis=1)]
        if len(P) >= 10 and not np.isnan(XR[i]).any():
            sc[i] = _rms(robust_z(XR[i], P))
    return sc


# ====================================================================== 내 방식 점수
def build_ref_nozero(p, period=('2020-09-01', '2020-11-30'), scales=None):
    """reference에서 0 모드 샷 제거 후 파생변수 계산 (기존 build_reference + 필터)"""
    shots = pd.read_csv(f'./base/{p}_shots.csv', parse_dates=['TimeStamp'])
    rs = build_reference(f'./linked/linked_unlabeled_{p}.csv', shots, p, period=period)
    z = rs[raw('Barrel_Temperature_1')] < 100
    print(f'[{p}] reference 0 모드 {int(z.sum())}샷 제거')
    rs = rs[~z].reset_index(drop=True)
    return add_phys(reference_features(rs, scales[p]['steps'], scales[p]['fr_step']))


def my_scores(feat, ref):
    """feat: base/{p}_features.csv (전 샷) + add_phys. 반환 {method: (score_feat, score_ref)}"""
    out = {}
    for kind in ('knn', 'maha'):
        det = UDetector(kind, FSETS['free']).fit(ref)
        out[f'M_{kind}_free'] = (det.raw_score(feat), det.s_ref)
    out['M_rule_phys'] = (rule_score(feat), rule_score(ref))
    return out


# ====================================================================== 평가
def random_ap(y, n=500):
    y = np.asarray(y)
    aps = [average_precision_score(y, np.random.RandomState(s).rand(len(y))) for s in range(n)]
    return float(np.mean(aps)), float(np.percentile(aps, 97.5))


def budget_f1(y, s, q):
    y, s = np.asarray(y), np.asarray(s, float)
    order = np.lexsort((np.random.RandomState(0).rand(len(s)), -s))
    k = int(round(len(s) * q))
    pred = np.zeros(len(s), int); pred[order[:k]] = 1
    tp = int((pred & y).sum())
    return tp, k - tp, int(y.sum()) - tp, 2 * tp / (2 * tp + (k - tp) + (y.sum() - tp))


def info_ceiling(rows):
    """같은 샷(같은 입력)의 행은 같은 판정 → 달성 가능한 최대 F1 (팀원 f1_ceiling과 같은 계산)"""
    g = rows.groupby('TimeStamp')['PassOrFail'].agg(p='sum', t='size')
    g['n'] = g['t'] - g['p']
    g = g[g['p'] > 0].assign(r=lambda x: x['p'] / x['t']).sort_values(['r', 'p'], ascending=False)
    P, tp, fp, best = rows['PassOrFail'].sum(), 0, 0, 0.0
    for _, r in g.iterrows():
        tp += r['p']; fp += r['n']
        best = max(best, 2 * tp / (2 * tp + fp + (P - tp)))
    return best


def frame1_eval(p, rows, shot_scores):
    """① 공식 2차, 행 단위. shot_scores: {method: Series(index=TimeStamp)}"""
    y = rows['PassOrFail'].values
    rmean, rhi = random_ap(y)
    ceil = info_ceiling(rows)
    out = []
    for m, s in shot_scores.items():
        sc = rows['TimeStamp'].map(s).fillna(0).values
        ap = average_precision_score(y, sc)
        r = {'제품': p, '방법': m, '행': len(y), '불량 행': int(y.sum()), 'AP': ap, '무작위 평균 대비': ap / rmean,
             '무작위 상단': rhi, '유의': 'O' if ap > rhi else '-'}
        order = np.lexsort((np.random.RandomState(0).rand(len(sc)), -sc))
        for k in (1, 5, 10):
            r[f'상위{k}%'] = f'{int(y[order[:max(1, int(len(sc) * k / 100))]].sum())}/{int(y.sum())}'
        for q in (0.05, 0.10):
            tp, fp, fn, f1 = budget_f1(y, sc, q)
            r[f'F1@{int(q*100)}%'] = f1
            r[f'TP@{int(q*100)}%'] = tp
        r['정보 상한 F1'] = ceil
        r['F1@5% / 상한'] = r['F1@5%'] / ceil
        out.append(r)
    return pd.DataFrame(out)


def frame2_oof(p, feat, shot_scores, ref_scores=None):
    """② 샷 단위 OOF 프레임 (evaluate_v2 입력). ref_scores가 있으면 pct도 계산"""
    parts = []
    for m, s in shot_scores.items():
        sc = feat['TimeStamp'].map(s).fillna(0).values
        if ref_scores and m in ref_scores:
            z, pc = to_ref_scale(sc, ref_scores[m])
        else:
            z, pc = sc, np.full(len(sc), np.nan)
        parts.append(pd.DataFrame({'product': p, 'method': m, 'run_id': feat['run_id'].values,
                                   'shot_seq': feat['shot_seq_in_run'].values, 'y': feat['PassOrFail'].values,
                                   'Reason': feat['Reason'].values, 'score': z, 'pct': pc}))
    return pd.concat(parts, ignore_index=True)


def frame2_eval(oof):
    s = evaluate_v2(oof)
    rows = []
    for (p, m), g in oof.groupby(['product', 'method'], sort=False):
        y, sc = g['y'].values, g['score'].values
        r = {'product': p, 'method': m}
        for q in (0.05, 0.10):
            tp, fp, fn, f1 = budget_f1(y, sc, q)
            r[f'F1@{int(q*100)}%(전체)'] = f1
        gg = g.assign(run_id='ALL').reset_index(drop=True)
        h, t = recall_at_k_by_run(gg, 0.10, gg['y'] == 1)
        r['전체 상위10%'] = f'{h}/{t}'
        rows.append(r)
    return s.merge(pd.DataFrame(rows), on=['product', 'method'])
