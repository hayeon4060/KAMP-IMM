"""
change_point.py — 46번: 변화점 탐지 / 순차 감시 (CUSUM · EWMA · MEWMA · Bayesian Online Change-Point)

왜: CN7 불량은 R1 한 에피소드에 몰려 있음 → '샷 하나가 이상한가'보다 '공정 상태가 언제부터 바뀌었나'를 보는 게 맞음.
    순차 감시는 작은 이상이 여러 샷 이어지면 누적해서 경보 → 단발 점수(최종 모델)가 놓치는 '서서히 나빠지는 구간'을 잡을 수 있음.
    라벨을 전혀 쓰지 않으므로 학습 데이터 누수가 구조적으로 없음.

신호 (1차원, 정상에서 ≈ N(0,1), 모두 라벨 미사용)
  final   : 최종 모델 pct → probit
  std4    : 흔들림 점수(배럴2·호퍼·쿠션·배럴5 직전 10샷 std) pct → probit
  thermal : 열 상태 6개(금형·호퍼 편차·상승·기울기) Mahalanobis pct → probit
방법 (설정값은 교과서 기본값으로 미리 고정)
  단발     : 신호 그대로 (비교 기준 — final이면 최종 모델 경보와 같음)
  CUSUM    : S_t = max(0, S_{t-1} + z_t − k), k = 0.5 (1σ 이동 탐지용) / 1.0
  EWMA     : E_t = λ z_t + (1−λ) E_{t-1}, λ = 0.2 (위쪽만)
  BOCPD    : Adams & MacKay(2007) 베이지안 온라인 변화점 — 정규(분산 1, 평균 미지), 위험률 1/250,
             점수 = '최근 10샷 안에 시작된 새 구간'의 사후확률 × (그 구간 평균 > 0.5)
  MEWMA    : 최종 모델 입력 17개(정상 기준 표준화)에 다변량 EWMA, λ = 0.1 / 0.3 → T²
  모두 재가동 구간마다 0에서 시작(5분 넘게 끊기면 새 구간)
경보 기준: 정상 기준 데이터(unlabeled)에서 경보 샷 비율이 1·3·5·10%가 되는 임계값 → 라벨 없이 결정
평가
  - 경보 샷 = 검사 샷으로 보고 Precision·Recall·F1·불량/가스/미성형 몇 개 중 몇 개
  - 같은 검사 샷 수(K)로 '최종 모델 점수 상위 K' · 'CN7 경보+std4 순위 상위 K'와 비교
  - 원형 이동 순열 p (몰림 구조 유지)
  - 불량 없는 Run의 경보율 (정상 기준에서 정한 경보율이 labeled 기간에서도 유지되나 = 기준 신뢰성)
  - 에피소드 표: 불량 Run마다 첫 경보 샷 · 첫 불량 샷 · 첫 미성형 샷 · 미성형 전에 미리 경보가 났나(몇 샷 전)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

import novel_common as NC

OUT = './outputs/change_point'
RATES = (0.01, 0.03, 0.05, 0.10)


# ====================================================================== 순차 통계량 (구간마다 0에서 시작)
def _z(x):
    return np.where(np.isfinite(x), x, 0.0)


def cusum(z, segs, k=0.5):
    z = _z(z)
    S = np.zeros(len(z))
    for ix in segs:
        s = 0.0
        for i in ix:
            s = max(0.0, s + z[i] - k)
            S[i] = s
    return S


def ewma(z, segs, lam=0.2):
    z = _z(z)
    E = np.zeros(len(z))
    for ix in segs:
        e = 0.0
        for i in ix:
            e = lam * z[i] + (1 - lam) * e
            E[i] = e
    return E


def mewma(X, segs, precision, lam=0.1):
    """X: (n, p) 정상 기준 표준화 입력, precision: 정상 기준 공분산의 역행렬 → T² (점근 공분산 λ/(2−λ)Σ 사용)"""
    X = np.nan_to_num(X)
    T = np.zeros(len(X))
    c = (2 - lam) / lam
    for ix in segs:
        z = np.zeros(X.shape[1])
        for i in ix:
            z = lam * X[i] + (1 - lam) * z
            T[i] = c * z @ precision @ z
    return T


def bocpd(z, segs, hazard=1 / 250, rmax=300, var0=1.0, varx=1.0, w=10, up=0.5):
    """
    Adams & MacKay 베이지안 온라인 변화점. 관측 z ~ N(μ, varx), 구간 평균 μ ~ N(0, var0) (정상 = 0)
    반환: 점수 = Σ_{r=1..w} P(현재 구간 길이 = r) · 1[그 구간 사후평균 > up]  (최근 w샷 안에 위로 바뀐 구간이 시작됐을 확률)
    """
    z = _z(z)
    out = np.zeros(len(z))
    for ix in segs:
        R = np.array([1.0])
        mu = np.array([0.0])
        var = np.array([var0])
        for i in ix:
            x = z[i]
            sd = np.sqrt(var + varx)
            pred = np.exp(-0.5 * ((x - mu) / sd) ** 2) / sd
            growth = R * pred * (1 - hazard)
            cp = (R * pred).sum() * hazard
            R = np.r_[cp, growth]
            tot = R.sum()
            R = R / tot if tot > 0 else np.r_[1.0, np.zeros(len(R) - 1)]
            nv = 1.0 / (1.0 / var + 1.0 / varx)
            nm = nv * (mu / var + x / varx)
            mu = np.r_[0.0, nm]
            var = np.r_[var0, nv]
            if len(R) > rmax:
                R, mu, var = R[:rmax], mu[:rmax], var[:rmax]
                R = R / R.sum()
            sl = slice(1, w + 1)
            out[i] = float((R[sl] * (mu[sl] > up)).sum())
    return out


# ====================================================================== 계산
def statistics(p):
    """반환: ref, lab, {방법 이름: (ref 통계량, lab 통계량)}"""
    ref, lab, model = NC.load(p)
    sig = NC.base_signals(ref, lab)
    sr, sl = NC.segments(ref), NC.segments(lab)
    S = {}
    for name, (zr, zl) in sig.items():
        S[f'단발 · {name}'] = (zr, zl)
        for k in (0.5, 1.0):
            S[f'CUSUM(k={k}) · {name}'] = (cusum(zr, sr, k), cusum(zl, sl, k))
        S[f'EWMA(λ=0.2) · {name}'] = (ewma(zr, sr, 0.2), ewma(zl, sl, 0.2))
        S[f'BOCPD · {name}'] = (bocpd(zr, sr), bocpd(zl, sl))
        print(f'  [{p}] {name} 완료', flush=True)
    Xr, Xl = model._prep(ref), model._prep(lab)
    prec = model.lw.precision_
    for lam in (0.1, 0.3):
        S[f'MEWMA(λ={lam}) · 최종 17변수'] = (mewma(Xr, sr, prec, lam), mewma(Xl, sl, prec, lam))
    return ref, lab, S


def comparison_scores(p, ref, lab):
    """같은 K 비교용 순위 점수: 최종 모델 점수 / (CN7) 경보 + std4"""
    out = {'최종 모델 점수 상위 K': lab['score'].values}
    s4l, _ = NC.std4_pct(ref, lab)
    out['경보 먼저 + std4 상위 K'] = np.where(lab['pct'].values >= 0.999, 2.0 + lab['pct'].values, s4l)
    return out


def run(products=('cn7', 'rg3'), rates=RATES, n_perm=1000, verbose=True):
    os.makedirs(OUT, exist_ok=True)
    rows, EP, STATS = [], [], {}
    for p in products:
        ref, lab, S = statistics(p)
        comp = comparison_scores(p, ref, lab)
        STATS[p] = {'ref': ref, 'lab': lab, 'S': S}
        for name, (st_r, st_l) in S.items():
            for rate in rates:
                thr = NC.threshold(st_r, rate)
                flag = st_l > thr
                r = NC.evaluate(lab, flag, name, 제품=p, 목표경보율=rate)
                r.update(NC.circ_null(lab, flag, n_perm))
                for cn, cs in comp.items():
                    ck = NC.evaluate(lab, NC.topk_flag(lab, cs, max(r['경보(검사) 샷'], 1)))
                    r[f'[같은 K] {cn} 불량'] = ck['불량']
                    for ty in NC.TYPES:
                        if ty in ck:
                            r[f'[같은 K] {cn} {ty}'] = ck[ty]
                rows.append(r)
        EP.append(episode_table(p, lab, S, rate=0.01))
        if verbose:
            print(f'[{p}] 완료', flush=True)
    T = pd.DataFrame(rows)
    E = pd.concat(EP, ignore_index=True)
    T.to_csv(f'{OUT}/summary.csv', index=False, encoding='utf-8-sig')
    E.to_csv(f'{OUT}/episodes.csv', index=False, encoding='utf-8-sig')
    return {'table': T, 'episodes': E, 'stats': STATS}


def episode_table(p, lab, S, rate=0.01, before=50):
    """불량이 있는 Run마다: 첫 불량·첫 미성형 seq, 방법별 첫 경보 seq, 미성형 직전 before샷 안 경보 수, 선행 샷 수"""
    ev = lab['eval'].values
    t = lab['type'].values
    y = (lab['PassOrFail'].values == 1) & ev
    rows = []
    for r in pd.unique(lab.loc[y, 'run_id']):
        m = (lab['run_id'] == r).values
        seq = lab['seq'].values
        fd = int(seq[m & y].min())
        sh = seq[m & ev & (t == '미성형')]
        fs = int(sh.min()) if len(sh) else None
        for name, (st_r, st_l) in S.items():
            thr = NC.threshold(st_r, rate)
            a = m & ev & (st_l > thr)
            first = int(seq[a].min()) if a.any() else None
            row = {'제품': p, 'Run': r, '방법': name, '경보율': rate, 'Run 평가 샷': int((m & ev).sum()),
                   'Run 불량': int((m & y).sum()), '첫 불량 seq': fd, '첫 미성형 seq': fs,
                   '첫 경보 seq': first, 'Run 안 경보 수': int(a.sum())}
            if fs is not None:
                pre = a & (seq < fs) & (seq >= fs - before)
                row[f'미성형 직전 {before}샷 경보 수'] = int(pre.sum())
                row['미성형보다 몇 샷 먼저 경보'] = int(fs - seq[pre].min()) if pre.any() else 0
            row['첫 불량보다 몇 샷 먼저 경보'] = (fd - first) if first is not None and first <= fd else None
            rows.append(row)
    return pd.DataFrame(rows)


def best_table(R, p, rate=0.03, sort='TP'):
    T = R['table']
    cols = ['방법', '경보(검사) 샷', 'TP', '불량', '가스', '미성형', 'Precision', 'Recall', 'F1', '정규화 Precision',
            '불량 없는 Run 경보율', 'p_불량', 'p_가스', 'p_미성형'] + [c for c in T.columns if c.startswith('[같은 K]')]
    t = T[(T['제품'] == p) & (T['목표경보율'] == rate)]
    return t[[c for c in cols if c in t.columns]].sort_values(sort, ascending=False).reset_index(drop=True)


def plot_run(R, p='cn7', methods=None, run=None, rate=0.01, save=None):
    """불량이 가장 많은 Run에서 방법별 통계량과 경보 기준선, 불량 위치"""
    import matplotlib.pyplot as plt
    NC.style()
    lab, S = R['stats'][p]['lab'], R['stats'][p]['S']
    ev, t = lab['eval'].values, lab['type'].values
    if run is None:
        run = lab.loc[ev & (lab['PassOrFail'] == 1).values, 'run_id'].value_counts().index[0]
    methods = methods or ['단발 · final', 'CUSUM(k=0.5) · final', 'CUSUM(k=0.5) · std4', 'BOCPD · final',
                          'MEWMA(λ=0.1) · 최종 17변수', 'CUSUM(k=0.5) · thermal']
    m = (lab['run_id'] == run).values
    seq = lab['seq'].values[m]
    fig, axes = plt.subplots(len(methods), 1, figsize=(13, 1.9 * len(methods)), sharex=True)
    for ax, name in zip(np.atleast_1d(axes), methods):
        st_r, st_l = S[name]
        thr = NC.threshold(st_r, rate)
        ax.plot(seq, st_l[m], color='#555', lw=1)
        ax.axhline(thr, color='#c0392b', ls='--', lw=1)
        for ty, c in (('가스', '#e67e22'), ('미성형', '#1f4e79')):
            mm = (t[m] == ty) & ev[m]
            for s in seq[mm]:
                ax.axvline(s, color=c, alpha=0.35, lw=1.2)
        ax.set_ylabel(name, fontsize=7, rotation=0, ha='right')
    np.atleast_1d(axes)[0].set_title(f'{p.upper()} {run} — 빨간 점선 = 정상 기준 경보율 {rate:.0%} 임계값, 주황 = 가스, 파랑 = 미성형',
                                     loc='left', fontsize=10)
    np.atleast_1d(axes)[-1].set_xlabel('Run 안 샷 순서')
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
