"""
active_learning.py — 49번: 능동 학습(사람이 검사한 결과로 바로 학습) 운영 시뮬레이션

왜: 라벨이 처음부터 많을 수 없고 불량이 한 에피소드에 몰림 → '라벨을 먼저 모아 학습'이 아니라
    현장처럼 시간 순으로 검사하면서, 검사원이 확인한 결과를 즉시 모델·검사 순서에 반영하는 방식이 맞는지 확인.
    (보고서 4장 '현장 적용 · 운영 시나리오'의 근거)

시뮬레이션 (labeled 평가 대상 샷, 시간 순)
  - Run마다 시간 순으로 block(기본 100샷) 단위 = 한 번에 검사하는 생산 묶음(로트)
  - 묶음마다 검사 예산 = 10% (전체 검사 샷 수는 모든 전략이 같음)
  - 검사한 샷만 라벨 공개 → 다음 선택·다음 묶음 학습에 사용 (미래 라벨은 절대 안 씀)
기본 점수 (라벨 미사용)
  'final' : 최종 모델 점수
  'combo' : 최종 모델 경보(pct≥0.999) 먼저 + std4 흔들림 순 (CN7 38번 규칙, RG3에도 같은 규칙 적용)
전략
  S0 무작위
  S1 고정 순위             : 기본 점수 순으로 검사 (학습 없음)
  S2 고정 + 주변 확장       : 불량을 찾으면 같은 묶음 안 앞뒤 nbr샷을 먼저 검사 (몰려 나오는 성질 이용, 학습 없음)
  S3 온라인 학습            : 지금까지 검사 결과(불량 ≥2개부터) + 정상 기준 3,000샷(정상, 가중치 0.3)으로 RF 재학습,
                              순위 = 기본 점수 순위와 RF 확률 순위의 평균, 예산의 20%는 무작위 탐색
  S4 온라인 + 주변 확장
  입력(RF): 최종 모델 입력 17개(정상 기준 표준화) + std4 흔들림(probit)
평가
  - 찾은 불량/가스/미성형 몇 개 중 몇 개, Precision(검사 샷 중 불량)·Recall·F1 (모든 전략 검사 샷 수 동일)
  - 누적 검사 샷 대비 누적 발견 불량 곡선
  - 불량 Run마다 '첫 불량을 찾기까지 검사 샷 수', '첫 발견 이후 그 Run에서 더 찾은 불량'
  - (선택) 학습 효과 검증: 라벨을 Run 안 원형 이동 → S3−S1 이득이 우연 수준인지
⚠ S2의 이득은 '불량이 몰려 나온다'는 구조에서 오므로 원형 이동 순열로는 없어지지 않음(구조 자체가 실제 성질).
  변수 학습 덕분인지는 S3−S1로 따로 봄.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import novel_common as NC

OUT = './outputs/active_learning'
STRATS = ('S0 무작위', 'S1 고정 순위', 'S2 고정 + 주변 확장', 'S3 온라인 학습', 'S4 온라인 + 주변 확장')


def prepare(p):
    ref, lab, model = NC.load(p)
    s4l, s4r = NC.std4_pct(ref, lab)
    Zl = np.c_[model._prep(lab), NC.probit(s4l)]
    Zr = np.c_[model._prep(ref), NC.probit(s4r)]
    base = {'final': lab['score'].values,
            'combo': np.where(lab['pct'].values >= 0.999, 2.0 + lab['pct'].values, s4l)}
    return {'p': p, 'lab': lab, 'Zl': Zl, 'Zr': Zr, 'base': base}


def blocks(lab, size=100):
    """평가 대상 샷을 Run 시작 시각 순 → Run 안 시간 순으로 size개씩 묶음 (행 위치 배열 목록)"""
    ev = lab['eval'].values
    order = lab[ev].groupby('run_id')['TimeStamp'].min().sort_values().index
    out = []
    for r in order:
        ix = np.flatnonzero(ev & (lab['run_id'] == r).values)
        ix = ix[np.argsort(lab['TimeStamp'].values[ix], kind='stable')]
        out += [ix[i:i + size] for i in range(0, len(ix), size)]
    return out


def _rank(v):
    return pd.Series(v).rank(pct=True).values


def simulate(D, strategy, base='combo', budget=0.10, block=100, nbr=5, explore=0.2, w_ref=0.3, n_ref=3000,
             seed=0, y_override=None):
    lab, Zl, Zr = D['lab'], D['Zl'], D['Zr']
    y = (lab['PassOrFail'].values == 1).astype(int) if y_override is None else y_override
    t = lab['type'].values if y_override is None else np.where(y == 1, '불량', '정상')
    seq, runs = lab['seq'].values, lab['run_id'].values
    bscore = np.asarray(D['base'][base], float)
    rng = np.random.default_rng(seed)
    online = strategy.startswith(('S3', 'S4'))
    expand = strategy.startswith(('S2', 'S4'))
    seen_idx, seen_y = [], []
    model = None
    log = []
    ref_pick = rng.choice(len(Zr), min(n_ref, len(Zr)), replace=False)
    for blk in blocks(lab, block):
        k = int(np.ceil(budget * len(blk)))
        if strategy.startswith('S0'):
            pri = rng.random(len(blk))
        elif online and model is not None:
            pri = 0.5 * _rank(bscore[blk]) + 0.5 * _rank(model.predict_proba(Zl[blk])[:, 1])
        else:
            pri = bscore[blk] + rng.random(len(blk)) * 1e-9
        pri = dict(zip(blk, pri))
        remaining = set(blk)
        queue = []
        for _ in range(k):
            if expand and queue:
                i = queue.pop(0)
            elif online and rng.random() < explore:
                i = rng.choice(sorted(remaining))
            else:
                i = max(remaining, key=lambda j: pri[j])
            remaining.discard(i)
            seen_idx.append(i)
            seen_y.append(y[i])
            log.append({'i': i, 'run': runs[i], 'seq': seq[i], 'y': y[i], 'type': t[i]})
            if expand and y[i] == 1:
                nb = [j for j in remaining if runs[j] == runs[i] and abs(seq[j] - seq[i]) <= nbr and j not in queue]
                queue += sorted(nb, key=lambda j: abs(seq[j] - seq[i]))
        if online and sum(seen_y) >= 2:
            X = np.vstack([Zl[seen_idx], Zr[ref_pick]])
            yy = np.r_[seen_y, np.zeros(len(ref_pick), int)]
            w = np.r_[np.ones(len(seen_idx)), w_ref * np.ones(len(ref_pick))]
            model = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3,
                                           class_weight='balanced_subsample', n_jobs=-1, random_state=seed).fit(X, yy, sample_weight=w)
    return pd.DataFrame(log)


def summarize_log(D, L, name, base, seed):
    lab = D['lab']
    ev = lab['eval'].values
    t = lab['type'].values
    Dn = int(((lab['PassOrFail'].values == 1) & ev).sum())
    K, tp = len(L), int(L['y'].sum())
    P, R = tp / K if K else 0.0, tp / Dn if Dn else 0.0
    row = {'제품': D['p'], '기본 점수': base, '전략': name, 'seed': seed, '검사 샷': K, '찾은 불량': tp, '전체 불량': Dn,
           'Precision': P, 'Recall': R, 'F1': 2 * P * R / (P + R) if P + R else 0.0}
    for ty in NC.TYPES:
        tm = NC.type_mask(lab, ty)
        row[f'{ty}'] = int(tm[L['i'].values.astype(int)].sum()) if len(L) else 0
        row[f'{ty} 전체'] = int((ev & tm).sum())
    return row


def episode_view(D, L, name, base, seed):
    """불량 Run마다: 첫 불량 발견까지 그 Run에서 검사한 샷 수, 첫 발견 이후 더 찾은 불량"""
    lab = D['lab']
    y = (lab['PassOrFail'].values == 1) & lab['eval'].values
    rows = []
    for r in pd.unique(lab.loc[y, 'run_id']):
        l = L[L['run'] == r].reset_index(drop=True)
        hit = np.flatnonzero(l['y'].values == 1)
        rows.append({'제품': D['p'], '기본 점수': base, '전략': name, 'seed': seed, 'Run': r,
                     'Run 불량': int(y[(lab['run_id'] == r).values].sum()), 'Run 검사 샷': len(l),
                     '첫 불량까지 검사 샷': int(hit[0] + 1) if len(hit) else None,
                     '첫 발견 이후 찾은 불량': int(len(hit) - 1) if len(hit) else 0})
    return rows


def run(products=('cn7', 'rg3'), bases=('combo', 'final'), seeds=(0, 1, 2, 3, 4), block=100, verbose=True, tag=None, **kw):
    tag = '' if (tag is None and block == 100) else (tag if tag is not None else f'_block{block}')
    os.makedirs(OUT, exist_ok=True)
    S, E, LOGS = [], [], {}
    for p in products:
        D = prepare(p)
        for base in bases:
            for st in STRATS:
                sds = seeds if st.startswith(('S0', 'S3', 'S4')) else seeds[:1]     # 무작위가 없는 전략은 1번이면 충분
                for sd in sds:
                    L = simulate(D, st, base, block=block, seed=sd, **kw)
                    LOGS[(p, base, st, sd)] = L
                    S.append(summarize_log(D, L, st, base, sd))
                    E += episode_view(D, L, st, base, sd)
                if verbose:
                    print(f'  [{p}] {base} · {st} 완료', flush=True)
    S, E = pd.DataFrame(S), pd.DataFrame(E)
    agg = S.groupby(['제품', '기본 점수', '전략'], sort=False).agg(
        검사_샷=('검사 샷', 'first'), 찾은_불량=('찾은 불량', 'mean'), 범위=('찾은 불량', lambda x: f'{x.min()}~{x.max()}'),
        전체_불량=('전체 불량', 'first'), 가스=('가스', 'mean'), 가스_전체=('가스 전체', 'first'),
        미성형=('미성형', 'mean'), 미성형_전체=('미성형 전체', 'first'),
        Precision=('Precision', 'mean'), Recall=('Recall', 'mean'), F1=('F1', 'mean')).reset_index()
    S.to_csv(f'{OUT}/by_seed{tag}.csv', index=False, encoding='utf-8-sig')
    agg.to_csv(f'{OUT}/summary{tag}.csv', index=False, encoding='utf-8-sig')
    E.to_csv(f'{OUT}/episodes{tag}.csv', index=False, encoding='utf-8-sig')
    return {'summary': agg, 'by_seed': S, 'episodes': E, 'logs': LOGS}


def learning_null(p='cn7', base='combo', n=20, seed=0, block=100, verbose=True):
    """라벨을 Run 안 원형 이동 → S3(온라인) − S1(고정) 이득의 우연 분포. 관측 이득이 무작위 95%를 넘는지"""
    D = prepare(p)
    lab = D['lab']
    ev = np.flatnonzero(lab['eval'].values)
    y0 = (lab['PassOrFail'].values == 1).astype(int)

    def gain(yv):
        a = simulate(D, 'S3 온라인 학습', base, block=block, seed=seed, y_override=yv)['y'].sum()
        b = simulate(D, 'S1 고정 순위', base, block=block, seed=seed, y_override=yv)['y'].sum()
        return int(a - b)

    obs = gain(y0)
    rng = np.random.default_rng(seed + 100)
    runs = lab['run_id'].values[ev]
    null = []
    for b in range(n):
        yy = y0.copy()
        for r in pd.unique(runs):
            ii = ev[runs == r]
            yy[ii] = np.roll(y0[ii], rng.integers(0, len(ii)))
        null.append(gain(yy))
        if verbose and b % 5 == 0:
            print('  순열', b, flush=True)
    null = np.asarray(null)
    return {'관측 S3−S1': obs, '무작위 평균': float(null.mean()), '무작위 95%': float(np.quantile(null, 0.95)),
            'p': float((1 + (null >= obs).sum()) / (n + 1))}


def plot_curves(R, p='cn7', base='combo', seed=0, save=None):
    """누적 검사 샷 대비 누적 발견 불량 (시간 순)"""
    import matplotlib.pyplot as plt
    NC.style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for st in STRATS:
        key = (p, base, st, seed) if (p, base, st, seed) in R['logs'] else (p, base, st, 0)
        L = R['logs'].get(key)
        if L is None:
            continue
        ax.plot(np.arange(1, len(L) + 1), L['y'].cumsum(), label=st)
    ax.set_xlabel('누적 검사 샷 (시간 순)'); ax.set_ylabel('누적 발견 불량')
    ax.set_title(f'{p.upper()} · 기본 점수 {base} — 능동 학습 시뮬레이션 (검사 예산 10%)', loc='left', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
