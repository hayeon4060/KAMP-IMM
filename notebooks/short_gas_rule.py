"""
short_gas_rule.py — 44번: 팀원 가설 검증 "미성형이 나오면, 그 직전 20샷 안에 가스 불량이 몰려 있다"

가설: CN7에서 미성형(가스+미성형 포함)은 모델이 잘 잡음 → 미성형을 기준점으로 직전 k샷을 보면 가스를 찾을 수 있다
     (팀원 보고: 직전 20샷에서 가스 9개)

확인하는 것
  1) 재현: 실제 라벨 기준으로 미성형 직전 k샷 안의 가스 수 (어떤 가스가 들어오는지 샷 번호까지)
  2) 운영형: 실제 라벨 대신 '최종 모델 경보(pct ≥ q)'를 기준점으로 쓰면? (현장에서는 미성형 라벨을 미리 모름)
     → 경보마다 직전 k샷을 역추적 검사할 때 검사 샷 수 · 가스 포착 · 정밀도 · 무작위 대비 배수
  3) 우연 검증: 가스 라벨을 Run 안에서 원형 이동(가스가 몰린 모양은 유지, 위치만 무작위) 5,000번
     → 같은 창에 우연히 들어오는 가스 수와 비교한 p
  4) 창 길이 민감도: k = 5, 10, 20, 30, 50
  5) 일반화: 미성형 '에피소드' 단위로 몇 개에서 성립하나 (CN7은 R1·R6 두 곳뿐)
     + RG3에서도 성립하는지 (독립 재현 — 다른 제품이라 참고)
평가 대상: 초기허용불량 제외, RG3 R6 제외 (final_model.EXCLUDE_RUNS)
"""
import os
import numpy as np
import pandas as pd

import cn7_alarm_plus_gas as C

OUT = './outputs/short_gas_rule'
KS = (5, 10, 20, 30, 50)


def load(p):
    ref, lab = C.load(p)
    return lab


def window_mask(lab, trigger, k):
    """trigger 샷마다 같은 Run의 직전 k샷(trigger 샷 제외)을 True로"""
    m = np.zeros(len(lab), bool)
    runs, seq = lab['run_id'].values, lab['seq'].values
    for i in np.flatnonzero(trigger):
        m |= (runs == runs[i]) & (seq >= seq[i] - k) & (seq < seq[i])
    return m


def describe(lab, mask, name=''):
    ev = lab['eval'].values
    t = lab['type'].values
    gas = ev & (t == '가스')
    nrm = ev & (t == '정상')
    sel = mask & ev
    g = int((sel & gas).sum())
    n_sel = int(sel.sum())
    base = gas.sum() / ev.sum()
    P = g / n_sel if n_sel else 0.0
    return {'규칙': name, '역추적 검사 샷': n_sel, '그 안의 가스': f'{g}/{int(gas.sum())}',
            '그 안의 정상': int((sel & nrm).sum()), '가스 Precision': P,
            '무작위 대비(배)': P / base if base else np.nan, '가스 Recall': g / gas.sum() if gas.sum() else np.nan}


def gas_positions(lab, mask):
    ev = lab['eval'].values
    t = lab['type'].values
    d = lab[mask & ev & (t == '가스')][['run_id', 'seq', 'TimeStamp']]
    return d


def circ_null(lab, mask, n=5000, seed=7):
    """가스 라벨만 Run 안에서 원형 이동 (미성형·창은 고정) → 창 안 가스 수의 우연 분포"""
    ev = np.flatnonzero(lab['eval'].values)
    runs = lab['run_id'].values[ev]
    isgas = (lab['type'].values[ev] == '가스')
    inwin = mask[ev]
    groups = [np.flatnonzero(runs == r) for r in pd.unique(runs)]
    rng = np.random.default_rng(seed)
    obs = int((isgas & inwin).sum())
    null = np.empty(n, int)
    for b in range(n):
        g = isgas.copy()
        for ii in groups:
            g[ii] = np.roll(isgas[ii], rng.integers(0, len(ii)))
        null[b] = (g & inwin).sum()
    return obs, float(null.mean()), float(np.quantile(null, 0.95)), float((1 + (null >= obs).sum()) / (n + 1))


def episodes(lab, trigger, gap=5):
    """연속(gap샷 이내)된 trigger 샷을 한 에피소드로 묶어 시작 샷만 반환"""
    out = np.zeros(len(lab), bool)
    runs, seq = lab['run_id'].values, lab['seq'].values
    last = {}
    for i in np.flatnonzero(trigger):
        r = runs[i]
        if r not in last or seq[i] - last[r] > gap:
            out[i] = True
        last[r] = seq[i]
    return out


def run(products=('cn7', 'rg3'), ks=KS, qs=(0.999, 0.99), n_perm=5000, verbose=True):
    os.makedirs(OUT, exist_ok=True)
    rows, nulls, eps, pos = [], [], [], {}
    for p in products:
        lab = load(p)
        ev = lab['eval'].values
        t = lab['type'].values
        triggers = {'실제 미성형 라벨(전부)': ev & (t == '미성형'),
                    '실제 미성형 라벨(에피소드 첫 샷)': episodes(lab, ev & (t == '미성형'))}
        for q in qs:
            triggers[f'최종 모델 경보 pct≥{q}'] = ev & (lab['pct'].values >= q)
        for tn, trig in triggers.items():
            for k in ks:
                mk = window_mask(lab, trig, k)
                r = describe(lab, mk, f'{tn} · 직전 {k}샷')
                r.update({'제품': p, '기준점': tn, 'k': k, '기준점 수': int(trig.sum())})
                o, mu, q95, pv = circ_null(lab, mk, n_perm)
                r.update({'무작위 위치 가스 평균': round(mu, 2), '무작위 95%': q95, '순열 p': round(pv, 4)})
                rows.append(r)
                if k == 20:
                    pos[(p, tn)] = gas_positions(lab, mk)
        # 에피소드 단위: 미성형 에피소드마다 직전 20샷에 가스가 있었나
        ep = episodes(lab, ev & (t == '미성형'))
        for i in np.flatnonzero(ep):
            one = np.zeros(len(lab), bool); one[i] = True
            mk = window_mask(lab, one, 20)
            eps.append({'제품': p, 'Run': lab['run_id'].iloc[i], '미성형 첫 샷 seq': int(lab['seq'].iloc[i]),
                        '직전 20샷 가스 수': int((mk & ev & (t == '가스')).sum()),
                        '그 Run 전체 가스 수': int(((lab['run_id'] == lab['run_id'].iloc[i]).values & ev & (t == '가스')).sum())})
        if verbose:
            print(f'[{p}] 완료', flush=True)
    R = pd.DataFrame(rows)
    E = pd.DataFrame(eps)
    cols = ['제품', '기준점', 'k', '기준점 수', '역추적 검사 샷', '그 안의 가스', '그 안의 정상', '가스 Precision', '가스 Recall',
            '무작위 대비(배)', '무작위 위치 가스 평균', '무작위 95%', '순열 p']
    R = R[cols]
    R.to_csv(f'{OUT}/rule_check.csv', index=False, encoding='utf-8-sig')
    E.to_csv(f'{OUT}/episodes.csv', index=False, encoding='utf-8-sig')
    return {'table': R, 'episodes': E, 'positions': pos}


def plot_r1(p='cn7', k=20, q=0.999, save=None):
    """가스가 몰린 Run: 샷 순서 위에 가스·미성형·경보 위치와 '미성형 직전 k샷' 창 표시"""
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    lab = load(p)
    ev, t = lab['eval'].values, lab['type'].values
    run = lab.loc[ev & (t == '가스'), 'run_id'].mode().iloc[0]
    r = lab[lab['run_id'] == run]
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.plot(r['seq'], r['pct'], color='#b5b4ad', lw=1, label='최종 모델 pct')
    for i in r.index[(r['type'] == '미성형').values]:
        s = r.loc[i, 'seq']
        ax.axvspan(s - k, s - 0.5, color='#1f4e79', alpha=0.06)
    for ty, c, mk in (('가스', '#e67e22', 'v'), ('미성형', '#1f4e79', 'v')):
        x = r.loc[r['type'] == ty, 'seq']
        ax.scatter(x, np.full(len(x), 1.04), color=c, marker=mk, s=30, label=ty)
    a = r[r['pct'] >= q]
    ax.scatter(a['seq'], a['pct'], color='k', marker='o', s=18, label=f'경보 pct≥{q}')
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(f'{run} 안 샷 순서'); ax.set_ylabel('pct')
    ax.set_title(f'{p.upper()} {run} — 파란 음영 = 미성형 직전 {k}샷', loc='left', fontsize=10)
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
