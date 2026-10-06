"""
thermal_cn7.py — CN7 전용: 재가동 직후를 빼면 '온도 교란 구간' 임계값이 의미가 생기나

28번(thermal_episode.py)과 같은 구간 점수 E를 쓰고, 바꾼 것만:
  1) 재가동 직후 N샷 제외 (N = 15 기본, 10·20 민감도)
     재가동 = Run 시작(60분 넘는 공백 뒤) 또는 Run 안 짧은 정지(5분 넘는 공백) 뒤
     → 워밍업 중 온도가 흔들리는 건 공정상 당연하므로 가스 결과와 무관하게 정할 수 있는 조건
  2) 임계값도 '재가동 직후를 뺀 정상 생산(unlabeled)'에서 다시 계산
  3) 임계값 하나를 고르지 않고 여러 오경보율(q)에서 결과를 표로 → 운영 옵션으로 제시

보는 것
  - q별: 정상 생산 1,000샷당 오경보 에피소드 / labeled 불량 없는 에피소드 / 가스 탐지 / 첫 미성형보다 몇 샷 먼저
  - R1 가스 언덕의 정점이 정상 생산 E 분포에서 상위 몇 %인지
⚠ 규칙의 방향은 R1을 보고 정했으므로 R1 결과는 사후 확인. 정상 생산 쪽 수치(오경보 빈도, 정점 백분위)는 라벨과 무관
"""
import numpy as np
import pandas as pd

from thermal_episode import prepare, episodes

STOP_GAP_SEC = 5 * 60
QS = (0.90, 0.95, 0.97, 0.98, 0.99, 0.995)


def add_since_restart(f):
    """Run 안에서 마지막 재가동(Run 시작 또는 5분 넘는 정지) 이후 몇 번째 샷인지"""
    f = f.sort_values(['run_id', 'TimeStamp']).copy()
    gap = f.groupby('run_id', sort=False)['TimeStamp'].diff().dt.total_seconds()
    restart = gap.isna() | (gap > STOP_GAP_SEC)
    seg = restart.groupby(f['run_id']).cumsum()
    f['since_restart'] = f.groupby([f['run_id'], seg]).cumcount() + 1
    return f


def load(p='cn7', base_dir='.', model_dir='./outputs/final_model'):
    lab, ref, _ = prepare(p, base_dir, model_dir)
    return add_since_restart(lab), add_since_restart(ref)


def _gas_episode_window(lab):
    """가스가 가장 많은 Run과 그 Run의 첫 미성형 seq"""
    g = lab[lab['in_eval'] & (lab['group'] == '가스')]
    run = g['run_id'].mode()[0]
    r = lab[lab['run_id'] == run]
    fs = r.loc[r['group'] == '미성형계열', 'seq'].min()
    return run, (None if pd.isna(fs) else int(fs))


def tradeoff(lab, ref, n_excl=15, qs=QS):
    ref_ok = ref[ref['since_restart'] > n_excl]
    lab_ok = lab['since_restart'] > n_excl
    e = lab[lab['in_eval']]
    run, first_short = _gas_episode_window(lab)
    rows = []
    for q in qs:
        t = float(ref_ok['E'].quantile(q))
        ref_al = (ref['E'] >= t) & (ref['since_restart'] > n_excl)
        ep_ref = episodes(ref.assign(al=ref_al), 'al')
        lab_al = (lab['E'] >= t) & lab_ok
        L = lab.assign(al=lab_al)
        ep_lab = episodes(L, 'al')
        if len(ep_lab):
            bad = (ep_lab['구간 내 불량'] > 0) | (ep_lab['직후 10샷 불량'] > 0)
        else:
            bad = pd.Series(dtype=bool)
        gas_hit = int((L.loc[e.index, 'al'] & (e['group'] == '가스')).sum())
        lead = None
        if first_short:
            r = L[(L['run_id'] == run) & (L['seq'] < first_short) & (L['seq'] >= first_short - 30) & L['al']]
            lead = int(first_short - r['seq'].min()) if len(r) else 0
        nrm = e['group'] == '정상'
        rows.append({'재가동 제외 샷': n_excl, 'q': q, '정상 오경보율': f'{100 * (1 - q):g}%', '임계값 E': t,
                     '정상생산 1,000샷당 에피소드': 1000 * len(ep_ref) / len(ref_ok),
                     'labeled 정상 샷 경보 비율': float(L.loc[e.index][nrm.values]['al'].mean()),
                     'labeled 에피소드': len(ep_lab),
                     '불량 관련 에피소드(구간 내·직후 10샷)': int(bad.sum()),
                     '불량 없는 에피소드': int((~bad).sum()) if len(ep_lab) else 0,
                     '가스 탐지': f'{gas_hit}/{int((e["group"] == "가스").sum())}',
                     '첫 미성형보다 먼저 울린 샷 수': lead})
    return pd.DataFrame(rows)


def peak_rank(lab, ref, n_excl=15):
    """R1 가스 언덕 정점(첫 미성형 직전까지 최대 E)이 재가동 제외 정상 생산 E에서 상위 몇 %인지"""
    run, fs = _gas_episode_window(lab)
    r = lab[(lab['run_id'] == run) & (lab['since_restart'] > n_excl)]
    g = r[r['group'] == '가스']
    win = r[(r['seq'] >= g['seq'].min()) & (r['seq'] < (fs if fs else r['seq'].max() + 1))]
    peak = float(win['E'].max())
    ref_ok = ref.loc[ref['since_restart'] > n_excl, 'E']
    # 같은 높이 이상의 '언덕'이 정상 생산에 몇 번 있었나
    ep = episodes(ref.assign(al=(ref['E'] >= peak) & (ref['since_restart'] > n_excl)), 'al')
    # labeled 다른 구간(가스 언덕 밖)에서 같은 높이 이상
    other = lab[(lab['since_restart'] > n_excl) & ~(lab.index.isin(win.index))]
    ep_lab = episodes(other.assign(al=other['E'] >= peak), 'al')
    return {'재가동 제외 샷': n_excl, 'R1 가스 언덕 정점 E': peak,
            '정상생산에서 상위 %': 100 * float((ref_ok >= peak).mean()),
            '정상생산에서 같은 높이 이상 언덕 수': len(ep),
            '정상생산 1,000샷당 그런 언덕': 1000 * len(ep) / len(ref_ok),
            'labeled 다른 구간의 같은 높이 이상 언덕 수': len(ep_lab)}


def plot_tradeoff(T, save=None):
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n, g in T.groupby('재가동 제외 샷'):
        x = g['정상생산 1,000샷당 에피소드'].values
        y = g['가스 탐지'].str.split('/').str[0].astype(int).values
        ax.plot(x, y, marker='o', label=f'재가동 후 {n}샷 제외')
        for xi, yi, q in zip(x, y, g['q']):
            ax.annotate(f'q{q}', (xi, yi), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('정상 생산 1,000샷당 오경보 에피소드')
    ax.set_ylabel('가스 탐지 수 (CN7, 총 11)')
    ax.set_title('CN7 온도 교란 구간 — 오경보 vs 가스 탐지', loc='left', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
