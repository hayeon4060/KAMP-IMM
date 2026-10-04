"""
r1_compare.py — CN7 R1 안에서 불량 vs 정상 비교 (설명용 분석, 예측 성능 아님)

왜 R1 안에서만 비교하나
  - CN7 공정불량 15개 중 14개가 R1 → Run 간 비교는 'Run 차이'와 '불량 차이'가 섞임
  - 같은 Run 안에서 비교하면 금형·재료·설정 같은 Run 공통 조건이 자동으로 통제됨

주의 (해석할 때 반드시 같이 볼 것)
  - Run 안에서도 시간이 흐르면 온도 등이 변함. 불량이 Run 후반에 몰려 있으면 '시간 추세'가 '불량 차이'로 보일 수 있음
    → compare_table(match_window=k): 불량 샷 앞뒤 k샷 안의 정상 샷만 비교군으로 쓴 결과를 같이 제공
  - 불량 14개 → p값보다 효과크기(Cliff's delta)·AUC를 중심으로 볼 것. 다중비교는 BH-FDR로 보정

지표
  med_정상 / med_불량 : 중앙값
  diff_IQR           : (불량 중앙값 − 정상 중앙값) / 정상 IQR → '정상 변동폭 몇 배만큼 다른가'
  diff_step          : 같은 차이를 기록 단위(0.1·0.01) 칸 수로 → 이산 기록이라 1~2칸 차이는 의미 약함
  cliffs_delta       : P(불량 > 정상) − P(불량 < 정상), −1~+1. |δ| ≥ 0.33 중간, ≥ 0.47 큼 (Romano 기준)
  AUC                : 변수 하나로 불량/정상을 가르는 능력 (0.5 = 무관, 방향 무관하게 보려면 |AUC−0.5|)
  p / q_FDR          : Mann-Whitney U 양측 검정 / BH 보정
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from features import ALL_RAW, raw, resolution
from plot_style import set_style, short, INK, SUB, MUTED, NORMAL, TYPE_COL, TYPE_ORDER, ACCENT

GROUPS = {'전체불량': ['가스', '가스+미성형', '미성형'],
          '가스': ['가스'],
          '미성형계열': ['미성형', '가스+미성형']}

DERIVED = ['dev_Max_Injection_Speed', 'dev_Filling_Time', 'dev_Max_Injection_Pressure',
           'dev_Max_Switch_Over_Pressure', 'dev_Average_Back_Pressure', 'dev_Cushion_Position',
           'dev_Plasticizing_Time', 'dev_Cycle_Time', 'fill_anomaly', 'ev_max_abs_dev',
           'dev_Mold_Temperature_3', 'dev_Hopper_Temperature', 'mold_rise', 'hopper_rise',
           'mold_slope20', 'hopper_slope20', 'flow_res', 'dev_flow_res']


def load_r1(path='./base/cn7_features.csv', run='cn7_R1'):
    d = pd.read_csv(path, parse_dates=['TimeStamp'])
    d['Reason'] = d['Reason'].fillna('정상')
    r1 = d[d['run_id'] == run].sort_values('shot_seq_in_run').reset_index(drop=True)
    print(f'{run}: {len(r1)}샷, 불량', r1.loc[r1['PassOrFail'] == 1, 'Reason'].value_counts().to_dict(),
          '| 불량 샷 순번:', r1.loc[r1['PassOrFail'] == 1, 'shot_seq_in_run'].tolist())
    return d, r1


def variable_list(d):
    raws = [raw(v) for v in ALL_RAW if raw(v) in d.columns and d[raw(v)].nunique() > 1]
    der = [c for c in DERIVED if c in d.columns]
    return raws, der


# ====================================================================== 지표
def _bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    o = np.argsort(pv)
    r = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(len(pv)); out[o] = np.clip(r, 0, 1)
    q[ok] = out
    return q


def _matched_normal(r1, pos_mask, window):
    """불량 샷 앞뒤 window샷 안에 있는 정상 샷 (시간 추세 통제용 비교군)"""
    seq = r1['shot_seq_in_run'].values
    dseq = seq[pos_mask]
    near = np.zeros(len(r1), bool)
    for s in dseq:
        near |= np.abs(seq - s) <= window
    return near & (r1['PassOrFail'].values == 0)


def compare_table(r1, cols, group='전체불량', match_window=None):
    """
    group        : GROUPS 키 (전체불량 / 가스 / 미성형계열)
    match_window : None = R1 정상 전체와 비교, k = 불량 앞뒤 k샷 안의 정상만 비교
    """
    pos = r1['Reason'].isin(GROUPS[group]).values
    neg = _matched_normal(r1, pos, match_window) if match_window else (r1['PassOrFail'].values == 0)
    rows = []
    for c in cols:
        a = r1.loc[pos, c].dropna().values.astype(float)
        b = r1.loc[neg, c].dropna().values.astype(float)
        if len(a) < 2 or len(b) < 2:
            continue
        if np.ptp(np.concatenate([a, b])) == 0:
            continue
        U, p = mannwhitneyu(a, b, alternative='two-sided')
        auc = U / (len(a) * len(b))
        iqr = np.subtract(*np.percentile(b, [75, 25]))
        step = resolution(r1[c]) if c.endswith('_raw') else np.nan
        diff = np.median(a) - np.median(b)
        rows.append({'변수': c, '이름': short(c), 'n불량': len(a), 'n정상': len(b),
                     'med_정상': np.median(b), 'med_불량': np.median(a),
                     'diff_IQR': diff / iqr if iqr > 0 else np.nan,
                     'diff_step': diff / step if step and step > 0 else np.nan,
                     'cliffs_delta': 2 * auc - 1, 'AUC': auc, '|AUC-0.5|': abs(auc - 0.5), 'p': p})
    t = pd.DataFrame(rows)
    t['q_FDR'] = _bh(t['p'].values)
    t['효과'] = pd.cut(t['cliffs_delta'].abs(), [-1, 0.147, 0.33, 0.47, 1.01],
                     labels=['무시', '작음', '중간', '큼'])
    t['비교군'] = '정상 전체' if not match_window else f'불량 ±{match_window}샷 정상'
    return t.sort_values('|AUC-0.5|', ascending=False).reset_index(drop=True)


def both_tables(r1, cols, group='전체불량', window=10):
    """정상 전체 vs 시간 맞춘 정상 결과를 나란히 → 둘 다 큰 변수만 '불량 특성' 후보"""
    a = compare_table(r1, cols, group)
    b = compare_table(r1, cols, group, match_window=window)
    m = a[['변수', '이름', 'med_정상', 'med_불량', 'diff_IQR', 'diff_step', 'cliffs_delta', 'AUC', 'q_FDR', '효과']].merge(
        b[['변수', 'cliffs_delta', 'AUC', 'q_FDR', '효과']], on='변수', suffixes=('', '_시간맞춤'))
    m['일관'] = (np.sign(m['cliffs_delta']) == np.sign(m['cliffs_delta_시간맞춤'])) & \
               (m['cliffs_delta'].abs() >= 0.33) & (m['cliffs_delta_시간맞춤'].abs() >= 0.33)
    return m.sort_values('cliffs_delta_시간맞춤', key=np.abs, ascending=False).reset_index(drop=True)


def run_context_table(d, cols, run='cn7_R1'):
    """R1 정상 샷 vs 다른 CN7 Run 정상 샷 → 'R1이라서 다른 것'(Run 특성)이 뭔지"""
    g = d[d['PassOrFail'] == 0]
    a, b = g[g['run_id'] == run], g[g['run_id'] != run]
    rows = []
    for c in cols:
        x, y = a[c].dropna(), b[c].dropna()
        if len(x) < 2 or len(y) < 2 or np.ptp(np.r_[x, y]) == 0:
            continue
        U, _ = mannwhitneyu(x, y, alternative='two-sided')
        iqr = np.subtract(*np.percentile(y, [75, 25]))
        rows.append({'변수': c, '이름': short(c), 'R1_정상_med': x.median(), '다른Run_정상_med': y.median(),
                     'diff_IQR': (x.median() - y.median()) / iqr if iqr > 0 else np.nan,
                     'cliffs_delta(R1-다른Run)': 2 * U / (len(x) * len(y)) - 1})
    return pd.DataFrame(rows).sort_values('cliffs_delta(R1-다른Run)', key=np.abs, ascending=False)


# ====================================================================== 그래프
def plot_effects(m, top=20, title='', save=None):
    """효과크기 점그래프: 정상 전체(회색) vs 시간 맞춘 정상(파랑). 둘 다 같은 쪽으로 크면 신뢰"""
    set_style()
    t = m.head(top).iloc[::-1]
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(t) + 1.6))
    for v in (-0.47, -0.33, 0.33, 0.47):
        ax.axvline(v, color=MUTED, lw=0.6, ls=':' if abs(v) == 0.33 else '--', zorder=0)
    ax.axvline(0, color=SUB, lw=0.8)
    ax.hlines(y, t['cliffs_delta'], t['cliffs_delta_시간맞춤'], color='#d6d5cc', lw=2, zorder=1)
    ax.scatter(t['cliffs_delta'], y, s=46, color=NORMAL, edgecolor='white', lw=1.5, zorder=3, label='비교군: R1 정상 전체')
    ax.scatter(t['cliffs_delta_시간맞춤'], y, s=46, color=ACCENT, edgecolor='white', lw=1.5, zorder=3,
               label='비교군: 불량 앞뒤 정상 (시간 맞춤)')
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}{'  *' if k else ''}" for n, k in zip(t['이름'], t['일관'])], fontsize=9)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Cliff's δ  (+ = 불량 쪽이 큼,  점선 0.33 중간 · 파선 0.47 큼)")
    ax.set_title(title or "불량 vs 정상 효과크기 — * = 두 비교군 모두 |δ|≥0.33, 같은 방향", loc='left')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()


def plot_timeline(r1, cols, title='', save=None):
    """R1 샷 순번 × 변수: 회색 선 = 전체 흐름, 색 점 = 불량 유형. 시간 추세와 불량 위치를 같이 봄"""
    set_style()
    fig, axes = plt.subplots(len(cols), 1, figsize=(13, 1.9 * len(cols)), sharex=True)
    axes = np.atleast_1d(axes)
    x = r1['shot_seq_in_run']
    stops = r1.loc[r1['after_stop'].astype(bool), 'shot_seq_in_run'] if 'after_stop' in r1 else []
    for ax, c in zip(axes, cols):
        ax.plot(x, r1[c], color=MUTED, lw=1, zorder=1)
        roll = r1[c].rolling(15, center=True, min_periods=5).median()
        ax.plot(x, roll, color=SUB, lw=1.6, zorder=2)
        for t in TYPE_ORDER:
            h = r1[r1['Reason'] == t]
            ax.scatter(h['shot_seq_in_run'], h[c], s=52, color=TYPE_COL[t], edgecolor='white', lw=1.5,
                       zorder=3, label=t)
        for s in stops:
            ax.axvline(s, color=INK, lw=0.6, ls='--', zorder=0)
        ax.set_ylabel(short(c), rotation=0, ha='right', va='center', fontsize=9)
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h + [plt.Line2D([], [], color=SUB, lw=1.6)], l + ['15샷 이동 중앙값'],
               loc='upper right', ncol=5, fontsize=8)
    axes[-1].set_xlabel('R1 샷 순번 (점선 = 짧은 정지 후 재가동)')
    fig.suptitle(title or 'CN7 R1 타임라인 — 불량이 시간 추세 위에 있는지, 순간적으로 튀는지', x=0.01, ha='left',
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save:
        fig.savefig(save)
    plt.show()


def plot_distributions(r1, cols, window=10, save=None):
    """변수별 분포: 정상 전체 / 시간 맞춘 정상 / 가스 / 미성형계열"""
    set_style()
    pos_any = r1['PassOrFail'].values == 1
    matched = _matched_normal(r1, pos_any, window)
    cats = [('정상 전체', r1['PassOrFail'] == 0, NORMAL),
            (f'정상 (불량 ±{window}샷)', pd.Series(matched), ACCENT),
            ('가스', r1['Reason'] == '가스', TYPE_COL['가스']),
            ('미성형계열', r1['Reason'].isin(GROUPS['미성형계열']), TYPE_COL['미성형'])]
    n = len(cols)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(0)
    for ax, c in zip(axes, cols):
        for i, (name, m, col) in enumerate(cats):
            v = r1.loc[m.values, c].dropna()
            if len(v) == 0:
                continue
            ax.boxplot(v, positions=[i], widths=0.5, showfliers=False,
                       medianprops=dict(color=INK), boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED),
                       capprops=dict(color=MUTED))
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=10 if i < 2 else 30, color=col,
                       alpha=0.5 if i < 2 else 1, edgecolor='none' if i < 2 else 'white', lw=1, zorder=3)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([c_[0] for c_ in cats], rotation=30, ha='right', fontsize=8)
        ax.set_title(short(c), fontsize=10)
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle('CN7 R1 변수 분포 — 정상(전체·시간 맞춤) vs 불량 유형', x=0.01, ha='left', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        fig.savefig(save)
    plt.show()


def plot_run_context(d, cols, run='cn7_R1', save=None):
    """CN7 Run별 정상 샷 분포 + R1 불량 샷 → R1이 Run 전체로 다른지(상태형), 불량만 다른지(이벤트형)"""
    set_style()
    runs = sorted(d['run_id'].unique())
    n = len(cols)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        data = [d[(d['run_id'] == r) & (d['PassOrFail'] == 0)][c].dropna() for r in runs]
        bp = ax.boxplot(data, positions=range(len(runs)), widths=0.55, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK))
        for patch, r in zip(bp['boxes'], runs):
            patch.set_facecolor('#f4d6d5' if r == run else '#efeee9')
            patch.set_edgecolor(MUTED)
        i = runs.index(run)
        for t in TYPE_ORDER:
            h = d[(d['run_id'] == run) & (d['Reason'] == t)][c]
            ax.scatter(np.full(len(h), i + 0.35), h, s=26, color=TYPE_COL[t], edgecolor='white', lw=1, zorder=3,
                       label=t)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([r.replace('cn7_', '') for r in runs], fontsize=8)
        ax.set_title(short(c), fontsize=10)
    for ax in axes[n:]:
        ax.axis('off')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper right', ncol=3, fontsize=8)
    fig.suptitle('CN7 Run별 정상 샷 분포(상자) + R1 불량 샷(점) — R1 상자 자체가 다르면 Run 특성', x=0.01, ha='left',
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save:
        fig.savefig(save)
    plt.show()
