"""
run_compare.py — 여러 Run에 걸친 불량 vs 정상 비교 (RG3 교차확인 + RG3 구분 가능성)

PART 1. CN7 R1 가스 패턴이 RG3에서도 나오나 (교차확인)
  - Run마다 수준이 달라서 Run끼리 그냥 섞으면 'Run 차이'가 '불량 차이'로 보임
    → 변수를 Run 안에서 표준화(run_z)한 뒤 합쳐서 비교 = '같은 Run 안에서 불량 샷이 정상 샷보다 높나/낮나'
    run_z = (x − 그 Run 전체 중앙값) / max(그 Run IQR, 기록 단위)   ※ 라벨 안 씀 (불량 ≤ 20%라 중앙값 영향 작음)
  - 비교군 2개: 같은 Run 정상 전체 / 같은 Run 불량 앞뒤 ±k샷 정상 (시간 추세 통제)
  - Run별 δ와 '같은 방향 Run 수' → 특정 Run 하나 때문인지, 여러 Run에서 반복되는지
  - 시간 군집 검정: 불량이 Run 안에서 몰려서 나오나 (CN7 R1은 49~58번째에 9개)
  - Run 수준: Run 불량률 vs Run 상태(금형·호퍼 온도 등) — RG3 Run 6개라 경향만

PART 2. RG3 정상 vs 불량 구분 가능성
  - 단변량 표(PART 1과 같은 run_z) + 상위 변수 2개 산점도
  - 다변량: RG3 Run별 LORO logit (run_z 변수) → AUC + 라벨 섞기(permutation) 귀무분포와 비교
    ※ run_z는 Run 전체(미래 샷 포함)로 계산 → 현장 배치용이 아니라 '구분 가능성' 확인용
  - 캐비티: RG3 2차 불량은 전부 RH → 공정값이 LH/RH 공통이라 캐비티 차이는 설명 불가 (표로 확인)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from features import ALL_RAW, raw, resolution
from scope import apply_scope
from r1_compare import DERIVED, _bh
from plot_style import set_style, short, INK, SUB, MUTED, NORMAL, TYPE_COL, TYPE_ORDER, ACCENT, DIV

SEED = 42
GROUPS = {'전체불량': ['가스', '가스+미성형', '미성형'], '가스': ['가스'], '미성형계열': ['미성형', '가스+미성형']}
# CN7 R1에서 나온 가스 가설 변수 (09 노트북 결과)
GAS_HYP = ['Hopper_Temperature_raw', 'dev_Hopper_Temperature', 'hopper_rise', 'hopper_slope20',
           'Mold_Temperature_3_raw', 'dev_Mold_Temperature_3', 'mold_slope20', 'Plasticizing_Time_raw']


# ====================================================================== 데이터
def load_product(p, path='./base/{}_features.csv'):
    d = pd.read_csv(path.format(p), parse_dates=['TimeStamp'])
    d = d[d['split'] != '제외']
    d = apply_scope(d, exclude_initial=True)
    d['Reason'] = d['Reason'].fillna('정상')
    d = d.sort_values(['run_id', 'shot_seq_in_run']).reset_index(drop=True)
    s = d.groupby('run_id').agg(샷=('PassOrFail', 'size'), 불량=('PassOrFail', 'sum'),
                                가스=('Reason', lambda r: (r == '가스').sum()),
                                미성형계열=('Reason', lambda r: r.isin(GROUPS['미성형계열']).sum()))
    if 'LH_fail' in d:
        s['LH불량'] = d.groupby('run_id')['LH_fail'].sum()
        s['RH불량'] = d.groupby('run_id')['RH_fail'].sum()
    s['불량률'] = (s['불량'] / s['샷']).round(3)
    print(f'[{p}] 샷 {len(d)}, 불량 {int(d["PassOrFail"].sum())}')
    return d, s


def variable_list(d):
    raws = [raw(v) for v in ALL_RAW if raw(v) in d.columns and d[raw(v)].nunique() > 1]
    return raws + [c for c in DERIVED if c in d.columns]


def add_run_z(d, cols):
    """Run 안 표준화 (라벨 미사용). 열 이름 = 원래 이름 그대로 반환되는 새 DataFrame"""
    z = pd.DataFrame(index=d.index)
    for c in cols:
        step = resolution(d[c]) if c.endswith('_raw') else 0.0
        g = d.groupby('run_id')[c]
        med = g.transform('median')
        iqr = g.transform(lambda s: s.quantile(.75) - s.quantile(.25))
        sd = g.transform('std')
        den = np.maximum(iqr, step)
        den = den.where(den > 1e-12, sd).where(lambda x: x > 1e-12, 1.0)
        z[c] = (d[c] - med) / den
    return z


def _matched(d, pos, window):
    """같은 Run 안에서 불량 앞뒤 window샷 안의 정상"""
    near = np.zeros(len(d), bool)
    seq, run = d['shot_seq_in_run'].values, d['run_id'].values
    for i in np.flatnonzero(pos):
        near |= (run == run[i]) & (np.abs(seq - seq[i]) <= window)
    return near & (d['PassOrFail'].values == 0)


def _delta(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 1 or len(b) < 1 or np.ptp(np.r_[a, b]) == 0:
        return np.nan, np.nan
    U, p = mannwhitneyu(a, b, alternative='two-sided')
    return 2 * U / (len(a) * len(b)) - 1, p


# ====================================================================== PART 1. 표
def pooled_table(d, Z, cols, group='가스', window=None, min_def_run=2):
    """
    run_z로 합친 불량 vs 정상 + Run별 δ
    window      : None = 같은 Run 정상 전체, k = 같은 Run 불량 앞뒤 k샷 정상
    min_def_run : 이 개수 이상 불량이 있는 Run만 Run별 δ·방향 일치 계산
    """
    pos = d['Reason'].isin(GROUPS[group]).values
    neg = _matched(d, pos, window) if window else (d['PassOrFail'].values == 0)
    runs = [r for r in d['run_id'].unique() if pos[d['run_id'].values == r].sum() >= min_def_run]
    rows = []
    for c in cols:
        x = Z[c].values
        dl, p = _delta(x[pos], x[neg])
        if np.isnan(dl):
            continue
        r = {'변수': c, '이름': short(c), 'n불량': int(pos.sum()), 'n정상': int(neg.sum()),
             'med_z_불량': np.nanmedian(x[pos]), 'cliffs_delta': dl, 'AUC': (dl + 1) / 2, 'p': p}
        per = {}
        for run in runs:
            m = d['run_id'].values == run
            per[run], _ = _delta(x[pos & m], x[neg & m])
        r.update({f'δ_{k}': v for k, v in per.items()})
        vals = np.array([v for v in per.values() if np.isfinite(v)])
        r['같은방향_Run'] = f"{int((np.sign(vals) == np.sign(dl)).sum())}/{len(vals)}" if len(vals) else '-'
        rows.append(r)
    t = pd.DataFrame(rows)
    t['q_FDR'] = _bh(t['p'].values)
    t['비교군'] = '같은 Run 정상 전체' if not window else f'같은 Run 불량 ±{window}샷 정상'
    return t.sort_values('cliffs_delta', key=np.abs, ascending=False).reset_index(drop=True)


def both_pooled(d, Z, cols, group='가스', window=10):
    a = pooled_table(d, Z, cols, group)
    b = pooled_table(d, Z, cols, group, window)
    keep = ['변수', 'cliffs_delta', 'q_FDR', '같은방향_Run']
    m = a.merge(b[keep], on='변수', suffixes=('', '_시간맞춤'))
    m['일관'] = (np.sign(m['cliffs_delta']) == np.sign(m['cliffs_delta_시간맞춤'])) & \
               (m['cliffs_delta'].abs() >= 0.33) & (m['cliffs_delta_시간맞춤'].abs() >= 0.33)
    return m.sort_values('cliffs_delta_시간맞춤', key=np.abs, ascending=False).reset_index(drop=True)


def hypothesis_check(t_cn7, t_rg3, hyp=GAS_HYP):
    """CN7 R1 가스 가설 변수: CN7 δ vs RG3 δ (둘 다 시간 맞춤 포함)"""
    cols = ['변수', '이름', 'cliffs_delta', 'cliffs_delta_시간맞춤']
    a = t_cn7[t_cn7['변수'].isin(hyp)][cols]
    b = t_rg3[t_rg3['변수'].isin(hyp)][cols + ['같은방향_Run', '같은방향_Run_시간맞춤', 'q_FDR']]
    m = a.merge(b, on=['변수', '이름'], suffixes=('_CN7', '_RG3'), how='outer')
    m['방향 재현'] = np.sign(m['cliffs_delta_시간맞춤_CN7']) == np.sign(m['cliffs_delta_시간맞춤_RG3'])
    m['RG3도 중간 이상'] = m['cliffs_delta_시간맞춤_RG3'].abs() >= 0.33
    return m


def cluster_test(d, group='가스', n_perm=5000, seed=SEED):
    """
    Run 안에서 불량이 몰려 나오나: 통계량 = 불량 샷 간 '가장 가까운 불량까지 거리' 평균
    귀무: 같은 Run 안에서 불량 위치를 무작위로 재배치 (불량 수 유지). p = 실제보다 작거나 같은 비율
    """
    rng = np.random.default_rng(seed)
    pos = d['Reason'].isin(GROUPS[group])

    def stat(pos_seq_by_run):
        nn = []
        for s in pos_seq_by_run:
            if len(s) < 2:
                continue
            s = np.sort(s)
            gap = np.diff(s)
            nn += list(np.minimum(np.r_[np.inf, gap], np.r_[gap, np.inf]))
        return np.mean(nn) if nn else np.nan

    runs = {r: (g['shot_seq_in_run'].values, pos[g.index].values) for r, g in d.groupby('run_id')}
    real = stat([seq[p] for seq, p in runs.values()])
    null = []
    for _ in range(n_perm):
        null.append(stat([rng.choice(seq, p.sum(), replace=False) for seq, p in runs.values() if p.sum()]))
    null = np.array(null)
    rows = [{'Run': r, '샷': len(seq), f'{group} 수': int(p.sum()), f'{group} 순번': list(seq[p])}
            for r, (seq, p) in runs.items() if p.sum()]
    return {'관측 평균 최근접 거리': real, '무작위 평균': np.nanmean(null),
            'p(몰림)': (np.sum(null <= real) + 1) / (len(null) + 1)}, pd.DataFrame(rows), null


def run_level_table(d, state=('Mold_Temperature_3_raw', 'Hopper_Temperature_raw', 'flow_res',
                              'Barrel_Temperature_4_raw', 'Plasticizing_Position_raw')):
    """Run 수준: 불량률 vs Run 상태 중앙값 (Run 수가 적어 Spearman은 경향만)"""
    g = d.groupby('run_id')
    t = pd.DataFrame({'샷': g.size(), '불량률': g['PassOrFail'].mean(),
                      '가스율': g['Reason'].apply(lambda r: (r == '가스').mean())})
    for c in state:
        if c in d:
            t[short(c)] = g[c].median()
    cor = []
    for c in state:
        if c in d:
            for y in ['불량률', '가스율']:
                rho, p = spearmanr(t[short(c)], t[y])
                cor.append({'상태': short(c), '대상': y, 'Spearman ρ': rho, 'p': p, 'Run 수': len(t)})
    return t, pd.DataFrame(cor)


# ====================================================================== PART 2. 구분 가능성
def separability_loro(d, Z, cols, n_perm=200, C=0.1, seed=SEED):
    """
    RG3 Run별 LORO logit (run_z 변수). 반환: (요약, OOF, 귀무 AUC 배열)
    귀무: 각 Run 안에서 라벨을 섞어 같은 LORO를 n_perm번 → '이 정도 AUC는 우연으로도 나오나'
    """
    rng = np.random.default_rng(seed)
    X = Z[cols].replace([np.inf, -np.inf], np.nan).values
    runs = d['run_id'].values

    def loro(y):
        s = np.zeros(len(y))
        for r in np.unique(runs):
            te = runs == r
            if y[~te].sum() < 2:
                continue
            m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                              LogisticRegression(C=C, class_weight='balanced', max_iter=5000))
            m.fit(X[~te], y[~te])
            s[te] = m.predict_proba(X[te])[:, 1]
        return s

    y = d['PassOrFail'].values
    s = loro(y)
    oof = pd.DataFrame({'run_id': runs, 'shot_seq': d['shot_seq_in_run'].values, 'y': y,
                        'Reason': d['Reason'].values, 'score': s})
    auc = roc_auc_score(y, s)
    per_run = {r: roc_auc_score(g['y'], g['score']) for r, g in oof.groupby('run_id') if 0 < g['y'].sum() < len(g)}
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for r in np.unique(runs):
            idx = np.flatnonzero(runs == r)
            yp[idx] = rng.permutation(yp[idx])
        null.append(roc_auc_score(yp, loro(yp)))
    null = np.array(null)
    summ = {'LORO AUC': auc, 'Run별 AUC 평균': np.mean(list(per_run.values())), 'Run별 AUC': per_run,
            '귀무 AUC 평균': null.mean(), '귀무 95%': np.percentile(null, 95),
            'p(우연)': (np.sum(null >= auc) + 1) / (len(null) + 1)}
    return summ, oof, null


def cavity_table(d):
    """불량 샷의 LH/RH 분포 (Run별)"""
    f = d[d['PassOrFail'] == 1].copy()
    lh, rh = f['LH_fail'].fillna(0) == 1, f['RH_fail'].fillna(0) == 1
    f['캐비티'] = np.select([lh & rh, lh, rh], ['LH+RH', 'LH만', 'RH만'], '기록없음')
    return pd.crosstab(f['run_id'], [f['Reason'], f['캐비티']], margins=True, margins_name='합계')


# ====================================================================== 그래프
def plot_hypothesis(h, save=None):
    """가스 가설 변수: CN7 R1(빨강) vs RG3(파랑), 시간 맞춤 δ"""
    set_style()
    t = h.dropna(subset=['cliffs_delta_시간맞춤_CN7', 'cliffs_delta_시간맞춤_RG3']).iloc[::-1]
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.6))
    for v in (-0.47, -0.33, 0.33, 0.47):
        ax.axvline(v, color=MUTED, lw=0.6, ls=':' if abs(v) == 0.33 else '--', zorder=0)
    ax.axvline(0, color=SUB, lw=0.8)
    ax.hlines(y, t['cliffs_delta_시간맞춤_CN7'], t['cliffs_delta_시간맞춤_RG3'], color='#d6d5cc', lw=2)
    ax.scatter(t['cliffs_delta_시간맞춤_CN7'], y, s=50, color=TYPE_COL['가스'], edgecolor='white', lw=1.5,
               zorder=3, label='CN7 R1 가스')
    ax.scatter(t['cliffs_delta_시간맞춤_RG3'], y, s=50, color=ACCENT, edgecolor='white', lw=1.5, zorder=3,
               label='RG3 가스 (Run 안 표준화 후 합침)')
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}   [RG3 같은방향 {r}]" for n, r in zip(t['이름'], t['같은방향_Run_시간맞춤'])],
                       fontsize=9)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Cliff's δ, 불량 앞뒤 ±10샷 정상 대비  (+ = 가스 쪽이 큼)")
    ax.set_title('CN7 R1 가스 가설이 RG3에서도 재현되나', loc='left')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()


def plot_per_run_heat(t, title='', save=None, top=15):
    """Run별 δ 히트맵 → 신호가 한 Run 때문인지, 여러 Run에서 반복되는지"""
    set_style()
    rc = [c for c in t.columns if c.startswith('δ_')]
    s = t.head(top)
    M = s[rc].values.astype(float)
    fig, ax = plt.subplots(figsize=(1.0 * len(rc) + 4, 0.36 * len(s) + 1.8))
    im = ax.imshow(M, cmap=DIV, vmin=-1, vmax=1, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f'{M[i, j]:+.2f}', ha='center', va='center', fontsize=7,
                        color='white' if abs(M[i, j]) > 0.6 else INK)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s['이름'], fontsize=8)
    ax.set_xticks(range(len(rc))); ax.set_xticklabels([c.replace('δ_', '').replace('rg3_', '') for c in rc])
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Run별 Cliff's δ")
    ax.set_title(title or 'Run별 효과크기 — 같은 색이 여러 Run에 반복되면 일반적인 신호', loc='left', fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()


def plot_run_timelines(d, cols, group_runs=None, save=None):
    """Run별 타임라인 (가스·미성형 위치 + 변수 흐름). Run마다 한 장씩 열로"""
    set_style()
    runs = group_runs or [r for r in d['run_id'].unique() if d.loc[d['run_id'] == r, 'PassOrFail'].sum()]
    fig, axes = plt.subplots(len(cols), len(runs), figsize=(4.2 * len(runs), 1.7 * len(cols)), sharey='row',
                             squeeze=False)
    for j, r in enumerate(runs):
        g = d[d['run_id'] == r]
        for i, c in enumerate(cols):
            ax = axes[i, j]
            ax.plot(g['shot_seq_in_run'], g[c], color=MUTED, lw=0.9)
            ax.plot(g['shot_seq_in_run'], g[c].rolling(9, center=True, min_periods=3).median(), color=SUB, lw=1.4)
            for t in TYPE_ORDER:
                h = g[g['Reason'] == t]
                ax.scatter(h['shot_seq_in_run'], h[c], s=34, color=TYPE_COL[t], edgecolor='white', lw=1.1,
                           zorder=3, label=t)
            if j == 0:
                ax.set_ylabel(short(c), rotation=0, ha='right', va='center', fontsize=8)
            if i == 0:
                ax.set_title(r, fontsize=9)
    h, l = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc='upper right', ncol=3, fontsize=8)
    fig.suptitle('Run별 타임라인 (x = Run 안 샷 순번, 진한 선 = 9샷 이동 중앙값)', x=0.01, ha='left', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save:
        fig.savefig(save)
    plt.show()


def plot_run_level(t, xcol, save=None):
    set_style()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(t[xcol], t['불량률'] * 100, s=60, color=ACCENT, edgecolor='white', lw=1.5, zorder=3)
    for r, row in t.iterrows():
        ax.annotate(r.split('_')[-1], (row[xcol], row['불량률'] * 100), xytext=(5, 3), textcoords='offset points',
                    fontsize=8, color=SUB)
    ax.set_xlabel(f'Run 중앙값: {xcol}'); ax.set_ylabel('Run 불량률 (%)')
    ax.set_title(f'Run 상태 vs 불량률 (Run {len(t)}개 — 경향만)', loc='left', fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()


def plot_top2(d, Z, t, save=None):
    """단변량 상위 변수 2개로 본 정상 vs 불량 유형 (Run 안 표준화 값)"""
    set_style()
    a, b = t['변수'].iloc[0], t['변수'].iloc[1]
    rng = np.random.default_rng(0)
    jit = lambda n: rng.uniform(-0.12, 0.12, n)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    n = d['PassOrFail'] == 0
    ax.scatter(Z.loc[n, a] + jit(n.sum()), Z.loc[n, b] + jit(n.sum()), s=8, color=NORMAL, alpha=0.5, label='정상')
    for tp in TYPE_ORDER:
        m = d['Reason'] == tp
        if m.any():
            ax.scatter(Z.loc[m, a], Z.loc[m, b], s=48, color=TYPE_COL[tp], edgecolor='white', lw=1.3, zorder=3,
                       label=tp)
    ax.set_xlabel(f'{short(a)} (Run 안 z)'); ax.set_ylabel(f'{short(b)} (Run 안 z)')
    ax.set_title('상위 변수 2개 — 불량이 정상 구름 밖에 있나', loc='left', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()


def plot_separability(summ, oof, null, save=None):
    """왼쪽: LORO 점수 분포(정상 vs 유형), 오른쪽: 라벨 섞은 귀무 AUC 분포 vs 실제"""
    set_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    rng = np.random.default_rng(0)
    cats = ['정상'] + [t for t in TYPE_ORDER if (oof['Reason'] == t).any()]
    for i, t in enumerate(cats):
        v = oof.loc[(oof['y'] == 0) if t == '정상' else (oof['Reason'] == t), 'score']
        a1.scatter(i + rng.uniform(-0.25, 0.25, len(v)), v, s=8 if t == '정상' else 40, color=TYPE_COL[t],
                   alpha=0.5 if t == '정상' else 1, edgecolor='none' if t == '정상' else 'white', lw=1.2)
        a1.boxplot(v, positions=[i], widths=0.6, showfliers=False, medianprops=dict(color=INK),
                   boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    a1.set_xticks(range(len(cats))); a1.set_xticklabels(cats)
    a1.set_ylabel('LORO 예측 확률'); a1.set_title(f"LORO 점수 — AUC {summ['LORO AUC']:.3f}", loc='left',
                                                fontsize=10)
    a2.hist(null, bins=30, color=NORMAL, edgecolor='white')
    a2.axvline(summ['LORO AUC'], color=TYPE_COL['가스'], lw=2, label=f"실제 {summ['LORO AUC']:.3f}")
    a2.axvline(summ['귀무 95%'], color=INK, ls='--', lw=1, label=f"귀무 95% {summ['귀무 95%']:.3f}")
    a2.set_xlabel('라벨을 Run 안에서 섞었을 때 LORO AUC'); a2.legend(fontsize=8)
    a2.set_title(f"우연으로 이 정도가 나올 확률 p = {summ['p(우연)']:.3f}", loc='left', fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()
