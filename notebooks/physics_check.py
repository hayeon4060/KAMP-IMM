"""
physics_check.py — 알려진 사출 불량 메커니즘을 데이터로 검증 (가설·방향을 결과 보기 전에 고정)

0. 물리 일관성: 흐름저항(사출압/속도, Hagen-Poiseuille상 점도에 비례)이
   온도가 오르면 내려가나 (Arrhenius·Cross-WLF: T↑ → η↓)
   → Run 안 편차끼리 회귀 (Run 고정효과), 라벨 미사용. 잔차 = '온도로 설명 안 되는 점도' (점도잔차)
1. 메커니즘 지표 (모두 라벨 미사용, 공정값으로만 계산) → Run 안 표준화 후 불량 vs 정상 비교
2. HYP 표에 적힌 예상 방향으로 단측 검정 + Run별 방향 일치 → 판정(지지 / 약한 지지 / 근거 없음 / 반대)

지표 정의 (+ = 해당 메커니즘 쪽)
  재료부족   = (−충전시간 + 사출속도 − 사출압) / 3   (Run 안 z)  앞에 쌓인 수지가 적으면 저항↓ → 빨리·짧게·낮은 압력
  계량지연   = 계량시간 (Run 안 z)                               재료 공급 불안정
  쿠션부족   = −쿠션 위치 (Run 안 z)                            쿠션이 줄면 보압 여유 없음
  압력한계   = 사출압 / Run 99% 분위                             설정 압력 상한 근처
  점도잔차   = 0번 회귀 잔차                                     온도로 설명 안 되는 점도↑
  금형냉각   = −금형온도3 (Run 안 z)                            조기 고화
  디젤(속도) = 사출속도 (Run 안 z)                               빠른 사출 → 공기 단열압축
  체류시간   = 사이클시간 (Run 안 z)                             배럴 안에 머무는 시간 대리 (길수록 분해↑)
  배럴최고온도 = 배럴1~6 중 최고온도 (Run 안 z)                   가공온도 상한 근처일수록 분해↑
  ※ 예전 '열분해 = ln(사이클) − Ea/RT'는 Run 안 변동에서 온도 항이 수십 배 커서 Ea≥약 20이면 사실상
    배럴온도와 같고 Ea=0이면 사이클과 같음 → Ea를 가정하지 않고 두 지표로 나눠서 봄
  호퍼(탐색) = 호퍼온도 (Run 안 z)                               방향 가설 없음 (CN7 R1에서 본 신호 확인용)

한계: 용융수지 온도·금형 내 압력·벤트 상태·수분·설정값 없음 → 배럴온도·최대값 기반 대리 지표
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from features import raw
from run_compare import add_run_z, _matched, GROUPS
from plot_style import set_style, INK, SUB, MUTED, NORMAL, TYPE_COL, TYPE_ORDER, DIV, ACCENT

BARRELS = [raw(f'Barrel_Temperature_{i}') for i in range(1, 7)]

# ---------------------------------------------------------------- 사전 등록 가설 (결과 보기 전에 고정)
#  (메커니즘, 지표, 불량군, 예상 부호, 설명)   부호 +1: 불량 쪽이 큼 / −1: 작음 / 0: 탐색(양측)
HYP = [
    ('미성형① 재료부족', '재료부족', '미성형계열', +1, '충전↓·속도↑·압력↓ (수지 부족 → 저항↓)'),
    ('미성형① 재료부족', '계량지연', '미성형계열', +1, '계량시간↑ (공급 불안정)'),
    ('미성형① 재료부족', '쿠션부족', '미성형계열', +1, '쿠션↓'),
    ('미성형② 압력한계', '압력한계', '미성형계열', +1, '사출압이 Run 상한 근처'),
    ('미성형② 압력한계', '점도잔차', '미성형계열', +1, '온도로 설명 안 되는 점도↑'),
    ('미성형③ 조기고화', '금형냉각', '미성형계열', +1, '금형온도↓'),
    ('가스① 디젤효과', '디젤(속도)', '가스', +1, '사출속도↑ → 공기 단열압축'),
    ('가스② 열분해', '체류시간', '가스', +1, '사이클↑ → 배럴 체류↑'),
    ('가스② 열분해', '배럴최고온도', '가스', +1, '배럴 최고온도↑'),
    ('가스② 열분해', '점도잔차', '가스', -1, '분해 → 분자량↓ → 점도↓'),
    ('탐색', '호퍼(탐색)', '가스', 0, 'CN7 R1 호퍼 신호 재확인 (방향 가설 없음)'),
]


# ====================================================================== 0. 물리 일관성
def physics_consistency(d, temps=None):
    """
    Run 안 편차끼리: log(흐름저항) ~ 배럴1~6 + 금형온도3 + 호퍼온도 (Run 고정효과 = Run 평균 빼기)
    예상: 배럴·금형 계수 < 0 (온도↑ → 점도↓). 호퍼는 수지가 녹기 전 구간이라 방향 가설 없음
    반환: (계수표, 점도잔차 Series, within R²)
    """
    temps = temps or BARRELS + [raw('Mold_Temperature_3'), raw('Hopper_Temperature')]
    temps = [c for c in temps if c in d.columns and d[c].nunique() > 1]
    y = np.log(d['flow_res'])
    X = d[temps].astype(float)
    g = d['run_id']
    yd = y - y.groupby(g).transform('mean')
    Xd = X - X.groupby(g).transform('mean')
    ok = yd.notna() & Xd.notna().all(axis=1) & np.isfinite(yd)
    A, b = Xd[ok].values, yd[ok].values
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = b - A @ beta
    dof = max(len(b) - A.shape[1] - g[ok].nunique(), 1)
    se = np.sqrt(np.diag((res @ res / dof) * np.linalg.pinv(A.T @ A)))
    exp = {c: (0 if 'Hopper' in c else -1) for c in temps}
    t = pd.DataFrame({'변수': temps, '계수(%/℃)': beta * 100, 'SE': se * 100, 't': beta / se,
                      '예상 부호': [exp[c] for c in temps]})
    t['일치'] = np.where(t['예상 부호'] == 0, '탐색',
                       np.where(np.sign(t['계수(%/℃)']) == t['예상 부호'],
                                np.where(t['t'].abs() >= 2, '일치(유의)', '일치(약함)'),
                                np.where(t['t'].abs() >= 2, '반대(유의)', '반대(약함)')))
    r2 = 1 - (res @ res) / (b @ b) if (b @ b) > 0 else np.nan
    resid = pd.Series(np.nan, index=d.index)
    resid[ok] = res
    return t, resid, r2


# ====================================================================== 1. 지표
def build_indices(d, visc_resid):
    """라벨 미사용. 반환: 지표 원값 DataFrame (Run 안 표준화 전)"""
    Z = add_run_z(d, [raw(v) for v in ['Filling_Time', 'Max_Injection_Speed', 'Max_Injection_Pressure',
                                         'Plasticizing_Time', 'Cushion_Position', 'Mold_Temperature_3',
                                         'Hopper_Temperature', 'Cycle_Time']])
    I = pd.DataFrame(index=d.index)
    I['재료부족'] = (-Z[raw('Filling_Time')] + Z[raw('Max_Injection_Speed')] - Z[raw('Max_Injection_Pressure')]) / 3
    I['계량지연'] = Z[raw('Plasticizing_Time')]
    I['쿠션부족'] = -Z[raw('Cushion_Position')]
    p = d[raw('Max_Injection_Pressure')]
    I['압력한계'] = p / p.groupby(d['run_id']).transform(lambda s: s.quantile(0.99))
    I['점도잔차'] = visc_resid
    I['금형냉각'] = -Z[raw('Mold_Temperature_3')]
    I['디젤(속도)'] = Z[raw('Max_Injection_Speed')]
    I['체류시간'] = Z[raw('Cycle_Time')]
    I['배럴최고온도'] = d[[c for c in BARRELS if c in d.columns]].max(axis=1)
    I['호퍼(탐색)'] = Z[raw('Hopper_Temperature')]
    return I


def standardize_indices(d, I):
    """지표를 Run 안 표준화 (불량 vs 정상 비교용)"""
    tmp = I.copy()
    tmp['run_id'] = d['run_id'].values
    return add_run_z(tmp, list(I.columns))


# ====================================================================== 2. 검정
def _verdict(delta, p, sign):
    if sign == 0 or not np.isfinite(delta):
        return '탐색' if sign == 0 else '-'
    s = delta * sign                         # 예상 방향으로 잰 효과크기
    if s >= 0.33 and p < 0.05:
        return '지지'
    if s >= 0.147:
        return '약한 지지'
    if s <= -0.147:
        return '반대'
    return '근거 없음'


def test_hypotheses(d, IZ, product, hyp=HYP, window=10):
    rows = []
    run = d['run_id'].values
    for mech, idx, grp, sign, desc in hyp:
        pos = d['Reason'].isin(GROUPS[grp]).values
        if pos.sum() == 0 or idx not in IZ:
            continue
        x = IZ[idx].values.astype(float)
        r = {'제품': product, '메커니즘': mech, '지표': idx, '불량군': grp, '예상': {1: '+', -1: '−', 0: '?'}[sign],
             '설명': desc, 'n불량': int(pos.sum())}
        for tag, neg in [('전체', d['PassOrFail'].values == 0), (f'±{window}샷', _matched(d, pos, window))]:
            a, b = x[pos & np.isfinite(x)], x[neg & np.isfinite(x)]
            if len(a) == 0 or len(b) == 0 or np.ptp(np.r_[a, b]) == 0:
                r[f'δ_{tag}'], r[f'p_{tag}'] = np.nan, np.nan
                continue
            U, _ = mannwhitneyu(a, b, alternative='two-sided')
            dl = 2 * U / (len(a) * len(b)) - 1
            alt = 'greater' if sign > 0 else ('less' if sign < 0 else 'two-sided')
            r[f'δ_{tag}'], r[f'p_{tag}'] = dl, mannwhitneyu(a, b, alternative=alt).pvalue
            r[f'판정_{tag}'] = _verdict(dl, r[f'p_{tag}'], sign)
        # Run별 방향 일치 (불량 1개 이상인 Run)
        agree, tot = 0, 0
        for rr in np.unique(run[pos]):
            m = run == rr
            a = x[pos & m & np.isfinite(x)]
            b = x[(d['PassOrFail'].values == 0) & m & np.isfinite(x)]
            if len(a) and len(b):
                U, _ = mannwhitneyu(a, b, alternative='two-sided')
                dl = 2 * U / (len(a) * len(b)) - 1
                tot += 1
                agree += int(np.sign(dl) == (sign if sign else np.sign(r.get('δ_전체', 0))))
        r['예상방향_Run'] = f'{agree}/{tot}'
        rows.append(r)
    return pd.DataFrame(rows)


# ====================================================================== 그래프
def plot_consistency(tabs, save=None):
    """제품별 온도 계수 (±2SE). 배럴·금형이 0보다 왼쪽(−)이면 물리와 일치"""
    set_style()
    fig, axes = plt.subplots(1, len(tabs), figsize=(6 * len(tabs), 4.2), sharex=False)
    for ax, (p, (t, r2)) in zip(np.atleast_1d(axes), tabs.items()):
        s = t.iloc[::-1]
        y = np.arange(len(s))
        col = np.where(s['예상 부호'] == 0, MUTED, np.where(s['일치'].str.startswith('일치'), ACCENT, TYPE_COL['가스']))
        ax.hlines(y, s['계수(%/℃)'] - 2 * s['SE'], s['계수(%/℃)'] + 2 * s['SE'], color='#d6d5cc', lw=3)
        ax.scatter(s['계수(%/℃)'], y, s=50, color=col, edgecolor='white', lw=1.5, zorder=3)
        ax.axvline(0, color=SUB, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([v.replace('_raw', '').replace('Barrel_Temperature_', '배럴')
                            .replace('Mold_Temperature_3', '금형온도3').replace('Hopper_Temperature', '호퍼')
                            for v in s['변수']], fontsize=9)
        ax.set_xlabel('흐름저항 변화 (%/℃), 막대 = ±2SE')
        ax.set_title(f'{p.upper()} — within R² {r2:.3f}  (파랑 = 물리 예상과 일치, 빨강 = 반대)', loc='left',
                     fontsize=10)
    fig.suptitle('물리 일관성: 온도↑ → 흐름저항(점도)↓ 인가', x=0.01, ha='left', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save:
        fig.savefig(save)
    plt.show()


def plot_verdict(res, tag='±10샷', save=None):
    """행 = 가설, 열 = 제품. 색 = 예상 방향으로 잰 δ (+ 파랑계열 아님: 빨강 = 예상대로, 파랑 = 반대)"""
    set_style()
    res = res.copy()
    res['가설'] = res['메커니즘'] + ' · ' + res['지표']
    sign = res['예상'].map({'+': 1, '−': -1, '?': 1})
    res['s'] = res[f'δ_{tag}'] * sign
    rows = list(dict.fromkeys(res['가설']))
    cols = list(dict.fromkeys(res['제품']))
    M = np.full((len(rows), len(cols)), np.nan)
    L = [['' for _ in cols] for _ in rows]
    for _, r in res.iterrows():
        i, j = rows.index(r['가설']), cols.index(r['제품'])
        M[i, j] = r['s']
        L[i][j] = f"{r[f'δ_{tag}']:+.2f}\n{r.get(f'판정_{tag}', '')}\nRun {r['예상방향_Run']}"
    fig, ax = plt.subplots(figsize=(2.3 * len(cols) + 4.5, 0.62 * len(rows) + 1.8))
    im = ax.imshow(M, cmap=DIV, vmin=-1, vmax=1, aspect='auto')
    for i in range(len(rows)):
        for j in range(len(cols)):
            if L[i][j]:
                ax.text(j, i, L[i][j], ha='center', va='center', fontsize=7,
                        color='white' if np.isfinite(M[i, j]) and abs(M[i, j]) > 0.6 else INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=9)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.upper() for c in cols])
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label=f"예상 방향으로 잰 Cliff's δ ({tag} 비교군)\n빨강 = 예상대로, 파랑 = 반대")
    ax.set_title('알려진 불량 메커니즘 검증 — 칸 = δ / 판정 / 예상 방향 일치 Run 수', loc='left', fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()


def plot_index_strips(d, IZ, product, save=None):
    """지표별 Run 안 z 분포: 정상 / 가스 / 미성형계열. 제목에 예상 방향"""
    set_style()
    idxs = list(IZ.columns)
    expect = {}
    for mech, idx, grp, sign, _ in HYP:
        expect.setdefault(idx, []).append(f"{grp} {'+' if sign > 0 else ('−' if sign < 0 else '?')}")
    cats = [('정상', d['PassOrFail'] == 0, NORMAL), ('가스', d['Reason'] == '가스', TYPE_COL['가스']),
            ('미성형계열', d['Reason'].isin(GROUPS['미성형계열']), TYPE_COL['미성형'])]
    ncol = 3
    nrow = int(np.ceil(len(idxs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(0)
    for ax, c in zip(axes, idxs):
        for i, (name, m, col) in enumerate(cats):
            v = IZ.loc[m.values, c].dropna().clip(-10, 10)
            if len(v) == 0:
                continue
            ax.boxplot(v, positions=[i], widths=0.55, showfliers=False, medianprops=dict(color=INK),
                       boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=6 if i == 0 else 30, color=col,
                       alpha=0.4 if i == 0 else 1, edgecolor='none' if i == 0 else 'white', lw=1, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xticks(range(3)); ax.set_xticklabels([c_[0] for c_ in cats], fontsize=8)
        ax.set_title(f"{c}  (예상: {', '.join(expect.get(c, ['-']))})", fontsize=9)
    for ax in axes[len(idxs):]:
        ax.axis('off')
    fig.suptitle(f'{product.upper()} — 메커니즘 지표 분포 (Run 안 z, ±10에서 자름)', x=0.01, ha='left', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        fig.savefig(save)
    plt.show()
