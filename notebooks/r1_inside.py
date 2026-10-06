"""
r1_inside.py — CN7 R1 '안에서' 정상 샷과 불량 샷은 무엇이 다른가 (Run 간 차이를 뺀 비교)

비교 대상 (평가 범위: 제외 구간·초기허용불량 제외)
  가스 / 미성형계열  vs  ① R1 정상 전체   ② 시간 맞춘 정상 (각 불량 샷 앞뒤 ±10샷 안의 정상)
  → ②는 'R1 안에서도 시간이 흐르며 생기는 변화'까지 맞춘 비교라 가장 엄격함

변수: 원값 21개 + 최종 모델 파생변수 17개 + 최종 모델 점수
지표: 중앙값 차이, Cliff's δ(−1~1, |δ|≥0.33 중간·≥0.47 큼), Mann-Whitney p, BH-FDR q (변수 수만큼 검정해서 보정)

추가
  - 불량 샷의 Run 내 위치(몇 번째 샷인지)와 군집 여부
  - 불량 샷 하나하나의 최종 모델 점수·이상 원인 후보·상위 기여 변수 (alarm_explain 재사용)
  - R1 타임라인 그래프 (차이가 큰 변수, 불량 위치 표시)
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from final_model import (raw, ALL_RAW, FEATURES, INITIAL, TYPE_GROUPS, load_labeled_shots, FinalModel,
                         make_features)
from alarm_explain import contributions, cause_hint, LABEL

RAW_VARS = [raw(v) for v in ALL_RAW]
WIN = 10


def _cliff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(((a[:, None] > b[None, :]).mean() - (a[:, None] < b[None, :]).mean()))


def _bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    o = np.argsort(pv)
    r = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(len(pv)); out[o] = np.minimum(r, 1)
    q[ok] = out
    return q


def load_run(p='cn7', run=None, base_dir='.', model_dir='./outputs/final_model'):
    """반환: (R1 샷 DataFrame — 원값·파생변수·점수·그룹, 모델, Run 이름)"""
    model = FinalModel.load(f'{model_dir}/{p}_final_model.pkl')
    shots = load_labeled_shots(p, base_dir)
    f = make_features(shots, model.steps, model.fr_step)
    sc = pd.read_csv(f'{model_dir}/{p}_scores.csv', parse_dates=['TimeStamp'], usecols=['TimeStamp', 'score', 'pct'])
    f = f.merge(sc, on='TimeStamp', how='left')
    if run is None:
        e = f[(f['split'] != '제외') & (f['Reason'] != INITIAL)]
        run = e.groupby('run_id')['PassOrFail'].sum().idxmax()
    d = f[f['run_id'] == run].sort_values('TimeStamp').reset_index(drop=True)
    d['seq'] = np.arange(1, len(d) + 1)                              # Run 안 순번 (전체 시간 흐름 기준)
    d['group'] = np.select([d['Reason'].isin(['가스']), d['Reason'].isin(TYPE_GROUPS['미성형']),
                            d['Reason'] == INITIAL, d['PassOrFail'] == 0], ['가스', '미성형계열', '초기허용', '정상'], '기타')
    d['평가범위'] = d['split'] != '제외'
    return d, model, run


def compare(d, group, vars_=None, win=WIN):
    """group vs (① 정상 전체, ② 시간 맞춘 정상) 변수별 차이"""
    vars_ = vars_ or (RAW_VARS + FEATURES + ['score'])
    e = d[d['평가범위']]
    g = e[e['group'] == group]
    nm = e[e['group'] == '정상']
    near = set()
    for s in g['seq']:
        near |= set(range(s - win, s + win + 1))
    nm_t = nm[nm['seq'].isin(near)]
    rows = []
    for v in vars_:
        if v not in e.columns:
            continue
        for lab, ref in [('① R1 정상 전체', nm), (f'② 시간 맞춘 정상(±{win}샷)', nm_t)]:
            a, b = g[v].dropna(), ref[v].dropna()
            pv = mannwhitneyu(a, b).pvalue if len(a) and len(b) and (a.nunique() > 1 or b.nunique() > 1) else np.nan
            rows.append({'불량': group, '비교': lab, '변수': v, '이름': LABEL.get(v, v.replace('_raw', '')),
                         'n불량': len(a), 'n정상': len(b), '불량 중앙값': a.median(), '정상 중앙값': b.median(),
                         '차이': a.median() - b.median(), "Cliff's δ": _cliff(a, b), 'p': pv})
    t = pd.DataFrame(rows)
    t['q(BH)'] = t.groupby('비교')['p'].transform(lambda s: pd.Series(_bh(s.values), index=s.index))
    t['효과'] = pd.cut(t["Cliff's δ"].abs(), [-1, 0.147, 0.33, 0.474, 1.01], labels=['무시', '작음', '중간', '큼'])
    t['|δ|'] = t["Cliff's δ"].abs()
    return t.sort_values(['비교', '|δ|'], ascending=[True, False]).drop(columns='|δ|')


def positions(d):
    """불량 샷의 Run 내 위치 + 군집 검정 (같은 수의 샷을 무작위로 뽑았을 때 간격이 이만큼 좁을 확률)"""
    rows = []
    rng = np.random.default_rng(0)
    e = d[d['평가범위']]
    n = len(d)
    for gname in ['가스', '미성형계열']:
        s = np.sort(e.loc[e['group'] == gname, 'seq'].values)
        if len(s) < 2:
            rows.append({'불량': gname, '샷 수': len(s), '위치(seq)': list(s)})
            continue
        span = s.max() - s.min()
        sim = [np.ptp(rng.choice(np.arange(1, n + 1), len(s), replace=False)) for _ in range(5000)]
        rows.append({'불량': gname, '샷 수': len(s), '위치(seq)': list(s), '범위(첫~끝 간격)': int(span),
                     'Run 길이': n, '군집 p(이만큼 좁게 모일 확률)': float(np.mean(np.array(sim) <= span))})
    return pd.DataFrame(rows)


def defect_cases(d, model, k=3):
    """불량 샷 하나하나: 순번, 점수, 백분위, 이상 원인 후보, 상위 기여 변수"""
    e = d[d['평가범위'] & d['group'].isin(['가스', '미성형계열'])].copy()
    if e.empty:
        return e
    C, d2 = contributions(model, e)
    tops = []
    for j, i in enumerate(e.index):
        top = C.loc[i].sort_values(ascending=False).head(k)
        tops.append(', '.join(f'{LABEL.get(c, c)}({e.loc[i, c]:+.1f}, {100 * v / d2[j]:.0f}%)' for c, v in top.items()))
    e['이상 원인 후보'] = e.apply(cause_hint, axis=1)
    e['상위 기여 변수 (dev, 비율)'] = tops
    return e[['seq', 'TimeStamp', 'group', 'Reason', 'score', 'pct', '이상 원인 후보', '상위 기여 변수 (dev, 비율)']]


def plot_timeline(d, vars_, run, save=None):
    """R1 순번에 따른 변수 흐름 (주황 = 가스, 빨강 = 미성형계열, 회색 점선 = 초기허용)"""
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    fig, axes = plt.subplots(len(vars_), 1, figsize=(12, 1.9 * len(vars_)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, vars_):
        ax.plot(d['seq'], d[v], color='#9db9d5', lw=1)
        for gname, col in [('가스', '#e67e22'), ('미성형계열', '#c0392b')]:
            m = d['group'] == gname
            ax.scatter(d.loc[m, 'seq'], d.loc[m, v], color=col, s=30, zorder=3, edgecolor='k', lw=0.4, label=gname)
        for s in d.loc[d['group'] == '초기허용', 'seq']:
            ax.axvline(s, color='#999999', ls=':', lw=0.8)
        ax.set_ylabel(LABEL.get(v, v.replace('_raw', '')), fontsize=8)
    axes[0].legend(fontsize=7, loc='upper right')
    axes[0].set_title(f'{run} — Run 안 흐름 (x = Run 내 순번)', loc='left', fontsize=10)
    axes[-1].set_xlabel('Run 내 순번 (seq)')
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
