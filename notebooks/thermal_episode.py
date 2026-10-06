"""
thermal_episode.py — '온도 교란 구간' 규칙 점검: ① 오경보가 얼마나 자주 나는가  ② RG3 가스에서도 재현되는가

규칙 (26번 R1 결과에서 방향만 가져오고, 임계값은 라벨 없이 정상 기준(unlabeled)에서 정함 — 사전 고정)
  샷 성분 c_t = max(호퍼온도 추세 z, 0) + max(−금형온도 편차, 0) + 0.5 × max(호퍼온도 편차, 0)
      호퍼온도 추세 z = (hopper_slope20 − 정상 중앙값) / 정상 IQR
      금형온도 편차·호퍼온도 편차 = 최종 모델의 dev_* (같은 Run 직전 20샷 대비 robust z)
  구간 점수 E_t = 같은 Run의 최근 5샷 c 평균 (과거 샷만 → 실시간 계산 가능)
  경보: E_t ≥ 정상 기준 E의 q 분위수 (q = 0.99 / 0.995 → 정상 생산에서 1% / 0.5%만 울리도록)
  구간(에피소드): 같은 Run에서 연속으로 경보가 울린 샷 묶음

⚠ 규칙의 '방향'은 R1 가스 에피소드 하나를 보고 정했음 → CN7 R1 성능은 사후 확인일 뿐, 독립 검증은 ①(오경보)과 ②(RG3)
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

from final_model import (FinalModel, INITIAL, TYPE_GROUPS, load_labeled_shots, build_reference, make_features,
                         recall_at_k_by_run)
from r1_inside import _cliff, _bh
from alarm_explain import LABEL

W = 5
QS = (0.99, 0.995)
T_VARS = ['hopper_slope20', 'dev_Hopper_Temperature', 'dev_Mold_Temperature_3', 'hopper_rise', 'mold_rise',
          'mold_slope20']


# ====================================================================== 점수
def _components(f, med, iqr):
    zs = ((f['hopper_slope20'] - med) / iqr).fillna(0)
    c = (zs.clip(0, 10) + (-f['dev_Mold_Temperature_3'].fillna(0)).clip(0, 10)
         + 0.5 * f['dev_Hopper_Temperature'].fillna(0).clip(0, 20))
    return c


def _episode_score(f, med, iqr):
    f = f.sort_values(['run_id', 'TimeStamp']).copy()
    f['c'] = _components(f, med, iqr).values
    f['E'] = f.groupby('run_id', sort=False)['c'].transform(lambda s: s.rolling(W, min_periods=1).mean())
    f['seq'] = f.groupby('run_id', sort=False).cumcount() + 1
    return f


def prepare(p, base_dir='.', model_dir='./outputs/final_model'):
    model = FinalModel.load(f'{model_dir}/{p}_final_model.pkl')
    shots = load_labeled_shots(p, base_dir)
    ref_shots, _ = build_reference(p, shots, base_dir)
    ref = make_features(ref_shots, model.steps, model.fr_step)
    lab = make_features(shots, model.steps, model.fr_step)
    hs = ref['hopper_slope20'].dropna()
    med = hs.median()
    iqr = max(hs.quantile(0.75) - hs.quantile(0.25), 1e-6)
    ref = _episode_score(ref, med, iqr)
    lab = _episode_score(lab, med, iqr)
    lab['in_eval'] = (lab['split'] != '제외') & (lab['Reason'] != INITIAL)
    lab['group'] = np.select([lab['Reason'] == '가스', lab['Reason'].isin(TYPE_GROUPS['미성형']),
                              lab['PassOrFail'] == 0], ['가스', '미성형계열', '정상'], '기타')
    th = {q: float(ref['E'].quantile(q)) for q in QS}
    return lab, ref, th


def episodes(f, alarm_col):
    """연속 경보 묶음 → 에피소드 표"""
    rows = []
    for r, g in f.groupby('run_id', sort=False):
        a = g[alarm_col].values
        seq = g['seq'].values
        start = None
        for i in range(len(a) + 1):
            on = i < len(a) and a[i]
            if on and start is None:
                start = i
            if not on and start is not None:
                seg = g.iloc[start:i]
                after = g.iloc[i:i + 10]                                    # 경보 끝난 뒤 10샷
                rows.append({'run_id': r, '시작 seq': int(seq[start]), '끝 seq': int(seq[i - 1]),
                             '길이(샷)': i - start,
                             '구간 내 불량': int(seg.get('PassOrFail', pd.Series(0)).fillna(0).sum()),
                             '구간 내 가스': int((seg.get('group', pd.Series('')) == '가스').sum()),
                             '직후 10샷 불량': int(after.get('PassOrFail', pd.Series(0)).fillna(0).sum())})
                start = None
    return pd.DataFrame(rows)


# ====================================================================== ① 오경보 빈도
def false_alarm_table(p, lab, ref, th):
    rows = []
    for q, t in th.items():
        ref_al = ref['E'] >= t
        ep_ref = episodes(ref.assign(al=ref_al), 'al')
        e = lab[lab['in_eval']]
        lab_al = lab['E'] >= t
        ep_lab = episodes(lab.assign(al=lab_al), 'al')
        nrm = e['group'] == '정상'
        rows.append({'제품': p, 'q': q, '임계값 E': t,
                     '정상기준 경보 비율': ref_al.mean(),
                     '정상기준 에피소드 수': len(ep_ref),
                     '정상기준 1,000샷당 에피소드': 1000 * len(ep_ref) / len(ref),
                     'labeled 정상 샷 경보 비율': (e.loc[nrm, 'E'] >= t).mean(),
                     'labeled 에피소드 수': len(ep_lab),
                     '불량 포함 에피소드': int((ep_lab['구간 내 불량'] > 0).sum()) if len(ep_lab) else 0,
                     '불량 없는 에피소드(오경보)': int((ep_lab['구간 내 불량'] == 0).sum()) if len(ep_lab) else 0,
                     '가스 경보/전체 가스': f"{int((e.loc[e['group'] == '가스', 'E'] >= t).sum())}/{int((e['group'] == '가스').sum())}",
                     '미성형계열 경보/전체': f"{int((e.loc[e['group'] == '미성형계열', 'E'] >= t).sum())}/{int((e['group'] == '미성형계열').sum())}"})
    return pd.DataFrame(rows)


def run_breakdown(p, lab, th, q=0.99):
    t = th[q]
    e = lab[lab['in_eval']]
    rows = []
    for r, g in e.groupby('run_id'):
        ep = episodes(lab[lab['run_id'] == r].assign(al=lambda d: d['E'] >= t), 'al')
        rows.append({'제품': p, 'Run': r, '샷': len(g), '가스': int((g['group'] == '가스').sum()),
                     '미성형계열': int((g['group'] == '미성형계열').sum()), '경보 샷': int((g['E'] >= t).sum()),
                     '에피소드': len(ep), '가스 경보': int(((g['E'] >= t) & (g['group'] == '가스')).sum()),
                     '정상 샷 경보 비율': ((g['E'] >= t) & (g['group'] == '정상')).sum() / max(1, (g['group'] == '정상').sum())})
    return pd.DataFrame(rows)


def ranking(p, lab):
    """구간 점수 E로 순위를 매겼을 때 가스·미성형 AP (평가 범위)"""
    e = lab[lab['in_eval']].reset_index(drop=True)
    y = e['PassOrFail'].values.astype(int)
    s = e['E'].values + np.random.default_rng(0).random(len(e)) * 1e-9
    g = pd.DataFrame({'run_id': e['run_id'], 'score': s})
    r = {'제품': p, '불량률': y.mean(), 'AP(전체 불량)': average_precision_score(y, s)}
    for name, gname in [('가스', '가스'), ('미성형', '미성형계열')]:
        pm = (e['group'] == gname).values
        if pm.sum():
            ev = pm | (y == 0)
            r[f'AP_{name}'] = average_precision_score(pm[ev].astype(int), s[ev])
            r[f'{name} 기준선(비율)'] = pm[ev].mean()
            r[f'Run별 상위10%_{name}'] = '%d/%d' % recall_at_k_by_run(g, 0.10, pd.Series(pm))
    return r


def lead_time(lab, th, q=0.99, run=None):
    """첫 미성형 샷 이전에 경보가 처음 울린 시점 (CN7 R1 사후 확인용)"""
    t = th[q]
    e = lab if run is None else lab[lab['run_id'] == run]
    first_short = e.loc[e['group'] == '미성형계열', 'seq'].min()
    if pd.isna(first_short):
        return None
    before = e[(e['seq'] < first_short) & (e['seq'] >= first_short - 30) & (e['E'] >= t)]
    return {'첫 미성형 seq': int(first_short),
            '그 전 30샷 중 첫 경보 seq': int(before['seq'].min()) if len(before) else None,
            '선행 샷 수': int(first_short - before['seq'].min()) if len(before) else None}


# ====================================================================== ② 시간 맞춘 비교 (모든 Run 합침)
def matched_compare(lab, group='가스', vars_=T_VARS + ['E'], win=10):
    e = lab[lab['in_eval']]
    A, B = {v: [] for v in vars_}, {v: [] for v in vars_}
    nd = 0
    for r, g in e.groupby('run_id'):
        d = g[g['group'] == group]
        if d.empty:
            continue
        nd += len(d)
        near = set()
        for s in d['seq']:
            near |= set(range(s - win, s + win + 1))
        nm = g[(g['group'] == '정상') & g['seq'].isin(near)]
        for v in vars_:
            A[v] += d[v].dropna().tolist()
            B[v] += nm[v].dropna().tolist()
    rows = []
    for v in vars_:
        a, b = np.array(A[v]), np.array(B[v])
        pv = mannwhitneyu(a, b).pvalue if len(a) and len(b) else np.nan
        rows.append({'불량': group, '변수': LABEL.get(v, '구간 점수 E' if v == 'E' else v), 'n불량': len(a),
                     'n주변정상': len(b), '불량 중앙값': np.median(a) if len(a) else np.nan,
                     '주변정상 중앙값': np.median(b) if len(b) else np.nan, "Cliff's δ": _cliff(a, b), 'p': pv})
    t = pd.DataFrame(rows)
    t['q(BH)'] = _bh(t['p'].values)
    return t


def plot_runs(p, lab, th, q=0.99, save=None):
    """Run별 구간 점수 E 흐름 (점선 = 임계값, 주황 = 가스, 빨강 = 미성형계열)"""
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    runs = sorted(lab['run_id'].unique())
    fig, axes = plt.subplots(len(runs), 1, figsize=(12, 1.6 * len(runs)), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, runs):
        g = lab[lab['run_id'] == r]
        ax.plot(g['seq'], g['E'], color='#9db9d5', lw=1)
        ax.axhline(th[q], color='#555555', ls='--', lw=0.8)
        for gname, col in [('가스', '#e67e22'), ('미성형계열', '#c0392b')]:
            m = g['group'] == gname
            ax.scatter(g.loc[m, 'seq'], g.loc[m, 'E'], color=col, s=22, zorder=3, edgecolor='k', lw=0.4,
                       label=gname)
        ax.set_ylabel(r, fontsize=8)
    axes[0].legend(fontsize=7, loc='upper right')
    axes[0].set_title(f'{p.upper()} — 온도 교란 구간 점수 E (점선 = 정상 기준 q={q} 임계값)', loc='left', fontsize=10)
    axes[-1].set_xlabel('Run 내 순번')
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
