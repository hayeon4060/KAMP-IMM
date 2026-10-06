"""
level_check.py — '정상 대비 금형온도·계량위치 수준'이 불량(특히 CN7 가스)과 관련 있는지 검토 → 최종 모델과 합쳐 다시 평가

기준(라벨 미사용): 최종 모델 학습에 쓴 reference = unlabeled 정상 생산 샷(2020-09~11, 0 모드 제거)
  수준 z = (값 − reference 중앙값) / reference IQR   (Run 간 차이를 그대로 담는 '절대 수준' 지표)
  ※ 최종 모델 변수(dev_*)는 '같은 Run 직전 20샷 대비'라 Run 전체 수준 차이는 의도적으로 지워져 있음

A. 검토
  A-1 Run별 수준: 정상·불량 샷 중앙값, reference 1~99% 범위를 벗어난 샷 비율
  A-2 차이 검정: (a) R1 정상 vs 다른 Run 정상  (b) R1 안에서 가스 vs R1 정상  — Cliff's δ, Mann-Whitney p
      → (a)만 크고 (b)가 작으면 = 'R1이라는 구간'이 다를 뿐, R1 안에서 가스를 가려 주지는 못함

B. 결합 모델 (결과 보기 전에 고정)
  L_level   : 수준 점수 = √(z_금형² + z_계량위치²)  (각 z는 ±20 클리핑), reference 분포 대비 백분위
  M_final   : 최종 모델
  C_mean    : 두 점수의 이상도(−log10(1−pct)) 평균
  C_max     : 두 점수의 이상도 중 큰 값  (둘 중 하나라도 이상하면 경보)
  평가(② 전체·샷·초기허용 제외 + ① 공식 2차·행):
    AP, 유형별 AP, Run별 상위10%(Run 내부 순위 → 'R1 찾기'로는 오르지 않음),
    R1 내부 AP(R1 샷끼리만 순위 → 가스를 R1 정상과 가르는지), 경보 수와 R1 정상 샷 경보 비율
  최종 모델 대비 ΔAP는 paired bootstrap 95% CI
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

from final_model import (raw, INITIAL, TYPE_GROUPS, load_labeled_shots, load_official_rows, build_reference,
                         recall_at_k_by_run, ap_ci)
from compare_final import _paired_delta

LEVEL_VARS = ['Mold_Temperature_3', 'Plasticizing_Position']
NAME = {'Mold_Temperature_3': '금형온도3', 'Plasticizing_Position': '계량위치'}
CLIP = 20
QS = (0.99, 0.999)


# ====================================================================== 데이터
def _cliff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(((a[:, None] > b[None, :]).mean() - (a[:, None] < b[None, :]).mean()))


def prepare(p, base_dir='.', model_dir='./outputs/final_model'):
    shots = load_labeled_shots(p, base_dir)
    ref, _ = build_reference(p, shots, base_dir)
    cols = [raw(v) for v in LEVEL_VARS]
    med = ref[cols].median()
    q75, q25 = ref[cols].quantile(0.75), ref[cols].quantile(0.25)
    iqr = (q75 - q25).where(q75 - q25 > 1e-9, ref[cols].std()).replace(0, 1.0)

    def level(df):
        z = ((df[cols] - med) / iqr).clip(-CLIP, CLIP)
        z.columns = [f'z_{NAME[v]}' for v in LEVEL_VARS]
        return z, np.sqrt((z ** 2).sum(axis=1)).values

    zr, sr = level(ref)
    s_ref = np.sort(sr)
    sc = pd.read_csv(f'{model_dir}/{p}_scores.csv', parse_dates=['TimeStamp'])
    d = sc.merge(shots[['TimeStamp'] + cols], on='TimeStamp', how='left')
    z, s = level(d)
    d = pd.concat([d, z], axis=1)
    d['level_score'] = s
    d['level_pct'] = np.searchsorted(s_ref, s, side='right') / (len(s_ref) + 1)
    band = ref[cols].quantile([0.01, 0.99])
    info = {'reference 샷': len(ref), 'median': med, 'iqr': iqr, 'band': band}
    return d, ref, info


def scope2(d):
    return d[(d['split'] != '제외') & (d['Reason'] != INITIAL)].reset_index(drop=True)


# ====================================================================== A. 검토
def run_table(p, d, info):
    """A-1 Run별 수준 (평가 범위)"""
    d = scope2(d)
    rows = []
    for r, g in d.groupby('run_id'):
        row = {'제품': p, 'Run': r, '샷': len(g), '불량': int(g['PassOrFail'].sum()),
               '가스': int((g['Reason'] == '가스').sum()),
               '미성형계열': int(g['Reason'].isin(TYPE_GROUPS['미성형']).sum())}
        for v in LEVEL_VARS:
            c = raw(v)
            lo, hi = info['band'].loc[0.01, c], info['band'].loc[0.99, c]
            row[f'{NAME[v]} 정상 중앙값'] = g.loc[g['PassOrFail'] == 0, c].median()
            row[f'{NAME[v]} 불량 중앙값'] = g.loc[g['PassOrFail'] == 1, c].median()
            row[f'{NAME[v]} 기준범위 밖 비율'] = ((g[c] < lo) | (g[c] > hi)).mean()
        rows.append(row)
    t = pd.DataFrame(rows)
    ref_row = {'제품': p, 'Run': 'reference(정상 기준)', '샷': info['reference 샷']}
    for v in LEVEL_VARS:
        c = raw(v)
        ref_row[f'{NAME[v]} 정상 중앙값'] = info['median'][c]
        ref_row[f'{NAME[v]} 기준범위 밖 비율'] = 0.02
    return pd.concat([t, pd.DataFrame([ref_row])], ignore_index=True)


def diff_tests(p, d, focus_run=None, defect='가스'):
    """A-2 (a) 집중 Run 정상 vs 다른 Run 정상  (b) 집중 Run 안에서 해당 불량 vs 정상"""
    d = scope2(d)
    if focus_run is None:                                     # 해당 불량이 가장 많은 Run
        cnt = d[d['Reason'] == defect].groupby('run_id').size()
        if cnt.empty:
            return pd.DataFrame()
        focus_run = cnt.idxmax()
    fr = d['run_id'] == focus_run
    rows = []
    for v in LEVEL_VARS + ['__score__']:
        c = 'score' if v == '__score__' else raw(v)
        nm = '최종 모델 점수(참고)' if v == '__score__' else NAME[v]
        a1, b1 = d.loc[fr & (d['PassOrFail'] == 0), c], d.loc[~fr & (d['PassOrFail'] == 0), c]
        a2, b2 = d.loc[fr & (d['Reason'] == defect), c], d.loc[fr & (d['PassOrFail'] == 0), c]
        for lab, a, b in [(f'(a) {focus_run} 정상 vs 다른 Run 정상', a1, b1),
                          (f'(b) {focus_run} 안: {defect} vs 정상', a2, b2)]:
            pv = mannwhitneyu(a, b).pvalue if len(a) > 0 and len(b) > 0 else np.nan
            rows.append({'제품': p, '변수': nm, '비교': lab, 'n(앞)': len(a), 'n(뒤)': len(b),
                         '중앙값(앞)': a.median(), '중앙값(뒤)': b.median(), "Cliff's δ": _cliff(a, b), 'p': pv})
    return pd.DataFrame(rows)


def plot_levels(p, d, info, save=None):
    """Run별 금형온도·계량위치 (회색 띠 = reference 1~99%, 빨강 = 불량, 주황 = 가스)"""
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    d = scope2(d)
    runs = sorted(d['run_id'].unique())
    fig, axes = plt.subplots(1, len(LEVEL_VARS), figsize=(6 * len(LEVEL_VARS), 4.2))
    rng = np.random.default_rng(0)
    for ax, v in zip(np.atleast_1d(axes), LEVEL_VARS):
        c = raw(v)
        lo, hi = info['band'].loc[0.01, c], info['band'].loc[0.99, c]
        ax.axhspan(lo, hi, color='#d9d9d9', alpha=0.6, label='정상 기준 1~99%')
        for i, r in enumerate(runs):
            g = d[d['run_id'] == r]
            x = i + rng.uniform(-0.25, 0.25, len(g))
            nm = g['PassOrFail'] == 0
            ax.scatter(x[nm.values], g.loc[nm, c], s=5, color='#9db9d5', alpha=0.5)
            gas = (g['Reason'] == '가스').values
            oth = ((g['PassOrFail'] == 1) & (g['Reason'] != '가스')).values
            ax.scatter(x[gas], g.loc[gas, c], s=28, color='#e67e22', edgecolor='k', lw=0.4,
                       label='가스' if i == 0 else None)
            ax.scatter(x[oth], g.loc[oth, c], s=28, color='#c0392b', edgecolor='k', lw=0.4,
                       label='기타 불량' if i == 0 else None)
        ax.set_xticks(range(len(runs))); ax.set_xticklabels(runs, rotation=30, fontsize=8)
        ax.set_title(f'{p.upper()} — {NAME[v]}', loc='left', fontsize=10)
        ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()


# ====================================================================== B. 결합 모델
def _surprise(pct):
    return -np.log10(1 - np.clip(np.asarray(pct, float), 0, 1 - 1e-9))


def add_scores(d):
    d = d.copy()
    sm, sl = _surprise(d['pct']), _surprise(d['level_pct'])
    d['S_M_final'] = d['score']
    d['S_L_level'] = d['level_score']
    d['S_C_mean'] = (sm + sl) / 2 + 1e-6 * np.tanh(d['score'] / 10)       # 동점 깨기
    d['S_C_max'] = np.maximum(sm, sl) + 1e-6 * np.tanh(d['score'] / 10)
    # 경보 판정용 이상도(정상 기준 백분위 기반): 결합은 '정상에서 같은 비율이 경보'가 되도록 q를 이상도로 변환
    d['A_M_final'], d['A_L_level'] = sm, sl
    d['A_C_mean'], d['A_C_max'] = (sm + sl) / 2, np.maximum(sm, sl)
    return d


METHODS = ['M_final', 'L_level', 'C_mean', 'C_max']


def evaluate(p, d, ref_alarm_rates=None):
    """② 평가 + 경보. ref_alarm_rates: {method: {q: reference 이상도 임계값}}"""
    d = add_scores(scope2(d))
    y = d['PassOrFail'].values.astype(int)
    big = d.groupby('run_id')['PassOrFail'].sum().idxmax()          # 불량이 가장 많은 Run
    inr = (d['run_id'] == big).values
    rows = []
    base = d['S_M_final'].values
    for m in METHODS:
        s = d[f'S_{m}'].values
        g = pd.DataFrame({'run_id': d['run_id'], 'score': s, 'y': y})
        ap, lo, hi = ap_ci(y, s)
        r = {'제품': p, '방법': m, 'AP': ap, 'AP_lo': lo, 'AP_hi': hi}
        for name, rs in TYPE_GROUPS.items():
            pm = d['Reason'].isin(rs).values
            if pm.sum():
                ev = pm | (y == 0)
                r[f'AP_{name}'] = average_precision_score(pm[ev].astype(int), s[ev])
        r['Run별 상위10%'] = '%d/%d' % recall_at_k_by_run(g, 0.10, g['y'] == 1)
        r['전체 상위10%'] = '%d/%d' % recall_at_k_by_run(g.assign(run_id='ALL'), 0.10, g['y'] == 1)
        r[f'{big} 내부 AP'] = average_precision_score(y[inr], s[inr]) if y[inr].sum() and (y[inr] == 0).any() else np.nan
        r[f'{big} 내부 불량률'] = y[inr].mean()
        if ref_alarm_rates:
            for q in QS:
                th = ref_alarm_rates[m][q]
                al = d[f'A_{m}'].values >= th
                tp = int((al & (y == 1)).sum())
                r[f'q{q} 경보'] = int(al.sum())
                r[f'q{q} 적중'] = tp
                r[f'q{q} {big} 정상 경보 비율'] = (al & inr & (y == 0)).sum() / max(1, (inr & (y == 0)).sum())
        if m != 'M_final':
            o, l, h, v = _paired_delta(y, s, base)
            r.update({'ΔAP(vs 최종)': o, 'Δ CI': f'[{l:.3f}, {h:.3f}]', '판정': v.replace('기준', '최종')})
        rows.append(r)
    return pd.DataFrame(rows)


def reference_thresholds(p, ref_d):
    """경보 임계값: reference(정상) 샷에서 각 방법의 이상도 q 분위수 → 정상 오경보율 q가 같도록"""
    out = {}
    for m in METHODS:
        a = ref_d[f'A_{m}'].values
        out[m] = {q: float(np.quantile(a, q)) for q in QS}
    return out


def reference_scores(p, ref, info, model_dir='./outputs/final_model', base_dir='.'):
    """reference 샷에 같은 결합 점수 계산 (경보 임계값용). 최종 모델 점수는 저장된 모델로 다시 계산"""
    from final_model import FinalModel, make_features
    model = FinalModel.load(f'{model_dir}/{p}_final_model.pkl')
    f = make_features(ref, model.steps, model.fr_step)
    s = model.score(f)
    f['score'] = s
    f['pct'] = model.pct(s)
    cols = [raw(v) for v in LEVEL_VARS]
    z = ((f[cols] - info['median']) / info['iqr']).clip(-CLIP, CLIP)
    ls = np.sqrt((z ** 2).sum(axis=1)).values
    f['level_score'] = ls
    s_ref = np.sort(ls)
    f['level_pct'] = np.searchsorted(s_ref, ls, side='right') / (len(s_ref) + 1)
    f['split'], f['Reason'], f['PassOrFail'] = '정상기준', '정상', 0
    return add_scores(f)


def eval_frame1(p, d):
    """① 공식 2차·행 AP"""
    rows = load_official_rows(p)
    d = add_scores(d)
    y = rows['PassOrFail'].values.astype(int)
    out = {'제품': p}
    for m in METHODS:
        s = rows['TimeStamp'].map(pd.Series(d[f'S_{m}'].values, index=d['TimeStamp'].values)).fillna(0).values
        out[f'① AP {m}'] = average_precision_score(y, s)
    return out
