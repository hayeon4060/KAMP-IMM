"""
inspect_curve.py — 42번: 검사 비율별 Precision / Recall / F1 (검사 우선순위 곡선)

점수 높은 순으로 평가 대상 샷의 r%를 검사한다고 할 때 (r = 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30%)
  검사 샷 수, 잡은 불량(TP), Precision, Recall, F1, 무작위 대비 배수(lift), 유형별(가스·미성형) 몇 개 중 몇 개
+ 무작위 검사 기준선 (Recall = r, Precision = 불량률)

평가 대상: 초기허용불량·RG3 R6 제외 (lab['eval']), 불량 = PassOrFail == 1
"""
import numpy as np
import pandas as pd

RATES = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30)


def _types(lab):
    if 'type' in lab.columns:
        return lab['type'].values
    r = lab['Reason'].fillna('정상').values
    return np.where(r == '가스', '가스', np.where(np.isin(r, ['미성형', '가스+미성형']), '미성형',
                                               np.where(lab['PassOrFail'].values == 1, '기타', '정상')))


def _tmask(lab, idx, t, ty):
    """'gas_any' 열이 있으면 가스 = 가스 + 가스+미성형 (혼합은 양쪽에 셈)"""
    if ty == '가스' and 'gas_any' in lab.columns:
        return lab['gas_any'].values.astype(bool)[idx]
    return t == ty


def curve(lab, score, rates=RATES, name=''):
    """반환: 검사 비율별 표"""
    m = lab['eval'].values
    idx = np.flatnonzero(m)
    y = (lab['PassOrFail'].values[idx] == 1).astype(int)
    t = _types(lab)[idx]
    s = np.asarray(score, float)[idx]
    ok = np.isfinite(s)
    s = np.where(ok, s, (s[ok].min() - 1) if ok.any() else 0) + np.random.default_rng(0).random(len(idx)) * 1e-9
    order = np.argsort(-s)
    base = y.mean()
    rows = []
    for r in rates:
        k = int(np.ceil(r * len(idx)))
        sel = np.zeros(len(idx), bool)
        sel[order[:k]] = True
        tp = int((sel & (y == 1)).sum())
        P = tp / k if k else 0.0
        R = tp / y.sum() if y.sum() else 0.0
        row = {'모델': name, '검사 비율': f'{r * 100:g}%', '검사 샷': k, '불량': f'{tp}/{int(y.sum())}',
               'Precision': P, 'Recall': R, 'F1': 2 * P * R / (P + R) if P + R else 0.0,
               'Precision 최대(불량수/검사샷)': min(1.0, y.sum() / k) if k else np.nan,
               '정규화 Precision': tp / min(k, y.sum()) if k and y.sum() else np.nan,
               '무작위 대비(배)': P / base if base else np.nan,
               '무작위 기대 불량': round(r * y.sum(), 1), '평가 샷': len(idx)}
        for ty in ('가스', '미성형'):
            tm = _tmask(lab, idx, t, ty)
            n = int(tm.sum())
            if n:
                row[ty] = f'{int((sel & tm).sum())}/{n}'
        rows.append(row)
    return pd.DataFrame(rows)


def plot_curves(tables, title='', save=None):
    """tables: {이름: curve() 결과}. 왼쪽 Recall, 가운데 Precision, 오른쪽 F1 (+ 무작위 기준선)"""
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, T in tables.items():
        x = T['검사 비율'].str.rstrip('%').astype(float)
        for ax, col in zip(axes, ('Recall', 'Precision', 'F1')):
            ax.plot(x, T[col], marker='o', label=name)
    T0 = next(iter(tables.values()))
    x = T0['검사 비율'].str.rstrip('%').astype(float)
    base = int(T0['불량'].iloc[0].split('/')[1]) / T0['평가 샷'].iloc[0]
    axes[0].plot(x, x / 100, 'k--', lw=1, label='무작위 검사')
    axes[1].axhline(base, color='k', ls='--', lw=1, label=f'무작위 (불량률 {base:.3f})')
    for ax, col in zip(axes, ('Recall (잡은 불량 / 전체 불량)', 'Precision (검사 샷 중 불량)', 'F1')):
        ax.set_xlabel('검사 비율 (%)'); ax.set_ylabel(col)
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.suptitle(title, x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()


# ====================================================================== 모델별 점수 준비
def load_final(p):
    """최종 모델 점수 (20b 결과) — R6 제외·평가 대상 표시 포함"""
    from final_model import INITIAL, apply_exclusions
    lab = pd.read_csv(f'./outputs/final_model/{p}_scores.csv', parse_dates=['TimeStamp'])
    lab['Reason'] = lab['Reason'].fillna('정상')
    lab = apply_exclusions(lab, p)
    lab['eval'] = (lab['split'] != '제외') & (lab['Reason'] != INITIAL)
    return lab


def cn7_alarm_plus_gas_score(k=10):
    """CN7: 최종 모델 경보(pct≥0.999) 샷을 맨 앞에 두고, 나머지는 std4 흔들림 점수 순 → 하나의 순위 점수로"""
    import cn7_alarm_plus_gas as C
    ref, lab = C.load('cn7')
    g = C.gas_std4(ref, lab, k)
    g = np.where(np.isfinite(g), g, np.nanmin(g[np.isfinite(g)]) - 1)
    rank = pd.Series(g).rank(pct=True).values                     # 0~1
    score = np.where(lab['pct'].values >= C.ALARM_Q, 2.0 + lab['pct'].values, rank)
    return lab, score


def rg3_gas_rf_score(seed=0):
    """RG3: 가스 전용 RF (std 24개, Run 하나씩 빼고 검증) 점수"""
    import rg3_gas_short as G
    lab, X, _ = G.load()
    s = G.loro(lab, X, '가스', 'gas', seed=seed)
    return lab, s


# ====================================================================== RG3 가스 + 미성형 결합 (39번 구성) 곡선
def split_curve(lab, gas_s, short_s, gas_share=0.68, rates=RATES, name=''):
    """검사 비율 r마다: 검사 샷 K 중 gas_share×K를 가스 모델 상위로, 나머지를 미성형 모델 순위로 채움 (39번과 같은 규칙)"""
    m = lab['eval'].values
    idx = np.flatnonzero(m)
    y = (lab['PassOrFail'].values[idx] == 1).astype(int)
    t = _types(lab)[idx]
    jit = np.random.default_rng(0).random(len(idx)) * 1e-9
    fill = lambda s: np.where(np.isfinite(s), s, np.nanmin(s[np.isfinite(s)]) - 1 if np.isfinite(s).any() else 0)
    g = fill(np.asarray(gas_s, float)[idx]) + jit
    sh = fill(np.asarray(short_s, float)[idx]) + jit
    og, osh = np.argsort(-g), np.argsort(-sh)
    base = y.mean()
    rows = []
    for r in rates:
        K = int(np.ceil(r * len(idx)))
        sel = np.zeros(len(idx), bool)
        sel[og[:int(round(gas_share * K))]] = True
        for i in osh:
            if sel.sum() >= K:
                break
            sel[i] = True
        tp = int((sel & (y == 1)).sum())
        P, R = tp / K, tp / y.sum()
        row = {'모델': name, '검사 비율': f'{r * 100:g}%', '검사 샷': K, '불량': f'{tp}/{int(y.sum())}',
               'Precision': P, 'Recall': R, 'F1': 2 * P * R / (P + R) if P + R else 0.0,
               'Precision 최대(불량수/검사샷)': min(1.0, y.sum() / K), '정규화 Precision': tp / min(K, y.sum()),
               '무작위 대비(배)': P / base, '무작위 기대 불량': round(r * y.sum(), 1), '평가 샷': len(idx)}
        for ty in ('가스', '미성형'):
            tm = _tmask(lab, idx, t, ty)
            row[ty] = f'{int((sel & tm).sum())}/{int(tm.sum())}'
        rows.append(row)
    return pd.DataFrame(rows)


def rg3_combo_scores(short_method='중첩', seed=0):
    """RG3 가스 RF 점수 + 미성형 모델 점수. short_method: 'under3_smote'·'under10_adasyn'·'balanced_rf'·'weight'·'중첩'"""
    import rg3_gas_short as G
    lab, X, _ = G.load()
    gas = G.loro(lab, X, '가스', 'gas', seed=seed)
    short = G.nested_short(lab, X, seed=seed, verbose=False)[0] if short_method == '중첩' \
        else G.loro(lab, X, '미성형', short_method, seed=seed)
    return lab, gas, short


# ====================================================================== 검사 예산과 무관한 '분류' 지표
def rank_metrics(lab, score, recalls=(0.25, 0.5, 0.75)):
    """
    검사 비율을 정하지 않고 '정상과 불량을 얼마나 잘 가르나'를 보는 지표 (최대값 1)
      R-Precision      : 불량 수(D)만큼만 검사했을 때의 Precision (= 그때의 Recall). 무작위 = 불량률
      Precision@Recall : 불량의 25·50·75%를 잡을 때까지 검사하면, 검사한 샷 중 불량 비율
      필요 검사 비율    : 그 Recall에 도달하려면 전체의 몇 %를 검사해야 하나
      ROC-AUC          : 불량 하나와 정상 하나를 뽑았을 때 불량 점수가 더 높을 확률 (무작위 0.5)
      AP               : PR 곡선 아래 면적(계단식)
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    m = lab['eval'].values
    y = (lab['PassOrFail'].values[m] == 1).astype(int)
    s = np.asarray(score, float)[m]
    ok = np.isfinite(s)
    s = np.where(ok, s, (s[ok].min() - 1) if ok.any() else 0) + np.random.default_rng(0).random(len(s)) * 1e-9
    o = np.argsort(-s)
    cum = np.cumsum(y[o])
    D = int(y.sum())
    r = {'불량 수': D, 'R-Precision': cum[D - 1] / D if D else np.nan, 'ROC-AUC': roc_auc_score(y, s),
         'AP': average_precision_score(y, s), '무작위 기준(불량률)': y.mean()}
    for rc in recalls:
        need = int(np.ceil(rc * D))
        k = int(np.searchsorted(cum, need) + 1) if need else 0
        r[f'Precision@Recall{int(rc * 100)}%'] = need / k if k else np.nan
        r[f'Recall{int(rc * 100)}% 필요 검사 비율'] = k / len(s)
    return r


def inner_classification(lab, score, pct, budget=0.10, q=0.999):
    """
    2단계 판정: 점수 상위 budget(예: 10%)을 검사 대상으로 고른 뒤, 그 안에서 pct ≥ q면 '불량', 아니면 '정상'으로 판정
    → 검사 대상 안에서의 혼동행렬·Precision·Recall·Accuracy (검사 대상 안의 불량 기준)
    """
    m = lab['eval'].values
    y = (lab['PassOrFail'].values[m] == 1).astype(int)
    s = np.asarray(score, float)[m] + np.random.default_rng(0).random(m.sum()) * 1e-9
    pc = np.asarray(pct, float)[m]
    K = int(np.ceil(budget * m.sum()))
    sel = np.argsort(-s)[:K]
    yy, pred = y[sel], (pc[sel] >= q).astype(int)
    tp, fp = int(((pred == 1) & (yy == 1)).sum()), int(((pred == 1) & (yy == 0)).sum())
    fn, tn = int(((pred == 0) & (yy == 1)).sum()), int(((pred == 0) & (yy == 0)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {'검사 대상 샷': K, '그 안의 불량': int(yy.sum()), '불량 판정': tp + fp, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Precision(판정)': P, 'Recall(검사 대상 안)': R, 'F1': 2 * P * R / (P + R) if P + R else 0.0,
            'Accuracy': (tp + tn) / K}
