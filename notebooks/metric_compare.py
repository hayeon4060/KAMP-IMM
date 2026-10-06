"""
metric_compare.py — 54번: M_maha_free(최종 모델 단독) vs 현재 최종 모델(제품별 통합 모델) — 샷 단위 / 행 단위 지표 비교

평가 틀 3개
  행 (공식 2차, LH/RH) : 대회 공식 형태. CN7 1,211행·불량 17행 (팀원 표의 AP 0.373이 이 틀). 같은 샷의 LH/RH 행은 같은 점수
  샷 (2차)            : 2차 데이터 샷 단위. 샷의 LH·RH 중 하나라도 불량이면 불량 샷
  샷 (1차+2차)        : labeled 전체 샷 단위 (초기허용 제외, RG3 R6 제외는 모든 틀 공통)

모델
  M_maha_free               : 최종 모델 점수 단독 (비지도, 라벨 0개)
  CN7 P 통합                : 최종 모델 경보(pct≥0.999) 맨 위 + 나머지 std4 흔들림 순 (라벨 0개)
  RG3 P 통합 (시간 순 / LORO): 가스 RF 순위 : N1 순위 = 68 : 32 로 합친 순위 (시드 3개 평균)

지표
  순위 전체 : AP(계단식), PR-AUC(사다리꼴), ROC-AUC, R-Precision, 무작위 AP 평균·97.5%(1,000번), AP/무작위, 유의(AP > 무작위 97.5%)
              + 행 틀은 상한 F1(같은 샷 LH/RH가 같은 점수라 도달 가능한 최대 F1, 팀원 표와 같음)
  검사 5·10·15·20% : 검사 수 K(= round(전체 × 비율), 팀원 budget_f1과 같음), TP·FP·FN, Precision, Recall, F1,
              예산 상한 F1(K개를 완벽히 골랐을 때), F1 / 예산 상한, 정규화 Precision(TP / min(K, 불량 수)),
              무작위 대비(배) = Precision / 불량률, 가스·미성형 몇 개 중 몇 개 (가스+미성형은 양쪽에 셈)
동점 처리: 점수가 같으면 무작위 순서 (시간 순서 덕에 유리해지지 않게, 팀원 _order와 같음)
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import semi_final as SF
import unified_model as UM
from final_model import random_band, info_ceiling, pr_auc_trapz, _order

OUT = './outputs/metric_compare'
RATES = (0.05, 0.10, 0.15, 0.20)
MODELS = {'cn7': {'M_maha_free (최종 모델 단독)': 'B1 최종 모델 점수 (비지도)', 'P 통합: 경보 먼저 + std4': UM.P_NAME['cn7']},
          'rg3': {'M_maha_free (최종 모델 단독)': 'B1 최종 모델 점수 (비지도)',
                  'P 통합: 가스 RF(시간 순) + N1': UM.P_NAME['rg3_fwd'],
                  'P 통합: 가스 RF(LORO) + N1': UM.P_NAME['rg3_loro']}}


# ====================================================================== 지표
def _masks(T):
    t = T['type'].values
    gas = T['gas_any'].values.astype(bool) if 'gas_any' in T.columns else (t == '가스')
    return {'가스': gas, '미성형': t == '미성형'}


def rank_metrics(y, s, rows=None, n_rand=1000):
    y = np.asarray(y, int)
    s = np.nan_to_num(np.asarray(s, float), nan=-1e18, neginf=-1e18)
    ap = average_precision_score(y, s)
    rmean, rhi = random_band(y, n_rand)
    o = _order(s)
    D = int(y.sum())
    r = {'평가 대상': len(y), '불량': D, '불량률': y.mean(), 'AP': ap, 'PR-AUC(사다리꼴)': pr_auc_trapz(y, s),
         'ROC-AUC': roc_auc_score(y, s), 'R-Precision': y[o[:D]].mean() if D else np.nan,
         '무작위 AP 평균': rmean, 'AP / 무작위': ap / rmean, '무작위 AP 97.5%': rhi, '유의': 'O' if ap > rhi else '-'}
    if rows is not None:
        r['상한 F1 (같은 샷 동점)'] = info_ceiling(rows)
    return r


def budget_metrics(y, s, masks, rates=RATES):
    y = np.asarray(y, int)
    s = np.nan_to_num(np.asarray(s, float), nan=-1e18, neginf=-1e18)
    o = _order(s)
    D, N = int(y.sum()), len(y)
    out = []
    for q in rates:
        k = int(round(N * q))
        sel = np.zeros(N, bool)
        sel[o[:k]] = True
        tp = int((sel & (y == 1)).sum())
        fp, fn = k - tp, D - tp
        P = tp / k if k else 0.0
        R = tp / D if D else 0.0
        F1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        tpm = min(D, k)
        f1max = 2 * tpm / (2 * tpm + (k - tpm) + (D - tpm)) if k else 0.0
        row = {'검사 비율': f'{int(q * 100)}%', '검사 수 K': k, 'TP': tp, 'FP': fp, 'FN': fn, '불량': f'{tp}/{D}',
               'Precision': P, 'Recall': R, 'F1': F1, '예산 상한 F1': f1max, 'F1 / 예산 상한': F1 / f1max if f1max else np.nan,
               '정규화 Precision': tp / tpm if tpm else np.nan, '무작위 대비(배)': P / (D / N) if D else np.nan}
        for ty, m in masks.items():
            if m.sum():
                row[ty] = f'{int((sel & m).sum())}/{int(m.sum())}'
        out.append(row)
    return out


# ====================================================================== 실행
def _frames_scores(p, seeds):
    """평가 틀별 (평가 표, 모델별 시드별 점수)"""
    fn = UM.scores_cn7 if p == 'cn7' else UM.scores_rg3
    out = {}
    for fr in ('2차', '전체'):
        D, S, _ = fn(fr, seeds)
        lab = D['lab']
        ev = lab['eval'].values
        T = lab[ev].reset_index(drop=True)
        sc = {disp: {sd: np.asarray(S[sd][key], float)[ev] for sd in S} for disp, key in MODELS[p].items()}
        out[f'샷 ({"2차" if fr == "2차" else "1차+2차"})'] = (T, sc, None)
        if fr == '2차':
            rw = SF.row_frame(p, lab)
            scr = {disp: {sd: SF.row_score(rw, lab, S[sd][key]) for sd in S} for disp, key in MODELS[p].items()}
            out['행 (공식 2차, LH/RH)'] = (rw, scr, rw[['TimeStamp', 'PassOrFail']])
    return out


def run(products=('cn7', 'rg3'), seeds=(0, 1, 2), rates=RATES, verbose=True):
    os.makedirs(OUT, exist_ok=True)
    RK, BG = [], []
    for p in products:
        for frame, (T, sc, rows) in _frames_scores(p, seeds).items():
            y = T['PassOrFail'].values.astype(int)
            masks = _masks(T)
            for disp, by_seed in sc.items():
                for sd, s in by_seed.items():
                    if disp.startswith('M_maha') and sd != min(by_seed):
                        continue
                    RK.append({'제품': p, '평가 틀': frame, '모델': disp, 'seed': sd, **rank_metrics(y, s, rows)})
                    BG += [{'제품': p, '평가 틀': frame, '모델': disp, 'seed': sd, **r} for r in budget_metrics(y, s, masks, rates)]
            if verbose:
                print(f'[{p}] {frame} 완료', flush=True)
    rk, bg = pd.DataFrame(RK), pd.DataFrame(BG)
    R = {'rank': _avg_rank(rk), 'budget': _avg_budget(bg), 'rank_by_seed': rk, 'budget_by_seed': bg}
    R['compare'] = compare(R)
    for k in ('rank', 'budget', 'compare'):
        R[k].to_csv(f'{OUT}/{k}.csv', index=False, encoding='utf-8-sig')
    return R


def _avg_rank(rk):
    keys = ['제품', '평가 틀', '모델']
    num = rk.drop(columns=['seed']).groupby(keys, sort=False).mean(numeric_only=True).reset_index()
    sig = rk.groupby(keys, sort=False)['유의'].agg(lambda x: f'{(x == "O").sum()}/{len(x)}').reset_index()
    return num.merge(sig, on=keys)


def _avg_budget(bg):
    keys = ['제품', '평가 틀', '모델', '검사 비율']
    b = bg.copy()
    for c in ('불량', '가스', '미성형'):
        if c in b.columns:
            b[c + ' 수'] = b[c].map(lambda v: float(str(v).split('/')[0]) if isinstance(v, str) else np.nan)
            b[c + ' 전체'] = b[c].map(lambda v: int(str(v).split('/')[1]) if isinstance(v, str) else np.nan)
    num = b.groupby(keys, sort=False).mean(numeric_only=True).reset_index()
    for c in ('불량', '가스', '미성형'):
        if c + ' 수' in num.columns:
            num[c] = num[c + ' 수'].round(1).astype(str) + '/' + num[c + ' 전체'].fillna(0).astype(int).astype(str)
    cols = keys + ['검사 수 K', '불량', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1', '예산 상한 F1', 'F1 / 예산 상한',
                   '정규화 Precision', '무작위 대비(배)', '가스', '미성형']
    return num[[c for c in cols if c in num.columns]]


def compare(R):
    """M_maha_free vs 통합 모델 나란히 — 평가 틀 × 검사 비율, 핵심 지표 + 차이"""
    b = R['budget']
    rk = R['rank'][['제품', '평가 틀', '모델', 'AP', 'ROC-AUC']]
    b = b.merge(rk, on=['제품', '평가 틀', '모델'])
    rows = []
    for (p, fr, q), g in b.groupby(['제품', '평가 틀', '검사 비율'], sort=False):
        base = g[g['모델'].str.startswith('M_maha')]
        if base.empty:
            continue
        base = base.iloc[0]
        for _, r in g[~g['모델'].str.startswith('M_maha')].iterrows():
            rows.append({'제품': p, '평가 틀': fr, '검사 비율': q, '통합 모델': r['모델'],
                         'AP (maha → 통합)': f'{base["AP"]:.3f} → {r["AP"]:.3f}',
                         'ROC-AUC (maha → 통합)': f'{base["ROC-AUC"]:.3f} → {r["ROC-AUC"]:.3f}',
                         '불량 (maha → 통합)': f'{base["불량"]} → {r["불량"]}',
                         'Precision (maha → 통합)': f'{base["Precision"]:.3f} → {r["Precision"]:.3f}',
                         'Recall (maha → 통합)': f'{base["Recall"]:.3f} → {r["Recall"]:.3f}',
                         'F1 (maha → 통합)': f'{base["F1"]:.3f} → {r["F1"]:.3f}',
                         'ΔTP': r['TP'] - base['TP'],
                         '가스 (maha → 통합)': f'{base.get("가스", "-")} → {r.get("가스", "-")}',
                         '미성형 (maha → 통합)': f'{base.get("미성형", "-")} → {r.get("미성형", "-")}'})
    return pd.DataFrame(rows)


def view(R, p, frame=None):
    """제품(·평가 틀)별 보기 좋은 표 3개: 순위 지표, 검사 비율별, 비교"""
    f = (lambda d: d[(d['제품'] == p) & ((d['평가 틀'] == frame) if frame else True)])
    return f(R['rank']), f(R['budget']), f(R['compare'])


def plot(R, p, metric='Recall', save=None):
    """평가 틀별 검사 비율 vs 지표 (maha 점선, 통합 실선)"""
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    b = R['budget'][R['budget']['제품'] == p]
    frames = list(dict.fromkeys(b['평가 틀']))
    fig, axes = plt.subplots(1, len(frames), figsize=(4.6 * len(frames), 3.6), sharey=True)
    for ax, fr in zip(np.atleast_1d(axes), frames):
        g = b[b['평가 틀'] == fr]
        for m, gg in g.groupby('모델', sort=False):
            x = gg['검사 비율'].str.rstrip('%').astype(int)
            ax.plot(x, gg[metric], marker='o', ls='--' if m.startswith('M_maha') else '-', label=m)
        ax.set_title(fr, fontsize=10)
        ax.set_xlabel('검사 비율 (%)')
    np.atleast_1d(axes)[0].set_ylabel(metric)
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.suptitle(f'{p.upper()} — M_maha_free vs 통합 모델 ({metric})', x=0.01, ha='left')
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
