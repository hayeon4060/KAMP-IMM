"""
model_phys.py — (1) 계량시간 → 사이클 연결 확인  (2) 물리 검증 결과 기반 변수 재설정 + 모델 비교

(1) 가설: RG3 가스 샷의 '사이클↑'은 체류시간(열분해)이 아니라 계량 지연의 흔적
    - Run 안에서 계량시간과 사이클이 같이 움직이나 (Spearman, Run별·합침)
    - 사이클을 시간 구성요소(계량·사출·형체)로 Run 안 회귀 → 무엇이 사이클 변동을 만드나
    - 사이클에서 계량 영향을 뺀 잔차로도 가스 차이가 남나 → 남으면 계량 말고 다른 지연

(2) 변수 재설정 근거 (12_physics_check 결과)
    채택: 재료 공급·계량 계통 — 계량지연(지지: RG3 4/5 Run), 쿠션부족·재료부족(지지: CN7)
          + 그 구성 변수(충전·속도·압력 편차) + 사이클 편차 + 재가동 맥락
    제외: 배럴온도(점도 일관성 실패), 금형온도 수준(조기고화 반대/근거 없음), 호퍼(CN7 R1에서만),
          흐름저항·점도잔차(일관성 약함), 사출속도 단독(디젤 근거 없음)
    ※ 모델 입력은 모두 '인과적' 변수(dev_* = 같은 Run 과거 샷 기준) → 현장에서 그대로 계산 가능
       (12번의 Run 안 z는 미래 샷까지 쓰므로 분석용이었음)

    비교 (같은 reference·같은 LORO·같은 평가):
      E2_free   : v2 메인 재현 (free 변수, 이상탐지 3종 + 통합 logit)
      E2_phys   : 같은 구조, 물리 변수
      R_phys    : 학습 없는 물리 규칙 = max(재료부족, 쿠션부족, 계량지연) — 결과 보기 전에 고정
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

from features import raw
from run_compare import add_run_z, _matched, GROUPS
from model_v2 import (RefDetector, to_ref_scale, run_unsupervised, run_pooled_supervised, ensemble,
                      _oof_frame, FSETS, SUP_FSETS, CONTEXT)
from plot_style import set_style, INK, SUB, MUTED, NORMAL, TYPE_COL, TYPE_ORDER


# ====================================================================== (1) 계량 → 사이클
def _within_ols(y, X, g):
    """Run 평균 빼고 OLS. 반환 (표준화 계수, R², 잔차)"""
    yd = y - y.groupby(g).transform('mean')
    Xd = X - X.groupby(g).transform('mean')
    ok = yd.notna() & Xd.notna().all(axis=1)
    sy, sx = yd[ok].std(), Xd[ok].std().replace(0, 1)
    A, b = (Xd[ok] / sx).values, (yd[ok] / sy).values
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = b - A @ beta
    r2 = 1 - (res @ res) / (b @ b) if (b @ b) > 0 else np.nan
    resid = pd.Series(np.nan, index=y.index)
    resid[ok] = res
    return pd.Series(beta, index=X.columns), r2, resid


def metering_cycle_link(d, product):
    """반환: (Run별 상관표, 사이클 구성요소 회귀표, 불량 vs 정상 δ 표, Run 안 z DataFrame)"""
    met, cyc = raw('Plasticizing_Time'), raw('Cycle_Time')
    parts = [c for c in [met, raw('Injection_Time'), raw('Filling_Time'), raw('Clamp_Close_Time')]
             if c in d.columns and d[c].nunique() > 1]
    Z = add_run_z(d, list(dict.fromkeys([met, cyc] + parts)))
    good = d['PassOrFail'] == 0

    rows = []
    for r, g in d.groupby('run_id'):
        m = d['run_id'] == r
        rho_all = spearmanr(Z.loc[m, met], Z.loc[m, cyc], nan_policy='omit')[0]
        rho_good = spearmanr(Z.loc[m & good, met], Z.loc[m & good, cyc], nan_policy='omit')[0]
        x, y = d.loc[m, met], d.loc[m, cyc]
        slope = np.polyfit(x, y, 1)[0] if x.nunique() > 1 else np.nan
        rows.append({'제품': product, 'Run': r, '샷': int(m.sum()), 'ρ(계량,사이클) 전체': rho_all,
                     'ρ 정상만': rho_good, '기울기(사이클초/계량초)': slope,
                     '계량 중앙값(s)': x.median(), '사이클 중앙값(s)': y.median(),
                     '사이클 IQR(s)': y.quantile(.75) - y.quantile(.25)})
    per_run = pd.DataFrame(rows)
    pooled = spearmanr(Z[met], Z[cyc], nan_policy='omit')[0]
    per_run = pd.concat([per_run, pd.DataFrame([{'제품': product, 'Run': '합침(Run 안 z)',
                                                  'ρ(계량,사이클) 전체': pooled}])], ignore_index=True)

    coef, r2, _ = _within_ols(d[cyc], d[parts], d['run_id'])
    comp = pd.DataFrame({'제품': product, '구성요소': coef.index, '표준화 계수': coef.values, 'within R²': r2})

    # 사이클에서 계량 영향 뺀 잔차
    _, _, cyc_res = _within_ols(d[cyc], d[[met]], d['run_id'])
    V = pd.DataFrame({'계량시간': Z[met], '사이클': Z[cyc], '사이클|계량 잔차': cyc_res})
    drows = []
    for grp in ['가스', '미성형계열']:
        pos = d['Reason'].isin(GROUPS[grp]).values
        if pos.sum() == 0:
            continue
        for tag, neg in [('전체', good.values), ('±10샷', _matched(d, pos, 10))]:
            for c in V.columns:
                a, b = V[c].values[pos], V[c].values[neg]
                a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                if len(a) == 0 or np.ptp(np.r_[a, b]) == 0:
                    continue
                U, p = mannwhitneyu(a, b, alternative='two-sided')
                drows.append({'제품': product, '불량군': grp, '비교군': tag, '변수': c, 'n불량': len(a),
                              'δ': 2 * U / (len(a) * len(b)) - 1, 'p': p})
    return per_run, comp, pd.DataFrame(drows), V


def plot_metering_cycle(D, V, save=None):
    """제품별: Run 안 z 계량시간 vs 사이클 (정상 회색, 불량 유형 색)"""
    set_style()
    fig, axes = plt.subplots(1, len(D), figsize=(6 * len(D), 4.8))
    rng = np.random.default_rng(0)
    for ax, p in zip(np.atleast_1d(axes), D):
        d, v = D[p], V[p]
        j = lambda n: rng.uniform(-0.08, 0.08, n)
        n = (d['PassOrFail'] == 0).values
        ax.scatter(v.loc[n, '계량시간'] + j(n.sum()), v.loc[n, '사이클'] + j(n.sum()), s=7, color=NORMAL,
                   alpha=0.5, label='정상')
        for t in TYPE_ORDER:
            m = (d['Reason'] == t).values
            if m.any():
                ax.scatter(v.loc[m, '계량시간'], v.loc[m, '사이클'], s=44, color=TYPE_COL[t], edgecolor='white',
                           lw=1.3, zorder=3, label=t)
        rho = spearmanr(v['계량시간'], v['사이클'], nan_policy='omit')[0]
        ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
        ax.set_xlabel('계량시간 (Run 안 z)'); ax.set_ylabel('사이클 (Run 안 z)')
        ax.set_title(f'{p.upper()} — Run 안 Spearman ρ = {rho:.2f}', loc='left', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('계량이 길어진 샷은 사이클도 길어지나 (±8에서 자름)', x=0.01, ha='left', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save:
        fig.savefig(save)
    plt.show()


# ====================================================================== (2) 물리 기반 변수
PHYS_BASE = ['dev_Plasticizing_Time', 'dev_Cushion_Position', 'dev_Filling_Time', 'dev_Max_Injection_Speed',
             'dev_Max_Injection_Pressure', 'dev_Cycle_Time', 'fill_anomaly']
PHYS_DERIVED = ['phys_shortfall', 'phys_cushion_low', 'phys_metering']


def add_phys(d):
    """인과적 dev_* 로 물리 지표 계산 (라벨 미사용). Run 첫 5샷 NaN은 0(편차 없음)"""
    d = d.copy()
    f = lambda c: d[c].fillna(0)
    d['phys_shortfall'] = (-f('dev_Filling_Time') + f('dev_Max_Injection_Speed')
                           - f('dev_Max_Injection_Pressure')) / 3          # 재료부족
    d['phys_cushion_low'] = -f('dev_Cushion_Position')                       # 쿠션부족
    d['phys_metering'] = f('dev_Plasticizing_Time')                          # 계량지연
    return d


PHYS_FSETS = {'phys': PHYS_BASE + PHYS_DERIVED, 'phys_core': PHYS_DERIVED}
PHYS_SUP = {'sup_phys': PHYS_BASE + PHYS_DERIVED + ['log_since_stop', 'is_transient']}
COMPARE_FSETS = {'free': FSETS['free']}
COMPARE_SUP = {'sup_free': SUP_FSETS['sup_free']}


def rule_score(d):
    """학습 없는 물리 규칙: 세 지표 중 최대 (사전 고정)"""
    return d[PHYS_DERIVED].max(axis=1).values


def run_phys_models(FEAT, REF, kinds=('iforest', 'maha', 'knn')):
    """
    FEAT, REF: {'cn7': df, 'rg3': df} (add_phys 적용 후)
    반환: (OOF 전체, REF_Z, SUP_REF)
    """
    parts, REF_Z = [], {}
    for p in FEAT:
        o, rz = run_unsupervised(FEAT[p], REF[p], p, fsets={**COMPARE_FSETS, **PHYS_FSETS}, kinds=kinds)
        # 물리 규칙 (reference 척도로 변환)
        s_ref = rule_score(REF[p])
        sc, pc = to_ref_scale(rule_score(FEAT[p]), s_ref)
        rz['U_rule|phys'] = to_ref_scale(s_ref, s_ref)[0]
        parts += [o, _oof_frame(FEAT[p], p, 'U_rule|phys', sc, pc)]
        REF_Z[p] = rz
    oof_s, SUP_REF = run_pooled_supervised(FEAT, REF, fsets={**COMPARE_SUP, **PHYS_SUP})
    ALL = pd.concat(parts + [oof_s], ignore_index=True)

    ens = {'E2_free': ['U_iforest|free', 'U_maha|free', 'U_knn|free', 'S_logit|sup_free'],
           'E2_phys': ['U_iforest|phys', 'U_maha|phys', 'U_knn|phys', 'S_logit|sup_phys'],
           'E_phys+rule': ['U_knn|phys', 'U_rule|phys', 'S_logit|sup_phys']}
    E = [ensemble(ALL, m, n, REF_Z, SUP_REF) for n, m in ens.items()]
    return pd.concat([ALL] + E, ignore_index=True), REF_Z, SUP_REF
