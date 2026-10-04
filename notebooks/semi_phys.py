"""
semi_phys.py — 준지도 학습(가짜 라벨) 실험: 성능이 실제로 오르는지 '정직하게' 확인

설계 (누수 방지가 핵심)
  - LORO fold마다:
      ① 학습 Run(라벨)으로 통합 지도모델 학습 (물리 변수 sup_phys)
      ② 그 모델로 unlabeled reference(9~10월, labeled 샷·Run 시간창 제거) 점수 계산
      ③ 가짜 라벨: 점수 상위 q_pos → '가짜 불량', 하위 q_neg 이하 → '가짜 정상'
         (옵션) 가짜 불량은 물리 규칙도 상위 1%인 샷만 = 물리적으로 말이 되는 것만
      ④ 라벨 + 가짜 라벨(가중치 w_pseudo)로 다시 학습 (n_iter회 반복 가능)
      ⑤ 테스트 Run 예측 → 평가는 '진짜 라벨'로만
  - 테스트 Run은 ①~④ 어디에도 안 들어감 (reference에서 labeled 시간창이 이미 제거됨)

비교 (같은 fold·같은 평가)
  S0 라벨만 / S1 + 가짜 정상 / S2 + 가짜 정상·불량 / S3 + 가짜 정상·불량(물리 필터)

진단
  - fold별 가짜 라벨 수
  - 가짜 불량이 진짜 불량과 물리 지표가 비슷한가 (그럴듯함 점검)
  - 날짜별 가짜 불량 수 vs ERR_FACT_QTY (컬럼 의미 확인 후 해석)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model_v2 import make_sup, _X, to_ref_scale, _oof_frame
from model_phys import PHYS_SUP, PHYS_DERIVED, rule_score
from plot_style import set_style, INK, SUB, MUTED, NORMAL, TYPE_COL, ACCENT

SEED = 42
COLS = PHYS_SUP['sup_phys']
VARIANTS = {
    'S0_라벨만':            dict(neg=False, pos=False, phys=False),
    'S1_+가짜정상':          dict(neg=True, pos=False, phys=False),
    'S2_+가짜정상·불량':      dict(neg=True, pos=True, phys=False),
    'S3_+가짜정상·불량(물리)': dict(neg=True, pos=True, phys=True),
}


def _fit(model, X, y, w):
    m = make_sup(model)
    m.fit(X, y, **{m.steps[-1][0] + '__sample_weight': w})
    return m


def _product_weights(tr):
    cnt = tr['product'].map(tr['product'].value_counts())
    return (len(tr) / tr['product'].nunique() / cnt).values


def pseudo_masks(s_ref, rule_ref, q_pos, q_neg, phys, rule_q=0.99):
    pos = s_ref >= np.quantile(s_ref, q_pos)
    if phys:
        pos &= rule_ref >= np.quantile(rule_ref, rule_q)
    neg = s_ref <= np.quantile(s_ref, q_neg)
    return pos, neg


def run_semi(FEAT, REF, variants=VARIANTS, model='logit', q_pos=0.995, q_neg=0.5, w_pseudo=0.3,
             n_iter=1, max_neg=5000, seed=SEED):
    """
    FEAT, REF: {'cn7','rg3'} — add_phys 적용된 labeled feature / reference feature
    반환: (OOF, fold별 가짜 라벨 통계)
    """
    rng = np.random.default_rng(seed)
    A = pd.concat([FEAT[p].assign(product=p) for p in FEAT], ignore_index=True)
    XR = {p: _X(REF[p], COLS) for p in REF}
    RR = {p: rule_score(REF[p]) for p in REF}
    parts, stats = [], []
    for test_run in A['run_id'].unique():
        tr, te = A[A['run_id'] != test_run], A[A['run_id'] == test_run]
        p_te = te['product'].iloc[0]
        Xtr, ytr, wtr = _X(tr, COLS), tr['PassOrFail'].values, _product_weights(tr)
        base = _fit(model, Xtr, ytr, wtr)
        for vname, cfg in variants.items():
            m = base
            n_pos = n_neg = 0
            if cfg['neg'] or cfg['pos']:
                for _ in range(n_iter):
                    Xs, ys, ws = [Xtr], [ytr], [wtr]
                    n_pos = n_neg = 0
                    for p in REF:
                        s_ref = m.predict_proba(XR[p])[:, 1]
                        pos, neg = pseudo_masks(s_ref, RR[p], q_pos, q_neg, cfg['phys'])
                        if cfg['pos'] and pos.any():
                            Xs.append(XR[p][pos]); ys.append(np.ones(pos.sum(), int))
                            ws.append(np.full(pos.sum(), w_pseudo)); n_pos += int(pos.sum())
                        if cfg['neg']:
                            idx = np.flatnonzero(neg)
                            if len(idx) > max_neg:
                                idx = rng.choice(idx, max_neg, replace=False)
                            Xs.append(XR[p][idx]); ys.append(np.zeros(len(idx), int))
                            ws.append(np.full(len(idx), w_pseudo)); n_neg += len(idx)
                    m = _fit(model, np.vstack(Xs), np.concatenate(ys), np.concatenate(ws))
            s_te = m.predict_proba(_X(te, COLS))[:, 1]
            s_ref = m.predict_proba(XR[p_te])[:, 1]
            sc, pc = to_ref_scale(s_te, s_ref)
            parts.append(_oof_frame(te, p_te, f'{vname}|{model}', sc, pc))
            stats.append({'test_run': test_run, 'variant': vname, 'model': model,
                          '학습 진짜 불량': int(ytr.sum()), '가짜 불량': n_pos, '가짜 정상': n_neg})
        print(f'  {test_run} 완료')
    return pd.concat(parts, ignore_index=True), pd.DataFrame(stats)


# ====================================================================== 진단 (전체 labeled로 학습한 모델 기준 — 평가용 아님)
def pseudo_diagnostics(FEAT, REF, model='logit', q_pos=0.995, phys=False):
    """반환: {p: reference에 가짜 불량 표시한 DataFrame}, 비교표(가짜 불량 vs 진짜 불량 물리 지표)"""
    A = pd.concat([FEAT[p].assign(product=p) for p in FEAT], ignore_index=True)
    m = _fit(model, _X(A, COLS), A['PassOrFail'].values, _product_weights(A))
    out, rows = {}, []
    for p in REF:
        s = m.predict_proba(_X(REF[p], COLS))[:, 1]
        pos, _ = pseudo_masks(s, rule_score(REF[p]), q_pos, 0.5, phys)
        r = REF[p][['run_id', 'TimeStamp'] + PHYS_DERIVED].copy()
        r['score'], r['가짜불량'] = s, pos
        out[p] = r
        lab = FEAT[p]
        for name, df in [('reference 전체', r), ('가짜 불량', r[r['가짜불량']]),
                         ('진짜 정상', lab[lab['PassOrFail'] == 0]),
                         ('진짜 가스', lab[lab['Reason'] == '가스']),
                         ('진짜 미성형계열', lab[lab['Reason'].isin(['미성형', '가스+미성형'])])]:
            row = {'제품': p, '그룹': name, 'n': len(df)}
            row.update({c: df[c].median() for c in PHYS_DERIVED})
            rows.append(row)
    return out, pd.DataFrame(rows)


def err_fact_table(unl_path, ref_marked):
    """
    reference 샷에 생산계획일·ERR_FACT_QTY를 붙여 날짜별로: 샷 수, 가짜 불량 수, ERR_FACT_QTY(고유값)
    ※ ERR_FACT_QTY 의미(일별 불량 수량? 누적?)부터 확인: 같은 날짜 안에서 값이 하나인지 '고유값 수'로 봄
    """
    u = pd.read_csv(unl_path, parse_dates=['TimeStamp'], low_memory=False,
                    usecols=lambda c: c in ('TimeStamp', 'PART_FACT_PLAN_DATE', 'ERR_FACT_QTY', 'cavity'))
    u = u.drop_duplicates('TimeStamp')
    r = ref_marked.merge(u, on='TimeStamp', how='left')
    g = r.groupby('PART_FACT_PLAN_DATE')
    t = pd.DataFrame({'샷': g.size(), '가짜 불량': g['가짜불량'].sum(),
                      'ERR_FACT_QTY 고유값 수': g['ERR_FACT_QTY'].nunique(),
                      'ERR_FACT_QTY 최소': g['ERR_FACT_QTY'].min(),
                      'ERR_FACT_QTY 최대': g['ERR_FACT_QTY'].max()})
    t['가짜 불량률'] = t['가짜 불량'] / t['샷']
    return t.reset_index()


def plot_semi_compare(summ, product, metric_cols=('AP', 'R@10%'), save=None):
    set_style()
    s = summ[summ['product'] == product]
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5.5 * len(metric_cols), 3.6))
    for ax, mcol in zip(np.atleast_1d(axes), metric_cols):
        y = np.arange(len(s))
        ax.barh(y, s[mcol], color=[MUTED if m.startswith('S0') else ACCENT for m in s['method']], height=0.6)
        if mcol == 'AP':
            ax.hlines(y, s['AP_lo'], s['AP_hi'], color=INK, lw=1)
        ax.set_yticks(y); ax.set_yticklabels(s['method'], fontsize=8)
        ax.set_title(mcol, loc='left', fontsize=10)
        ax.grid(axis='y', visible=False)
    fig.suptitle(f'{product.upper()} — 가짜 라벨 추가 효과 (회색 = 라벨만)', x=0.01, ha='left', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    if save:
        fig.savefig(save)
    plt.show()
