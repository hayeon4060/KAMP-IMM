"""확률 층 (사전등록 docs/PREREG_BETA_W.md)

최종 점수(LW Mahalanobis) → 풀링 로지스틱(log 점수) → Venn-Abers 보정·구간 [p0, p1]
→ 측면 배분 w (Beta 사전 κ=10, 외부 ERR 집계 w_prior.json) + 캐비티 기저(Jeffreys, 전향은 비경보 행 순차 갱신)
→ 행(부품) 확률 p = 1 − (1 − w·p_shot)(1 − p_base), 구간은 p0/p1을 같은 식에 통과.
보정 OOF = 공식 Run 단위 leave-one-run-out(두 제품 풀링). 전향 = 공식 전체 적합 고정.
출력: outputs/prob/prob_results_beta.pkl, prob_summary.json
"""
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from paths import FINAL, LINKED, PROB, W_PRIOR
from shots_data import load_all

warnings.filterwarnings('ignore')

EV = ['Max_Injection_Speed', 'Filling_Time', 'Max_Injection_Pressure', 'Max_Switch_Over_Pressure',
      'Average_Back_Pressure', 'Cushion_Position', 'Plasticizing_Time', 'Cycle_Time']
SHORT = {'Max_Injection_Speed': '사출속도', 'Filling_Time': '충전시간', 'Max_Injection_Pressure': '사출압력',
         'Max_Switch_Over_Pressure': '전환압력', 'Average_Back_Pressure': '평균배압',
         'Cushion_Position': '쿠션', 'Plasticizing_Time': '계량시간', 'Cycle_Time': '사이클'}
PCT_BASE, TAU_Q = 0.5, 0.95
KAPPA = float(os.environ.get('KAPPA', '10'))      # Beta 사전 강도 (사전등록 기본 10)


def venn_abers(z_cal, y_cal, z_test):
    z_cal = np.asarray(z_cal, float); y_cal = np.asarray(y_cal, float)
    p0 = np.empty(len(z_test)); p1 = np.empty(len(z_test))
    for i, zt in enumerate(np.asarray(z_test, float)):
        for lab, out in ((0.0, p0), (1.0, p1)):
            ir = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip')
            ir.fit(np.append(z_cal, zt), np.append(y_cal, lab))
            out[i] = ir.predict([zt])[0]
    return p0, p1


def feats(d):
    """로지스틱 입력: log(점수) (맥락 항은 사전등록 판정 규칙 ①에 따라 제외)"""
    return np.column_stack([np.log(np.maximum(d['score'].values, 1e-6))])


def fit_logistic(X, y):
    return LogisticRegression(C=1e6, max_iter=2000).fit(X, y)


def brier_logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4); y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def f1_at(y, s, q=0.05):
    y = np.asarray(y, int); s = np.asarray(s, float)
    k = int(np.ceil(q * len(y))); o = np.argsort(-s, kind='stable')
    tp = int(y[o[:k]].sum())
    return 2 * tp / (k + y.sum()) if (k + y.sum()) else 0.0, tp, k


def load_shots(p):
    sh = pd.read_csv(FINAL / f'{p}_scores.csv')
    sh = sh[sh.split != '제외'].reset_index(drop=True)
    sh['prod'] = p
    sh['since_start'] = sh.groupby('run_id').cumcount()
    sh['y_shot'] = sh.PassOrFail.astype(int)
    devs = sh[[f'dev_{v}' for v in EV]].values
    order = np.argsort(-np.abs(np.nan_to_num(devs)), 1)[:, :3]
    sh['top3'] = [', '.join(f'{SHORT[EV[j]]}({devs[i, j]:+.1f})' for j in order[i]) for i in range(len(sh))]
    rel = []
    for fa, fr in zip(sh['fill_anomaly'].values, sh['dev_flow_res'].values):
        parts = []
        if np.isfinite(fa) and abs(fa) >= 3: parts.append('충전시간×사출속도')
        if np.isfinite(fr) and abs(fr) >= 3: parts.append('사출압력/속도(흐름저항)')
        rel.append(' + '.join(parts))
    sh['relation'] = rel
    return sh


def load_rows(p, L):
    norm = lambda s: pd.to_datetime(s).dt.strftime('%Y-%m-%d %H:%M:%S')
    rows = pd.read_csv(LINKED / 'linked_labeled.csv', parse_dates=['TimeStamp'])
    rows = rows[rows['product'] == p].copy()
    rows['TimeStamp'] = rows['TimeStamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    rows['y'] = rows.PassOrFail.astype(int)
    Le = L[(L['prod'] == p.upper()) & (L.period == 'extra')]
    ext_rows = pd.DataFrame({'product': p, 'TimeStamp': norm(Le.TimeStamp).values,
                             'cavity': Le['side'].values, 'Reason': Le.reason.values, 'y': Le.y.values})
    Lo = L[(L['prod'] == p.upper()) & (L.period == 'official')]
    oid = Lo.set_index([norm(Lo.TimeStamp), Lo['side']]).oid
    rows['oid'] = rows.set_index(['TimeStamp', 'cavity']).index.map(oid)
    assert set(rows.cavity.unique()) <= {'LH', 'RH'}
    return rows, ext_rows


def main():
    L = load_all()['L']
    prior_all = json.load(open(W_PRIOR))
    SH = {p: load_shots(p) for p in ('cn7', 'rg3')}
    OFF = pd.concat([SH[p][SH[p].split == '2차'] for p in SH], ignore_index=True)
    X_off, y_off = feats(OFF), OFF['y_shot'].values

    # ---- 보정 OOF: 공식 Run 단위 leave-one-run-out (풀링) ----
    p0 = np.full(len(OFF), np.nan); p1 = np.full(len(OFF), np.nan)
    for r in OFF.run_id.unique():
        te = (OFF.run_id == r).values; tr = ~te
        m = fit_logistic(X_off[tr], y_off[tr])
        p0[te], p1[te] = venn_abers(m.decision_function(X_off[tr]), y_off[tr], m.decision_function(X_off[te]))
    OFF['p0'], OFF['p1'] = p0, p1
    OFF['p_shot'] = p1 / (1 - p0 + p1)

    # ---- 전향: 공식 전체 적합 고정 ----
    m_full = fit_logistic(X_off, y_off)
    z_cal = m_full.decision_function(X_off)
    coefs = dict(zip(['log_score'], m_full.coef_[0].round(3).tolist()))
    coefs['intercept'] = round(float(m_full.intercept_[0]), 3)

    R = {}
    for p in ('cn7', 'rg3'):
        sh = SH[p]
        off = OFF[OFF['prod'] == p].reset_index(drop=True)
        ext = sh[sh.split == '추가'].reset_index(drop=True)
        tau = float(np.quantile(off['score'], TAU_Q))
        if len(ext):
            e0, e1 = venn_abers(z_cal, y_off, m_full.decision_function(feats(ext)))
            ext['p0'], ext['p1'] = e0, e1
            ext['p_shot'] = e1 / (1 - e0 + e1)

        rows_off, rows_ext = load_rows(p, L)
        cols = ['TimeStamp', 'run_id', 'score', 'pct', 'p_shot', 'p0', 'p1', 'since_start', 'top3', 'relation']
        r_off = rows_off.merge(off[cols], on='TimeStamp', how='left')

        # 측면 배분: Beta 사후평균 = (κ·share_hist + n_s) / (κ + n_불량쇼트)
        dsh = r_off[r_off['y'] == 1].groupby('TimeStamp')['cavity'].agg(set)
        w_raw = {s: (float(np.mean([s in c for c in dsh])) if len(dsh) else 0.5) for s in ['LH', 'RH']}
        prior = prior_all[p]
        n_def = len(dsh)
        w = {s: (KAPPA * prior[f'share_{s}'] + sum(s in c for c in dsh)) / (KAPPA + n_def) for s in ['LH', 'RH']}

        # 기저: 공식 pct<0.5 행, Jeffreys
        pool = r_off[r_off['pct'] < PCT_BASE]
        ab0 = {s: [pool[pool.cavity == s]['y'].sum() + 0.5, len(pool[pool.cavity == s]) + 1.0] for s in ['LH', 'RH']}
        p_base = {s: ab0[s][0] / ab0[s][1] for s in ['LH', 'RH']}

        def rowp(rr, pb_map):
            ws = rr['cavity'].map(w).values; pb = rr['cavity'].map(pb_map).values
            f = lambda ps: 1 - (1 - ws * np.nan_to_num(ps)) * (1 - pb)
            return f(rr['p_shot'].values), f(rr['p0'].values), f(rr['p1'].values)

        r_off['p'], r_off['p_lo'], r_off['p_hi'] = rowp(r_off, p_base)
        na = r_off['p_shot'].isna()
        for c in ('p', 'p_lo', 'p_hi'):
            r_off.loc[na, c] = r_off.loc[na, 'cavity'].map(p_base)

        # 전향 행: 기저 순차 갱신(비경보 쇼트 행만), prequential
        if len(ext):
            r_ext = rows_ext.merge(ext[cols], on='TimeStamp', how='inner')
            r_ext = r_ext.sort_values(['TimeStamp', 'cavity']).reset_index(drop=True)
            ab = {s: list(ab0[s]) for s in ['LH', 'RH']}
            P = np.empty((len(r_ext), 3))
            for t, g in r_ext.groupby('TimeStamp', sort=True):
                for i, rr in g.iterrows():
                    pb = ab[rr.cavity][0] / ab[rr.cavity][1]; ws = w[rr.cavity]
                    P[i] = [1 - (1 - ws * x) * (1 - pb) for x in (rr.p_shot, rr.p0, rr.p1)]
                if g['score'].iloc[0] < tau:
                    for i, rr in g.iterrows():
                        ab[rr.cavity][0] += float(rr.y); ab[rr.cavity][1] += 1.0
            r_ext['p'], r_ext['p_lo'], r_ext['p_hi'] = P[:, 0], P[:, 1], P[:, 2]
        else:
            r_ext = pd.DataFrame()

        # ---- 평가 ----
        base_rate = float(r_off['y'].mean())
        res = {}
        for tag, rr in (('official_oof', r_off), ('prospective', r_ext)):
            if not len(rr): continue
            y = rr['y'].values
            b, ll = brier_logloss(y, rr['p'].values)
            b0, ll0 = brier_logloss(y, np.full(len(y), base_rate))
            prank = rr['p'].values + 1e-9 * np.nan_to_num(rr['score'].values)   # isotonic 동률 깨기(2차 키=점수)
            f1p, tpp, k = f1_at(y, prank)
            f1s, tps, _ = f1_at(y, rr['score'].fillna(0).values)
            res[tag] = dict(n=len(y), defects=int(y.sum()), brier=round(b, 5), brier_base=round(b0, 5),
                            BSS=round(1 - b / b0, 3), logloss=round(ll, 4), logloss_base=round(ll0, 4),
                            F1_5pct_prob=round(f1p, 3), TP_prob=tpp, F1_5pct_score=round(f1s, 3),
                            TP_score=tps, k=k, interval_width=round(float(np.nanmean(rr['p_hi'] - rr['p_lo'])), 5))
        R[p] = dict(summary=dict(prod=p, tau=tau, coefs=coefs, w_side={k: round(v, 3) for k, v in w.items()},
                                 w_raw={k: round(v, 3) for k, v in w_raw.items()}, kappa=KAPPA,
                                 p_base={k: round(v, 5) for k, v in p_base.items()},
                                 p_base_ext_final=({s: round(ab[s][0] / ab[s][1], 5) for s in ab} if len(ext) else None),
                                 base_rate_row_off=round(base_rate, 5),
                                 rows_without_score_off=int(r_off['p_shot'].isna().sum()), rows=res),
                    off=off, ext=ext, rows_off=r_off, rows_ext=r_ext)
        s = R[p]['summary']['rows']
        print(f'  [{p}] w={R[p]["summary"]["w_side"]} | ' +
              ' | '.join(f'{k}: BSS {v["BSS"]} (Brier {v["brier"]} vs {v["brier_base"]})' for k, v in s.items()))
    pickle.dump(R, open(PROB / 'prob_results_beta.pkl', 'wb'))
    json.dump({p: R[p]['summary'] for p in R}, open(PROB / 'prob_summary.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    return R


if __name__ == '__main__':
    main()
