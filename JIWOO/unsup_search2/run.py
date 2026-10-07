"""
CN7 공식 기간 '같은 검사량' 개선 탐색 — PREREG.md 고정 규칙 그대로 실행 (탐색)

후보 10개(A1~A3, B1·B2, C1~C3, D1·D2) × 학습 구간 2개(W1 = 10/16 이전, W2 = 평가일 3일 제외)
비교: R0 제출(원장), R1 GMM·dev26, R2 제출 방식(c17 + LW-Mahalanobis) 재학습
출력: outputs/ (git 제외) — settings.csv, top5_defects.csv, null.json
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'unsup_search'))
import common as C  # noqa: E402
import models as M  # noqa: E402
from s1_slices import for_product  # noqa: E402

OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
P = 'cn7'
BUDGETS = {'0.8%': 10, '1.65%': 20, '5%': 61}
EVENT = pd.to_datetime(['2020-10-16 05:55:21', '2020-10-16 05:56:22', '2020-10-16 05:57:20'])
EVAL_DAYS = [pd.Timestamp(d).date() for d in ('2020-10-16', '2020-10-19', '2020-10-20')]
SEEDS = range(5)
Q_ALARM = 0.999
N_NULL = 1000
raw = C.raw


# ====================================================================== 새 물리 관계 12개 (PREREG 표)
def add_new_relations(d):
    """d: Run → 시간 순 정렬된 샷 표 (*_raw 포함). n1~n12 원값 → 인과 편차 dv_n*"""
    d = d.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
    with np.errstate(all='ignore'):
        lg = lambda c: np.log(d[raw(c)].where(d[raw(c)] > 0))
        hold = (d[raw('Injection_Time')] - d[raw('Filling_Time')])
        rel = {
            'n1': lg('Max_Injection_Pressure') + lg('Filling_Time'),
            'n2': lg('Max_Switch_Over_Pressure') + np.log(hold.where(hold > 0)),
            'n3': lg('Max_Injection_Pressure') - lg('Max_Injection_Speed'),
            'n4': lg('Max_Screw_RPM') + lg('Plasticizing_Time'),
            'n5': lg('Average_Back_Pressure') + lg('Plasticizing_Time'),
        }
        for i, (a, b) in enumerate([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], start=6):
            rel[f'n{i}'] = d[raw(f'Barrel_Temperature_{a}')] - d[raw(f'Barrel_Temperature_{b}')]
        rel['n11'] = d[raw('Mold_Temperature_3')] - d[raw('Mold_Temperature_4')]
        rel['n12'] = d[raw('Hopper_Temperature')] - d[raw('Barrel_Temperature_6')]
    for k, v in rel.items():
        d[k] = pd.Series(v, index=d.index).replace([np.inf, -np.inf], np.nan)
        step = 0.001 if k in ('n1', 'n2', 'n3', 'n4', 'n5') else 0.1
        d[f'dv_{k}'] = C.fm._by_run(d, k, lambda a, s=step: C.fm.causal_robust_dev(a, s)).values
    return d


NEW = [f'dv_n{i}' for i in range(1, 13)]
DEV26 = C.FEAT_DEV + C.FEAT_REL
DEV38 = DEV26 + NEW


def Xof(d, cols):
    return d[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)


def pct_of(ref_scores, s):
    r = np.sort(ref_scores)
    return np.searchsorted(r, np.asarray(s, float), side='right') / (len(r) + 1)


# ====================================================================== 점수기 (학습 → 라벨 샷 점수, 학습 구간 점수)
def gmm(ref, lab, cols):
    pr = M.Prep().fit(Xof(ref, cols))
    A, B = pr(Xof(ref, cols)), pr(Xof(lab, cols))
    sl, sr = [], []
    for sd in SEEDS:
        m = M.GMM(2, seed=sd).fit(A)
        sl.append(m.score(B)); sr.append(m.score(A))
    return np.mean(sl, 0), np.mean(sr, 0)


def lw(ref, lab, cols):
    pr = M.Prep().fit(Xof(ref, cols))
    A, B = pr(Xof(ref, cols)), pr(Xof(lab, cols))
    m = M.MahaLW().fit(A)
    return m.score(B), m.score(A)


def c17_submission_style(ref, lab):
    m = C.fm.FinalModel(C.FEAT_C17).fit(ref)
    return m.score(lab), m.score(ref)


def local_maha(ref, lab, cols, k):
    pr = M.Prep().fit(Xof(ref, cols))
    A, B = pr(Xof(ref, cols)), pr(Xof(lab, cols))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(A)

    def score(Q, drop_self):
        idx = nn.kneighbors(Q, return_distance=False)
        idx = idx[:, 1:] if drop_self else idx[:, :k]
        out = np.empty(len(Q))
        for i, q in enumerate(Q):
            nb = A[idx[i]]
            lwm = LedoitWolf().fit(nb)
            out[i] = np.sqrt(lwm.mahalanobis(q[None])[0])
        return out
    return score(B, False), score(A, True)


def lof(ref, lab, cols, k):
    pr = M.Prep().fit(Xof(ref, cols))
    A, B = pr(Xof(ref, cols)), pr(Xof(lab, cols))
    m = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(A)
    return -m.score_samples(B), -m.negative_outlier_factor_


# ====================================================================== 평가 (공식 행)
def official_rows():
    l = pd.read_csv(C.FINAL_INTER / 'linked' / 'linked_labeled.csv', parse_dates=['TimeStamp'])
    e = l[(l['product'] == P) & l['in_secondary']].copy()
    e['Reason'] = e['Reason'].fillna('정상')
    e = e.sort_values(['TimeStamp', 'cavity']).reset_index(drop=True)     # 동점 깨기: 시간 → LH → RH
    assert len(e) == 1211 and int(e['PassOrFail'].sum()) == 17
    return e


def order_from_score(s):
    return np.argsort(-np.asarray(s, float), kind='stable')


def ap_from_order(y, o):
    ys = y[o]
    return float((ys * np.cumsum(ys) / np.arange(1, len(ys) + 1)).sum() / ys.sum())


def rows_needed(y, o, n):
    c = np.cumsum(y[o])
    return int(np.argmax(c >= n) + 1) if c[-1] >= n else None


def policy_order(e, shots, alarm_shot, alarm_score, fill_score):
    """D: ① 경보 행(경보 성분 점수 순) ② 소급 RH(경보 3연속 → 첫 쇼트 직전 10쇼트 RH, 가까운 순) ③ 채우기 점수 순"""
    ts = e['TimeStamp'].values
    order = []
    al_rows = np.flatnonzero(np.isin(ts, shots.loc[alarm_shot, 'TimeStamp'].values))
    order += list(al_rows[np.argsort(-alarm_score[al_rows], kind='stable')])
    retro_ts = []
    for _, g in shots.groupby('run_id', sort=False):
        a = np.asarray(alarm_shot)[g.index.values]
        streak = 0
        for j in range(len(g)):
            streak = streak + 1 if a[j] else 0
            if streak == 3:
                first = j - 2
                retro_ts += list(g['TimeStamp'].values[max(0, first - 10):first][::-1])
    for t in retro_ts:
        r = np.flatnonzero((ts == t) & (e['cavity'].values == 'RH'))
        order += [i for i in r if i not in order]
    n_rule = len(order)
    rest = np.setdiff1d(np.arange(len(e)), order)
    order += list(rest[np.argsort(-fill_score[rest], kind='stable')])
    return np.array(order), n_rule


def main():
    F = C.build_cache()
    LAB = C.build_labeled_feats()
    ws = (F['equip'] == C.EQUIP_W2) & (F['prod'] == 'CN7')
    U = add_new_relations(for_product(F[ws], P))
    lab = add_new_relations(for_product(LAB[P], P))
    t = U['TimeStamp']
    WIN = {'W1': U[t < '2020-10-16'], 'W2': U[~t.dt.date.isin(EVAL_DAYS)]}
    C.log(f'학습 구간 W1 {len(WIN["W1"]):,}샷, W2 {len(WIN["W2"]):,}샷')

    e = official_rows()
    y = e['PassOrFail'].values.astype(int)
    shot_map = lambda s: e['TimeStamp'].map(pd.Series(np.asarray(s, float), index=lab['TimeStamp'].values)).values
    shots = lab[lab['split'] == '2차'].sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
    shot_pos = pd.Series(np.arange(len(lab)), index=lab['TimeStamp'].values)

    settings = {}   # name → dict(order, ref_cut_fn, window, kind)

    def add_score(name, win, kind, s_lab, s_ref, null=True):
        rs = shot_map(s_lab)
        settings[name] = dict(order=order_from_score(rs), row_score=rs, ref=np.asarray(s_ref), win=win, kind=kind, null=null)

    for wn, ref in WIN.items():
        C.log(f'--- {wn}')
        r1 = gmm(ref, lab, DEV26); add_score(f'R1 GMM·dev26|{wn}', wn, '비교', *r1)
        r2 = c17_submission_style(ref, lab); add_score(f'R2 제출방식 재학습|{wn}', wn, '비교', *r2)
        add_score(f'A1 dev38 GMM|{wn}', wn, 'A', *gmm(ref, lab, DEV38))
        add_score(f'A2 dev38 LW-Maha|{wn}', wn, 'A', *lw(ref, lab, DEV38))
        add_score(f'A3 n12만 GMM|{wn}', wn, 'A', *gmm(ref, lab, NEW))
        # B: 각자 학습 구간 기준 백분위
        p1l, p1r = pct_of(r1[1], r1[0]), pct_of(r1[1], r1[1])
        p2l, p2r = pct_of(r2[1], r2[0]), pct_of(r2[1], r2[1])
        add_score(f'B1 순위평균(R2,R1)|{wn}', wn, 'B', (p1l + p2l) / 2, (p1r + p2r) / 2)
        add_score(f'B2 최대(R2,R1)|{wn}', wn, 'B', np.maximum(p1l, p2l), np.maximum(p1r, p2r))
        C.log('C 국소 밀도 계산')
        add_score(f'C1 국소Maha k100|{wn}', wn, 'C', *local_maha(ref, lab, DEV26, 100))
        add_score(f'C2 국소Maha k300|{wn}', wn, 'C', *local_maha(ref, lab, DEV26, 300))
        add_score(f'C3 LOF k10|{wn}', wn, 'C', *lof(ref, lab, DEV26, 10))
        # D: 정책
        r1_row, r2_row = shot_map(r1[0]), shot_map(r2[0])
        a1 = pd.Series(pct_of(r1[1], r1[0])[shot_pos[shots['TimeStamp']].values] >= Q_ALARM)
        a2 = pd.Series(pct_of(r2[1], r2[0])[shot_pos[shots['TimeStamp']].values] >= Q_ALARM)
        for nm, al, al_sc, fill, fill_ref, null in (
                (f'D1 GMM경보+소급RH+GMM|{wn}', a1, r1_row, r1_row, r1[1], True),
                (f'D2 제출방식경보+소급RH+GMM|{wn}', a2, r2_row, r1_row, r1[1], True),
                (f'(참고) 제출방식경보+소급RH+제출방식|{wn}', a2, r2_row, r2_row, r2[1], False)):
            o, n_rule = policy_order(e, shots, al.values, al_sc, fill)
            settings[nm] = dict(order=o, row_score=fill, ref=np.asarray(fill_ref), win=wn,
                                kind='D' if null else '참고', null=null, n_rule=n_rule)

    # R0 원장 (고정 점수) + 원장 경보 소급 규칙 (재현 확인)
    Lg = pd.read_csv(C.FINAL / 'outputs' / 'ledger' / 'ledger_v1.csv', parse_dates=['TimeStamp'])
    Lg = Lg[(Lg['product'] == 'CN7') & (Lg['period'] == 'official')]
    key = lambda d: d['TimeStamp'].astype(str) + d['cavity']
    sc0 = key(e).map(pd.Series(Lg['score'].values, index=key(Lg).values)).values
    al0 = key(e).map(pd.Series(Lg['alarm'].values, index=key(Lg).values)).values.astype(int)
    assert al0.sum() == 10, al0.sum()
    ref0 = c17_submission_style(U[(t >= '2020-09-01')], lab)[1]      # 제출 학습 구간(09-01~) 기준 데이터
    settings['R0 제출(원장)'] = dict(order=order_from_score(sc0), row_score=sc0, ref=ref0, win='원장', kind='비교', null=False)
    al_shot0 = shots['TimeStamp'].isin(e.loc[al0 == 1, 'TimeStamp']).values
    o, n_rule = policy_order(e, shots, al_shot0, sc0, sc0)
    settings['(참고) 원장경보+소급RH+원장'] = dict(order=o, row_score=sc0, ref=ref0, win='원장', kind='참고', null=False,
                                         n_rule=n_rule)

    # ------------------------------------------------------------------ 지표
    rows, top5 = [], []
    for name, s in settings.items():
        o = s['order']
        r = {'설정': name, '구간': s['win'], '계열': s['kind'], 'AP': ap_from_order(y, o)}
        for b, k in BUDGETS.items():
            tp = int(y[o[:k]].sum())
            r[f'TP@{b}'] = tp
            r[f'F1@{b}'] = 2 * tp / (k + y.sum())
        cut = s['row_score'][o[60]]                         # 5% 경보선 (61번째 행 점수)
        r['5%경보선_정상초과율'] = float((s['ref'] >= cut).mean())
        if 'n_rule' in s:
            r['규칙 행 수'] = s['n_rule']
            if s['n_rule'] > 60:
                r['5%경보선_정상초과율'] = np.nan
        r['rows_needed_10'] = rows_needed(y, o, 10)
        rank = np.empty(len(o), int); rank[o] = np.arange(1, len(o) + 1)
        r['_rank'] = rank
        rows.append(r)
        for i in np.flatnonzero((y == 1) & (rank <= 61)):
            top5.append({'설정': name, '시각': e.loc[i, 'TimeStamp'].strftime('%H:%M:%S'), '측면': e.loc[i, 'cavity'],
                         '유형': e.loc[i, 'Reason'], '순위': int(rank[i]),
                         '사건쇼트': e.loc[i, 'TimeStamp'] in set(EVENT)})
    R = pd.DataFrame(rows)

    # ------------------------------------------------------------------ 판정 (PREREG 고정)
    def crit1(name, base):
        a = R.set_index('설정').loc[name]; b = R.set_index('설정').loc[base]
        if a['TP@5%'] > b['TP@5%']:
            return True, '더 많이'
        need_a = rows_needed(y, settings[name]['order'], int(b['TP@5%']))
        need_b = rows_needed(y, settings[base]['order'], int(b['TP@5%']))
        if a['TP@5%'] == b['TP@5%'] and need_a is not None and need_b is not None and need_a < need_b:
            return True, f'같은 수를 {need_a}행에 (기준 {need_b}행)'
        return False, ''

    def gained(name, base, k=61):
        ra, rb = R.set_index('설정').loc[name, '_rank'], R.set_index('설정').loc[base, '_rank']
        idx = np.flatnonzero((y == 1) & (ra <= k) & (rb > k))
        return [(e.loc[i, 'TimeStamp'], e.loc[i, 'cavity'], int(ra[i]), int(rb[i])) for i in idx]

    verdict = []
    for name in R['설정']:
        if not name[:2] in ('A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2', 'C3', 'D1', 'D2') or not name.endswith('|W1'):
            continue
        cand = name[:-3]
        ok1, how1 = crit1(f'{cand}|W1', 'R1 GMM·dev26|W1')
        ok2, how2 = crit1(f'{cand}|W2', 'R1 GMM·dev26|W2')
        g = gained(f'{cand}|W1', 'R1 GMM·dev26|W1')
        g_ne = [x for x in g if x[0] not in set(EVENT)]
        ok3 = ok1 and (len(g_ne) > 0 if how1 == '더 많이' else True)
        verdict.append({'후보': cand, '판정1(W1)': f'{"O" if ok1 else "X"} {how1}', '판정2(W2)': f'{"O" if ok2 else "X"} {how2}',
                        '판정3 늘어난 행': '; '.join(f'{a.strftime("%H:%M:%S")} {c} {ra}위(기준 {rb}위)' for a, c, ra, rb in g) or '-',
                        '판정': '개선' if (ok1 and ok2 and ok3) else '개선 없음'})
    V = pd.DataFrame(verdict)

    # ------------------------------------------------------------------ 탐색 보정 (24개 설정, 순환 이동 1,000회)
    null_names = [n for n, s in settings.items() if s['null']]
    assert len(null_names) == 24, len(null_names)
    obs = max(ap_from_order(y, settings[n]['order']) for n in null_names)
    rng = np.random.default_rng(2026)
    O = np.stack([settings[n]['order'] for n in null_names])
    mx = np.empty(N_NULL)
    rk = np.arange(1, len(y) + 1)
    for b in range(N_NULL):
        yb = np.roll(y, rng.integers(len(y) // 20, len(y) - len(y) // 20))
        Ys = yb[O]
        mx[b] = ((Ys * np.cumsum(Ys, 1) / rk).sum(1) / yb.sum()).max()
    null = {'n_settings': len(null_names), 'obs_max_AP': obs,
            'obs_best': max(null_names, key=lambda n: ap_from_order(y, settings[n]['order'])),
            'null_max_mean': float(mx.mean()), 'null_max_95': float(np.quantile(mx, 0.95)),
            'p': float((1 + (mx >= obs).sum()) / (N_NULL + 1))}

    R.drop(columns='_rank').to_csv(OUT / 'settings.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(top5).to_csv(OUT / 'top5_defects.csv', index=False, encoding='utf-8-sig')
    V.to_csv(OUT / 'verdict.csv', index=False, encoding='utf-8-sig')
    json.dump(null, open(OUT / 'null.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    pd.set_option('display.width', 250)
    print(R.drop(columns='_rank').round(3).to_string(index=False))
    print(V.to_string(index=False))
    print(null)


if __name__ == '__main__':
    main()
