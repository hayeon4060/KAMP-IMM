"""동일 조건 비교표 + 기준선

ledger_v1_plus.csv : 원장 + z(과거기준선 이탈도, 비지도) + rf(RandomForest 지도 기준선)
    rf: 24개 공정변수, RandomForest(200, random_state=0). 공식 기간은 쇼트 순서대로 5개 연속 블록 fold,
        시험 블록 앞뒤 ±10쇼트는 학습에서 제외. 전향 기간은 공식 전체로 적합한 모델 적용.
compare_table_v1.csv : 5% 예산(k 명시)에서 모델별 AP·TP·FP·FN·F1·Brier (행 단위).
    '기저확률만'은 점수가 상수라 동률이 많아 TP 등은 동률 무작위 정렬 200회 기대값으로 기재.
final_system_sim.csv : 경보 ∪ 초품검사(T=5분, N=10) ∪ 불량 발견 후 다음 5쇼트 연쇄 검사 시뮬레이션
baseline_3sigma.csv, baseline_3sigma_exRPM.csv : 현장 관행 3σ 관리도 (기준 = 10/16 이전 비라벨 정상모드)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from paths import LEDGER
from score_z import past_baseline_z, rms
from shots_data import VARS, load_all, stream


def _p1(clf, X):
    pr = clf.predict_proba(X)
    return pr[:, 1] if pr.shape[1] == 2 else np.full(len(X), float(clf.classes_[0]))


def build_plus(d):
    L = pd.read_csv(LEDGER / 'ledger_v1.csv', parse_dates=['TimeStamp'])
    S = d['S']
    zmap = {}
    for prod in ['CN7', 'RG3']:
        st = stream(d['S'], d['U'], prod).reset_index(drop=True)
        st['z'] = rms(past_baseline_z(st, 10))
        for ts, v in zip(st.TimeStamp, st.z): zmap[(prod, ts)] = v
    L['z'] = [zmap.get((p, t.strftime('%Y-%m-%d %H:%M:%S')), np.nan) for p, t in zip(L['product'], L.TimeStamp)]

    rfmap = {}
    for prod in ['CN7', 'RG3']:
        sh = S[S['prod'] == prod].sort_values('t').reset_index(drop=True)
        off = sh[sh.period == 'official'].reset_index(drop=True)
        ext = sh[sh.period == 'extra'].reset_index(drop=True)
        X, y = off[VARS].fillna(0).to_numpy(), off.y.to_numpy()
        n = len(off); oof = np.full(n, np.nan)
        for f in np.array_split(np.arange(n), 5):          # 연속 블록 + 경계 ±10 제외
            test = np.zeros(n, bool); test[f] = True
            near = np.zeros(n, bool); near[max(0, f[0] - 10):f[-1] + 11] = True
            rf = RandomForestClassifier(200, random_state=0, n_jobs=4).fit(X[~near], y[~near])
            oof[test] = _p1(rf, X[test])
        rf_full = RandomForestClassifier(200, random_state=0, n_jobs=4).fit(X, y)
        for ts, v in zip(off.TimeStamp, oof): rfmap[(prod, ts)] = v
        if len(ext):
            for ts, v in zip(ext.TimeStamp, _p1(rf_full, ext[VARS].fillna(0).to_numpy())): rfmap[(prod, ts)] = v
    L['rf'] = [rfmap.get((p, t.strftime('%Y-%m-%d %H:%M:%S')), np.nan) for p, t in zip(L['product'], L.TimeStamp)]
    L.to_csv(LEDGER / 'ledger_v1_plus.csv', index=False, encoding='utf-8-sig')


def _at_budget(g, s, q=0.05):
    y = g.y.to_numpy(); s = np.asarray(s, float)
    k = int(np.ceil(q * len(y))); o = np.argsort(-np.nan_to_num(s, nan=-1), kind='stable')
    tp = int(y[o[:k]].sum()); fp = k - tp; fn = int(y.sum()) - tp
    f1 = 2 * tp / (k + y.sum()) if (k + y.sum()) else 0
    return k, tp, fp, fn, round(f1, 3)


def _brier(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return float(np.mean((p - np.asarray(y, float)) ** 2))


def compare_table():
    L = pd.read_csv(LEDGER / 'ledger_v1_plus.csv', parse_dates=['TimeStamp'])
    rows_out, sim_out = [], []
    for (prod, per), g in L.groupby(['product', 'period']):
        g = g.sort_values(['TimeStamp', 'cavity']).reset_index(drop=True)
        y = g.y.to_numpy(); n = len(g); P = int(y.sum()); br = P / n
        base_off = L[(L['product'] == prod) & (L.period == 'official')].y.mean()
        pb = g.cavity.map(g.groupby('cavity').y.mean() if per == 'official' else
                          L[(L['product'] == prod) & (L.period == 'official')].groupby('cavity').y.mean())
        models = [
            ('무작위', np.random.default_rng(0).random(n), None),
            ('기저확률만', pb.to_numpy(), pb.to_numpy()),
            ('RF 지도(±10간격)', g.rf.to_numpy(), g.rf.to_numpy()),
            ('이탈도 z(비지도)', g.z.to_numpy(), None),
            ('Mahalanobis(동결)', g.score.to_numpy(), None),
            ('Maha+보정확률(Beta)', g.p.to_numpy(), g.p.to_numpy()),
        ]
        for nm, s, pp in models:
            m = ~np.isnan(np.asarray(s, float))
            ap = round(float(average_precision_score(y[m], np.asarray(s, float)[m])), 3) if P > 0 else np.nan
            if nm == '무작위': ap = round(br, 3)
            k, tp, fp, fn, f1 = _at_budget(g[m], np.asarray(s, float)[m])
            rows_out.append(dict(제품=prod, 기간=per, 모델=nm, AP=ap, k5pct=k, TP=tp, FP=fp, FN=fn,
                                 F1_5pct=f1, Brier=(round(_brier(y[m], pp[m]), 5) if pp is not None else None),
                                 Brier기준=round(_brier(y, np.full(n, base_off)), 5)))
        # 최종 시스템 시뮬레이션: 경보 ∪ 초품(T=5분, N=10) ∪ 불량 발견 → 다음 5쇼트 (연쇄)
        sh = g.drop_duplicates('TimeStamp')[['TimeStamp']].reset_index(drop=True)
        sh['gap'] = sh.TimeStamp.diff().dt.total_seconds().fillna(1e9) / 60
        seg = (sh.gap > 5).cumsum(); k_ = sh.groupby(seg).cumcount()
        first_ts = set(sh.TimeStamp[k_ < 10])
        tslist = sh.TimeStamp.tolist(); pos = {t: i for i, t in enumerate(tslist)}
        insp_ts = set(g.TimeStamp[g.alarm == 1]) | first_ts
        ydict = g.groupby('TimeStamp').y.max().to_dict()
        changed = True
        while changed:
            changed = False
            for t in list(insp_ts):
                if ydict.get(t, 0) == 1:
                    i = pos[t]
                    for j in range(i + 1, min(i + 6, len(tslist))):
                        if tslist[j] not in insp_ts:
                            insp_ts.add(tslist[j]); changed = True
        mask = g.TimeStamp.isin(insp_ts)
        sim_out.append(dict(제품=prod, 기간=per, 검사행=int(mask.sum()), 검사율=round(mask.mean() * 100, 1),
                            TP=int(g.y[mask].sum()), 전체불량=P))
    pd.DataFrame(rows_out).to_csv(LEDGER / 'compare_table_v1.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(sim_out).to_csv(LEDGER / 'final_system_sim.csv', index=False, encoding='utf-8-sig')

    # '기저확률만'은 상수 점수 → 안정 정렬이면 파일 순서 인공물이 생김. 동률 무작위 정렬 200회 기대값으로 교체
    T = pd.read_csv(LEDGER / 'compare_table_v1.csv')
    for c in ['TP', 'FP', 'FN']: T[c] = T[c].astype(float)
    rng = np.random.default_rng(1)
    for (prod, per), g in L.groupby(['product', 'period']):
        y = g.y.to_numpy(); n = len(g); k = int(np.ceil(0.05 * n))
        pb_map = (g.groupby('cavity').y.mean() if per == 'official' else
                  L[(L['product'] == prod) & (L.period == 'official')].groupby('cavity').y.mean())
        s = g.cavity.map(pb_map).to_numpy()
        tps = [int(y[np.argsort(-(s + rng.random(n) * 1e-9))[:k]].sum()) for _ in range(200)]
        tp = float(np.mean(tps))
        m = (T.제품 == prod) & (T.기간 == per) & (T.모델 == '기저확률만')
        T.loc[m, 'TP'] = round(tp, 1); T.loc[m, 'FP'] = round(k - tp, 1)
        T.loc[m, 'FN'] = round(float(y.sum()) - tp, 1); T.loc[m, 'F1_5pct'] = round(2 * tp / (k + y.sum()), 3)
    T['비고'] = np.where(T.모델 == '기저확률만', '동률 무작위 기대값(200회)', '')
    T.to_csv(LEDGER / 'compare_table_v1.csv', index=False, encoding='utf-8-sig')
    show = T[T.모델.isin(['이탈도 z(비지도)', 'RF 지도(±10간격)', 'Mahalanobis(동결)', 'Maha+보정확률(Beta)'])]
    print(show.pivot_table(index=['제품', '기간'], columns='모델', values='AP').round(3).to_string())
    return T


def baseline_3sigma(d):
    """3σ 관리도: 기준 = 10/16 이전 비라벨 정상모드 전역 평균·표준편차, 변수 중 하나라도 |z|>3이면 검사"""
    L = pd.read_csv(LEDGER / 'ledger_v1_plus.csv', parse_dates=['TimeStamp'])
    S, U = d['S'], d['U']
    for fname, vs in [('baseline_3sigma.csv', VARS), ('baseline_3sigma_exRPM.csv', [v for v in VARS if 'RPM' not in v])]:
        rows = []
        for prod in ['CN7', 'RG3']:
            ref = U[(U['prod'] == prod) & (U.t < '2020-10-16')][vs]
            mu, sd = ref.mean(), ref.std().replace(0, np.nan)
            sh = S[S['prod'] == prod].copy()
            Z = (sh[vs] - mu) / sd
            sh['s3'] = (Z.abs() > 3).any(axis=1).astype(int)
            m = dict(zip(sh.TimeStamp, sh.s3))
            for per, g in L[L['product'] == prod].groupby('period'):
                flag = g.TimeStamp.dt.strftime('%Y-%m-%d %H:%M:%S').map(m)
                k = int(flag.sum()); tp = int(g.y[flag == 1].sum())
                rows.append(dict(제품=prod, 기간=per, 검사행=k, 검사율=round(k / len(g) * 100, 1),
                                 TP=tp, 전체불량=int(g.y.sum())))
        pd.DataFrame(rows).to_csv(LEDGER / fname, index=False, encoding='utf-8-sig')


def main():
    d = load_all()
    build_plus(d)
    T = compare_table()
    baseline_3sigma(d)
    return T


if __name__ == '__main__':
    main()
