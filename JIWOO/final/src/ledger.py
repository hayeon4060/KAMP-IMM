"""원장(ledger) 생성 + 헤드라인 JSON + SHA-256 매니페스트

ledger_v1.csv : 확률 층 결과(공식 OOF 행 + 전향 행, 두 제품)를 행(부품) 단위 한 장으로 고정.
                alarm = pct ≥ 0.999 (참조 99.9 백분위, 라벨 무사용), rank_period = 기간 내 점수 순위.
headline_v1.json : 보고서 헤드라인 수치(전부 원장에서 도출) + 입력·원장 파일 해시.
"""
import hashlib
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from paths import ROOT, PROB, LEDGER, W_PRIOR, DOCS

ALARM_Q = 0.999


def build_ledger():
    R = pickle.load(open(PROB / 'prob_results_beta.pkl', 'rb'))
    frames = []
    for p in ['cn7', 'rg3']:
        for per, key in [('official', 'rows_off'), ('extra', 'rows_ext')]:
            d = R[p][key].copy()
            d['product'] = p.upper(); d['period'] = per
            keep = ['product', 'period', 'TimeStamp', 'cavity', 'y', 'Reason', 'run_id', 'score', 'pct',
                    'p_shot', 'p', 'p_lo', 'p_hi', 'since_start', 'top3', 'relation']
            if 'oid' in d.columns: keep.append('oid')
            frames.append(d[keep])
    L = pd.concat(frames, ignore_index=True)
    L['alarm'] = (L.pct >= ALARM_Q).astype(int)
    L['rank_period'] = L.groupby(['product', 'period'])['score'].rank(ascending=False, method='min')
    L = L.sort_values(['product', 'period', 'TimeStamp', 'cavity']).reset_index(drop=True)
    L.to_csv(LEDGER / 'ledger_v1.csv', index=False, encoding='utf-8-sig')
    return L


def _ap(d):
    m = d.score.notna()
    return round(float(average_precision_score(d.y[m], d.score[m])), 3)


def sha16(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


def headline(L):
    H = {}
    for (p, per), g in L.groupby(['product', 'period']):
        al = g[g.alarm == 1]
        H[f"{p}_{per}"] = dict(
            rows=len(g), defects=int(g.y.sum()), AP_score=_ap(g),
            alarms=len(al), alarm_rate_pct=round(len(al) / len(g) * 100, 2),
            TP_alarm=int(al.y.sum()),
            TP_top5pct=int(g.nlargest(int(np.ceil(0.05 * len(g))), 'score').y.sum()))
    ce = L[(L['product'] == 'CN7') & (L.period == 'extra')]
    alo = ce[ce.alarm == 1]
    H['CN7_extra']['alarm_TP_by_reason'] = {k: int(v) for k, v in alo[alo.y == 1].Reason.value_counts().items()}
    H['CN7_extra']['missed'] = {k: int(v) for k, v in ce[(ce.y == 1) & (ce.alarm == 0)].Reason.value_counts().items()}

    files = [PROB / 'prob_results_beta.pkl', W_PRIOR, DOCS / 'PREREG_BETA_W.md', LEDGER / 'ledger_v1.csv']
    man = {str(f.relative_to(ROOT)): sha16(f) for f in files}
    out = dict(headline=H, manifest=man, operating_point="pct>=0.999 (참조 99.9백분위, 라벨 무사용)",
               prob_version=f"beta_kappa{float(os.environ.get('KAPPA', '10')):g}")
    json.dump(out, open(LEDGER / 'headline_v1.json', 'w'), ensure_ascii=False, indent=1)
    for k in sorted(H):
        h = H[k]
        print(f'  {k:13s} 행 {h["rows"]:5d} 불량 {h["defects"]:3d} AP {h["AP_score"]:.3f} '
              f'경보 {h["alarms"]:3d} TP {h["TP_alarm"]}')
    return out


def main():
    L = build_ledger()
    return headline(L)


if __name__ == '__main__':
    main()
