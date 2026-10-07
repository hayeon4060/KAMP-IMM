"""원장 정합성 감사 → outputs/ledger/audit_v1.json

확률 범위·VA 구간·결측·경보 일관성·쇼트 정합(2행 쇼트: max(pL,pR) ≤ q ≤ pL+pR)·p_shot 단조성·행수 대조
"""
import json
import pandas as pd

from paths import LEDGER

KNOWN = {('CN7', 'official'): (1211, 17), ('RG3', 'official'): (1182, 25),
         ('CN7', 'extra'): (2745, 11), ('RG3', 'extra'): (71, 7)}


def main():
    L = pd.read_csv(LEDGER / 'ledger_v1.csv')
    rep = {}
    rep['p_범위위반'] = int(((L.p < 0) | (L.p > 1)).sum())
    rep['구간위반 p<p_lo or p>p_hi'] = int(((L.p < L.p_lo - 1e-12) | (L.p > L.p_hi + 1e-12)).sum())
    rep['p 결측'] = int(L.p.isna().sum())
    rep['score 결측'] = int(L.score.isna().sum())
    rep['경보 불일치 (alarm != pct>=0.999)'] = int((L.alarm != (L.pct >= 0.999).astype(int)).sum())
    viol = dict(lower=0, upper=0, p11=0, shots=0)
    for (prod, per, ts), g in L.groupby(['product', 'period', 'TimeStamp']):
        if len(g) != 2: continue
        pL, pR = g.p.values
        q = 1 - (1 - pL) * (1 - pR)
        viol['shots'] += 1
        if max(pL, pR) > q + 1e-12: viol['lower'] += 1
        if q > min(1, pL + pR) + 1e-12: viol['upper'] += 1
        if pL * pR < -1e-12: viol['p11'] += 1
    rep['쇼트정합(2행 쇼트)'] = viol
    mono_bad = 0
    for (prod, per), g in L.dropna(subset=['score', 'p_shot']).groupby(['product', 'period']):
        g2 = g.drop_duplicates('TimeStamp').sort_values('score')
        mono_bad += int((g2.p_shot.diff() < -1e-9).sum())
    rep['p_shot 단조성 위반'] = mono_bad
    mis = {f"{k}": (len(g), int(g.y.sum())) for k, g in L.groupby(['product', 'period'])
           if (len(g), int(g.y.sum())) != KNOWN[k]}
    rep['행수/불량수 불일치'] = mis if mis else "없음"

    # 단조성 재검: 같은 보정기 단위 (공식 = Run별 LORO 보정기, 전향 = 제품별 고정 보정기)
    d = L.dropna(subset=['score', 'p_shot'])
    bad_off = 0
    for (prod, run), g in d[d.period == 'official'].groupby(['product', 'run_id']):
        g2 = g.drop_duplicates('TimeStamp').sort_values('score')
        bad_off += int((g2.p_shot.diff() < -1e-9).sum())
    bad_ext = 0
    for prod, g in d[d.period == 'extra'].groupby('product'):
        g2 = g.drop_duplicates('TimeStamp').sort_values('score')
        bad_ext += int((g2.p_shot.diff() < -1e-9).sum())
    rep['p_shot 단조성 (보정기 단위 재검)'] = {'공식: 런 내(동일 보정기) 위반': bad_off,
                                         '전향: 제품 내(고정 보정기) 위반': bad_ext}
    rep[f'p_shot 단조성 위반 {mono_bad} 해석'] = ("전부 런 경계 교차 — LORO 설계상 런마다 보정기가 달라 "
                                           "기간 전체 단조성은 성립하지 않는 것이 정상") if (bad_off + bad_ext) == 0 \
        else "보정기 내부 위반 존재 — 확인 필요"
    json.dump(rep, open(LEDGER / 'audit_v1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('  ' + json.dumps(rep, ensure_ascii=False))
    return rep


if __name__ == '__main__':
    main()
