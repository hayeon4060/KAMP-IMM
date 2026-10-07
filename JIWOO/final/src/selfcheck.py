"""재현 확인: 재생성한 수치를 동결 기준값과 비교 (허용오차 0.0015, 정수는 완전 일치)

기준값
- 점수 기반 헤드라인(행·불량·AP·경보·TP)과 비교표의 z·RF AP: 확률 층과 무관 → κ와 상관없이 항상 확인
- 확률 지표(BSS)와 확률 기반 비교표 행: κ(측면 배분 Beta 사전 강도)에 따라 다름
    κ=10 (사전등록 기본값, 보고서 기준 원장 v2): CN7 BSS 전향 0.285 / 공식 OOF 0.056,
                              Maha+보정확률 AP CN7 전향 0.813·공식 0.194, ledger_v1.csv SHA-256 앞 16자 c107e5a443f110ff
    κ=20 (민감도)           : CN7 BSS 전향 0.297 / 공식 OOF 0.063, Maha+보정확률 AP CN7 전향 0.836·공식 0.193,
                              ledger_v1.csv SHA-256 앞 16자 221ec962e52fd43d
"""
import json
import os

import pandas as pd

from paths import LEDGER, PROB

TOL = 0.0015
REF_HEADLINE = {
    'CN7_official': dict(rows=1211, defects=17, AP_score=0.373, alarms=10, TP_alarm=6),
    'CN7_extra': dict(rows=2745, defects=11, AP_score=0.797, alarms=16, TP_alarm=10),
    'RG3_official': dict(rows=1182, defects=25, AP_score=0.024, alarms=6, TP_alarm=0),
    'RG3_extra': dict(rows=71, defects=7, AP_score=0.115, alarms=0),
}
REF_COMPARE_AP = {('CN7', 'official', '이탈도 z(비지도)'): 0.191, ('CN7', 'extra', '이탈도 z(비지도)'): 0.250,
                  ('CN7', 'official', 'RF 지도(±10간격)'): 0.014, ('CN7', 'extra', 'RF 지도(±10간격)'): 0.161}
REF_BY_KAPPA = {
    10: dict(bss={('cn7', 'prospective'): 0.285, ('cn7', 'official_oof'): 0.056},
             compare={('CN7', 'extra', 'Maha+보정확률(Beta)'): 0.813, ('CN7', 'official', 'Maha+보정확률(Beta)'): 0.194},
             ledger_sha='c107e5a443f110ff'),
    20: dict(bss={('cn7', 'prospective'): 0.297, ('cn7', 'official_oof'): 0.063},
             compare={('CN7', 'extra', 'Maha+보정확률(Beta)'): 0.836, ('CN7', 'official', 'Maha+보정확률(Beta)'): 0.193},
             ledger_sha='221ec962e52fd43d'),
}


def main():
    kappa = float(os.environ.get('KAPPA', '10'))
    ref_k = REF_BY_KAPPA.get(int(kappa)) if kappa.is_integer() else None
    rows = []

    def add(item, ref, got, exact):
        ok = (got == ref) if exact else (got is not None and abs(float(got) - float(ref)) <= TOL)
        rows.append(dict(항목=item, 기준=ref, 재생성=got, 판정='PASS' if ok else 'FAIL'))

    H = json.load(open(LEDGER / 'headline_v1.json'))
    for key, ref in REF_HEADLINE.items():
        for k, v in ref.items():
            add(f'{key}.{k}', v, H['headline'][key][k], exact=(k != 'AP_score'))
    T = pd.read_csv(LEDGER / 'compare_table_v1.csv')

    def cmp_ap(prod, per, m):
        got = T[(T.제품 == prod) & (T.기간 == per) & (T.모델 == m)].AP
        return float(got.iloc[0]) if len(got) else None
    for (prod, per, m), v in REF_COMPARE_AP.items():
        add(f'compare AP {prod} {per} {m}', v, cmp_ap(prod, per, m), exact=False)
    sha = H['manifest'].get('outputs/ledger/ledger_v1.csv')
    if ref_k is None:
        print(f'κ={kappa:g}: 확률 지표 기준값 없음 → 점수 기반 항목만 확인')
    else:
        S = json.load(open(PROB / 'prob_summary.json'))
        for (p, tag), v in ref_k['bss'].items():
            add(f'{p.upper()} BSS {tag} (κ={kappa:g})', v, S[p]['rows'][tag]['BSS'], exact=False)
        for (prod, per, m), v in ref_k['compare'].items():
            add(f'compare AP {prod} {per} {m} (κ={kappa:g})', v, cmp_ap(prod, per, m), exact=False)
        if ref_k['ledger_sha']:
            add(f'ledger_v1.csv SHA-256[:16] = 동결 원장 (κ={kappa:g})', ref_k['ledger_sha'], sha, exact=True)

    R = pd.DataFrame(rows)
    R.to_csv(LEDGER / 'selfcheck.csv', index=False, encoding='utf-8-sig')
    with pd.option_context('display.width', 200, 'display.max_rows', 100):
        print(R.to_string(index=False))
    n_fail = int((R.판정 == 'FAIL').sum())
    print(f'\n재현 확인 (κ={kappa:g}): {len(R) - n_fail}/{len(R)} PASS' + ('' if n_fail == 0 else f' — FAIL {n_fail}건'))
    return n_fail == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
