"""재현 확인: 재생성한 수치를 동결 기준값과 비교 (허용오차 0.0015, 정수는 완전 일치)

기준값
- 점수 기반 헤드라인(행·불량·AP·경보·TP)과 비교표의 z·RF AP: 확률 층과 무관 → κ와 상관없이 항상 확인
- 확률 지표(BSS)와 확률 기반 비교표 행: κ(측면 배분 Beta 사전 강도)에 따라 다름
    κ=10 (사전등록 기본값, 1008 국소 Mahalanobis 원장): CN7 BSS 전향 0.215 / 공식 OOF 0.02,
                              Maha+보정확률 AP CN7 전향 0.558·공식 0.179, ledger_v1.csv SHA-256 앞 16자 c3c520b85a0fc23b
    κ=20 (민감도)           : CN7 BSS 전향 0.242 / 공식 OOF 0.032, Maha+보정확률 AP CN7 전향 0.602·공식 0.175,
                              ledger_v1.csv SHA-256 앞 16자 a3b938c0641089c1
- 원장 판정은 값 지문(ledger.value_fingerprint)으로 한다: 실수가 아닌 열의 해시는 완전 일치, 실수 열의 가중합은 상대오차 1e-9 이내.
  기준값은 src/ledger_ref.json. 바이트 해시는 운영체제·CPU에 따라 부동소수 마지막 자리가 달라지면 바뀔 수 있으므로
  참고로만 출력한다(기준 환경: 리눅스, Python 3.11).
"""
import json
import os
from pathlib import Path

import pandas as pd

from paths import LEDGER, PROB

FP_REF = Path(__file__).resolve().parent / 'ledger_ref.json'
FP_RTOL = 1e-9

TOL = 0.0015
REF_HEADLINE = {
    'CN7_official': dict(rows=1211, defects=17, AP_score=0.465, alarms=6, TP_alarm=6),
    'CN7_extra': dict(rows=2745, defects=11, AP_score=0.608, alarms=10, TP_alarm=8),
    'RG3_official': dict(rows=1182, defects=25, AP_score=0.022, alarms=0, TP_alarm=0),
    'RG3_extra': dict(rows=71, defects=7, AP_score=0.102, alarms=0),
}
REF_COMPARE_AP = {('CN7', 'official', '이탈도 z(비지도)'): 0.191, ('CN7', 'extra', '이탈도 z(비지도)'): 0.250,
                  ('CN7', 'official', 'RF 지도(±10간격)'): 0.014, ('CN7', 'extra', 'RF 지도(±10간격)'): 0.161}
REF_BY_KAPPA = {
    10: dict(bss={('cn7', 'prospective'): 0.215, ('cn7', 'official_oof'): 0.02},
             compare={('CN7', 'extra', 'Maha+보정확률(Beta)'): 0.558, ('CN7', 'official', 'Maha+보정확률(Beta)'): 0.179},
             ledger_sha='c3c520b85a0fc23b'),
    20: dict(bss={('cn7', 'prospective'): 0.242, ('cn7', 'official_oof'): 0.032},
             compare={('CN7', 'extra', 'Maha+보정확률(Beta)'): 0.602, ('CN7', 'official', 'Maha+보정확률(Beta)'): 0.175},
             ledger_sha='a3b938c0641089c1'),
}


def main():
    kappa = float(os.environ.get('KAPPA', '10'))
    ref_k = REF_BY_KAPPA.get(int(kappa)) if kappa.is_integer() else None
    rows = []

    def add(item, ref, got, exact):
        ok = (got == ref) if exact else (got is not None and abs(float(got) - float(ref)) <= TOL)
        rows.append(dict(항목=item, 기준=ref, 재생성=got, 판정='PASS' if ok else 'FAIL'))

    H = json.load(open(LEDGER / 'headline_v1.json', encoding='utf-8'))
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
        S = json.load(open(PROB / 'prob_summary.json', encoding='utf-8'))
        for (p, tag), v in ref_k['bss'].items():
            add(f'{p.upper()} BSS {tag} (κ={kappa:g})', v, S[p]['rows'][tag]['BSS'], exact=False)
        for (prod, per, m), v in ref_k['compare'].items():
            add(f'compare AP {prod} {per} {m} (κ={kappa:g})', v, cmp_ap(prod, per, m), exact=False)
        fp = H.get('ledger_fingerprint', {})
        ref_fp = json.load(open(FP_REF, encoding='utf-8')).get(f'{kappa:g}') if FP_REF.exists() else None
        if ref_fp is None:
            rows.append(dict(항목=f'ledger_v1.csv 값 지문 = 동결 원장 (κ={kappa:g})', 기준='(없음)', 재생성=fp.get('exact'), 판정='FAIL'))
        else:
            rel = max(abs(fp['wsum'].get(c, float('nan')) - v) / max(abs(v), 1e-12) for c, v in ref_fp['wsum'].items())
            ok = fp.get('exact') == ref_fp['exact'] and set(fp['wsum']) == set(ref_fp['wsum']) and rel <= FP_RTOL
            rows.append(dict(항목=f'ledger_v1.csv 값 지문 = 동결 원장 (κ={kappa:g})',
                             기준=f"{ref_fp['exact']} / 실수 {len(ref_fp['wsum'])}열",
                             재생성=f"{fp.get('exact')} / 최대 상대오차 {rel:.1e}", 판정='PASS' if ok else 'FAIL'))
        same = '일치' if sha == ref_k['ledger_sha'] else '불일치(다른 운영체제·CPU면 정상일 수 있음, 판정은 값 지문)'
        print(f'참고: ledger_v1.csv 바이트 SHA-256[:16] {sha} — 기준 환경 {ref_k["ledger_sha"]}와 {same}')

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
