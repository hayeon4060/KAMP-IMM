"""검증 실험 재현 확인 (기본 재현 확인과 별도 표). 실수 허용오차 0.0015 (Fisher p는 1e-5), 정수는 완전 일치.

기준값 = 보고서에 인용한 실험 결과(2026-10-04 실행분)."""
import json

import numpy as np
import pandas as pd

from paths import EXP

TOL = 0.0015
REF_SHUFFLE = {   # (제품, 방법): 진짜 라벨 F1, 섞은 라벨 F1 평균, 섞기 횟수, 순열 p
    ('CN7', 'guidebook'): dict(real=0.990698, shuf_mean=0.985609, n_shuf=20, p=0.047619),
    ('CN7', 'honest_rf'): dict(real=0.046154, shuf_mean=0.021795, n_shuf=20, p=0.190476),
    ('CN7', 'deviation_K10'): dict(real=0.179487, shuf_mean=0.022795, n_shuf=1000, p=0.000999),
    ('RG3', 'guidebook'): dict(real=0.979129, shuf_mean=0.978909, n_shuf=20, p=0.52381),
    ('RG3', 'honest_rf'): dict(real=0.023529, shuf_mean=0.024706, n_shuf=20, p=0.619048),
    ('RG3', 'deviation_K10'): dict(real=0.023529, shuf_mean=0.028282, n_shuf=1000, p=0.698302),
}
REF_PROSPECTIVE = {
    'CN7': dict(tau=1.2704635879886383, shots=1389, defects=6, alarms=113, alarm_rate=0.08135349172066235,
                good_alarm_rate=0.07881417208966016, defect_alarmed=4, fisher_p=0.0005492276500914675, top5pct_k=70,
                defect_in_top5pct=4, alarms_within10_after_restart=17, restarts=5),
    'RG3': dict(tau=1.259953867835512, shots=38, defects=7, alarms=14, alarm_rate=0.3684210526315789,
                good_alarm_rate=0.3548387096774194, defect_alarmed=3, fisher_p=0.5167838116754523, top5pct_k=2,
                defect_in_top5pct=0, alarms_within10_after_restart=7, restarts=3),
}
REF_RESTART = {'CN7': {'3': (15, 0), '5': (25, 0), '10': (50, 0)}, 'RG3': {'3': (5, 0), '5': (7, 0), '10': (12, 0)}}


def main():
    rows = []

    def add(item, ref, got, tol=None):
        ok = (got == ref) if tol is None else (got is not None and abs(float(got) - float(ref)) <= tol)
        rows.append(dict(실험=item.split(' ')[0], 항목=item, 기준=ref, 재생성=got, 판정='PASS' if ok else 'FAIL'))

    A = {(r['prod'], r['method']): r for r in json.load(open(EXP / 'label_shuffle.json', encoding='utf-8'))}
    for key, ref in REF_SHUFFLE.items():
        r = A[key]; sh = np.array(r['shuf'])
        nm = f'A {key[0]} {key[1]}'
        add(f'{nm} F1(진짜 라벨)', ref['real'], round(float(r['real']), 6), TOL)
        add(f'{nm} F1(섞은 라벨 평균)', ref['shuf_mean'], round(float(sh.mean()), 6), TOL)
        add(f'{nm} 섞기 횟수', ref['n_shuf'], len(sh))
        add(f'{nm} 순열 p', ref['p'], round(float((np.sum(sh >= r['real']) + 1) / (len(sh) + 1)), 6), TOL)
    B = json.load(open(EXP / 'prospective.json', encoding='utf-8'))
    for p, ref in REF_PROSPECTIVE.items():
        for k, v in ref.items():
            tol = 1e-5 if k == 'fisher_p' else (TOL if isinstance(v, float) else None)
            add(f'B {p} {k}', v, B[p][k], tol)
        for N, (ins, cau) in REF_RESTART[p].items():
            got = B[p]['restart_rule'][N]
            add(f'B {p} 재기동 첫 {N}쇼트 검사/적중', f'{ins}/{cau}', f"{got['inspected']}/{got['caught']}")
    for f in ['fig_A_label_shuffle.png', 'fig_B_prospective_cn7.png', 'pred_prospective_baseline.csv']:
        add(f'파일 {f}', '생성', '생성' if (EXP / f).exists() and (EXP / f).stat().st_size > 0 else '없음')

    R = pd.DataFrame(rows)
    R.to_csv(EXP / 'exp_selfcheck.csv', index=False, encoding='utf-8-sig')
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(R.drop(columns='실험').to_string(index=False))
    n_fail = int((R.판정 == 'FAIL').sum())
    print(f'\n검증 실험 재현 확인: {len(R) - n_fail}/{len(R)} PASS' + ('' if n_fail == 0 else f' — FAIL {n_fail}건'))
    return n_fail == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
