#!/usr/bin/env python3
"""전처리 → 학습 → 추론 → 결과 생성 → 재현 확인 자동 실행 (단일 진입점)

사용법:  python run_all.py               # 사전등록 기본값 κ=10 (측면 배분 Beta 사전 강도)
         python run_all.py --kappa 20    # 민감도 κ=20 (동결 원장 ledger_v1.csv와 바이트 동일하게 재현)
         python run_all.py --experiments # 1~9단계 후 검증 실험 A(라벨 섞기)·B(사전등록 전향) + 그림 + 별도 재현 확인
데이터:  ./data/dataset/ 에 대회 원본 csv 배치 (data/dataset/README.md)
결과:    ./outputs/ (단계별 산출물은 README.md 표 참고)
종료 코드: 재현 확인(9단계, --experiments 시 실험 재현 확인 포함) 전부 PASS면 0, 하나라도 FAIL이면 1
"""
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

if '--kappa' in sys.argv:                     # 확률 층 모듈이 import 시 읽으므로 가장 먼저 설정
    os.environ['KAPPA'] = sys.argv[sys.argv.index('--kappa') + 1]
os.environ.setdefault('KAPPA', '10')

import numpy as np  # noqa: E402

random.seed(0)
np.random.seed(0)          # 모든 난수는 각 모듈에서 시드를 고정해 사용 (여기는 전역 안전장치)

import paths  # noqa: E402


def step(no, title, fn, total='9'):
    t = time.time()
    print(f'\n[{no}/{total}] {title}', flush=True)
    r = fn()
    print(f'[{no}/{total}] 완료 ({time.time() - t:.1f}초)', flush=True)
    return r


def s1_prep():
    import prep_link, prep_base, shots_data, w_prior
    prep_link.main()          # 1차↔2차 연결 (linked/)
    prep_base.main()          # labeled 샷 테이블·Run·기준값 (base/)
    shots_data.build()        # 원본 라벨 행·쇼트·비라벨 정상 쇼트 캐시
    w_prior.main()            # 측면 배분 사전 (외부 ERR 집계)


def s2_model():
    import final_model as fm
    fm.train(verbose=True)
    R = fm.test(verbose=True)
    print(R['check'].to_string(index=False))


def s3_prob():
    import prob_beta
    prob_beta.main()


def s4_ledger():
    import ledger
    ledger.main()


def s5_audit():
    import audit
    audit.main()


def s6_compare():
    import compare
    compare.main()


def s7_ops():
    import ops_tables
    ops_tables.main()


def s8_pred():
    import predictions
    predictions.main()


def s9_check():
    import selfcheck
    return selfcheck.main()


def e1_shuffle():
    import exp_label_shuffle
    exp_label_shuffle.main()


def e2_prospective():
    import exp_prospective
    exp_prospective.main()


def e3_figs():
    import fig_experiments
    fig_experiments.main()


def e4_check():
    import exp_selfcheck
    return exp_selfcheck.main()


if __name__ == '__main__':
    T0 = time.time()
    print(f'κ (측면 배분 Beta 사전 강도) = {os.environ["KAPPA"]}')
    paths.check_data()
    paths.ensure_dirs()
    step(1, '전처리: 1차↔2차 연결, 샷·Run 테이블, 기준값, 측면 사전', s1_prep)
    step(2, '최종 탐지기 학습(비라벨 정상) · 점수(라벨 샷)', s2_model)
    step(3, '확률 층 (log 점수 → Venn-Abers → Beta 측면 배분 → 행 확률)', s3_prob)
    step(4, '원장·헤드라인·SHA 매니페스트', s4_ledger)
    step(5, '원장 정합성 감사', s5_audit)
    step(6, '비교표 (z, RF, 3σ 기준선 포함)', s6_compare)
    step(7, '초품 그리드·비용표·Brier 분해', s7_ops)
    step(8, '제출용 예측 파일', s8_pred)
    ok = step(9, '재현 확인 (동결 기준값 대비)', s9_check)
    if '--experiments' in sys.argv:
        T1 = time.time()
        step('E1', '검증 실험 A: 라벨 섞기 (가이드북식 · 누설 차단 RF · 과거기준선 이탈도)', e1_shuffle, 'E4')
        step('E2', '검증 실험 B: 사전등록 전향 검증 (과거기준선 이탈도)', e2_prospective, 'E4')
        step('E3', '실험 그림 (fig_A, fig_B)', e3_figs, 'E4')
        ok_e = step('E4', '검증 실험 재현 확인', e4_check, 'E4')
        print(f'\n검증 실험 소요 {time.time() - T1:.0f}초 — {"PASS" if ok_e else "FAIL"}')
        ok = ok and ok_e
    print(f'\n전체 소요 {time.time() - T0:.0f}초 — {"PASS" if ok else "FAIL"}')
    sys.exit(0 if ok else 1)
