# 사출성형 공정데이터 기반 품질불량 사전예측 및 검사 우선순위 결정 — 코드 재현 패키지

[팀명]

**목적**: 원본 데이터만으로 전처리 → 학습 → 추론 → 결과 생성 → 재현 확인까지 명령 한 줄로 다시 만든다.

## 1. 실행 방법

```bash
pip install -r requirements.txt          # Python 3.11 에서 검증 (버전 고정)
# 원본 csv 5개를 data/dataset/ 에 배치 (data/dataset/README.md 참고)
python run_all.py
```

- 마지막 9단계에서 동결 기준값과 비교한 PASS/FAIL 표를 출력하고, 전부 PASS면 종료 코드 0, 하나라도 FAIL이면 1.
- 소요 시간: 약 45초 (2코어 리눅스 기준. 1단계 원본 읽기 약 25초, 2단계 탐지기 학습·점수 약 8초). `--experiments` 포함 약 2분 40초.
- 난수: 모든 난수는 고정 시드 사용 (탐지기 SEED=42, RandomForest random_state=0, 비교표 무작위 기준 seed 0·1).
  같은 환경에서 몇 번을 돌려도 결과 파일이 같다.
- `python run_all.py --kappa 20` : 확률 층의 측면 배분 사전 강도 κ 민감도 실행 (아래 5절).
- `python run_all.py --experiments` : 1~9단계 뒤에 보고서의 검증 실험 2개(A 라벨 섞기, B 기준선 설정 고정 후향 평가)와
  그림을 재생성하고 별도 재현 확인 표(E4)를 출력한다. 기본 실행 결과는 바뀌지 않는다. 추가 소요 약 2분(2코어).
  그림의 한글 글꼴(Noto Sans CJK 등)이 없으면 기본 글꼴로 그리며(글자만 깨짐), 수치·파일은 동일하다.

## 2. 폴더 구조

```
run_all.py              단일 진입점 (1~9단계, 단계별 소요 시간 출력)
requirements.txt        패키지 버전 고정
docs/PREREG_BETA_W.md   측면 배분 Beta 사전 — 결과 보기 전 고정한 규칙 (SHA 매니페스트 대상)
docs/PREREG_BASELINE_z_1004.md  기준선(강건 z) 추가 기간 평가 — 점수 계산 전 고정한 설정·예측 (불량 개수·유형은 이미 알던 상태)
docs/PREREG_PINN_1006.md   물리 제약 신경망 탐색 실험 — 실행 전 고정 (탐색, 제출 모델 불변)
docs/PREREG_MIL_1006.md    로트 집계 약라벨(MIL) 탐색 실험 — 실행 전 고정 (탐색, 제출 모델 불변)
docs/PREREG_POLICY_1006.md 검사 정책 비교 규칙 — 실행 전 고정 (탐색)
data/dataset/           원본 데이터 배치 위치 (패키지에는 데이터 없음, README만)
src/
  paths.py              경로 설정 (모두 패키지 루트 기준 상대 경로)
  prep_link.py          1차 ↔ 2차 연결 (labeled 행 검증, unlabeled 제품별 원값)
  prep_base.py          labeled 샷 단위 테이블·Run 구분·split, 기록 단위(reference_scales)
  shots_data.py         원본 라벨 행·쇼트 표·비라벨 정상 쇼트 (확률 층·기준선 공용)
  w_prior.py            측면(LH/RH) 배분 사전 — 비라벨 ERR_FACT_QTY 집계
  final_model.py        제출 모델(탐지기): 인과적 파생변수 17개 → Ledoit-Wolf Mahalanobis, 백분위(pct)
  prob_beta.py          확률 층: 로지스틱(log 점수) → Venn-Abers → Beta 측면 배분 → 행 확률·구간
  ledger.py             원장·헤드라인·SHA 매니페스트
  audit.py              원장 정합성 감사
  score_z.py            기준선(강건 z): 직전 10쇼트 대비 변수별 강건 z의 rms (비지도, 학습 없음)
  compare.py            동일 조건 비교표 (z, RF, 3σ 관리도 포함)
  ops_tables.py         초품검사 그리드·비용표·Brier 기여 분해·경보 덩어리
  predictions.py        제출용 예측 파일
  selfcheck.py          재현 확인 (동결 기준값 대비)
  exp_label_shuffle.py  [--experiments] 검증 실험 A: 라벨 섞기 (가이드북식·누설 차단 RF·강건 z)
  exp_prospective.py    [--experiments] 검증 실험 B: 기준선(강건 z) 설정 고정 후향 평가
  fig_experiments.py    [--experiments] 그림 fig_A, fig_B
  exp_selfcheck.py      [--experiments] 검증 실험 재현 확인 (별도 표)
  plot_font.py          그림 한글 글꼴 설정 (없으면 기본 글꼴로 대체)
  exp_tabpfn.py         [선택, 탐색] TabPFN vs RF 지도 기준선 (같은 분할·채점, pip install tabpfn, 가중치 내려받기 필요)
  explore/              [선택, 탐색] 보고서 5장 탐색 실험 — run_all.py 다음에 개별 실행, torch 필요 (requirements 밖)
    pinn_lite.py          물리 제약 신경망 (docs/PREREG_PINN_1006.md)        → outputs/experiments/explore/pinn_result.json, 약 45초
    mil_build_bags.py     MIL 로트 묶음 조립 (docs/PREREG_MIL_1006.md)        → mil_bags.pkl
    mil_run.py            MIL 학습·평가                                      → mil_result.json, 약 1분
    mil_sens_alpha.py     MIL 정규화 강도 사후 민감도 (사전 기록 밖)          → mil_sens_alpha.json
    policy_compare.py     검사 정책 비교 (docs/PREREG_POLICY_1006.md)        → policy_result.json/.csv, 약 8분
    public_md_check.py    공개 분석 사례(정적 Mahalanobis) 방법 재구현 — 보고서 2-2절 → public_md_check.txt, 약 10초 (torch 불필요)
    exp_chronos.py        시계열 파운데이션 모델(Chronos) 예측 오차 점수 비교 (docs/PREREG_CHRONOS_1006.md) → chronos_compare.csv. pip install chronos-forecasting, 가중치 내려받기 필요. MOCK=1로 흐름 점검 가능
outputs/                실행 시 생성
```

## 3. 단계별 산출물

| 단계 | 내용 | 산출물 (`outputs/`) |
|---|---|---|
| 1 | 전처리: 1차↔2차 연결(z-score 1e-6 재현·라벨 100% 일치 assert), labeled 샷 테이블·Run·split, 기록 단위, 원본 라벨 행·쇼트 캐시, 측면 배분 사전 | `intermediate/linked/linked_labeled.csv`, `linked_unlabeled_{cn7,rg3}.csv`, `intermediate/base/{cn7,rg3}_shots.csv`, `reference_scales.json`, `intermediate/shots_cache.pkl`, `intermediate/w_prior.json` |
| 2 | 제출 모델 학습(비라벨 정상 샷만, 라벨 0개) → 라벨 샷 점수·평가 | `final_model/{cn7,rg3}_final_model.pkl`, `{p}_scores.csv`, `data_log.csv`, `train_params.csv`, `frame1.csv`, `frame2.csv`, `threshold.csv`, `check.csv` |
| 3 | 확률 층 (공식 기간 Run 단위 leave-one-run-out OOF, 추가 기간은 공식 전체 적합 고정) | `prob/prob_results_beta.pkl`, `prob/prob_summary.json` |
| 4 | 원장(행 단위 5,209행) + 헤드라인 + SHA-256 매니페스트 | `ledger/ledger_v1.csv`, `ledger/headline_v1.json` |
| 5 | 원장 정합성 감사 (확률 범위·구간·경보 일관성·쇼트 정합·단조성) | `ledger/audit_v1.json` |
| 6 | 비교표: 무작위·기저확률·RF 지도·기준선(강건 z)·제출 모델(Mahalanobis)·보정확률 (5% 예산), 최종 시스템 시뮬레이션, 3σ 관리도 | `ledger/ledger_v1_plus.csv`, `compare_table_v1.csv`, `final_system_sim.csv`, `baseline_3sigma.csv`, `baseline_3sigma_exRPM.csv` |
| 7 | 초품검사 그리드(T×N), 비용표(1:10·1:30·1:100), CN7 추가 기간 Brier 기여 분해, 경보 덩어리 | `ledger/firstarticle_grid_{제품}_{기간}.csv`, `cost_table_beta.csv`, `brier_decomp_cn7_extra.csv`, `alarm_clusters.csv` |
| 8 | 제출용 예측 파일 | `pred/pred_official.csv` (공식 2차 2,393행, OOF), `pred/pred_prospective.csv` (추가 기간 2,816행; 파일 이름의 prospective는 '추가 기간'을 뜻함) |
| 9 | 재현 확인 | 화면 표 + `ledger/selfcheck.csv` |
| E1 (`--experiments`) | 검증 실험 A 라벨 섞기: 같은 절차를 진짜 라벨과 섞은 라벨(RF 20회, 강건 z 1,000회)에 적용 — ① 분할 전 SMOTE→RF→무작위 5-fold(가이드북식) ② 쇼트 단위 분할·±10행 간격·fold 안 SMOTE RF ③ 기준선 강건 z(라벨 없음) | `experiments/label_shuffle.json` |
| E2 (`--experiments`) | 검증 실험 B 기준선 설정 고정 후향 평가(`docs/PREREG_BASELINE_z_1004.md`): 공식 기간에서 고정한 강건 z·임계값(공식 상위 5%)을 추가 기간에 그대로 적용. 출력의 Fisher p는 참고값(아래 5절) | `experiments/prospective.json`, `pred_prospective_baseline.csv`, `prospective_shots.pkl` |
| E3 (`--experiments`) | 그림 | `experiments/fig_A_label_shuffle.png`, `fig_B_prospective_cn7.png` |
| E4 (`--experiments`) | 검증 실험 재현 확인 (별도 PASS 표) | 화면 표 + `experiments/exp_selfcheck.csv` |

예측 파일 열: `원행번호`(공식 파일만, 대회 2차 파일의 `Unnamed: 0`), `제품`, `쇼트시각`, `좌우`, `불량확률`,
`확률구간_하`, `확률구간_상`, `검사대상(경보)`(점수 백분위 ≥ 0.999, 라벨 무사용), `검사우선순위`(제품·기간 내 확률 순위),
`이상변수Top3`(같은 Run 직전 기준선 대비 편차 상위 3개), `깨진관계`(충전시간×사출속도, 사출압력/속도), `실제불량`(평가용).

## 4. 외부 데이터 사용 위치

대회 제공 2차 파일(`moldset_labeled_{cn7,rg3}.csv`) 외에 같은 데이터셋의 **1차 원본**을 사용한다.
연결 방법·검증은 `src/prep_link.py`, `src/prep_base.py`의 assert로 매 실행 확인된다.

| 원본 파일 | 사용 위치 | 용도 |
|---|---|---|
| `labeled_data.csv` | `prep_base.py`, `shots_data.py` | 원래 단위 공정값·시각·LH/RH·불량 사유, 추가 기간 행 |
| `moldset_labeled.csv` | `prep_link.py`, `shots_data.py` | 2차 행 ↔ 1차 행 연결 (`_id`, 원행번호) |
| `unlabeled_data.csv` | `prep_link.py`, `prep_base.py`, `final_model.py` | 탐지기 학습용 정상 기준 샷(2020-09-01~11-30, 우진2호기, 라벨 샷·라벨 Run ±60분 제외), 변수별 기록 단위 |
| `unlabeled_data.csv` (`ERR_FACT_QTY`) | `w_prior.py` → `prob_beta.py` | 측면 배분 Beta 사전: 평가 이전 마감 로트(계획일 CN7 ≤ 10/15, RG3 ≤ 10/20)의 측면별 불량 수 비율 (CN7 LH 0.4591, RG3 LH 0.5181) |
| `unlabeled_data.csv` | `shots_data.py` → `compare.py`, `exp_label_shuffle.py`, `exp_prospective.py` | 강건 z의 비교 쇼트 흐름(라벨·비라벨 쇼트를 시각순으로 합침), 3σ 관리도 기준 = 10/16 이전 정상 모드 |

라벨(불량 여부·사유)은 평가와 확률 보정에만 쓰고, 탐지기 학습에는 쓰지 않는다.

## 5. 재현 확인

`python run_all.py`의 9단계가 아래 동결 기준값과 자동 비교한다 (허용오차 0.0015, 개수는 완전 일치).

| 항목 | CN7 공식 | CN7 추가 | RG3 공식 | RG3 추가 |
|---|---|---|---|---|
| 행 / 불량 | 1,211 / 17 | 2,745 / 11 | 1,182 / 25 | 71 / 7 |
| AP (탐지기 점수) | 0.373 | 0.797 | 0.024 | 0.115 |
| 경보(pct ≥ 0.999) / 그중 불량 | 10 / 6 | 16 / 10 | 6 / 0 | 0 / 0 |
| 확률 BSS (κ=10) | 0.056 | 0.285 | - | - |
| 비교표 AP: 기준선(강건 z) / RF | 0.191 / 0.014 | 0.250 / 0.161 | - | - |

**κ(측면 배분 Beta 사전 강도)에 대해**: 사전등록 기본값은 κ=10이며 `python run_all.py`가 이 값으로 실행된다.
점수 기반 수치(행·불량·AP·경보·TP, z·RF 비교, 초품 그리드, 3σ, 감사)는 κ와 무관하다.
확률 기반 수치(`p`, `p_lo`, `p_hi` 열과 이를 쓰는 비교표 '보정확률' 행·비용표·Brier 분해)는 κ에 따라 달라진다.
`python run_all.py --kappa 20`은 민감도 실행이며, 이때 원장 `ledger_v1.csv`가 동결 원장과 **바이트 단위로 동일**하게
재생성되는지(SHA-256)까지 9단계에서 확인한다.

`--experiments`의 E4는 보고서 인용 실험값과 비교한다 (57항목). 주요 기준값:
CN7 가이드북식 F1 진짜 0.9907 / 섞은 라벨 평균 0.9856 (라벨이 보탠 몫 +0.005), 누설 차단 RF 0.046 / 0.022,
기준선(강건 z) 0.179 / 0.023 (행 단위 순열 p 0.001). 기준선 설정 고정 후향 평가 CN7: 불량 6쇼트 중 4쇼트 경보, 경보율 8.1%(113/1,389),
Fisher p = 0.00055 — 단, 쇼트를 독립으로 보는 검정이라 시간 군집(10/27 연속 사건)을 무시하므로 보고서에서는 근거로 쓰지 않는다
(보고서 2-6절은 사건 구조를 보존한 순환 이동 검정 p = 0.007을 쓴다). RG3: 7쇼트 중 3쇼트, 경보율 36.8%, p = 0.52.
제출 모델(Mahalanobis)의 추가 기간 수치는 임계값과 설정을 결과를 본 뒤 정했으므로 '탐색 평가'다.

새 폴더 재현 확인 절차: 패키지를 빈 폴더에 복사 → `data/dataset/`에 원본 배치 → `python run_all.py` →
`outputs/ledger/headline_v1.json`의 `headline` 값과 9단계 PASS 확인.
