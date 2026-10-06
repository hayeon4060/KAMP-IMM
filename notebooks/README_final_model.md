# 최종 모델 실행 가이드 (M_maha_free)

사출성형 공정데이터 기반 품질불량 사전예측 및 검사 우선순위 결정 — 최종 모델 학습·테스트

## 1. 모델 한눈에 보기

| 항목 | 내용 |
|---|---|
| 방식 | **비지도 이상탐지**: 정상 생산 샷만 학습하고, 라벨은 평가에만 사용 |
| 학습 데이터 | unlabeled(1차) 정상 생산 샷, 2020-09-01 ~ 11-30, 650톤 우진2호기 |
| 테스트 데이터 | labeled. ① 공식 2차 행 단위 / ② labeled 전체 샷 단위(초기허용불량 제외) |
| 입력 변수 | 인과적 파생변수 17개. 같은 Run의 **과거 샷만** 사용하므로 미래 정보가 없음 |
| 알고리즘 | Ledoit-Wolf 공분산 + Mahalanobis 거리 |
| 출력 | 샷별 이상 점수와 정상 대비 백분위(pct). `pct ≥ q`면 검사 대상 |

## 2. 파일 구성

```
notebooks/
├── final_model.py              ← 최종 모델 코드 (이 파일 하나로 동작)
├── 20a_final_train.ipynb       ← ① 학습 (unlabeled → 모델 저장)
├── 20b_final_test.ipynb        ← ② 테스트 (저장된 모델 → labeled 평가)
├── 20c_pr_auc.ipynb            ← ③ PR-AUC·PR 곡선 (저장된 모델 사용)
├── 20_final_model.ipynb        ← (참고) 학습+테스트 한 번에
├── README_final_model.md       ← 이 문서
├── linked/
│   ├── linked_unlabeled_cn7.csv
│   ├── linked_unlabeled_rg3.csv
│   └── linked_labeled.csv
└── base/
    ├── cn7_shots.csv
    ├── rg3_shots.csv
    └── reference_scales.json
```

### 입력 파일과 만드는 노트북

| 파일 | 내용 | 만드는 노트북 |
|---|---|---|
| `linked/linked_unlabeled_{cn7,rg3}.csv` | 1차 unlabeled 원값 (`TimeStamp`, `*_raw`) | `01_link_primary_secondary.ipynb` |
| `linked/linked_labeled.csv` | 공식 2차 행 ↔ 1차 시각 연결 (`product`, `in_secondary`, `TimeStamp`, `cavity`, `PassOrFail`, `Reason`) | `01_link_primary_secondary.ipynb` |
| `base/{cn7,rg3}_shots.csv` | labeled 샷 단위 (`run_id`, `split`, `PassOrFail`, `Reason`, `*_raw`) | `00_base_labeled.ipynb` |
| `base/reference_scales.json` | 변수별 기록 단위, 흐름저항 하한 | `06_loro_compare.ipynb` |

- 입력 파일이 이미 있으면 위 노트북은 다시 돌릴 필요가 없어요.
- `reference_scales.json`이 없어도 실행은 돼요(reference로 대신 계산). 다만 기존 결과와 수치가 조금 달라질 수 있어요.

## 3. 환경

```bash
pip install numpy pandas scikit-learn jupyter
```

- Python 3.9 이상
- 팀 공용 conda 환경(`kamp`)이 있으면 그대로 사용

## 4. 실행 방법

### 방법 A: 노트북 (권장)

1. `notebooks` 폴더에서 Jupyter 실행
2. **`20a_final_train.ipynb`** 실행 → 학습
   - unlabeled만 사용 (라벨 X)
   - 결과: `outputs/final_model/{cn7,rg3}_final_model.pkl`, 학습 데이터 처리 기록, 학습 파라미터, 임계값
3. **`20b_final_test.ipynb`** 실행 → 테스트
   - 저장된 `.pkl`을 불러와 labeled에 점수 → 평가
   - **0번 셀 `일치`가 모두 `O`인지 확인**
4. 각 노트북 저장(Ctrl+S) → 출력이 노트북에 남음

학습을 한 번 해두면 테스트는 `20b`만 다시 돌리면 돼요.

### 방법 B: 파이썬 코드

```python
# 작업 폴더가 notebooks 여야 함
from final_model import train, test

T = train()                  # 학습 → outputs/final_model/{p}_final_model.pkl
print(T['data_log'])         # 단계별 샷 수
print(T['params'])           # 변수별 중앙값·IQR

R = test()                   # 저장된 모델로 테스트
print(R['check'])            # 재현 확인
print(R['frame1'])           # 평가 ① 공식 2차 · 행 단위
print(R['frame2'])           # 평가 ② 전체 · 샷 단위
print(R['threshold'])        # 임계값별 경보 성능
```

다른 폴더에서 실행하면 경로를 지정하세요: `train(base_dir='경로/notebooks', out_dir='경로/outputs/final_model')` (test도 동일)

## 5. 출력 파일 (`outputs/final_model/`)

| 파일 | 내용 |
|---|---|
| `check.csv` | 재현 확인: 18번 AP와 일치 여부 |
| `data_log.csv` | [학습] reference 처리 단계별 샷 수 (unlabeled → 필터 → 최종 학습 샷) |
| `train_params.csv` | [학습] 변수별 중앙값·IQR |
| `{cn7,rg3}_train_features.csv` | [학습] **모델에 실제로 들어간 학습 데이터** — Run·시각, 원값 21개, 파생변수 17개, 학습 데이터 자신의 점수·백분위 (CN7 5,179샷) |
| `frame1.csv` | 평가 ① 지표 |
| `frame2.csv` | 평가 ② 지표 |
| `threshold.csv` | q 임계값별 혼동행렬, Precision, Recall, F1 |
| `{cn7,rg3}_scores.csv` | [테스트] **모델에 들어간 테스트 데이터** — labeled 샷별 17개 변수 값, 점수(`score`), 백분위(`pct`), 라벨 |
| `{cn7,rg3}_final_model.pkl` | [학습] 학습된 모델 (기준값 포함) — 테스트·새 샷 적용에 사용 |

## 6. 기대 결과 (재현 기준)

| 제품 | ① 공식 2차 · 행 AP | ② 전체 · 샷 AP |
|---|---|---|
| CN7 | 0.373 | 0.212 |
| RG3 | 0.024 | (R6 제외 후 새 값) |

- **RG3 R6 제외 (팀 결정)**: R6(2020-11-04)은 공식 2차 범위 밖 → `final_model.py`의 `EXCLUDE_RUNS`로 평가·지도학습에서 제외
  - 공식 2차 기준과 같은 불량 수: RG3 가스 17 / 미성형 8 (R6 포함 시 22 / 10)
  - 행은 지우지 않고 `split='제외'`로만 표시 → 파생변수는 전체 시간 흐름 그대로, 학습(reference)에서도 이 시간대는 계속 빠짐

## 7. 처리 과정

### 학습 데이터 (reference) 만들기

1. LH/RH 두 행 → 샷 1행
2. labeled 샷과 labeled Run 시간대 ±60분 제거 (평가 데이터 누수 방지)
3. 기간 2020-09-01 ~ 11-30
4. 끼어든 이상 기록 제거 (간격 50초 미만 + 사이클이 그날 중앙값에서 2초 이상 차이)
5. Run 구분 (60분 공백 = 새 Run), 10샷 미만 Run 제거
6. 0 모드(배럴1 온도 < 100℃, 비가동 기록) 제거

### 파생변수 17개 (학습·테스트에 같은 함수, 같은 기준값 적용)

| 계열 | 변수 | 의미 |
|---|---|---|
| 이벤트 편차 (8) | `dev_` + 사출속도·충전시간·사출압·전환압·배압·쿠션위치·계량시간·사이클 | 같은 Run 직전 20샷 대비 robust z |
| 이벤트 요약 (2) | `fill_anomaly`, `ev_max_abs_dev` | 충전시간 편차 − 속도 편차(음수 = 미성형형), 편차 최대 절대값 |
| 열 상태 (6) | `dev_Mold_Temperature_3`, `dev_Hopper_Temperature`, `mold_rise`, `hopper_rise`, `mold_slope20`, `hopper_slope20` | 온도 순간 이탈, Run 시작 대비 상승량, 최근 20샷 기울기 |
| 흐름저항 (1) | `dev_flow_res` | 사출압 / 속도(점도 대리 지표)의 편차 |

**편차 계산식**: `dev = (x − 기준선 중앙값) / max(1.4826·MAD, 기록 단위)`

- 기준선 = 같은 Run 과거 샷 중 정상으로 받아들인 최근 20샷
- |dev| > 5인 샷은 기준선에 넣지 않음
- 5샷 연속으로 제외되면 기준선 재설정
- Run 첫 5샷은 편차 없음(0)으로 처리

### 학습 · 점수

1. reference 중앙값·IQR로 스케일 → ±20 클리핑
2. Ledoit-Wolf 공분산 학습
3. 점수 = Mahalanobis 거리
4. pct = reference 점수 분포 대비 백분위

모든 설정값은 결과를 보기 전에 고정했어요(`final_model.py` 상단 '설정').

## 8. 평가지표

| 지표 | 의미 |
|---|---|
| **AP** (주 지표) | 불량을 순위 위쪽에 얼마나 몰았는지. 무작위면 불량률과 같음. PR 곡선 아래 면적을 계단식으로 계산한 PR-AUC |
| PR-AUC(사다리꼴) | PR 곡선 아래 면적을 점 사이 직선으로 계산. 불량이 적으면 AP보다 다소 높게(낙관적으로) 나옴 → 참고 지표 |
| 무작위 평균 대비 · 유의 | 무작위 순위 1,000번 대비 배수. 97.5% 지점을 넘으면 유의(O) |
| 상위 k% | 점수 상위 k%만 검사할 때 잡는 불량 수 |
| F1@5% · 10% | 상위 k%를 불량으로 판정했을 때의 F1 (라벨로 임계값을 고르지 않음) |
| F1(최적, 참고) | 정답을 보고 고른 최적 임계값의 F1. 참고용 |
| 상한 F1 | 같은 입력(LH/RH)은 같은 판정을 받는다는 제약에서의 최대 F1 |
| AP 95% CI | 부트스트랩 신뢰구간 (불량 수가 적어 넓음) |
| Run별 상위 k% | Run마다 상위 k%를 검사할 때 |
| 유형별 AP | 가스 / 미성형 각각 |
| q 임계값 성능 | `pct ≥ q`로 판정. 경보 비율 ≈ 1 − q 이면 보정이 잘 된 것 |

## 9. 새 샷에 적용하기

```python
from final_model import score_new_shots

f = score_new_shots(new_shots, 'cn7')    # new_shots: run_id, TimeStamp, *_raw
alarm = f[f['pct'] >= 0.999]             # 정상 오경보율 0.1% 기준 검사 대상
```

- 편차 계산에 **같은 Run의 직전 20샷**이 필요하므로 과거 샷도 함께 넣어야 해요.
- 비가동(0 모드) 샷은 점수 계산 전에 걸러내야 해요.

## 10. 문제 해결

| 증상 | 확인할 것 |
|---|---|
| `... final_model.pkl 없음` | `20a_final_train.ipynb`(train)를 먼저 실행 |
| `FileNotFoundError` | 작업 폴더가 `notebooks`인지, 입력 파일 4종이 2장 구조대로 있는지 |
| 0번 셀 `확인 필요` | `reference_scales.json` 유무 확인 → 4번 셀(캐시 비교)에서 값이 다른 변수 확인 |
| `KeyError: '..._raw'` | 입력 CSV가 1차 원값(`*_raw`) 열을 가진 버전인지 |
| 실행이 느림 | 정상이에요. Run별 기울기 계산 때문에 수 분 걸려요 |
