# KAMP-IMM

## 제품별 실행 패키지 (2026-10-08)

- **CN7:** [사용 안내](cn7/README.md) — 지우 최신 final 통합 이상 경보. `01_preprocess_train.ipynb` → `02_predict.ipynb`.
- **RG3:** [사용 안내](rg3/README.md) — 하연 가스12개 + 최고 미성형 저장 모델. `01_preprocess_train.ipynb` → `02_predict.ipynb`.
- **데이터:** [원본/KAMP/가공 데이터 구분](dataset/README.md).
- **기존 실험:** [hayeon 보관 폴더](hayeon/README.md).

두 제품은 검증 환경이 다르므로 각각 가상환경을 먼저 생성·활성화한 뒤 의존성을 설치한다. 예측 노트북은 저장 모델을 로드하며 재학습하지 않는다. 샷 기준/행 기준 지표 및 개발검증/새 독립검증을 구분한다.

## 이전 RG3 추론 안내 (과거50개 모델, 보관용)

이전158/159/157실험은 `hayeon/` 아래로이동했다. 아래안내는이전50개가스모델기록이며새배포실행순서는위제품별README를따른다.

이전 연결 재현은 [157번 보관 노트북](hayeon/notebooks/157_rg3_existing_connected_pipeline_runner.ipynb), [실행 범위](hayeon/reports/rg3_existing_pipeline_reproduction.md)를 참고한다.

[155번 이전 가스 추론](hayeon/notebooks/155_rg3_new_data_inference.ipynb), [사용 안내](hayeon/reports/rg3_inference_usage.md), [156번 검증](hayeon/notebooks/156_rg3_inference_selfcheck.ipynb)은이전모델보관용이다.

155번은 S14/RG3 저장 가스 모델의 새 입력 추론입니다. 이 경로의 가스 음성은 `미성형 모델 미연결`로 표시합니다. 기존 최고 미성형·가스 연결의 학습/개발 평가 재실행은157번, 기존 미성형 비라벨 적용은145번으로 구분합니다.

> **2026 KAMP 제조 데이터 분석 경진대회** — 사출성형(Injection Molding Machine) 과제 팀 레포

---

## 1. 과제 개요

### ① 사출성형 공정데이터 기반 품질불량 사전예측 및 검사 우선순위 결정

| 항목 | 내용 |
|---|---|
| 데이터셋 | KAMP **사출성형기 AI 데이터셋** |
| 출제 배경 | 제조기업은 공정이 끝난 뒤 품질검사로 이상을 확인하기 때문에 불량에 선제적으로 대응하기 어려움 |
| 요구사항 1 | 제공된 제조데이터로 **최종검사 이전에 제조 이상 여부를 예측하는 AI 모델** 개발 |
| 요구사항 2 | 모델이 **어떤 조건에서 정확하게 작동하고, 어떤 조건에서 실패하는지** 분석 |
| 요구사항 3 (과제명) | 예측 결과를 활용한 **검사 우선순위 결정** |

### 과제 해석 (우리 팀 관점)
- 사출성형은 **1사이클(샷, shot)마다 제품 1개**가 나오고, 데이터의 한 행이 곧 그 제품 1개의 공정 기록
- 지금은 모든 제품을 공정이 끝난 뒤 검사함 → 공정 데이터로 **제품별 불량 위험도를 미리 예측**하고, 위험도가 높은 제품부터 먼저·꼼꼼히 검사하도록 **검사 우선순위**를 정하는 것이 목표
- 따라서 단순 분류 정확도보다 **"위험도 상위 제품만 검사했을 때 불량을 얼마나 잡아내는가"**가 핵심 성과 지표
- 여기에 모델이 잘 맞는 조건과 틀리는 조건(제품, 공정 조건 등)을 함께 분석해서 제시

---

## 2. 데이터셋 정보 (가이드북 기준)

### 2.1 수집 개요
| 항목 | 내용 |
|---|---|
| 제조 분야 | 자동차 앞유리 사이드 몰딩 사출 공정 |
| 제품 | **CN7**, **RG3** (CN7 W/S SIDE MLD'G LH/RH 등) |
| 설비 예시 | 650톤-우진2호기 |
| 수집 장비 | 공장 내 사출성형기 데이터 + 품질 데이터(MES) |
| 수집 기간 | 2020-10-16 ~ 2020-11-19 (약 1개월) |
| 수집 주기 | 0.2초 |
| 라벨 방식 | 검수자가 샷마다 육안으로 양품/불량 판정 (labeled), 일별 불량 개수만 집계 (unlabeled) |

### 2.2 제공 파일 (대회 제공분)
> 가이드북에는 1·2차 가공 파일이 모두 소개되지만, **이번 대회에서는 아래 4개 파일만 제공**됨

| 파일 | 제품 | 라벨 | 행 수 | 불량(1) 수 |
|---|---|---|---|---|
| `moldset_labeled_cn7.csv` | CN7 | 있음 | 1,211 | 17 |
| `moldset_unlabeled_cn7.csv` | CN7 | 없음 | 35,239 | - |
| `moldset_labeled_rg3.csv` | RG3 | 있음 | 1,182 | 25 |
| `moldset_unlabeled_rg3.csv` | RG3 | 없음 | 35,941 | - |

- 구성: `Unnamed: 0`(원본 행 인덱스) + PassOrFail(labeled만) + **z-score 표준화된 공정변수 24개**
- TimeStamp, 설비, 불량유형(Reason) 같은 메타정보는 들어 있지 않음

### 2.3 변수 정의
| 그룹 | 변수 | 단위 |
|---|---|---|
| 시간 | Injection_Time, Filling_Time, Plasticizing_Time, Cycle_Time, Clamp_Close_Time | sec |
| 위치 | Cushion_Position, Plasticizing_Position, Clamp_Open_Position | mm |
| 속도 | Max_Injection_Speed, Max_Screw_RPM, Average_Screw_RPM | mm/s |
| 압력 | Max_Injection_Pressure, Max_Switch_Over_Pressure, Max_Back_Pressure, Average_Back_Pressure | MPa |
| 온도 | Barrel_Temperature_1~6, Hopper_Temperature, Mold_Temperature_3, Mold_Temperature_4 | °C |
| 타깃 | PassOrFail | 원본: Y(양품)/N(불량) → moldset 파일: **0 = 양품, 1 = 불량** |

- 제공 파일 기준 공정변수 24개 (원본에 있던 Switch_Over_Position, Barrel_Temperature_7, Mold_Temperature_1·2·5~12는 값이 모두 0이거나 불필요해서 가공 단계에서 제거됨)

### 2.4 도메인 요약 (가이드북 1.1)
- **사출 3단계**: 충전(Filling) → 가압(Pressurisation) → 보압(Compensating)
- **1사이클**: 건조·가소화 → 형체 → 사출 → 보압·냉각 → 계량 → 형개 → 취출 → 형체
- **주요 성형조건**: 실린더 온도, 사출압력/속도, 금형온도, 사출시간, 냉각시간
- **주요 불량 유형**: 쇼트숏(미성형), 플래시, 싱크마크, 플로우마크, 은조흔, 광택불량, 웰드라인, 기포, 변형/휨
- **현장 Pain Point**
  - 미성형·가스자국 불량이 너무 많음
  - 불량 원인 분석이 어렵고 조치에 시간이 오래 걸림
  - 원인 제거와 문제 해결 노하우가 쌓이지 않음
  - 레시피(소재량/온도/압력/속도) 세팅을 경험에 의존함
  - 제품이 냉각·수축된 **뒤에야** 불량이 확인됨 → 불량 시점에 공정을 멈추거나 제어하기 어려움

---

## 3. 가이드북 베이스라인 요약

### 3.1 데이터 품질 전처리
- 5대 품질지수: 완전성 / 유일성 / 유효성 / 일관성 / 정확성
- Null 비율이 30% 이상인 컬럼 삭제, 모든 값이 0인 컬럼 삭제, 제품(CN7·RG3)별로 추출

### 3.2 방법 A — 지도학습 기반 이상탐지 (Denoising AutoEncoder)
1. 양품/불량 분리 → 양품으로만 AE 학습 (드롭아웃으로 노이즈 추가)
2. 학습/평가 분할은 약 6:4, 스케일링 적용
3. 복원 오차(reconstruction error)의 **3σ**를 임계값으로 사용 → 넘으면 불량
4. 평가: Confusion Matrix, Accuracy / Precision / Recall / F1 (불량을 놓치면 안 되므로 **Recall 중시**)
5. 가이드북의 결론: Accuracy는 높지만 나머지 지표는 낮음 → 원인은 불량 데이터 부족, 임계값(3σ/4σ/5σ) 조정 필요

### 3.3 방법 B — 준지도학습 (Pseudo-labeling + Entropy Minimization)
1. labeled 데이터로 SVM / Random Forest / Gaussian NB / DNN(Keras) 학습 (층화추출, 하이퍼파라미터 탐색)
2. unlabeled 데이터를 예측 → **엔트로피가 낮은(확신이 높은) 상위 10%**에 pseudo-label 부여
3. labeled 데이터에 합쳐서 재학습 → unlabeled 데이터를 다 쓸 때까지 반복
4. 불균형 대응으로 오버/언더샘플링 언급, DNN은 0.5 기준으로 분류

### 3.4 환경
- Python, Jupyter, pandas, matplotlib, scikit-learn, (Keras/TensorFlow)

---

## 4. 레포 운영

- 브랜치: `main` (통합) / 개인 브랜치 (`taein` 등)에서 작업 후 PR
- 폴더 구조 (예정)
  ```
  KAMP-IMM/
  ├── data/          # 원본·가공 데이터 (git 제외)
  ├── notebooks/     # EDA, 실험 노트북
  ├── src/           # 전처리·모델 코드
  ├── reports/       # 결과보고서, 그림
  └── README.md
  ```
