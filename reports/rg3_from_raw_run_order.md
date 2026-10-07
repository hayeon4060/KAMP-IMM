# RG3 실행 순서 — 로컬 원본부터 시작

## 한눈에 보기

**로컬 원본7개 준비 → 158 전처리 → 159 가스 중간 결과 → 157 연결 실행**

검증: 코드와 원본7개만 있는 임시 폴더에서 전처리·가스 중간 결과·연결8단계를 실행해 기존 결과를 재현했다. 필요한148/151 및op_metrics 누락도 이 과정에서 보완했다. 기록은 `tables/rg3_from_raw_execution_check.json`에 있다.

이번 보충 커밋에는 전처리·실행 코드와 안내만 추가한다. 원본 CSV는 각자 로컬에 준비한다. 이전 커밋에는 일부 데이터가 실제로 들어가 있으며 이번에 자동 삭제하지 않는다.

## 1. 환경 준비

hayeon 브랜치를 내려받고 저장소 루트에서 실행한다. 현재 검증 환경은 Python3.14이며 requirements.txt 버전을 사용한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

노트북 커널도 해당 가상환경의 Python으로 선택한다. 환경 생성·활성화 후 설치한다.

## 2. 입력 파일

아래7개를 **data/raw_input/**에 둔다.

|파일|구분|
|---|---|
|moldset_labeled.csv|제공받은1차 원본 라벨|
|labeled_data.csv|제공받은1차 전체 라벨|
|unlabeled_data.csv|제공받은1차 비라벨|
|moldset_labeled_cn7.csv|대회2차 가공 라벨|
|moldset_labeled_rg3.csv|대회2차 가공 라벨|
|moldset_unlabeled_cn7.csv|대회2차 가공 비라벨|
|moldset_unlabeled_rg3.csv|대회2차 가공 비라벨|

**원본을 제품별로 자른 파일을 대회2차 파일 자리에 넣으면 안 된다.** 같은 파일명이어도 센서 스케일·행 구성이 다르다. supervised_label_cn7.csv는 이 RG3 흐름에서 사용하지 않는다.

## 3. 실행할 세 노트북

|순서|파일|하는 일|주요 출력|
|---|---|---|---|
|1|158_rg3_prepare_from_local_raw.ipynb|원본 분리·과거 센서 피처·1차/2차 연결·Shot/Run·정상 참조 준비|data/original_split; original_labeled_shot_features.csv; data/taein/linked; base/rg3_shots.csv; outputs/final_model/rg3_train_features.csv 및 rg3_scores.csv|
|2|159_rg3_rebuild_gas_intermediates.ipynb|가스 학습·구간별 경계·중간 예측을 재생성|reports/tables의rg3_nested/rank_fusion/innovation/pdf/pressure 결과|
|3|157_rg3_existing_connected_pipeline_runner.ipynb|미성형 후보 및 가스→미성형 연결 실행|rg3_shortshot_143_fixed_candidate_* 및 rg3_153_gas_best_shortshot_cascade_*|

모두 저장소 루트 기준으로 실행한다. 개인 컴퓨터 절대경로를 수정하지 않는다.

## 4. 내부 실행 순서

158번이 아래 전처리를 자동 수행한다.

1. **09_original_product_splits**: 원본을 제품별로 분리.
2. **10_original_defect_report_revision**: Shot 통합 및 과거10 Shot 피처.
3. **preprocessing/01_link_primary_secondary**: 1차 센서·Timestamp와 대회 가공본 연결 → data/taein/linked.
4. **preprocessing/00_base_labeled**: 전체 라벨의 Shot/Run 정리 → data/taein/base.
5. **final_model.train(products=('rg3',))**: 기존20a의 RG3 참조 학습 함수 → rg3_train_features.
6. **final_model.test(products=('rg3',))**: 기존20b의 labeled 피처·점수 함수 → rg3_scores.

20a/20b 노트북을 따로 돌릴 필요는 없다.158이 기존 함수들을 직접 호출한다. Git에는 공통 코드semi_normal.py/semi_sup.py/final_model.py/op_metrics.py를 함께 제공한다.

159번 내부 순서:

```text
16 → 22 → 42 → 46 → 47 → 54 → 56 → 57 → 58 → 59 → 60
```

157번 내부 순서:

```text
133 → 139 → 141 → 143 → 148 → 151 → 152 → 153
```

## 5. 최종 결과 확인

**reports/tables/rg3_153_gas_best_shortshot_cascade_metrics.csv**를 확인한다.

|평가|TP|FP|FN|TN|F1|
|---|---:|---:|---:|---:|---:|
|가스만|8|6|9|560|0.5161|
|미성형만|6|11|2|555|0.4800|
|연결 전체 불량|14|17|11|549|0.5000|

기존 정상566/가스17/미성형8 자료와 설정의 개발 평가 재현 목표다. 숫자가 다르면 assertion을 지우거나 결과를 맞추지 말고 입력 버전·대회 파일·환경부터 확인한다. 새로운 독립 테스트 성적이 아니다.

## 6. 다른 실행 목적

- **기존 미성형 비라벨 적용**:157에서 RUN_UNLABELED_APPLICATION=True로145 추가 실행. 정답이 없어F1 계산 불가.
- **학습 없이 저장 가스 모델로 새 CSV 추론**:155. 원본 센서 단위와 이전 이력이 필요하다.
- **가스 저장 모델 재내보내기**:126 및154는 별도 적용/내보내기 흐름으로 추가 원본 대조 입력이 필요하다. 연결 평가 재현에는 실행하지 않는다.

가스 제거 후 과거 센서 창을 다시 계산하지 않으며 실제 정답으로 단계 분기하지 않는다.
