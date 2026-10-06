# RG3 가스 후보 선택 및 입력 데이터

브랜치: hayeon. 기존 코드·예측 결과는 73b61b1에 보존되어 있으며, 이번 변경은 후보별 안내·입력 데이터·검증 노트북을 보충한다. CN7 실험, 새 가스19개 데이터, SmartFactory 모델은 이 묶음에 포함하지 않는다.

## 선정 후보

| 후보 | 찾은 가스 TP | 정상 오탐 FP | 놓친 가스 FN | 정상 맞춤 TN | F1 |
|---|---:|---:|---:|---:|---:|
|22 RF 표준편차 + 정상 관계 잔차|4|6|13|560|0.296|
|47 현재 변화 확인 + 3 seed 평균|5|6|12|560|0.357|
|60 온도 다수결 + 압력 다수결|7|6|10|560|0.467|
|60 온도 탐색 앵커 + 압력 다수결|8|6|9|560|0.516|

모두 정상566+가스17=583 Shot의 관측구간 제외 개발 평가다. 미성형8은 정상으로 바꾸지 않고 제외한다. 여러 실험을 보고 선정한 탐색 성적이며 공식 테스트 또는 새 데이터 독립 성적이 아니다. 마지막 두 후보의 과거→이후 평가는 TP3/FP4/FN10/TN423/F1=0.300이다. 추가37개에 대한 다른 고정 프로토콜의 검증도 실패했으며, 위 8/6 앙상블의 새로운 일반화 성적을 입증한 결과는 없다.

## 실제 사용 입력 파일

- `data/original_split/moldset_labeled_rg3.csv`: original/moldset_labeled.csv에서 제품 RG3를 추출한 1,182행. RH/LH 두 행 구조, Timestamp와 원본 센서 및 정답 포함.
- `data/original_split/moldset_unlabeled_rg3.csv`: original/unlabeled_data.csv의 RG3 센서 이력 72,484행. 정상/불량 정답으로 새 라벨링하지 않고 과거 센서 파생변수 계산에 활용.
- `data/audit/rg3_rh_labeled_with_timestamp.csv`: 기존 라벨의 RG3 RH 591행. 정상566, 가스17, 미성형8. 원본 정답 및 Timestamp 대조용.
- `reports/tables/original_labeled_shot_features.csv`: 기존 10번에서 생성한 실제 입력 테이블. CN7/RG3 전체1,304 Shot이며 각 모델에서 RG3 591 Shot만 선택한다. 현재−직전 차이, 과거10 Shot 표준편차·편차 등이 포함된다. 실제 사용 파일을 보존하기 위해 CN7 행을 삭제하거나 새 계산으로 바꾸지 않았다.

전체 original CSV, PDF, 가상환경은 포함하지 않는다. 데이터 라이선스는 원 KAMP 제공 조건을 따르며 별도 자유 라이선스를 부여하지 않는다. 위 파일은 user 요청으로 기존 data/ Git 제외 규칙의 예외로 명시적으로 추가했다.

## 실행 및 확인

저장소 루트에서 기존 `.venv`를 활성화하고 requirements.txt 환경을 사용한다. 소스는 모두 ipynb다.

1. `125_rg3_selected_candidate_data_verification.ipynb`로 데이터 행수·정답·Timestamp·SHA256 및 저장 예측의 혼동행렬을 확인한다. 이 노트북은 재학습이 아니라 패키징 검증이다.
2. 학습부터 다시 실행하려면 기존 `22 → 42 → 46 → 47 → 54 → 56 → 57 → 58 → 59 → 60` 순서를 따른다. 일부 노트북은 이전 소스 셀과 저장 앵커 예측을 사용하므로 마지막 노트북만 단독 실행하는 구조가 아니다.
3. `61_rg3_f1target_candidate_verification.ipynb`로 최고 후보를 RH 원본 정답과 대조한다.

22번의 마지막 재현 비교에는 이미 추적 중인 `rg3_relational_feature_oof_predictions.csv`가 필요하다. 처음부터 전체 original을 추출하는 09/10번은 대용량 원본이 별도로 필요하지만, 이번에 동봉한 기존 입력 테이블로는 해당 추출 단계를 반복하지 않고 모델 단계부터 실행할 수 있다.

자세한 모델·피처·근거는 `rg3_gas_8tp_6fp_snapshot.md` 및 `rg3_f1_050_target_review.md`를 참고한다. 후보별 집계는 `tables/rg3_selected_candidates_verified.csv`, 파일 무결성은 `tables/rg3_selected_input_manifest.csv`에 기록한다.
