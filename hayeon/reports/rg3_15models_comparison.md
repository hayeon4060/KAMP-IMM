# RG3 가스: 15가지 모델 비교

## 핵심 결론

15모델 모두 실행 완료했다. 기존 RF 예측은1023개 전부 재현됐다. 구간제외 최대점수 결합의 AP는 ExtraTrees0.1749가 RF0.1686보다 조금 높지만, 보수적 기준 TP2/FP10은 RF TP4/FP6보다 나쁘다. 과거→이후 구간에서는 ExtraTrees AP0.0608/TP0/FP7, RF AP0.0812/TP2/FP4이다. 따라서 이번 결과로 ExtraTrees를 기존 모델의 개선판이라고 채택하지 않는다. 현재 피처에서 모델 교체만으로 안정적인 개선을 확인하지 못했다.

GaussianNB는 상수 특성 구간에서 분산이0이 되는 문제를 막기 위해 학습된 분산에1e-8 하한을 적용했다. 선형/거리 기반 모델은 학습집합의 StandardScaler를 적용하고 트리 모델은 원래 값으로 학습했다. QDA 등 수치 관련 경고는 별도 경고 CSV에 보관했다.

RH 정상566/가스17. 기존 흔들림4개와 관계잔차4개는 그대로 유지하고, 각 모델마다 두 분류기를 따로 학습한다. 두 점수 중 최대값 결합과 두 단독 채널을 모두 저장한다. 미성형은 제외한다.
잔차 학습은 학습 구간 내 정상 참조를 이용하며, 분류기는 정상/가스 라벨을 모두 사용한다. 구간 제외 검증과 과거 구간만 학습한 검증을 별도 보고한다. imputer/scaler는 학습 데이터에만 적합한다.
판정 기준은 학습 내부 구간검증 정상 점수의 상위1/3/5% 경계로 고정하며, 외부 평가 정상의 오탐률을 보장하지 않는다. SVC/Ridge의 결정점수는 sigmoid로 변환하지만 확률 보정이 아니다. 모델별 출력 척도가 달라 max 결합의 효과도 함께 달라질 수 있다.
하이퍼파라미터는 이번 비교 전에 고정했으며 15모델 중 최고를 동일 평가자료에서 선택하므로 독립적인 최종 평가가 아니다. 기존 반복탐색 영향도 있다. AP는 PR-AUC와 관련되지만 동일한 계산 방식은 아니다.
XGBoost/LightGBM/CatBoost는 설치되어 있지 않아 제외하고 scikit-learn 15모델을 비교한다. 새로운 패키지는 설치하지 않았다.
이전10샷 원본 센서 이력을 쓸 수 있다는 조건이며 대회 테스트 이력 제공 여부는 확인 필요. RPM 정의 불확실한 기존 잔차는 원인 설명에 쓰지 않는다. 초반 두구간만 내부 학습에 남으면 잔차 학습특성이 결측일 수 있으며 학습집합 median/0 대체를 한다.

## 최대점수 결합, 보수적 기준

| 평가 | 모델 | AP | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|---:|
| leave_one_block_out | ExtraTrees | 0.1749 | 2 | 10 | 15 | 556 |
| leave_one_block_out | RandomForest | 0.1686 | 4 | 6 | 13 | 560 |
| leave_one_block_out | GaussianNB | 0.1236 | 0 | 1 | 17 | 565 |
| leave_one_block_out | KNN | 0.1196 | 2 | 17 | 15 | 549 |
| leave_one_block_out | HistGradientBoosting | 0.0804 | 2 | 14 | 15 | 552 |
| leave_one_block_out | QDA | 0.0786 | 2 | 19 | 15 | 547 |
| leave_one_block_out | SVC_RBF | 0.0628 | 0 | 8 | 17 | 558 |
| leave_one_block_out | AdaBoost | 0.0593 | 0 | 0 | 17 | 566 |
| leave_one_block_out | LDA | 0.0559 | 1 | 8 | 16 | 558 |
| leave_one_block_out | LogisticRegression | 0.0528 | 1 | 8 | 16 | 558 |
| leave_one_block_out | RidgeClassifier | 0.0496 | 1 | 9 | 16 | 557 |
| leave_one_block_out | GradientBoosting | 0.0476 | 1 | 15 | 16 | 551 |
| leave_one_block_out | SGDLogistic | 0.0468 | 1 | 9 | 16 | 557 |
| leave_one_block_out | SVC_Linear | 0.0402 | 0 | 7 | 17 | 559 |
| leave_one_block_out | DecisionTree | 0.0397 | 0 | 6 | 17 | 560 |
| past_blocks_only | RandomForest | 0.0812 | 2 | 4 | 11 | 423 |
| past_blocks_only | GradientBoosting | 0.0808 | 2 | 19 | 11 | 408 |
| past_blocks_only | GaussianNB | 0.0707 | 1 | 8 | 12 | 419 |
| past_blocks_only | QDA | 0.0694 | 1 | 39 | 12 | 388 |
| past_blocks_only | ExtraTrees | 0.0608 | 0 | 7 | 13 | 420 |
| past_blocks_only | HistGradientBoosting | 0.0602 | 1 | 11 | 12 | 416 |
| past_blocks_only | KNN | 0.0528 | 3 | 42 | 10 | 385 |
| past_blocks_only | LDA | 0.0507 | 0 | 0 | 13 | 427 |
| past_blocks_only | LogisticRegression | 0.0495 | 0 | 0 | 13 | 427 |
| past_blocks_only | RidgeClassifier | 0.0469 | 0 | 0 | 13 | 427 |
| past_blocks_only | AdaBoost | 0.0438 | 0 | 21 | 13 | 406 |
| past_blocks_only | SVC_RBF | 0.0431 | 1 | 47 | 12 | 380 |
| past_blocks_only | SGDLogistic | 0.0423 | 0 | 0 | 13 | 427 |
| past_blocks_only | SVC_Linear | 0.0419 | 0 | 3 | 13 | 424 |
| past_blocks_only | DecisionTree | 0.0412 | 0 | 3 | 13 | 424 |

전체 채널 및 기준별 결과는 rg3_15models_metrics.csv, 샷별 결과는 rg3_15models_predictions.csv에 저장했다. RF 최대점수 예측1023개는 기존 결과와 정확히 일치하는지 확인했다.

실패 모델: []
모델 경고 건수: 0. 경고 파일을 함께 확인한다.
