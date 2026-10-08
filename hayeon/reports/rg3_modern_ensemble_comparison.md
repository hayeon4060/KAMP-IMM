# RG3 최신 설치 모델과 교차 모델 앙상블

## 결론

신규 부스팅도 포함한 평균/max 앙상블은 기존 RF 결합을 개선하지 못했다. 전체5개 평균 AP0.1396/TP2/FP9, 최대 AP0.0778/TP2/FP10으로 기존 RF AP0.1686/TP4/FP6보다 낮다. CatBoost 두신호 결합 AP0.1666/TP2/FP5는 오탐1개가 적지만 찾은가스도2개 적고 과거구간 AP0.0543으로 일관된 개선이 아니다.

CatBoost는 내부검증에서 모든 잔차특성이 상수/결측인 경우 학습할 수 없어 해당 채널만중립점수0.5로 처리했다. 이는 정상/불량 균형가중치의 상수 예측에 대응하는 폴백이며 성능을 높이려 테스트라벨로 정한 값이 아니다. 설치된 최신 안정버전에서 정한 소규모설정 비교이지 모든 최신 알고리즘/최적튜닝을 소진한 결과는 아니다.

참고 설치문서: [XGBoost](https://xgboost.readthedocs.io/en/stable/install.html), [LightGBM](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html), [CatBoost](https://catboost.ai/docs/en/installation/python-installation-method-pip-install).

{'xgboost': '3.4.1', 'lightgbm': '4.7.0', 'catboost': '1.2.10', 'sktime': '1.2.0', 'numba': '0.68.0'}

RH 정상566/가스17, 기존 두 신호(흔들림4·관계잔차4) 유지. 각 알고리즘의 두점수 max결합 이후 사전고정 부스팅3개 평균, 전체5개 평균 및 최대값을 비교한다. 비지도 방식이 아니다.
모델별 설정은 사전고정이며 XGBoost 불량가중치는 각 학습집합에서만 계산한다. 결합가중치는 동일하며 테스트 정답으로 선택하지 않는다. 모든 임계값은 학습집합 내부 구간검증 정상점수의1/3/5% 상위경계다. 모델별 점수는 완전 확률보정되지 않아 평균/max에는 척도영향이 있다.
평가구간은 센서 관측30분 단절 구간이며 정확한 실제 Run ID가 아니다. 전체구간 제외와 과거학습을 분리한다. 두 결과의 평가가스는17/13으로 다르다. 원본 이전센서이력 사용가능 조건이다. 동일자료 반복탐색이며 독립 최종평가가 아니다.

| 평가 | 모델/결합 | AP | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|---:|
| leave_one_block_out | ExtraTrees | 0.1749 | 2 | 10 | 15 | 556 |
| leave_one_block_out | RandomForest | 0.1686 | 4 | 6 | 13 | 560 |
| leave_one_block_out | CatBoost | 0.1666 | 2 | 5 | 15 | 561 |
| leave_one_block_out | all5_mean | 0.1396 | 2 | 9 | 15 | 557 |
| leave_one_block_out | boost3_mean | 0.0914 | 1 | 7 | 16 | 559 |
| leave_one_block_out | XGBoost | 0.0887 | 2 | 9 | 15 | 557 |
| leave_one_block_out | all5_max | 0.0778 | 2 | 10 | 15 | 556 |
| leave_one_block_out | LightGBM | 0.0724 | 1 | 12 | 16 | 554 |
| past_blocks_only | RandomForest | 0.0812 | 2 | 4 | 11 | 423 |
| past_blocks_only | all5_mean | 0.0692 | 0 | 5 | 13 | 422 |
| past_blocks_only | boost3_mean | 0.0682 | 0 | 4 | 13 | 423 |
| past_blocks_only | XGBoost | 0.0675 | 0 | 7 | 13 | 420 |
| past_blocks_only | ExtraTrees | 0.0608 | 0 | 7 | 13 | 420 |
| past_blocks_only | LightGBM | 0.0549 | 0 | 4 | 13 | 423 |
| past_blocks_only | CatBoost | 0.0543 | 1 | 6 | 12 | 421 |
| past_blocks_only | all5_max | 0.0456 | 0 | 5 | 13 | 422 |
