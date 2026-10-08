# RG3 가스: 변동성 증가·반복 이벤트·동시 변화

## 이번에 새로 확인한 정보

- pressure_variability_growth: 직전10개 압력 변화량 표준편차 / 그 앞20개 변화량 표준편차. 분모 하한0.1, 비율 상한20. 값 수준이 아닌 변화량의 불안정 증가.
- pressure_large_change_count: 직전10개 변화 중 압력 변화가 학습 정상 변화량 절댓값95% 분위수보다 큰 횟수.
- multi_sensor_change_count: 직전10개 변화에서 사출압력·평균배압·쿠션 중2개 이상이 동시에 각 큰변화 기준을 넘은 횟수.
- 95% 기준은 테스트 구간을 제외한 학습 정상에서만 계산. 기준 하한은 압력·배압0.1, 쿠션0.01(기록 단위). 이벤트는 현재Shot을 제외한 과거에서 센다.
- 기존 단일 delta1/과거 표준편차/평균편차/경로 효율/센서 상관을 새 이름으로 재사용한 것과 달리 변화량 분산의 상대 증가와 반복·동시 이벤트를 세는 후보이다.

## 평가

- RG3 RH 정상566 + 가스17, 총583개 유지. 미성형8개 제외, LH 합치지 않음.
- 기존4변수·새3변수·현재압력2개+새패턴2개를 비교. 현재압력2개만의 대조군도 함께 넣어 원값 추가 효과를 구별.
- 같은 RF200, depth8, leaf3, balanced_subsample, seed42. 확률0.5 고정, 결측은 학습 중앙값으로 대체.
- 구간 전체 제외5회: 기존과 직접 비교. 과거구간만 학습: 후반440개/가스13개 별도 평가.
- 30분 이력 초기화 규칙은 실제Run으로 확정하지 않는다. 충분한 이력이 없는 행도 제거하지 않음.
- 상위10%는 각 평가구간별 고정 검사 예산. 미래의 구간 전체 순위를 사용하므로 배치 검사 순위이며 실시간 경보 성능과 다름.
- 반복 탐색에서 같은 데이터를 재사용하고 있으므로 독립 최종 성능을 주장하지 않음. Timestamp/연속 관측이 테스트에 제공되는지도 추가 확인 필요.

## 이력 확보

        shots  past10_complete  past30_complete
group                                          
gas        17               16               15
normal    566              556              537

## leave_one_block_out

| model | inputs | shots | gas | TP | FP | FN | TN | AP | top10_TP | top10_FP | top10_selected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_std4 | 4 | 583 | 17 | 1 | 6 | 16 | 560 | 0.1387 | 6 | 56 | 62 |
| new_temporal3 | 3 | 583 | 17 | 0 | 18 | 17 | 548 | 0.0325 | 1 | 61 | 62 |
| pressure_level_plus_temporal4 | 4 | 583 | 17 | 0 | 3 | 17 | 563 | 0.0321 | 2 | 60 | 62 |
| raw_pressure2_control | 2 | 583 | 17 | 2 | 71 | 15 | 495 | 0.0364 | 1 | 61 | 62 |

## past_blocks_only

| model | inputs | shots | gas | TP | FP | FN | TN | AP | top10_TP | top10_FP | top10_selected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_std4 | 4 | 440 | 13 | 0 | 2 | 13 | 425 | 0.1133 | 4 | 42 | 46 |
| new_temporal3 | 3 | 440 | 13 | 0 | 15 | 13 | 412 | 0.0251 | 0 | 46 | 46 |
| pressure_level_plus_temporal4 | 4 | 440 | 13 | 0 | 5 | 13 | 422 | 0.0400 | 2 | 44 | 46 |
| raw_pressure2_control | 2 | 440 | 13 | 1 | 58 | 12 | 369 | 0.0487 | 1 | 45 | 46 |

- leave_one_block_out / new_temporal3: 기존대비 AP -0.1062, TP -1, FP +12, 같은 검사예산 발견수 -5.
- leave_one_block_out / pressure_level_plus_temporal4: 기존대비 AP -0.1066, TP -1, FP -3, 같은 검사예산 발견수 -4.
- past_blocks_only / new_temporal3: 기존대비 AP -0.0882, TP +0, FP +13, 같은 검사예산 발견수 -4.
- past_blocks_only / pressure_level_plus_temporal4: 기존대비 AP -0.0733, TP +0, FP +3, 같은 검사예산 발견수 -2.

## 해석

이번 새 두 조합은 채택할 개선 근거가 없다. 전체583개에서 기존 AP0.1387/TP1/FP6에 비해 새 시간패턴3개 AP0.0325/TP0/FP18, 현재압력+새패턴4개 AP0.0321/TP0/FP3이었다. 같은62개 검사 예산에서도 기존은가스6개, 신규는각각1개·2개로 낮았다.

압력 원값2개 대조군은 TP2/FP71/AP0.0364로 탐지1개 증가보다 정상 오탐 증가가 훨씬 크다. 과거순440개/가스13개 평가에서도 신규 두 조합은TP0이며 기존 순위 성능을 넘지 못했다. 변수 추가만으로 개선되는 방향은 확인되지 않았다.

확률0.5에서 TP를 늘려도 FP가 크게 늘면 실용적 개선이라고 하지 않는다. 검사 순위와 자동판정은 별개이며, 설정을 전체평가 결과로 선택한 뒤 최고값만 보고하면 낙관적이다. 추가 임계값 조정은 이번 실험에서 하지 않았다. 비라벨은 센서 이력일 뿐 정상 정답이나 가짜 불량 라벨로 사용하지 않았다.

## 산출물

- notebooks/21_rg3_temporal_event_features.ipynb
- reports/tables/rg3_temporal_event_metrics.csv
- reports/tables/rg3_temporal_event_predictions.csv
- reports/tables/rg3_temporal_event_training_thresholds.csv
- reports/tables/rg3_temporal_event_fold_features.csv
