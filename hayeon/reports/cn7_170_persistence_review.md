# CN7 가스: 놓친 8개와 지속형 관계 이상

## 기존 후보

{'TP': 5, 'FP': 2, 'FN': 8, 'TN': 590, 'precision': 0.7142857142857143, 'recall': 0.38461538461538464, 'F1': 0.5}

기존165: 가스13 중5 탐지, 정상오탐2. 시험 결과를 보고 만든 개발 후보이며 독립 성능이 아니다.

## 이번 변경

기존 순간 잔차에 절대편차와 과거5샷 지속 강도를 추가. 정상 관계는 각 학습 구간의 정상으로만 계산한다.
RF·ET·LR 3모델 평균 앙상블. 같은 시간블록/purge10 유지, 평가 기록은 학습 이력에서 제거.
학습은 평가 구간보다 미래 자료도 포함할 수 있어 실시간 미래검증은 아니다.

## 안쪽 검증으로 선택한 결과

               model            rule     policy   N  gas  TP  FP  FN  TN  precision   recall  F1  AP_standalone
inner_selected_mean3 inner_selection standalone 605   13   0   0  13 592   0.000000 0.000000 0.0       0.036512
inner_selected_mean3 inner_selection add_to_165 605   13   5   2   8 590   0.714286 0.384615 0.5       0.036512

## 고정 후보 탐색 결과(외부 성적을 본 사후 최고; 독립 성능 아님)

                 model     rule     policy   N  gas  TP  FP  FN  TN  precision   recall       F1  AP_standalone
    persistent_abs4_ET normal95 add_to_165 605   13   7   4   6 588   0.636364 0.538462 0.583333       0.030621
    persistent_abs4_ET normal97 add_to_165 605   13   7   4   6 588   0.636364 0.538462 0.583333       0.030621
   relation_signed4_ET normal99 add_to_165 605   13   5   2   8 590   0.714286 0.384615 0.500000       0.042827
relation_signed4_mean3 normal97 add_to_165 605   13   5   2   8 590   0.714286 0.384615 0.500000       0.036447
relation_signed4_mean3 normal99 add_to_165 605   13   5   2   8 590   0.714286 0.384615 0.500000       0.036447
      relation_abs4_LR normal99 add_to_165 605   13   5   2   8 590   0.714286 0.384615 0.500000       0.043408
      relation_abs4_RF normal99 add_to_165 605   13   5   2   8 590   0.714286 0.384615 0.500000       0.017728
      relation_abs4_ET normal99 add_to_165 605   13   5   2   8 590   0.714286 0.384615 0.500000       0.038545
   relation_abs4_mean3 normal99 add_to_165 605   13   5   2   8 590   0.714286 0.384615 0.500000       0.017808
    persistent_abs4_LR normal99 add_to_165 605   13   5   2   8 590   0.714286 0.384615 0.500000       0.015864
    persistent_abs4_RF normal99 add_to_165 605   13   5   2   8 590   0.714286 0.384615 0.500000       0.019382
    persistent_abs4_ET normal99 add_to_165 605   13   5   2   8 590   0.714286 0.384615 0.500000       0.030621

## 기존에서 놓친8개

          TimeStamp heldout_block  train_gas  Max_Injection_Pressure_residual_z  Max_Injection_Pressure_abs_normal_percentile  Max_Injection_Pressure_past5_abs_mean  Plasticizing_Time_residual_z  Plasticizing_Time_abs_normal_percentile  Plasticizing_Time_past5_abs_mean  Cushion_Position_residual_z  Cushion_Position_abs_normal_percentile  Cushion_Position_past5_abs_mean  Mold_Temperature_4_residual_z  Mold_Temperature_4_abs_normal_percentile  Mold_Temperature_4_past5_abs_mean
2020-10-16 05:37:29   cn7_R1__B02          1                          -0.321592                                      0.333922                               1.322004                     -0.632075                                 0.846290                          0.496081                    -0.147823                                0.234982                         0.066883                       1.039157                                  0.551237                           1.059172
2020-10-16 05:45:26   cn7_R1__B02          1                          -0.321592                                      0.333922                               0.696112                     -0.318854                                 0.522968                          0.494981                     2.001453                                0.991166                         0.103453                       0.949145                                  0.491166                           1.047599
2020-10-16 05:48:24   cn7_R1__B02          1                          -1.269628                                      0.885159                               0.346816                     -0.808069                                 0.927562                          0.293305                    -0.064177                                0.100707                         0.507642                       1.013439                                  0.535336                           1.052373
2020-10-16 05:50:25   cn7_R1__B02          1                          -1.169613                                      0.862191                               0.593567                     -0.469518                                 0.687279                          0.275199                    -0.064177                                0.100707                         0.479265                       1.330268                                  0.810954                           1.113167
2020-10-16 05:51:23   cn7_R1__B02          1                          -1.938739                                      0.948763                               0.763172                     -0.596830                                 0.821555                          0.305332                     0.019456                                0.033569                         0.091810                       1.354135                                  0.819788                           1.189392
2020-10-16 05:52:23   cn7_R1__B02          1                          -0.833455                                      0.614841                               1.122348                     -0.985602                                 0.964664                          0.411544                    -0.064177                                0.100707                         0.066136                       1.746267                                  0.984099                           1.258817
2020-10-16 05:53:23   cn7_R1__B02          1                           0.035868                                      0.017668                               1.276210                     -0.834938                                 0.938163                          0.582779                    -0.089562                                0.113074                         0.049407                       1.621380                                  0.966431                           1.345874
2020-10-16 05:54:21   cn7_R1__B03          2                           0.288429                                      0.270796                               1.018201                     -0.785855                                 0.916814                          0.593978                    -0.134632                                0.230088                         0.046802                       1.287245                                  0.791150                           1.475363

정상 대비 백분위는 이상 강도이며 불량 확률이 아니다. 정상과 겹치는 사례는 센서만으로 구분 불가능하다고 단정할 수 없지만, 이 관계 피처에서 분리된다고도 주장할 수 없다.