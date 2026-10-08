# CN7 가스 추가 개선: 상대적인 충전 상태와 동시 변화

정상592/가스13, 샷 기준. 170번의 7TP/4FP 후보를 고정하고 새로운 작은 전문가를 추가한다.
같은 시간블록20/purge10 및 중첩검증5 유지. 평가 센서가 학습 이력에 들어가는 경우0.
F1이 높은 고정 후보는 평가 결과를 보고 고른 사후 개발 최고이며 독립 성능이 아니다.
비교표의 add_to_165는 코드 재사용 명칭으로 실제로는 165+170 기준에 추가한 것이다.

## 기준
{'TP': 7, 'FP': 4, 'FN': 6, 'TN': 588, 'precision': 0.6363636363636364, 'recall': 0.5384615384615384, 'F1': 0.5833333333333334}

## 안쪽 선택
               model            rule     policy   N  gas  TP  FP  FN  TN  precision   recall       F1  AP_standalone
inner_selected_mean3 inner_selection standalone 605   13   0  14  13 578       0.00 0.000000 0.000000       0.014933
inner_selected_mean3 inner_selection add_to_165 605   13   7  18   6 574       0.28 0.538462 0.368421       0.014933

## 고정 후보 탐색
                    model     rule     policy   N  gas  TP  FP  FN  TN  precision   recall       F1  AP_standalone
          flow_signed6_LR normal99 add_to_165 605   13   9   8   4 584   0.529412 0.692308 0.600000       0.086394
           meter_load6_ET normal99 add_to_165 605   13   7   6   6 586   0.538462 0.538462 0.538462       0.020270
        meter_load6_mean3 normal99 add_to_165 605   13   7   7   6 585   0.500000 0.538462 0.518519       0.013504
   thermal_flow_joint4_LR normal99 add_to_165 605   13   7   8   6 584   0.466667 0.538462 0.500000       0.168559
   thermal_flow_joint4_RF normal99 add_to_165 605   13   7   8   6 584   0.466667 0.538462 0.500000       0.016776
           meter_load6_LR normal99 add_to_165 605   13   7   8   6 584   0.466667 0.538462 0.500000       0.012965
           meter_load6_RF normal99 add_to_165 605   13   7   9   6 583   0.437500 0.538462 0.482759       0.015839
         flow_history8_RF normal99 add_to_165 605   13   7   9   6 583   0.437500 0.538462 0.482759       0.018808
       flow_signed6_mean3 normal99 add_to_165 605   13   7  10   6 582   0.411765 0.538462 0.466667       0.017376
thermal_flow_joint4_mean3 normal99 add_to_165 605   13   7  10   6 582   0.411765 0.538462 0.466667       0.016676
          flow_signed6_RF normal99 add_to_165 605   13   7  11   6 581   0.388889 0.538462 0.451613       0.014813
          flow_signed6_ET normal99 add_to_165 605   13   7  11   6 581   0.388889 0.538462 0.451613       0.018818

## 새 경보
          TimeStamp   gas  any_fail run_id  seq  previous_prediction  candidate_prediction  new_alarm
2020-10-16 05:37:29  True         1 cn7_R1   40                    0                     1       True
2020-10-16 05:38:29 False         0 cn7_R1   41                    0                     1       True
2020-10-16 05:45:26  True         1 cn7_R1   48                    0                     1       True
2020-10-16 06:06:15 False         0 cn7_R1   69                    0                     1       True
2020-10-16 06:13:12 False         0 cn7_R1   76                    0                     1       True
2020-10-16 06:25:06 False         0 cn7_R1   88                    0                     1       True

## 남은 미탐
          TimeStamp  gas  any_fail run_id  seq  previous_prediction  candidate_prediction  new_alarm
2020-10-16 05:51:23 True         1 cn7_R1   54                    0                     0      False
2020-10-16 05:52:23 True         1 cn7_R1   55                    0                     0      False
2020-10-16 05:53:23 True         1 cn7_R1   56                    0                     0      False
2020-10-16 05:54:21 True         1 cn7_R1   57                    0                     0      False