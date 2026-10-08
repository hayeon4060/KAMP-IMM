# CN7 가스: 방향과 크기 분리/조건부 정상 대비 충전 편차

기준173: 8TP/5FP/F1.615. 새 전문가는 CN7 라벨만으로 학습하고 기준 앙상블에 추가한다.
현재 충전6 절대편차, 절대편차6+방향2, 조건부 정상30개 대비 절대편차6, 동시강도4를 비교했다.
조건부 정상 선택은 배럴5/호퍼/금형4 운전상태에 근거하며 학습 정상만 사용한다.
RF/ET/LR 및 평균 앙상블. 같은20 시간블록/purge10/nested5.
고정 최고는 평가 결과를 본 사후 개발 후보이며 독립 성능이 아니다.

## 고정 후보
                       model            rule     policy   N  gas  TP  FP  FN  TN  precision   recall       F1  AP_standalone
           flow_absolute6_LR        normal99 add_to_173 605   13   9   7   4 585   0.562500 0.692308 0.620690       0.184098
     local_flow_absolute6_RF        normal99 add_to_173 605   13   9   7   4 585   0.562500 0.692308 0.620690       0.014672
           flow_absolute6_ET        normal99 add_to_173 605   13   8   7   5 585   0.533333 0.615385 0.571429       0.012971
     local_flow_absolute6_LR        normal99 add_to_173 605   13   8   7   5 585   0.533333 0.615385 0.571429       0.184287
         joint_magnitude4_LR        normal99 add_to_173 605   13   8   8   5 584   0.500000 0.615385 0.551724       0.166712
  local_flow_absolute6_mean3        normal99 add_to_173 605   13   9  11   4 581   0.450000 0.692308 0.545455       0.020027
        inner_selected_mean3 inner_selection add_to_173 605   13   9  11   4 581   0.450000 0.692308 0.545455       0.020027
         joint_magnitude4_RF        normal99 add_to_173 605   13   8   9   5 583   0.470588 0.615385 0.533333       0.016619
      joint_magnitude4_mean3        normal99 add_to_173 605   13   8  10   5 582   0.444444 0.615385 0.516129       0.017389
           flow_absolute6_RF        normal99 add_to_173 605   13   8  11   5 581   0.421053 0.615385 0.500000       0.014901
        flow_absolute6_mean3        normal99 add_to_173 605   13   8  11   5 581   0.421053 0.615385 0.500000       0.016297
flow_direction_magnitude8_RF        normal99 add_to_173 605   13   8  11   5 581   0.421053 0.615385 0.500000       0.012697

## 안쪽 선택
               model            rule     policy   N  gas  TP  FP  FN  TN  precision   recall       F1  AP_standalone
inner_selected_mean3 inner_selection standalone 605   13   1   6  12 586   0.142857 0.076923 0.100000       0.020027
inner_selected_mean3 inner_selection add_to_173 605   13   9  11   4 581   0.450000 0.692308 0.545455       0.020027

## 가스13개
          TimeStamp  gas  any_fail run_id  seq  baseline  candidate    score  threshold
2020-10-16 05:21:38 True         1 cn7_R1   24         1          1 0.366368   0.911056
2020-10-16 05:37:29 True         1 cn7_R1   40         0          0 0.004988   0.757406
2020-10-16 05:45:26 True         1 cn7_R1   48         1          1 0.001866   0.757406
2020-10-16 05:47:24 True         1 cn7_R1   50         1          1 0.112979   0.757406
2020-10-16 05:48:24 True         1 cn7_R1   51         1          1 0.002447   0.757406
2020-10-16 05:49:24 True         1 cn7_R1   52         1          1 0.066841   0.757406
2020-10-16 05:50:25 True         1 cn7_R1   53         1          1 0.026008   0.757406
2020-10-16 05:51:23 True         1 cn7_R1   54         0          0 0.570449   0.757406
2020-10-16 05:52:23 True         1 cn7_R1   55         0          0 0.001842   0.757406
2020-10-16 05:53:23 True         1 cn7_R1   56         0          1 0.788116   0.757406
2020-10-16 05:54:21 True         1 cn7_R1   57         0          0 0.482872   0.974856
2020-10-16 05:56:22 True         1 cn7_R1   59         1          1 1.000000   0.974856
2020-10-16 05:57:20 True         1 cn7_R1   60         1          1 1.000000   0.974856

## 새 경보
          TimeStamp   gas  any_fail run_id  seq  baseline  candidate    score  threshold
2020-10-16 05:53:23  True         1 cn7_R1   56         0          1 0.788116   0.757406
2020-10-16 07:09:46 False         0 cn7_R1  133         0          1 0.848318   0.807021
2020-10-20 06:18:16 False         0 cn7_R2  415         0          1 0.816788   0.797804

## 전체606샷
{'TP': 9, 'FP': 8, 'FN': 4, 'TN': 585, 'precision': 0.5294117647058824, 'recall': 0.6923076923076923, 'F1': 0.6}