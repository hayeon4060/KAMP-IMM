# CN7 가스: 같은 Run 정상 공분산 기반 관계 이상 보조 전문가

지도학습 기준165/170/171/173에 정상 기반 Mahalanobis 통계 전문가를 추가 비교했다.
현재 충전6/과거5 중앙값 대비 변화6/계량3/금형-속도3 변수군.
LW 수축공분산 및 ridge0.1/0.01 공분산. 모두 학습 정상만 사용한다.
같은20 시간블록/purge10/nested5. 고정 최고는 사후 개발 성적이며 독립 성능이 아니다.

## 고정 후보
                  model              rule     policy   N  gas  TP  FP  FN  TN  precision   recall       F1
          flow6_ridge01 inner_normal0.995 add_to_173 605   13   8   8   5 584   0.500000 0.615385 0.551724
       dynamic_flow6_LW inner_normal0.995 add_to_173 605   13   8   8   5 584   0.500000 0.615385 0.551724
  dynamic_flow6_ridge01 inner_normal0.995 add_to_173 605   13   8   8   5 584   0.500000 0.615385 0.551724
 dynamic_flow6_ridge001 inner_normal0.995 add_to_173 605   13   8   8   5 584   0.500000 0.615385 0.551724
      thermal_speed3_LW inner_normal0.995 add_to_173 605   13   8   8   5 584   0.500000 0.615385 0.551724
 thermal_speed3_ridge01 inner_normal0.995 add_to_173 605   13   8   8   5 584   0.500000 0.615385 0.551724
thermal_speed3_ridge001 inner_normal0.995 add_to_173 605   13   8   8   5 584   0.500000 0.615385 0.551724
          flow6_ridge01 inner_normal0.995 add_to_171 605   13   9  11   4 581   0.450000 0.692308 0.545455
       dynamic_flow6_LW inner_normal0.995 add_to_171 605   13   9  11   4 581   0.450000 0.692308 0.545455
  dynamic_flow6_ridge01 inner_normal0.995 add_to_171 605   13   9  11   4 581   0.450000 0.692308 0.545455
 dynamic_flow6_ridge001 inner_normal0.995 add_to_171 605   13   9  11   4 581   0.450000 0.692308 0.545455
      thermal_speed3_LW inner_normal0.995 add_to_171 605   13   9  11   4 581   0.450000 0.692308 0.545455
 thermal_speed3_ridge01 inner_normal0.995 add_to_171 605   13   9  11   4 581   0.450000 0.692308 0.545455
thermal_speed3_ridge001 inner_normal0.995 add_to_171 605   13   9  11   4 581   0.450000 0.692308 0.545455
               flow6_LW inner_normal0.995 add_to_173 605   13   8   9   5 583   0.470588 0.615385 0.533333

## 안쪽 선택
         model            rule     policy   N  gas  TP  FP  FN  TN  precision   recall       F1
inner_selected inner_selection standalone 605   13   2   9  11 583   0.181818 0.153846 0.166667
inner_selected inner_selection add_to_165 605   13   5  11   8 581   0.312500 0.384615 0.344828
inner_selected inner_selection add_to_170 605   13   7  13   6 579   0.350000 0.538462 0.424242
inner_selected inner_selection add_to_171 605   13   9  17   4 575   0.346154 0.692308 0.461538
inner_selected inner_selection add_to_173 605   13   8  14   5 578   0.363636 0.615385 0.457143

## 가스13개
          TimeStamp  gas  any_fail run_id  seq  baseline  candidate
2020-10-16 05:21:38 True         1 cn7_R1   24         1          1
2020-10-16 05:37:29 True         1 cn7_R1   40         0          0
2020-10-16 05:45:26 True         1 cn7_R1   48         1          1
2020-10-16 05:47:24 True         1 cn7_R1   50         1          1
2020-10-16 05:48:24 True         1 cn7_R1   51         1          1
2020-10-16 05:49:24 True         1 cn7_R1   52         1          1
2020-10-16 05:50:25 True         1 cn7_R1   53         1          1
2020-10-16 05:51:23 True         1 cn7_R1   54         0          0
2020-10-16 05:52:23 True         1 cn7_R1   55         0          0
2020-10-16 05:53:23 True         1 cn7_R1   56         0          0
2020-10-16 05:54:21 True         1 cn7_R1   57         0          0
2020-10-16 05:56:22 True         1 cn7_R1   59         1          1
2020-10-16 05:57:20 True         1 cn7_R1   60         1          1

## 전체606샷
{'TP': 8, 'FP': 9, 'FN': 5, 'TN': 584, 'precision': 0.47058823529411764, 'recall': 0.6153846153846154, 'F1': 0.5333333333333333}