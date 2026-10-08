# CN7 가스: 정상 기반 동시 변화 보조 전문가

지도학습 모델을 고정하고 정상 분포 기반 2변수 tail coincidence 전문가를 추가했다.
각 센서의 학습 정상 대비 백분위를 계산하고 두 신호 백분위의 최소값으로 동시성을 측정한다.
냉각+속도, 냉각+짧은 충전, 계량/쿠션+긴 가소화, 배압+RPM 등을 검토했다.
과거 변화는 현재 샷을 제외한 과거10샷 중앙값 대비. 평가 기록은 학습 이력에서 제거.
정상 순위 모델은 비지도/통계 보조 모델이며 전체를 순수 지도학습이라고 부르면 안 된다.
같은20 시간블록/purge10/nested5. 미래 시점이 학습에 포함될 수 있다.
고정 최고는 외부 평가 결과로 선택한 개발 성적이며 최종 독립 성능이 아니다.

## 고정 후보
              model              rule     policy   N  gas  TP  FP  FN  TN  precision   recall       F1
          cold_fast inner_normal0.995 add_to_171 605   13  10  11   3 581   0.476190 0.769231 0.588235
          cold_fast inner_normal0.995 add_to_170 605   13   8   7   5 585   0.533333 0.615385 0.571429
transient_fast_fill inner_normal0.995 add_to_171 605   13   9  10   4 582   0.473684 0.692308 0.562500
       cooling_fast inner_normal0.995 add_to_171 605   13  10  13   3 579   0.434783 0.769231 0.555556
      cold_flow_max inner_normal0.995 add_to_171 605   13  10  13   3 579   0.434783 0.769231 0.555556
  cooling_shortfill inner_normal0.995 add_to_171 605   13   9  11   4 581   0.450000 0.692308 0.545455
    speed_shortfill inner_normal0.995 add_to_171 605   13   9  11   4 581   0.450000 0.692308 0.545455
  meter_longplastic inner_normal0.995 add_to_171 605   13   9  11   4 581   0.450000 0.692308 0.545455
 load_highbp_lowrpm inner_normal0.995 add_to_171 605   13   9  11   4 581   0.450000 0.692308 0.545455
  cooling_shortfill  inner_normal0.99 add_to_171 605   13  10  14   3 578   0.416667 0.769231 0.540541
transient_fast_fill inner_normal0.995 add_to_170 605   13   7   6   6 586   0.538462 0.538462 0.538462
       cooling_fast inner_normal0.995 add_to_170 605   13   8   9   5 583   0.470588 0.615385 0.533333
      cold_flow_max inner_normal0.995 add_to_170 605   13   8   9   5 583   0.470588 0.615385 0.533333
     inner_selected   inner_selection add_to_171 605   13   9  12   4 580   0.428571 0.692308 0.529412
  meter_longplastic  inner_normal0.99 add_to_171 605   13  10  15   3 577   0.400000 0.769231 0.526316

## 안쪽 선택
         model            rule     policy   N  gas  TP  FP  FN  TN  precision   recall       F1
inner_selected inner_selection standalone 605   13   0   5  13 587   0.000000 0.000000 0.000000
inner_selected inner_selection add_to_165 605   13   5   6   8 586   0.454545 0.384615 0.416667
inner_selected inner_selection add_to_170 605   13   7   8   6 584   0.466667 0.538462 0.500000
inner_selected inner_selection add_to_171 605   13   9  12   4 580   0.428571 0.692308 0.529412

## 가스13개
          TimeStamp  gas  any_fail run_id  seq  baseline  candidate    score  threshold
2020-10-16 05:21:38 True         1 cn7_R1   24         1          1 0.208029   0.835938
2020-10-16 05:37:29 True         1 cn7_R1   40         1          1 0.064189   0.929012
2020-10-16 05:45:26 True         1 cn7_R1   48         1          1 0.064189   0.929012
2020-10-16 05:47:24 True         1 cn7_R1   50         1          1 0.165541   0.929012
2020-10-16 05:48:24 True         1 cn7_R1   51         1          1 0.064189   0.929012
2020-10-16 05:49:24 True         1 cn7_R1   52         1          1 0.165541   0.929012
2020-10-16 05:50:25 True         1 cn7_R1   53         1          1 0.165541   0.929012
2020-10-16 05:51:23 True         1 cn7_R1   54         0          0 0.486486   0.929012
2020-10-16 05:52:23 True         1 cn7_R1   55         0          1 0.949324   0.929012
2020-10-16 05:53:23 True         1 cn7_R1   56         0          0 0.418919   0.929012
2020-10-16 05:54:21 True         1 cn7_R1   57         0          0 0.173469   0.936275
2020-10-16 05:56:22 True         1 cn7_R1   59         1          1 0.173469   0.936275
2020-10-16 05:57:20 True         1 cn7_R1   60         1          1 0.173469   0.936275

## 새 경보
          TimeStamp   gas  any_fail run_id  seq  baseline  candidate    score  threshold
2020-10-16 05:11:42 False         0 cn7_R1   14         0          1 0.938356   0.929012
2020-10-16 05:23:36 False         0 cn7_R1   26         0          1 0.978102   0.835938
2020-10-16 05:31:31 False         0 cn7_R1   34         0          1 0.912409   0.835938
2020-10-16 05:52:23  True         1 cn7_R1   55         0          1 0.949324   0.929012

## 순수 미성형까지 포함한 전체606샷 가스 판정
{'TP': 10, 'FP': 12, 'FN': 3, 'TN': 581, 'precision': 0.45454545454545453, 'recall': 0.7692307692307693, 'F1': 0.5714285714285714}