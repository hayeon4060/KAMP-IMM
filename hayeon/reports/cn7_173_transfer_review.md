# CN7 가스: RG3 가스17개를 보조 학습한 상대 공정 전문가

CN7 정상592/가스13의 동일 시간블록20/purge10/nested5. CN7 평가 샷은 학습에 포함하지 않는다.
RG3 공식 정상566/가스17를 보조 학습으로 추가한다. 원시 절대값을 섞지 않고 각 제품/Run의 학습 정상 기준으로 표준화한다.
CN7 라벨만 사용하는 모델과 다르며 RG3 보조 학습 사용을 반드시 표시한다. 19가스 데이터는 사용하지 않는다.
보조 샘플 가중치는0.5 고정. RF/ET/LR 및 평균 앙상블.
고정 최고는 이미 본 평가 결과에서 후보를 고른 개발 성적이다. 새 Run 독립검증이 아니다.

## 최고 고정 후보
                 model              rule     policy   N  gas  TP  FP  FN  TN  precision   recall       F1
   mechanism_joint8_LR  inner_normal0.99 add_to_170 605   13   8   5   5 587   0.615385 0.615385 0.615385
   mechanism_joint8_LR  inner_normal0.99 add_to_171 605   13   9   8   4 584   0.529412 0.692308 0.600000
   mechanism_joint8_LR inner_normal0.995 add_to_171 605   13   9   8   4 584   0.529412 0.692308 0.600000
mechanism_joint8_mean3 inner_normal0.995 add_to_171 605   13   9   8   4 584   0.529412 0.692308 0.600000
   mechanism_joint8_LR inner_normal0.995 add_to_170 605   13   7   4   6 588   0.636364 0.538462 0.583333
mechanism_joint8_mean3 inner_normal0.995 add_to_170 605   13   7   4   6 588   0.636364 0.538462 0.583333
           flow6_mean3 inner_normal0.995 add_to_171 605   13   9   9   4 583   0.500000 0.692308 0.580645
              load6_RF inner_normal0.995 add_to_171 605   13   9   9   4 583   0.500000 0.692308 0.580645
              load6_ET inner_normal0.995 add_to_171 605   13   9   9   4 583   0.500000 0.692308 0.580645
           load6_mean3  inner_normal0.99 add_to_171 605   13   9   9   4 583   0.500000 0.692308 0.580645
           load6_mean3 inner_normal0.995 add_to_171 605   13   9   9   4 583   0.500000 0.692308 0.580645
   mechanism_joint8_RF inner_normal0.995 add_to_171 605   13   9   9   4 583   0.500000 0.692308 0.580645
mechanism_joint8_mean3  inner_normal0.99 add_to_171 605   13   9   9   4 583   0.500000 0.692308 0.580645
              flow6_RF inner_normal0.995 add_to_171 605   13   9  10   4 582   0.473684 0.692308 0.562500
              flow6_ET inner_normal0.995 add_to_171 605   13   9  10   4 582   0.473684 0.692308 0.562500

## 안쪽 선택
               model            rule     policy   N  gas  TP  FP  FN  TN  precision   recall       F1
inner_selected_mean3 inner_selection standalone 605   13   0   8  13 584   0.000000 0.000000 0.000000
inner_selected_mean3 inner_selection add_to_165 605   13   5  10   8 582   0.333333 0.384615 0.357143
inner_selected_mean3 inner_selection add_to_170 605   13   7  12   6 580   0.368421 0.538462 0.437500
inner_selected_mean3 inner_selection add_to_171 605   13   9  16   4 576   0.360000 0.692308 0.473684

## 가스13개
          TimeStamp  gas  any_fail run_id  seq  baseline  candidate
2020-10-16 05:21:38 True         1 cn7_R1   24         1          1
2020-10-16 05:37:29 True         1 cn7_R1   40         0          0
2020-10-16 05:45:26 True         1 cn7_R1   48         0          1
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

## 전체606샷 가스 판정
{'TP': 8, 'FP': 6, 'FN': 5, 'TN': 587, 'precision': 0.5714285714285714, 'recall': 0.6153846153846154, 'F1': 0.5925925925925926}