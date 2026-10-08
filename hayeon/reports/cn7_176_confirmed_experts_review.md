# CN7 가스: 여러 원인 전문가의 방향 확인 앙상블

정상592/가스13=605샷, 같은20 시간블록/purge10. 기준173을 고정한다.
175 조건부 충전편차 RF에 속도 상위5% 확인, 175 절대 충전편차 LR에 가소화 위치 상위5% 확인, 172 냉각+속도 정상통계 전문가에 속도 상위5% 확인을 붙였다.
각 확인 기준은 해당 평가 구간과 앞뒤10샷을 뺀 같은Run의 학습 정상95백분위다. 기준보다 엄격히 커야 한다.
속도·계량 확인은 이전 미탐/오탐을 보고 만든 사후 개발 가설이며 가스 원인을 입증하지 않는다.
95%는 이번 확인 실험 전에 고정했고 숫자 임계값을 평가 정답으로 최적화하지 않았다.
분류기·정상 기반 전문가·기존 조합 자체를 이미 본 평가 결과에서 골랐으므로 새 Run 독립 성능이 아니다.
미성형까지 포함한 전체606샷을 별도로 기록한다. 정상 통계 전문가가 포함돼 전체를 순수 지도 모델이라고 부르면 안 된다.

## 전체 비교
                    policy         cohort   N  gas  TP  FP  FN  TN  precision   recall       F1
                   base173 normal_and_gas 605   13   8   5   5 587   0.615385 0.615385 0.615385
                   base173   all_official 606   13   8   6   5 587   0.571429 0.615385 0.592593
           unconfirmed_all normal_and_gas 605   13  11  12   2 580   0.478261 0.846154 0.611111
           unconfirmed_all   all_official 606   13  11  13   2 580   0.458333 0.846154 0.594595
    base_plus_localRF_fast normal_and_gas 605   13   9   5   4 587   0.642857 0.692308 0.666667
    base_plus_localRF_fast   all_official 606   13   9   6   4 587   0.600000 0.692308 0.642857
base_plus_absoluteLR_meter normal_and_gas 605   13   8   5   5 587   0.615385 0.615385 0.615385
base_plus_absoluteLR_meter   all_official 606   13   8   6   5 587   0.571429 0.615385 0.592593
   base_plus_coldTail_fast normal_and_gas 605   13   9   8   4 584   0.529412 0.692308 0.600000
   base_plus_coldTail_fast   all_official 606   13   9   9   4 584   0.500000 0.692308 0.580645
           confirmed_RF_LR normal_and_gas 605   13   9   5   4 587   0.642857 0.692308 0.666667
           confirmed_RF_LR   all_official 606   13   9   6   4 587   0.600000 0.692308 0.642857
             confirmed_all normal_and_gas 605   13  10   8   3 584   0.555556 0.769231 0.645161
             confirmed_all   all_official 606   13  10   9   3 584   0.526316 0.769231 0.625000

## 가스13개
          TimeStamp  base173  confirmed_all  localRF_fast  absoluteLR_meter  coldTail_fast
2020-10-16 05:21:38        1              1             0                 0              0
2020-10-16 05:37:29        0              0             0                 0              0
2020-10-16 05:45:26        1              1             0                 0              0
2020-10-16 05:47:24        1              1             0                 0              0
2020-10-16 05:48:24        1              1             0                 0              0
2020-10-16 05:49:24        1              1             0                 0              0
2020-10-16 05:50:25        1              1             0                 0              0
2020-10-16 05:51:23        0              1             1                 0              0
2020-10-16 05:52:23        0              1             0                 0              1
2020-10-16 05:53:23        0              0             0                 0              0
2020-10-16 05:54:21        0              0             0                 0              0
2020-10-16 05:56:22        1              1             0                 0              0
2020-10-16 05:57:20        1              1             0                 0              0

## 새 경보
          TimeStamp   gas  any_fail run_id  seq  time_block  Max_Injection_Speed_raw  Plasticizing_Position_raw heldout_block  speed_cut  meter_cut  speed_high  meter_high  base173  unconfirmed_all  base_plus_localRF_fast  base_plus_absoluteLR_meter  base_plus_coldTail_fast  confirmed_RF_LR  confirmed_all  localRF_fast  absoluteLR_meter  coldTail_fast  raw_RF  raw_LR  raw_coldTail
2020-10-16 05:11:42 False         0 cn7_R1   14 cn7_R1__B00                55.799999                  68.839996   cn7_R1__B00  55.700001  68.860001           1           0        0                1                       0                           0                        1                0              1             0                 0              1       0       0             1
2020-10-16 05:23:36 False         0 cn7_R1   26 cn7_R1__B01                55.799999                  68.839996   cn7_R1__B01  55.700001  68.860001           1           0        0                1                       0                           0                        1                0              1             0                 0              1       0       0             1
2020-10-16 05:31:31 False         0 cn7_R1   34 cn7_R1__B01                55.799999                  68.839996   cn7_R1__B01  55.700001  68.860001           1           0        0                1                       0                           0                        1                0              1             0                 0              1       0       0             1
2020-10-16 05:51:23  True         1 cn7_R1   54 cn7_R1__B02                55.799999                  68.839996   cn7_R1__B02  55.700001  68.860001           1           0        0                1                       1                           0                        0                1              1             1                 0              0       1       0             0
2020-10-16 05:52:23  True         1 cn7_R1   55 cn7_R1__B02                55.799999                  68.839996   cn7_R1__B02  55.700001  68.860001           1           0        0                1                       0                           0                        1                0              1             0                 0              1       0       0             1

## 미탐
          TimeStamp  gas  any_fail run_id  seq  time_block  Max_Injection_Speed_raw  Plasticizing_Position_raw heldout_block  speed_cut  meter_cut  speed_high  meter_high  base173  unconfirmed_all  base_plus_localRF_fast  base_plus_absoluteLR_meter  base_plus_coldTail_fast  confirmed_RF_LR  confirmed_all  localRF_fast  absoluteLR_meter  coldTail_fast  raw_RF  raw_LR  raw_coldTail
2020-10-16 05:37:29 True         1 cn7_R1   40 cn7_R1__B02                55.299999                  68.839996   cn7_R1__B02  55.700001  68.860001           0           0        0                0                       0                           0                        0                0              0             0                 0              0       0       0             0
2020-10-16 05:53:23 True         1 cn7_R1   56 cn7_R1__B02                55.500000                  68.860001   cn7_R1__B02  55.700001  68.860001           0           0        0                1                       0                           0                        0                0              0             0                 0              0       0       1             0
2020-10-16 05:54:21 True         1 cn7_R1   57 cn7_R1__B03                55.799999                  68.839996   cn7_R1__B03  55.799999  68.860001           0           0        0                0                       0                           0                        0                0              0             0                 0              0       0       0             0