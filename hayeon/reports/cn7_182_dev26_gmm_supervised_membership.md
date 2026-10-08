# CN7 dev26: GMM 군집 소속도 + 지도 분류

181의인과dev26계산과동일한26피처/시간블록/purge10/nested5를사용한다.
GMM은학습샷센서로군집표현을학습하고,그군집소속확률2/3/5개를라벨로학습한balanced LR에넣는다. GMM만으로정답을붙인것이아니다.
GMM3LR＋RF 평균도비교한다.군집학습·LR·스케일·기준은해당평가구간을제외한학습만사용한다.
안쪽F1/정상오탐3%예산으로선택하고평가자료로기준을튜닝하지않는다.
전체학습배포기준도안쪽crossfit에서고정한다.비라벨RPM0/로그관계불능은보류한다.

## 라벨
                 model   N  gas  TP  FP  FN  TN  F1  precision  recall       AP
               GMM2_LR 605   13   0   2  13 590   0        0.0     0.0 0.183949
               GMM3_LR 605   13   0   1  13 591   0        0.0     0.0 0.199901
               GMM5_LR 605   13   0   0  13 592   0        0.0     0.0 0.097105
        GMM3LR_RF_mean 605   13   0   7  13 585   0        0.0     0.0 0.018696
primary_inner_selected 605   13   0   0  13 592   0        0.0     0.0 0.097363

## 비라벨
         model  selected_for_deployment  threshold  eligible_unknown  gas_candidates  alarm_percent
       GMM2_LR                        0   0.501484              9585            1584      16.525822
       GMM3_LR                        1   0.644757              9585            1570      16.379760
       GMM5_LR                        0   0.684101              9585               0       0.000000
GMM3LR_RF_mean                        0   0.503756              9585             423       4.413146

common.py가없어원본과완전히같은실험은아니다.기록단위고정·라벨/비라벨이력밀도·동일Run시간블록검증한계는181과같다.