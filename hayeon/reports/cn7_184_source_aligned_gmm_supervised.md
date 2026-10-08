# 지우 최신unsup_search 원본 정렬 + dev26 지도 분류

origin/jiwoo f02c9fd를병합없이받아common.py/models.py/final/src를임시스냅샷으로읽었다.
원본dev26의21+5/관계단위1e-3/기록단위JSON/causal_robust_dev/Prep/전체공분산GMM2를사용한다. 빠른dev26추출이원본ext_features와일치함을확인했다.
우진2 CN7만정제하고라벨시각/라벨Run±60분/이상기록/짧은Run/배럴1<100 비가동을제거했다.
과거참조4288샷, 마지막2020-10-15 23:26:18. GMM teacher는이참조만학습하며공식가스평가시각은겹치지않는다.
원본은비지도GMM음의로그밀도점수다. 우리는이를고정하고음의로그밀도+군집소속확률2개를라벨로학습한LR에입력하는지도classifier를추가했다.
RF26과GMM-LR＋RF평균도비교했다.같은605샷/시간블록20/purge10/nested5 유지.
지도모델선택·임계값은안쪽에서만계산했다.원본비지도기준99.9%는지도모델선택에서제외한별도비교다.
원본보고서는전체불량행기준AP이므로우리가스샷기준F1과직접비교하면안된다.
원본Prep가결측편차를0으로처리하므로엄격한26개모두유효필터를쓰지않았다. 비라벨정답없는경보비율만보고한다.
같은Run시간블록은미래라벨학습자료도포함할수있으며새Run/미래독립검증이아니다.참조GMM학습시각만공식평가이전으로고정했다.

## 라벨
                 model   N  gas  TP  FP  FN  TN   F1  precision   recall       AP
         reference_GMM 605   13   2   1  11 591 0.25   0.666667 0.153846 0.282031
           GMMscore_LR 605   13   0   3  13 589 0.00   0.000000 0.000000 0.018649
                  RF26 605   13   0   0  13 592 0.00   0.000000 0.000000 0.017362
         GMM_LR_RFmean 605   13   0   1  13 591 0.00   0.000000 0.000000 0.013478
primary_inner_selected 605   13   0   3  13 589 0.00   0.000000 0.000000 0.016358

## 정제비라벨
        model  deployment_choice  clean_unlabeled  alarms  alarm_percent  post_reference_n  post_reference_alarms
reference_GMM                  0             8007      11       0.137380              3719                      6
  GMMscore_LR                  1             8007       1       0.012489              3719                      0
         RF26                  0             8007       0       0.000000              3719                      0
GMM_LR_RFmean                  0             8007       0       0.000000              3719                      0