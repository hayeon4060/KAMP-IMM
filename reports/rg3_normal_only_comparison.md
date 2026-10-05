# RG3 정상만 학습하는 이상 탐지: 다른 접근

## 결론

학습정상만으로만든두표현의앙상블은구간제외평가에서가스를0개발견했다. 과거평가현재상태8개중앙값앙상블은TP1/FP4/F10.1111, 평균은TP1/FP10/F10.0833으로기존RF TP2/FP4/F10.2105보다낮다. IsolationForest단독은TP3/FP15/F10.1935로탐지는늘지만오탐부담이커진다. 이조건에서정상영역이탈을가스탐지로대체할수없다고판단하되다른정상참조/표현까지불가능하다고단정하지않는다.

지도분류RF의가중치/임계값튜닝과별개다. 각바깥학습구간에서라벨정상만선택하고가스는모델적합에서제외한다. 정상선별에는라벨을쓰므로정상샘플학습(one-class/novelty)방식이며비라벨전체정상가정이아니다.
8개현재센서(state8)와현재값-이전10샷중앙값을이전10샷표준편차로정규화한8개(local8)를분리해시험한다. 피처는압력/배압/쿠션/계량위치/배럴2/배럴5/호퍼/금형3이다. 결측대체및RobustScaler는모델학습정상만적합한다.
LedoitWolfMahalanobis/IsolationForest/10정상최근접평균거리로정상영역이탈을측정한다. 학습내부구간제외정상점수의CDF로척도를맞춰3개평균및중앙값(2개이상동의)앙상블을고정한다. 현재평가정답으로가중치를고르지않는다.
기준은학습내부정상상위0.5/1/2%로정하고평가오탐률보장없음. 내부CDF기준표본재사용이며독립확률보정아님. 가스모델학습은안하지만평가모집단을기존과맞추려고학습집합에두클래스가있는구간만평가한다. 구간제외583/가스17, 과거440/가스13.
같은자료반복탐색으로독립최종성능이아니다. 실제Run ID가아닌관측구간. 원본이력확보조건. 정상영역이탈을가스원인으로단정하지않는다.

| 평가 | 피처 | 모델/앙상블 | AP | TP | FP | FN | TN | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| leave_one_block_out | state8 | three_mean | 0.0580 | 0 | 0 | 17 | 566 | 0.0000 |
| leave_one_block_out | state8 | nearest | 0.0443 | 0 | 0 | 17 | 566 | 0.0000 |
| leave_one_block_out | state8 | mahalanobis | 0.0413 | 0 | 4 | 17 | 562 | 0.0000 |
| leave_one_block_out | state8 | three_consensus | 0.0407 | 0 | 0 | 17 | 566 | 0.0000 |
| leave_one_block_out | state8 | isolation | 0.0377 | 0 | 3 | 17 | 563 | 0.0000 |
| leave_one_block_out | local8 | mahalanobis | 0.0253 | 0 | 7 | 17 | 559 | 0.0000 |
| leave_one_block_out | local8 | three_mean | 0.0249 | 0 | 4 | 17 | 562 | 0.0000 |
| leave_one_block_out | local8 | nearest | 0.0246 | 0 | 3 | 17 | 563 | 0.0000 |
| leave_one_block_out | local8 | isolation | 0.0243 | 0 | 6 | 17 | 560 | 0.0000 |
| leave_one_block_out | local8 | three_consensus | 0.0240 | 0 | 7 | 17 | 559 | 0.0000 |
| past_blocks_only | state8 | isolation | 0.0886 | 3 | 15 | 10 | 412 | 0.1935 |
| past_blocks_only | state8 | three_consensus | 0.0567 | 1 | 4 | 12 | 423 | 0.1111 |
| past_blocks_only | state8 | three_mean | 0.0637 | 1 | 10 | 12 | 417 | 0.0833 |
| past_blocks_only | state8 | mahalanobis | 0.0414 | 0 | 6 | 13 | 421 | 0.0000 |
| past_blocks_only | state8 | nearest | 0.0348 | 0 | 4 | 13 | 423 | 0.0000 |
| past_blocks_only | local8 | three_mean | 0.0293 | 0 | 3 | 13 | 424 | 0.0000 |
| past_blocks_only | local8 | nearest | 0.0284 | 0 | 3 | 13 | 424 | 0.0000 |
| past_blocks_only | local8 | mahalanobis | 0.0270 | 0 | 13 | 13 | 414 | 0.0000 |
| past_blocks_only | local8 | isolation | 0.0270 | 0 | 4 | 13 | 423 | 0.0000 |
| past_blocks_only | local8 | three_consensus | 0.0268 | 0 | 5 | 13 | 422 | 0.0000 |
