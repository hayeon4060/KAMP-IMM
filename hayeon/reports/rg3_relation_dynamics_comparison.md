# RG3 관계변화4피처와 점수척도보정 앙상블

## 핵심 결론

과거평가 새CatBoost가2020-10-22 04:51:46가스1개를기존RF와별개로찾았다. 결과확인후추가탐색한RF판정ORCatBoost판정은구간제외TP4/FP10/FN13/TN556, 과거평가TP3/FP15/FN10/TN412다. 기존RF과거TP2/FP4대비가스1개를더찾지만오탐11개가증가한다. 이는테스트결과를보고아이디어를추가한탐색이며독립적인앙상블개선성능이아니다. 임계값자체는기존학습정상1%기준을변경하지않았다. 별도OR표는rg3_relation_dynamics_exploratory_or.csv에저장하며AP는산출하지않는다.

새 피처3모델 평균은 구간제외 AP0.0271/TP0/FP3으로 기존RF AP0.1686/TP4/FP6보다 낮다. 기존RF에새점수를최대결합해도 AP0.0670/TP2/FP4, 정상점수CDF보정평균은 AP0.0472/TP1/FP9로 개선되지 않았다. 과거→이후평가에서도새앙상블은가스를못찾고RF기존TP2/FP4를넘지못했다. 기본모델은변경하지않는다.

새16샷참조가확보된가스는16/17이며1개는이력부족으로4개피처를결측대체했다. 원본17개를제외없이평가하므로성능이좋은부분집합만골라보고하지않는다. 이4가지관계변화로현재RG3가스를구분하는증거를확인하지못했다는뜻이지, 가스에공정관계가없다는증명은아니다.

RH 정상566/가스17, 기존관측구간분리/과거평가. 이전16샷 관계를 현재샷에 적용해 pressure-cushion 및 pressure-backpressure 예측오차를 직전오차RMSE(하한1압력단위)로 나눈다. 참조16샷은 정상으로확인된자료가아니므로정상원인모델로해석하지않음.
직전16샷의 배럴2-5 온도차 추세가 앞8→뒤8에서 얼마나바뀌는지, pressure/cushion 15개차분의코사인일치도를추가한다. 총4피처. 17샷이력미만은결측으로유지하며모델학습집합에서대체;샷제외하지않음. 미래데이터사용없음.
작은RF/ExtraTrees/CatBoost3개고정평균및학습내부정상점수CDF로척도보정한평균. 기존RF두신호결합과추가결합. 정상CDF는각바깥학습내부예측만사용하고평가정답사용금지. CDF의내부정상표본재사용이므로평가오탐률보장아님. 정답기반최적가중치/임계값탐색하지않음.
이미반복탐색한동일자료이며AP로후보를고르면최종독립성능아님. 관측구간은실제Run ID아님. 원본센서이력확보조건. 새시계열이력의raw overlap purge는수행하지않아이전28번과검증조건이다름.

| 평가 | 모델/결합 | AP | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|---:|
| leave_one_block_out | RF_baseline | 0.1686 | 4 | 6 | 13 | 560 |
| leave_one_block_out | RF_physical_raw_max | 0.0670 | 2 | 4 | 15 | 562 |
| leave_one_block_out | RF_physical_rank_mean | 0.0472 | 1 | 9 | 16 | 557 |
| leave_one_block_out | physical_RF | 0.0285 | 0 | 5 | 17 | 561 |
| leave_one_block_out | physical3_rank_mean | 0.0282 | 0 | 3 | 17 | 563 |
| leave_one_block_out | physical_Cat | 0.0275 | 0 | 4 | 17 | 562 |
| leave_one_block_out | physical3_mean | 0.0271 | 0 | 3 | 17 | 563 |
| leave_one_block_out | physical_ET | 0.0266 | 0 | 0 | 17 | 566 |
| past_blocks_only | RF_baseline | 0.0812 | 2 | 4 | 11 | 423 |
| past_blocks_only | RF_physical_raw_max | 0.0433 | 0 | 7 | 13 | 420 |
| past_blocks_only | physical3_rank_mean | 0.0385 | 0 | 9 | 13 | 418 |
| past_blocks_only | physical3_mean | 0.0370 | 0 | 7 | 13 | 420 |
| past_blocks_only | RF_physical_rank_mean | 0.0355 | 0 | 8 | 13 | 419 |
| past_blocks_only | physical_Cat | 0.0341 | 1 | 11 | 12 | 416 |
| past_blocks_only | physical_ET | 0.0330 | 0 | 8 | 13 | 419 |
| past_blocks_only | physical_RF | 0.0329 | 0 | 3 | 13 | 424 |
