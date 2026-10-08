# CN7+RG3 RH 가스 공동학습 비교

## 결론

이번공동학습은기존RG3가스모델을개선하지못했다. 제품ID포함공동학습의RG3 AP0.1098/TP2/FP5/FN15/TN561이며, 제품별단독학습은AP0.1686/TP4/FP6/FN13/TN560이다(두설정모두학습정상기준보정). 제품ID없이공동학습한RG3는AP0.1034/TP2/FP2로더보수적이지만2개만찾았다. 오탐을줄인대신탐지를줄인것이지같은오탐에서의개선으로검증되지않았다.

CN7 공동학습+제품ID는AP0.0166/TP0/FP14/FN10/TN579로모든가스를놓쳤다. CN7의RH가스10개가첫관측구간에몰려있고대회범위관측구간은2개뿐이므로, 가스구간전체를빼면동일제품의학습가스가0개가된다. 다른제품가스에서전이되기를시험했으나이번설정에서는확인되지않았다. CN7단독의중첩기준보정은학습구간부족으로평가불가이며0TP라는모델성능으로대체하지않았다.

RH전용으로CN7 gas10+RG3 gas17, 정상593+566, 총1186개를사용했다. CN7 LH가스3건이빠진것은RH범위를명시적으로맞춘결과이며누락된실수나모델이틀린건수가아니다. 이결과를CN7전체캐비티가스탐지또는새생산에대한독립성능으로일반화하지않는다.

## 대상

         shots  gas  normal
product                    
CN7        603   10     593
RG3        583   17     566

- 이전 AP0.1686/TP4/FP6은 RG3 가스 실험이었다. 이번에는 CN7을 추가했다.
- 대회 범위 CN7에서 RH 가스10개만 포함. LH 가스3개는 RH 전용 대상이 아니므로 포함하지 않음. CN7 10/27 초기허용 기간 전체와 양제품 미성형 제외.
- LH/RH를 합치지 않고 raw RH 라벨을 사용. CN7 RH 정상에 LH만 불량인 Shot이 포함될 수 있으나 RH 정상 정답이므로 유지.
- 제품별로 따로학습 vs 공동학습+제품ID vs 공동학습 ID없음 비교. 제품ID는5번째입력이며 관계잔차는제품별정상에서따로학습.
- 같은RF200/depth8/leaf3/seed42, 같은구간전체제외. 잔차RF 학습변수는안쪽구간제외, 판정99% 기준도제품별학습정상안쪽예측만 사용.
- CN7은대회범위관측구간이2개로매우적다. 별도학습의바깥학습에1구간만남으면안쪽잔차/기준보정이부족하다. 이때결측과미평가를명시하고통합과확정비교하지않음.
- 기존RG3단독점수와기준은파일대조하여정확히재현검증.
- 미래학습구간이포함될수있는구간제외평가이며미래생산검증아님. 반복탐색후독립성한계가있음.
- 평균RPM관련잔차는정의불확실성을남겨둔기존모델재현용. 물리원인으로해석하지않음.

## 제품별 결과

| setting | model | product | rule | shots | gas | TP | FP | FN | TN | AP | top10_TP | top10_selected |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| separate_products | std_rf | CN7 | fixed05 | 603 | 10 | 0.0 | 10.0 | 10.0 | 583.0 | 0.0166 | 1.0 | 61.0 |
| separate_products | std_rf | CN7 | train_fpr1 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan |
| separate_products | residual_rf | CN7 | fixed05 | 603 | 10 | 0.0 | 0.0 | 10.0 | 593.0 | 0.0166 | 1.0 | 61.0 |
| separate_products | residual_rf | CN7 | train_fpr1 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan |
| separate_products | max_union | CN7 | fixed05 | 603 | 10 | 0.0 | 10.0 | 10.0 | 583.0 | 0.0166 | 1.0 | 61.0 |
| separate_products | max_union | CN7 | train_fpr1 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan |
| separate_products | std_rf | RG3 | fixed05 | 583 | 17 | 1.0 | 6.0 | 16.0 | 560.0 | 0.1387 | 6.0 | 62.0 |
| separate_products | std_rf | RG3 | train_fpr1 | 583 | 17 | 3.0 | 13.0 | 14.0 | 553.0 | 0.1387 | 6.0 | 62.0 |
| separate_products | residual_rf | RG3 | fixed05 | 583 | 17 | 1.0 | 1.0 | 16.0 | 565.0 | 0.0848 | 2.0 | 62.0 |
| separate_products | residual_rf | RG3 | train_fpr1 | 583 | 17 | 1.0 | 0.0 | 16.0 | 566.0 | 0.0848 | 2.0 | 62.0 |
| separate_products | max_union | RG3 | fixed05 | 583 | 17 | 2.0 | 7.0 | 15.0 | 559.0 | 0.1686 | 6.0 | 62.0 |
| separate_products | max_union | RG3 | train_fpr1 | 583 | 17 | 4.0 | 6.0 | 13.0 | 560.0 | 0.1686 | 6.0 | 62.0 |
| pooled_with_product | std_rf | CN7 | fixed05 | 603 | 10 | 0.0 | 2.0 | 10.0 | 591.0 | 0.0152 | 1.0 | 61.0 |
| pooled_with_product | std_rf | CN7 | train_fpr1 | 603 | 10 | 0.0 | 118.0 | 10.0 | 475.0 | 0.0152 | 1.0 | 61.0 |
| pooled_with_product | residual_rf | CN7 | fixed05 | 603 | 10 | 0.0 | 0.0 | 10.0 | 593.0 | 0.0166 | 1.0 | 61.0 |
| pooled_with_product | residual_rf | CN7 | train_fpr1 | 603 | 10 | 0.0 | 0.0 | 10.0 | 593.0 | 0.0166 | 1.0 | 61.0 |
| pooled_with_product | max_union | CN7 | fixed05 | 603 | 10 | 0.0 | 2.0 | 10.0 | 591.0 | 0.0166 | 1.0 | 61.0 |
| pooled_with_product | max_union | CN7 | train_fpr1 | 603 | 10 | 0.0 | 14.0 | 10.0 | 579.0 | 0.0166 | 1.0 | 61.0 |
| pooled_with_product | std_rf | RG3 | fixed05 | 583 | 17 | 1.0 | 1.0 | 16.0 | 565.0 | 0.1271 | 3.0 | 62.0 |
| pooled_with_product | std_rf | RG3 | train_fpr1 | 583 | 17 | 2.0 | 12.0 | 15.0 | 554.0 | 0.1271 | 3.0 | 62.0 |
| pooled_with_product | residual_rf | RG3 | fixed05 | 583 | 17 | 1.0 | 5.0 | 16.0 | 561.0 | 0.0579 | 2.0 | 62.0 |
| pooled_with_product | residual_rf | RG3 | train_fpr1 | 583 | 17 | 1.0 | 4.0 | 16.0 | 562.0 | 0.0579 | 2.0 | 62.0 |
| pooled_with_product | max_union | RG3 | fixed05 | 583 | 17 | 2.0 | 6.0 | 15.0 | 560.0 | 0.1098 | 4.0 | 62.0 |
| pooled_with_product | max_union | RG3 | train_fpr1 | 583 | 17 | 2.0 | 5.0 | 15.0 | 561.0 | 0.1098 | 4.0 | 62.0 |
| pooled_no_product | std_rf | CN7 | fixed05 | 603 | 10 | 0.0 | 1.0 | 10.0 | 592.0 | 0.0167 | 6.0 | 61.0 |
| pooled_no_product | std_rf | CN7 | train_fpr1 | 603 | 10 | 0.0 | 96.0 | 10.0 | 497.0 | 0.0167 | 6.0 | 61.0 |
| pooled_no_product | residual_rf | CN7 | fixed05 | 603 | 10 | 0.0 | 0.0 | 10.0 | 593.0 | 0.0166 | 1.0 | 61.0 |
| pooled_no_product | residual_rf | CN7 | train_fpr1 | 603 | 10 | 0.0 | 0.0 | 10.0 | 593.0 | 0.0166 | 1.0 | 61.0 |
| pooled_no_product | max_union | CN7 | fixed05 | 603 | 10 | 0.0 | 1.0 | 10.0 | 592.0 | 0.0166 | 1.0 | 61.0 |
| pooled_no_product | max_union | CN7 | train_fpr1 | 603 | 10 | 0.0 | 25.0 | 10.0 | 568.0 | 0.0166 | 1.0 | 61.0 |
| pooled_no_product | std_rf | RG3 | fixed05 | 583 | 17 | 1.0 | 1.0 | 16.0 | 565.0 | 0.1187 | 4.0 | 62.0 |
| pooled_no_product | std_rf | RG3 | train_fpr1 | 583 | 17 | 1.0 | 4.0 | 16.0 | 562.0 | 0.1187 | 4.0 | 62.0 |
| pooled_no_product | residual_rf | RG3 | fixed05 | 583 | 17 | 1.0 | 4.0 | 16.0 | 562.0 | 0.0540 | 2.0 | 62.0 |
| pooled_no_product | residual_rf | RG3 | train_fpr1 | 583 | 17 | 1.0 | 1.0 | 16.0 | 565.0 | 0.0540 | 2.0 | 62.0 |
| pooled_no_product | max_union | RG3 | fixed05 | 583 | 17 | 2.0 | 5.0 | 15.0 | 561.0 | 0.1034 | 3.0 | 62.0 |
| pooled_no_product | max_union | RG3 | train_fpr1 | 583 | 17 | 2.0 | 2.0 | 15.0 | 564.0 | 0.1034 | 3.0 | 62.0 |

AP를제품통합숫자하나로만보고하면CN7의쉬운사례가RG3향상처럼보일수있어제품별로보고한다. CN7 RH 대상과이전CN7 Shot합산대상의라벨수가달라그수치를직접비교하면안된다.

산출물: notebooks/23_cn7_rg3_rh_gas_joint_learning.ipynb, reports/tables/gas_product_pooling_metrics.csv 및 predictions/fold_audit/counts.
