# RG3 소규모 앙상블과 오탐 확인 필터

## 결론

구간제외RF+ET동일평균은AP0.1834로기존0.1686보다높지만TP4/FP8/F10.2759로기존TP4/FP6/F10.2963보다판정균형은나쁘다. 과거평가에서도AP0.0724/TP2/FP7로기존AP0.0812/TP2/FP4보다나쁘다. 순위신호의탐색후보로만보관하며새운영모델로채택하지않는다.

RF절반+ET/Cat각1/4은구간제외AP0.1826/TP4/FP7/F10.2857이지만과거평가F10.1905로기존보다낮다. 확인필터4개는두평가모두기존RF와판정샷이동일해오탐감소도없었다.

RF/ET/CatBoost 각각기존흔들림4개+관계잔차4개두모델max로정의한다. 앞선15모델및신규모델결과를보고3개를선택했으므로후보선택자체가동일평가자료의영향을받는다. 독립최종평가가아니다.
평균결합은고정RF-ET/RF-Cat/3개동일/가중치0.5:0.25:0.25를사전에정해비교한다. 임계값은각학습내부구간검증정상99%보다엄격한점수로정한다.
확인필터는RF기존판정에ET또는Cat점수가학습내부정상상위10%/25%경계를넘는경우만허용한다. 평가정답으로경계를고르지않음. 기존RF양성의부분집합이므로같은평가에서추가가스발견은불가능하고오탐감소/가스손실만확인한다. AND필터의AP는별도연속점수를정의하지않아산출하지않음.
바깥구간제외/과거평가를구분하며정상566가스17/정상427가스13. 센서이력및RPM정의불확실성조건은기존동일. 가스원인추정이아니다.

| 평가 | 결합 | AP | TP | FP | FN | TN | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| leave_one_block_out | RF_ET_mean | 0.1834 | 4 | 8 | 13 | 558 | 0.2759 |
| leave_one_block_out | RF_Cat_mean | 0.1786 | 2 | 6 | 15 | 560 | 0.1600 |
| leave_one_block_out | RF_ET_Cat_mean | 0.1778 | 3 | 8 | 14 | 558 | 0.2143 |
| leave_one_block_out | RF_half_ET_Cat | 0.1826 | 4 | 7 | 13 | 559 | 0.2857 |
| leave_one_block_out | RF_baseline | 0.1686 | 4 | 6 | 13 | 560 | 0.2963 |
| leave_one_block_out | RF_confirm_ET_10 | nan | 4 | 6 | 13 | 560 | 0.2963 |
| leave_one_block_out | RF_confirm_ET_25 | nan | 4 | 6 | 13 | 560 | 0.2963 |
| leave_one_block_out | RF_confirm_Cat_10 | nan | 4 | 6 | 13 | 560 | 0.2963 |
| leave_one_block_out | RF_confirm_Cat_25 | nan | 4 | 6 | 13 | 560 | 0.2963 |
| past_blocks_only | RF_ET_mean | 0.0724 | 2 | 7 | 11 | 420 | 0.1818 |
| past_blocks_only | RF_Cat_mean | 0.0710 | 0 | 5 | 13 | 422 | 0.0000 |
| past_blocks_only | RF_ET_Cat_mean | 0.0641 | 1 | 7 | 12 | 420 | 0.0952 |
| past_blocks_only | RF_half_ET_Cat | 0.0788 | 2 | 6 | 11 | 421 | 0.1905 |
| past_blocks_only | RF_baseline | 0.0812 | 2 | 4 | 11 | 423 | 0.2105 |
| past_blocks_only | RF_confirm_ET_10 | nan | 2 | 4 | 11 | 423 | 0.2105 |
| past_blocks_only | RF_confirm_ET_25 | nan | 2 | 4 | 11 | 423 | 0.2105 |
| past_blocks_only | RF_confirm_Cat_10 | nan | 2 | 4 | 11 | 423 | 0.2105 |
| past_blocks_only | RF_confirm_Cat_25 | nan | 2 | 4 | 11 | 423 | 0.2105 |
