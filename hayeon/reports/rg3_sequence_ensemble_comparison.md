# RG3 가스 시계열 분류와 RF 앙상블

## 결론

RF와세시계열모델 최대점수결합은 TP4/FP6/AP0.1653, 과거평가 TP2/FP4/AP0.0786이다. 두 평가 모두 기존 RF와 판정 샷이 완전히 같아 추가 가스를 발견하지 못했다. 단순 평균결합은 오히려 TP3/FP15/AP0.0992로 나쁘다. 지금 설정과32샷 표현에서 개선을 확인하지 못했으며 시계열 분석 전체가 불가능하다는 의미는 아니다.

MultiRocket 구현 호환을 위해 입력을float64로, pooling 개수를int32로 지정했다. 구현을 직접 수정하지 않았다. 사용모델은 [ROCKET 계열 공식 문서](https://www.sktime.net/docs/api-reference/sktimeclassificationkernel-basedrocketclassifier/) 및 [MultiRocket 공식 문서](https://www.sktime.net/docs/api-reference/sktimetransformationsrocketmultirocketmultivariate/)를 참고했다. LSTM/TCN/Transformer/사전학습 모델은 이번 실험에 포함하지 않았다.

샷 내부 압력파형이 아니라 샷 간 센서값의 시계열이다. 현재샷 포함32샷×8센서, 미래값 미사용. 30분 단절/제품전환으로 원본이력 구간을 끊는다. 짧은 이력은 가장 이른 관측값을 왼쪽에 복제하고583개/가스17 유지한다.
각 학습 시 평가윈도우와 원본 샷이 겹치는 학습윈도우는 제거(purge)한다. 결측대체·센서scale·ROCKET 변환은 학습집합에서만 적합한다. 각 윈도우 초기16샷중앙값을 빼서 절대센서수준보다 변화모양을 본다. padding/중심화는실시간 현재샷까지 관측가능 데이터만 사용한다.
ROCKET1000,MiniRocket1008,MultiRocket168커널,분류기는C0.1balancedLogistic 고정. 최신설치 구현이지만 모든 알고리즘이최신발표는아니다. 커널수가 작으므로 전체 알고리즘 최적성능을 대표하지 않는다.
TS3_mean은세시계열모델동일평균,RF_TS3_mean은기존RF두신호max와TS3_mean의평균,RF_TS3_max는두점수최대. 결합및기준은평가정답으로조정하지않는다. 점수는독립확률보정하지않음.
기준은학습내부구간검증정상상위1/3/5%경계이며평가오탐률보장아님. RF계열이력 purge는기존2신호정의유지로시계열채널purge와동일하지않아별도비교조건이다. 원본이력테스트제공미확인/동일자료반복탐색/최종독립평가아님.

| 평가 | 모델/결합 | AP | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|---:|
| leave_one_block_out | RF_TS3_max | 0.1653 | 4 | 6 | 13 | 560 |
| leave_one_block_out | RF_TS3_mean | 0.0992 | 3 | 15 | 14 | 551 |
| leave_one_block_out | ROCKET | 0.0435 | 1 | 23 | 16 | 543 |
| leave_one_block_out | TS3_mean | 0.0373 | 1 | 38 | 16 | 528 |
| leave_one_block_out | MiniRocket | 0.0363 | 1 | 35 | 16 | 531 |
| leave_one_block_out | MultiRocket | 0.0308 | 1 | 18 | 16 | 548 |
| past_blocks_only | RF_TS3_max | 0.0786 | 2 | 4 | 11 | 423 |
| past_blocks_only | RF_TS3_mean | 0.0672 | 1 | 6 | 12 | 421 |
| past_blocks_only | MultiRocket | 0.0392 | 1 | 16 | 12 | 411 |
| past_blocks_only | MiniRocket | 0.0386 | 2 | 17 | 11 | 410 |
| past_blocks_only | TS3_mean | 0.0374 | 0 | 10 | 13 | 417 |
| past_blocks_only | ROCKET | 0.0346 | 0 | 1 | 13 | 426 |
