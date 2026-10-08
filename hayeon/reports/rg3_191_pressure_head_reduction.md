# RG3 11개 내부 모델 점진 축소

정상566/가스17, 동일5구간 제외 검증. 각 축소 조합은 학습 내부 정상 점수99%로 임계값을 다시 계산했다. 저장된 원본11개와 예측 불일치 0개: 완전 재현으로 주장하지 않는다.

                           candidate  classifiers  TP  FP  FN  TN       F1
       preserved_anchor23_original11           32   8   6   9 560 0.516129
     preserved_anchor23_10_drop_LR10           31   8   6   9 560 0.516129
       preserved_anchor23_9_LR1_only           30   8   6   9 560 0.516129
  preserved_anchor23_9_two_seeds_LR3           30   7   6  10 560 0.466667
     preserved_anchor23_8_trees_only           29   7   6  10 560 0.466667
  preserved_anchor23_7_two_seeds_LR1           28   7   6  10 560 0.466667
preserved_anchor23_6_two_seeds_trees           27   7   6  10 560 0.466667
  preserved_anchor23_5_one_seed_each           26   8   6   9 560 0.516129
                          original11           11   5   6  12 560 0.357143
                        10_drop_LR10           10   5   6  12 560 0.357143
                     9_two_seeds_LR3            9   4   6  13 560 0.296296
                          9_LR1_only            9   5   6  12 560 0.357143
                        8_trees_only            8   4   6  13 560 0.296296
                     7_two_seeds_LR1            7   4   6  13 560 0.296296
                   6_two_seeds_trees            6   4   6  13 560 0.296296
                     5_one_seed_each            5   5   6  12 560 0.357143

## 모델 구성
{
  "original11": [
    "RF42",
    "RF7",
    "RF101",
    "ET42",
    "ET7",
    "ET101",
    "LR01",
    "LR1",
    "LR10"
  ],
  "10_drop_LR10": [
    "RF42",
    "RF7",
    "RF101",
    "ET42",
    "ET7",
    "ET101",
    "LR01",
    "LR1"
  ],
  "9_LR1_only": [
    "RF42",
    "RF7",
    "RF101",
    "ET42",
    "ET7",
    "ET101",
    "LR1"
  ],
  "8_trees_only": [
    "RF42",
    "RF7",
    "RF101",
    "ET42",
    "ET7",
    "ET101"
  ],
  "9_two_seeds_LR3": [
    "RF42",
    "RF7",
    "ET42",
    "ET7",
    "LR01",
    "LR1",
    "LR10"
  ],
  "7_two_seeds_LR1": [
    "RF42",
    "RF7",
    "ET42",
    "ET7",
    "LR1"
  ],
  "6_two_seeds_trees": [
    "RF42",
    "RF7",
    "ET42",
    "ET7"
  ],
  "5_one_seed_each": [
    "RF42",
    "ET42",
    "LR1"
  ]
}

수정 원인: 190번이 압력 prediction에 포함된 anchor를 압력 단독으로 오인했다. 수정 후 RF2+압력9의11개는TP5/FP6이며 완전 재현된다. preserved_anchor23은기존RF2+패턴RF12+열9의저장된구간제외출력을보존한다. 압력9개는실제로재학습해기존score/cut 및anchor결합583예측모두일치했다. 다른23개는이번에재학습하지않았다.
LR 숫자는C값, RF/ET 숫자는random seed다. RF2 정상 관계 피처에는Ridge4 전처리가 별도 필요하다.

## 한계
미리 명시한 축소 조합들의 개발 비교다. 평가 결과를 보고 최종 조합을 선택하면 독립검증 성능으로 부를 수 없다. 기존 이력 피처 조건을 재사용했으며 미관측 미래Run 검증이 아니다. 비라벨 경보율 확인 및 새 저장 추론 모델 교체는 하지 않았다.
