# RG3 12개 전체 새 학습 재현 검증

RF2+패턴RF4(seed42,2view×2mode)+열RF/ET/LR3+압력RF/ET/LR3. 이번 실행에서는 기존 예측을 학습 또는 결과 생성에 사용하지 않았다. 이전 CSV는 마지막 일치 검사에만 사용했다.

{
  "classifiers": 12,
  "TP": 8,
  "FP": 7,
  "FN": 9,
  "TN": 559,
  "F1": 0.5,
  "all583_predictions_match": true,
  "fit_calls": 26,
  "all_fit_timestamp_overlap_zero": true,
  "heldout_label_flip_first_block_maxdiff": {
    "baseline": 5.551115123125783e-17,
    "expert": 0.0,
    "thermal": 1.1102230246251565e-16,
    "pressure": 1.1102230246251565e-16
  },
  "input_sha256": "3531f192d80e082c9f9b4d83b72c0b6c9ef3b26530c195584969edc8c8c5dc27",
  "evaluation": "existing 5-block development reproduction, not unseen independent validation",
  "model_exported": false
}

학습과평가timestamp 및comparison block겹침없음. 임계값은각외부학습자료의내부구간제외정상점수로산정. 첫번째평가구간라벨반전후캐시전체비우고재학습한4개구성요소점수차이1e-12미만. 전처리Ridge4는분류기12개에포함하지않음.

## 주의
이미검토한평가결과에서조합을골랐으므로새로운독립검증F1이아니다. 기존센서과거이력피처를재사용했다. 이력피처에는과거센서가포함되지만미래값/불량라벨을피처로넣지않는다. 이전평가센서이력조건까지새로운미래Run배포일반화로해석할수없다. 이번실행은저장된인과피처입력에서의전체12개재학습이며raw전처리처음부터재생성은아니다. 비라벨경보율미확인,저장추론모델미교체,commit/push없음.
