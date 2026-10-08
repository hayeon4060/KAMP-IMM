# CN7 — 지우 최신 final

원본 브랜치 `origin/jiwoo`, 커밋 `e8e354f1712e980501c4c6bee09ce3a23cd1fc2b`. `vendor/jiwoo_final/`은 원본 소스를 보존한다. 새 실행 진입점은 두 노트북이다.

1. `01_preprocess_train.ipynb`: 압축 원본 복원 → 1차/2차 연결 → Run/샷/dev26 → 정상 참조 국소 Mahalanobis 학습 → 원본 전체 재현 확인 → `models/cn7_final_model.pkl`.
2. `02_predict.ipynb`: 저장된 모델로 `INPUT_CSV` 점수와 경보 생성. 재학습/불량 라벨 입력 불필요. 기본 입력은 `dataset/processed/cn7/cn7_shots.csv`; `PassOrFail`과 `Reason`은 제거하고 예측한다.

환경(Python3.11): 먼저 `py -3.11 -m venv .venv-cn7`, `.venv-cn7\Scripts\Activate.ps1`, 그 다음 `python -X utf8 -m pip install -r cn7/requirements.txt nbformat nbclient ipykernel`. 노트북 커널도 이 환경을 선택한다.

새 입력은 `run_id`, `TimeStamp`, 원본 공정 센서의 `*_raw` 컬럼을 포함한 샷 테이블이다. 같은 Run의 앞선 센서 이력이 포함되어야 한다. 이미 만든 dev값을 가져오는 대신 저장된 기록 단위로 다시 계산한다. 처음5샷 기준선 형성 규칙은 원본과 같다.

**가스/미성형 유형 분류가 아니라 통합 이상 경보**다. `defect_alarm`은 정상참조99.9%, `inspection_candidate`는97.5% 기준이다. 검사후보와불량확정은다르다. 예측노트북은탐지기 점수/경보를출력하며원본확률층의LH/RH행확률을새입력에추가적합하지않는다. 원본확률층/OOF공식행결과는01의outputs에별도생성된다.

원본 최신코드 재현 확인28/28 PASS. 공식행평가 AP.465,통합경보TP6/FP0/FN11/F1.522. 샷기준결과와행기준결과혼동금지. 모델은비라벨정상가정참조로학습,확률층은라벨보정사용. 추가기간은탐색평가이며독립최종성능이아니다.
