# Notebooks

- `01_injection_molding_correlation_analysis.ipynb`: PDF 정의를 반영한 데이터 품질 점검과 CN7/RG3 변수 상관관계 분석 노트북
- `02_eda_and_defect_type_analysis.ipynb`: 원본 `Reason` 라벨을 이용한 EDA와 가스·미성형·초기허용불량 유형별 공정변수 분석
- `03_run_and_within_run_eda.ipynb`: LH/RH 샷 통합, 원본 생산 이력을 이용한 run 추정, 연속 불량, 동일 관측 구간 비교, 이전 샷 기반 파생 피처 탐색. 브리핑 내용은 `reports/run_eda_briefing.md` 참고.
- `04_run_temperature_trends.ipynb`: 같은 후보 run 안의 금형·배럴·호퍼 온도 추세, 초반 상승과 장시간 변화 비교. 10분 중앙값 그래프와 run별 온도 변화표를 저장한다.
- `05_gas_defect_vs_normal.ipynb`: 가스 30샷 전체 추출, 미성형 동시 발생 2샷 분리, 같은 관측 구간 및 주변 정상 비교. 쉬운 설명은 `reports/gas_defect_easy_summary.md`, 사례별 수치는 `reports/tables/gas_casebook_with_normal_baselines.csv` 참고.
- `06_gas_mechanism_and_run_analysis.ipynb`: 건조·과열·배기 가설에 대응하는 관측 신호를 제품 전체/후보 run/같은 관측 구간/주변 정상에서 비교. 이전 30분 온도 기울기와 동일 온도값의 run별 사례를 분석한다. 브리핑은 `reports/gas_mechanism_run_summary.md` 참고.

저장소 루트에서 `.venv\Scripts\python.exe` 커널로 실행한다. 실행 결과의 표와 그림은 `reports/`에 저장된다.
