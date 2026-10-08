# 원본 데이터 배치 위치

대회 규정상 데이터는 이 패키지(저장소)에 포함하지 않습니다.
KAMP 「사출성형기 AI 데이터셋」의 아래 파일을 **이 폴더에 그대로** 넣은 뒤 `python run_all.py`를 실행하세요.

| 파일 | 내용 | 사용 단계 |
|---|---|---|
| `labeled_data.csv` | 1차 원본 라벨 데이터 (원래 단위 공정값, 시각, LH/RH, 불량 사유) | 1 (샷·Run 테이블, 원본 라벨 행) |
| `unlabeled_data.csv` | 1차 원본 비라벨 데이터 (약 225MB) — **외부 데이터** | 1 (정상 기준 학습 데이터, 기록 단위, 측면 배분 사전 ERR 집계, 과거기준선), 2 (탐지기 학습) |
| `moldset_labeled.csv` | 1차 라벨 데이터의 공정변수 표 (`_id` 포함) | 1 (2차 행 ↔ 1차 연결) |
| `moldset_labeled_cn7.csv` | 대회 제공 2차 라벨 데이터 CN7 (표준화 값) | 1 (연결 검증: z-score 1e-6 재현·라벨 100% 일치) |
| `moldset_labeled_rg3.csv` | 대회 제공 2차 라벨 데이터 RG3 (표준화 값) | 1 (연결 검증) |

같은 데이터셋의 `moldset_unlabeled_cn7.csv`, `moldset_unlabeled_rg3.csv`, `supervised_label_cn7.csv`는
있어도 되지만 사용하지 않습니다.
