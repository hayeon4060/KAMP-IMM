# RG3 가스 8개 탐지 / 정상 오탐 6개 보존본

Confluence 정리: https://happysally19.atlassian.net/wiki/spaces/~712020d9e86c720dff4d5485f983212f68575f/pages/20840457

## 결과와 범위

기존 RG3 정상 566 Shot + 가스 17 Shot만 평가한다. 미성형 8개, CN7, 추가 원본 37 Shot(가스 2개)은 이 수치에 포함하지 않는다.

| 후보 | TP | FP | FN | TN | F1 |
|---|---:|---:|---:|---:|---:|
| 54번 온도 앵커 + 60번 압력 2-of-3 | 8 | 6 | 9 | 560 | 0.516129 |
| 57번 온도 다수결 + 60번 압력 2-of-3 | 7 | 6 | 10 | 560 | 0.466667 |

8/6 후보 정밀도 57.1%, 재현율 47.1%. 과거 구간만 학습한 별도 평가(정상427·가스13)는 두 후보 모두 TP3/FP4/FN10/TN423/F1=0.300이다.

각 구간은 학습에서 제외했지만, 같은 자료의 누락·오탐을 보며 가설을 반복 개발했다. 0.516은 탐색 성적이며 새로운 독립검증 성적이나 공식 대회 점수가 아니다. 실제 설비 Run ID가 아니라 라벨 관측 30분 초과 공백으로 나눈 5개 비교 구간이다.

## 모델 구성

1. 22번: 과거10 Shot 표준편차4(배럴2·호퍼·쿠션·배럴5) RF와 정상 관계 잔차4 RF의 점수 최댓값. RF200/깊이8/리프3, 내부 정상99% 경계.
2. 42·46·47번: 학습 가스의 두 모드에 대한 RF 전문가를 추가한다. 세 seed 평균 점수가 정상99.5% 경계를 넘고 배럴5 현재−직전 변화가 정상95% 경계를 넘을 때만 추가한다.
3. 54번: 근접 정상 약30개 대비 열접촉 파생피처6의 편차를 RF·ExtraTrees·로지스틱으로 학습한다. 점수가 정상97.5% 경계 이상이고 배럴5 변화−금형 변화의 조건부 편차가 정상 하위5%보다 낮을 때 추가한다.
4. 58~60번: 압력 전환 파생피처6의 조건부 편차 전문가를 추가한다. 정상99% 점수 경계와 최고압 하락·속도 상승·전환압의 더 큰 하락 조건을 함께 통과해야 한다. 세 seed묶음 중2표를 요구한다.

최종 판정은 기존 판정 OR 확인조건을 통과한 새 분기다. 정상만 학습하는 비지도 모델이 아니라 정상·가스 라벨을 사용하는 지도학습이다. 구체적인 방향조건은 데이터 기반 가설이지 문헌의 확정된 가스 판정식이 아니다.

## 실행 및 입력 준비

기존 `.venv` 활성화 후 노트북을 실행한다. 원본 데이터·PDF는 저작권/크기 때문에 이 커밋에 넣지 않는다. 데이터는 로컬에서 별도 준비해야 한다.

- `data/original/moldset_labeled.csv`: 기존 범위의 라벨·Timestamp·센서.
- `data/original/unlabeled_data.csv`: 이전 Shot 센서 이력. 비라벨을 정상으로 새 라벨링하지 않는다.
- `data/moldset_labeled_cn7.csv`: 10번의 센서 컬럼 목록 확인용.
- 09 → 10번: 제품별 원본 분리 및 `reports/tables/original_labeled_shot_features.csv` 생성. 이 파생 센서 테이블도 원본에서 재생성하며 Git에는 넣지 않는다.
- 22번은 기존 `rg3_relational_feature_oof_predictions.csv`를 마지막 재현 비교에 사용한다. 기존 16번 노트북으로 해당 비교 파일을 생성하거나 별도 준비한다.
- 핵심 순서: 22 → 42 → 46 → 47 → 54 → 56 → 57 → 58 → 59 → 60. 50~53은 메커니즘 탐색 기록, 55·61은 결과 감사다.
- 61번 원본 대조에는 로컬 `data/audit/rg3_rh_labeled_with_timestamp.csv`가 필요하다. 해당 파일이 없으면 먼저 기존 Timestamp 추출 절차로 준비해야 한다.

저장된 예측/점수 CSV는 중간 앵커를 재사용하는 기존 노트북의 입력이기도 하다. 이 커밋은 전체 저장소를 무데이터 상태에서 자동 실행하는 새 파이프라인을 제공하지 않는다. 새 환경에서는 원본 준비 및 위 실행 순서가 필요하다.

## 근거와 출처

- 제공 가이드 `data/04. Guidebook_Molding.pdf`, PDF13~15쪽(인쇄12~14쪽), 은조흔·광택불량·기포: 열접촉·압력 대리피처의 발상 근거. 불량 세부원인은 확인되지 않았다.
- [Breiman, Random Forests (2001)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf): RF의 일반 방법론.
- [scikit-learn 앙상블 공식 문서](https://scikit-learn.org/stable/modules/ensemble.html): RF/ExtraTrees 구현 설명.
- [BASF 공정 문제 안내](https://plastics-rubber.basf.com/emea/en/performance_polymers/services/product_support_troubleshooting/injection_moulding_troubleshooter): 다양한 가스 관련 현상의 일반 배경. 이번 정리 시 추가 참고했으며 정확한 방향조건의 출처가 아니다.
- [사용자 제공 참고 Git](https://github.com/Chohyungun/NOVA50101-KAMP-Injection-Molding): 단계별 비교·누수 점검 참고. 이 8/6 앙상블을 그대로 이식한 원천은 아니다.

호퍼온도=건조 상태, 최고압=보압, 센서 변화=가스 원인으로 단정하지 않는다. 공식 테스트의 Timestamp와 이력 제공 여부도 별도 확인해야 한다.
