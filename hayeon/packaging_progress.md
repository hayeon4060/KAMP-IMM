# 배포 정리 진행 기록 (2026-10-08)

사용자 확정: cn7/은 origin/jiwoo 최신 final(e8e354f), rg3/은 최고 미성형143 및 가스12개(193 재현). 제품별 전처리/학습 ipynb와 저장모델 예측 ipynb 두 진입점. 기존 실험은 hayeon/ 보관. 원본/KAMP/가공 데이터 구분. 완료 후 hayeon 브랜치 커밋/push.

## 확인됨
- 모든 원격 브랜치 fetch, 병합 없음.
- 지우 최신 final 소스를 cn7/vendor/jiwoo_final로 복사(원본 유지).
- CN7은 가스/미성형을 분리하지 않는 통합 이상 경보. 유형 분류로 과장 금지.
- RG3 가스12개 전부 새 학습 검증: TP8 FP7 FN9 TN559 F1.500,583샷 이전예측 전부일치. 비라벨734/28710=2.56%,작동추정731/15797=4.63%.
- RG3 최고 미성형143: TP6 FP11 FN2 TN555 F1.480. 현재143재실행과145비라벨 함수는 있으나 저장추론 모델 구현/검증이 필요.
- 원본 unlabeled_data.csv 225,647,622bytes: 일반GitHub100MB초과. lossless gzip 동봉과자동복원 구성 예정.

## 구현/검증 완료

- cn7/ 및 rg3/의01_preprocess_train /02_predict 두실행진입점 완성.
- CN7 최신final 전체원본실행28/28 PASS,저장모델예측정답제거확인,정상학습참조4293개원값/피처가공CSV동봉.
- RG3 가스12개전부재현,미성형143평가재현,미성형38분류기실제저장/로드 점수원본fullfit591샷일치.
- 두제품 별도실행폴더01/02 재현성공. RF/ET/LR/SVC fit금지상태에서RG3예측성공.
- 새12가스+최고미성형OR개발평가TP14/FP18/F1.4912. 기존50가스cascade결과복사하지않음.
- 가스저장모델비라벨734/28710경보194와모두일치.
- 기존 notebooks/reports/src/models/examples/data는삭제없이hayeon/으로이동. 새데이터 original/KAMP/processed는dataset/.
- 비라벨원본gzip21,690,579bytes압축,CSV복원자동. dataset/manifest.json기록.
- 상세검증 RELEASE_VALIDATION.md,제품별실행안내README작성.

## 남은 작업(마지막Git배포)

선택적staging→누락/100MB초과/원치않는파일점검→commit/push→원격SHA확인.

## 최초 작업 목록 (기록)
1. 공용 data/original, data/kamp, data/processed 구성 및 압축원본 SHA검증.
2. CN7 원본 final을 노트북2개로 연결,예측단계 재학습 금지. Python3.11환경 활성화후의존성확인. 경로설정은vendor소스를복사하여기존함수정의 유지 또는노트북에서paths재바인딩; 실제latest final 재현확인.
3. RG3 가스12개 저장모델 exporter/predictor + 최고미성형 실제 학습모델 저장/추론 구현. OOF 개발결과와fullfit추론 구분. 결합평가 새로계산(기존50가스 cascade 숫자 재사용금지).
4. 각제품01_preprocess_train.ipynb /02_predict.ipynb 완성, 데이터 전처리/모델/사용자 입력 스키마 문서.
5. 기존 notebooks/reports/src/models/examples를정확한대상확인후hayeon/로이동하고참조경로수정. 데이터/venv/.git/다른작업 삭제금지. 미완성산출물 새배포폴더와구분.
6. 깨끗한폴더 원본→학습→예측재현, 예측 시 학습 호출없음 및라벨입력불필요 점검.
7. 선택적으로staging,대용량/민감정보/산출물과실험분리 검토,commit/push. 현재커밋/push아직없음.
