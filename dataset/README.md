# 데이터 준비 (Git 미포함)

아래 파일은 사용자가 직접 준비합니다. 데이터 내용과 중간 가공 CSV는 커밋하지 않습니다.

```text
dataset/
  original/
    labeled_data.csv
    moldset_labeled.csv
    unlabeled_data.csv.gz
  kamp/
    moldset_labeled_cn7.csv
    moldset_labeled_rg3.csv
    moldset_unlabeled_cn7.csv
    moldset_unlabeled_rg3.csv
  processed/  # 제품별 01 실행으로 자동 생성
```

original은 기존 프로젝트에서 받은 원본, kamp는 KAMP 제공 제품별 파일입니다. 파일 이름만 같은 다른 자료로 대체하면 재현을 보장하지 않습니다. unlabeled_data.csv.gz는 원본 unlabeled_data.csv를 gzip으로 압축한 파일이며, 두 제품의 01 노트북이 복원합니다.

RG3는 두 제품의 연결 파일이 필요하므로 CN7 파일도 준비하세요. 모델에 들어가는 값은 원단위와 과거 이력이 복원된 가공 피처이며 KAMP 표준화 센서 CSV를 02에 직접 넣지 않습니다. 01 실행 후 생성된 processed/rg3/rg3_unlabeled_input_features.csv 및 processed/cn7/cn7_shots.csv가 기본 예측 입력입니다. 이력 충분성/공정 유사성 조건을 통과한 비라벨 부분집합의 결과를 전체 비라벨 불량률로 해석하지 마세요.

저장 모델에는 학습·기준선에서 나온 통계와 참조 센서 배열이 포함됩니다. 원본 CSV를 제외하는 것이 모델에서 원천 정보까지 완전히 제거한다는 뜻은 아닙니다. 모델은 이 프로젝트의 공유 범위 안에서 사용하세요.
