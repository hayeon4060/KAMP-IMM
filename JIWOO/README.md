# 사출성형 불량 예측 — KAMP 제6회 경진대회

## 0. 준비 (팀원 각자 1회만)

### 0-1. 저장소 받기
```bash
git clone <저장소주소> kamp_injection
cd kamp_injection
```

### 0-2. 파이썬 환경
```bash
conda create -n kamp python=3.11 -y   # Python 3.11 (검증: 3.11.9 / 3.11.15)
conda activate kamp
pip install -r requirements.txt
python -m ipykernel install --user --name kamp --display-name "kamp"
```

### 0-3. 데이터 넣기 (git에 올리지 않음)
KAMP 포털에서 받은 파일을 `data/raw/` 에 복사합니다.
```
data/raw/
├── labeled_data.csv          # 1차 가공본 — 진단용 (시각·LH/RH·불량유형 있음)
├── unlabeled_data.csv        # 1차 가공본 — 비라벨 79.5만행
├── moldset_labeled_cn7.csv   # 2차 가공본 — 모델 입력 형식
├── moldset_labeled_rg3.csv
├── moldset_unlabeled_cn7.csv
└── moldset_unlabeled_rg3.csv
```

### 0-4. 확인
```bash
python src/check_env.py
```
`[OK] 환경 정상` 이 뜨면 준비 끝입니다.

---

## 1. 폴더 구조
```
kamp_injection/
├── data/raw/          원본 (git 제외)
├── notebooks/         탐색용 노트북 — 각자 이름으로 만들기
├── src/               공용 함수 — 모두가 import 해서 씀
├── reports/figures/   그림 저장 (png)
└── docs/              공유 문서
```

## 2. 작업 규칙
- **노트북은 각자 파일**: `notebooks/01_탐색_홍길동.ipynb` 처럼 이름을 넣습니다. 같은 파일을 둘이 고치면 git 충돌이 심합니다.
- **재사용할 코드는 src로**: 노트북에서 만든 함수가 쓸 만하면 `src/`로 옮기고 팀에 알립니다.
- **그림은 reports/figures에 저장**: 파일명은 `날짜_내용.png` (예: `0924_금형온도_하루추이.png`)
- **커밋 메시지**: `[탐색] 금형온도 3/4 하루 추이 확인` 처럼 대괄호로 분류를 답니다.

## 3. 브랜치
```bash
git checkout -b explore/홍길동-금형온도   # 작업 시작
git add . && git commit -m "[탐색] ..."
git push origin explore/홍길동-금형온도
```
main 브랜치에 직접 push 하지 않습니다. PR로 합칩니다.

## 4. 발견한 것 기록하기
그래프를 그려서 뭔가 보이면 `docs/findings.md` 에 한 줄 추가하고 그림 경로를 적습니다.
형식: `- [날짜] [이름] 무엇을 봤더니 어땠다 → 그림: reports/figures/xxx.png`

## 5. 문서 (docs/)
- `00_공정과_변수.md` — 사출 공정 8단계와 24개 변수 대응표
- `01_데이터_구조.md` — 파일별 성격, 주의사항
- `findings.md` — 발견 기록 (계속 추가)
