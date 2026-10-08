"""경로 설정 — 모든 경로는 패키지 루트(run_all.py가 있는 폴더) 기준 상대 경로."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'dataset'                 # 원본 데이터(대회 제공 파일) 배치 위치
OUT = ROOT / 'outputs'
INTER = OUT / 'intermediate'                     # 전처리 산출물 (linked/, base/, 캐시)
LINKED = INTER / 'linked'
BASE = INTER / 'base'
KAMP_CACHE = INTER / 'shots_cache.pkl'           # 원본 라벨 행(L)·쇼트(S)·비라벨 정상 쇼트(U)
W_PRIOR = INTER / 'w_prior.json'                 # 측면 배분 Beta 사전 (비라벨 ERR 집계)
FINAL = OUT / 'final_model'                      # 최종 탐지기 모델·샷별 점수
PROB = OUT / 'prob'                              # 확률 층 결과
LEDGER = OUT / 'ledger'                          # 원장·헤드라인·감사·비교표 등
PRED = OUT / 'pred'                              # 제출용 예측 파일
EXP = OUT / 'experiments'                        # 선택 단계: 검증 실험 A(라벨 섞기)·B(전향)
DOCS = ROOT / 'docs'

REQUIRED_FILES = ['labeled_data.csv', 'unlabeled_data.csv', 'moldset_labeled.csv',
                  'moldset_labeled_cn7.csv', 'moldset_labeled_rg3.csv']


def ensure_dirs():
    for d in (OUT, INTER, LINKED, BASE, FINAL, PROB, LEDGER, PRED, EXP):
        d.mkdir(parents=True, exist_ok=True)


def check_data():
    miss = [f for f in REQUIRED_FILES if not (DATA / f).exists()]
    if miss:
        raise SystemExit(f'원본 데이터 파일이 없습니다: {miss}\n'
                         f'→ {DATA.relative_to(ROOT)}/ 에 배치하세요 (data/dataset/README.md 참고)')
