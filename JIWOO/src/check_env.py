"""환경이 제대로 세팅됐는지 확인합니다.  실행: python src/check_env.py"""
import sys
from pathlib import Path

REQUIRED = ["pandas", "numpy", "matplotlib", "seaborn", "sklearn"]
DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = ["labeled_data.csv", "moldset_labeled_cn7.csv", "moldset_labeled_rg3.csv"]

ok = True
for m in REQUIRED:
    try:
        __import__(m)
    except ImportError:
        print(f"[없음] 패키지 {m} — pip install -r requirements.txt 를 실행하세요")
        ok = False

for f in FILES:
    if not (DATA / f).exists():
        print(f"[없음] data/raw/{f} — KAMP에서 받아 복사하세요")
        ok = False

if ok:
    print(f"[OK] 환경 정상 (python {sys.version.split()[0]})")
