"""그림용 한글 글꼴 설정 — 시스템에 CJK 글꼴이 있으면 사용, 없으면 기본 글꼴로 (그림은 그대로 생성)."""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

PREFERRED = ["Noto Sans CJK JP", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "Malgun Gothic", "AppleGothic"]


def setup(size=10):
    for f in fm.findSystemFonts():
        if "NotoSansCJK-Regular" in f or "NotoSansCJK-Medium" in f:
            try:
                fm.fontManager.addfont(f)
            except Exception:
                pass
    names = {f.name for f in fm.fontManager.ttflist}
    family = next((n for n in PREFERRED if n in names), None)
    if family is None:
        print("  [안내] 한글 글꼴을 찾지 못해 기본 글꼴로 그립니다 (글자가 네모로 보일 수 있으나 그림·수치는 동일).")
        family = "DejaVu Sans"
        warnings.filterwarnings("ignore", message=".*Glyph.*missing.*")
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": size})
    return family
