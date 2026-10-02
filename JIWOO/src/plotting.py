"""그래프 공용 설정. 한글 폰트와 저장 경로를 맞춥니다."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

FIG = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DEFECT_COLORS = {"가스": "#c0392b", "미성형": "#2c7fb8", "초기허용불량": "#f39c12"}


def setup():
    """한글이 깨지지 않게 폰트를 잡습니다. 노트북 맨 위에서 한 번 호출하세요.

    폰트가 없다는 경고가 뜨면:
      Windows  이미 맑은 고딕이 있어 보통 문제없습니다
      Mac      AppleGothic 이 기본으로 있습니다
      Linux    sudo apt install fonts-nanum 후 커널 재시작
    """
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic",
                 "Noto Sans CJK KR", "Noto Sans CJK JP"]:
        if any(name in f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            print(f"폰트: {name}")
            break
    else:
        print("한글 폰트를 못 찾았습니다. 라벨이 네모로 보일 수 있습니다.")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (14, 4)
    plt.rcParams["figure.dpi"] = 110


def save(fig, name):
    """reports/figures/ 에 저장합니다.  name 예: '0924_금형온도_하루추이'"""
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"저장: {path}")
    return path


def mark_defects(ax, day_df, x="ts"):
    """불량이 난 지점에 세로선을 긋습니다. 불량 유형별로 색이 다릅니다."""
    seen = set()
    for _, r in day_df[day_df["y"] == 1].iterrows():
        reason = r.get("Reason")
        color = DEFECT_COLORS.get(reason, "gray")
        label = reason if reason not in seen else None
        seen.add(reason)
        ax.axvline(r[x], color=color, alpha=0.75, lw=1.2, ls="--", label=label)
    if seen:
        ax.legend(fontsize=9, loc="upper right")


def plot_group(day_df, cols, title, x="ts", ax=None):
    """여러 변수를 한 그래프에 겹쳐 그리고 불량 시점을 표시합니다."""
    if ax is None:
        _, ax = plt.subplots()
    for c in cols:
        ax.plot(day_df[x], day_df[c], lw=0.9, label=c, alpha=0.85)
    mark_defects(ax, day_df, x=x)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    return ax
