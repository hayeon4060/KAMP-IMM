"""검증 실험 그림: fig_A(라벨 섞기), fig_B(CN7 전향 검증) → outputs/experiments/"""
import json

import numpy as np
import pandas as pd

from plot_font import setup   # Agg 백엔드 지정 후 pyplot 사용
import matplotlib
import matplotlib.dates  # noqa: F401
import matplotlib.pyplot as plt

from paths import EXP


def fig_a():
    setup(10)
    R = json.load(open(EXP / "label_shuffle.json"))
    BLUE, GRAY, INK, INK2, SURF = "#2a78d6", "#a8a7a1", "#0b0b0b", "#52514e", "#fcfcfb"
    NAMES = {"guidebook": "분할 전 SMOTE 절차 (가이드북 수치 재현)\nSMOTE → RF → 무작위 5-fold",
             "honest_rf": "지도학습 RF, 누설 차단\n쇼트 단위 분할 · ±10행 간격 · fold 안 SMOTE",
             "deviation_K10": "우리 모델: 과거기준선 이탈도\n라벨 없이 학습 · 검사 예산 5%"}
    ORDER = ["guidebook", "honest_rf", "deviation_K10"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURF)
    for ax, prod in zip(axes, ["CN7", "RG3"]):
        ax.set_facecolor(SURF)
        rows = {r["method"]: r for r in R if r["prod"] == prod}
        for i, m in enumerate(ORDER):
            r = rows[m]; sh = np.array(r["shuf"]); y0 = len(ORDER) - 1 - i
            ax.barh(y0 + 0.18, r["real"], height=0.32, color=BLUE, edgecolor=SURF, linewidth=2)
            ax.barh(y0 - 0.18, sh.mean(), height=0.32, color=GRAY, edgecolor=SURF, linewidth=2)
            lo, hi = np.quantile(sh, [0.025, 0.975])
            ax.plot([lo, hi], [y0 - 0.18] * 2, color=INK2, lw=1.2)
            ax.text(r["real"] + 0.015, y0 + 0.18, f"{r['real']:.3f}", va="center", color=INK, fontsize=9.5)
            ax.text(max(sh.mean(), hi) + 0.015, y0 - 0.18, f"{sh.mean():.3f}", va="center", color=INK2, fontsize=9.5)
            gain = r["real"] - sh.mean()
            ax.text(1.0, y0 + 0.42, f"라벨이 보탠 몫 {gain:+.3f}", ha="right", va="center", fontsize=8.5,
                    color=BLUE if (m == "deviation_K10" and gain > 0.05) else INK2,
                    fontweight="bold" if (m == "deviation_K10" and gain > 0.05) else "normal")
        ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, len(ORDER) - 0.35)
        ax.set_title(prod, loc="left", fontsize=12, fontweight="bold", color=INK)
        ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([NAMES[m] for m in ORDER[::-1]], fontsize=9, color=INK)
        ax.xaxis.grid(True, color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
        for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2"); ax.tick_params(colors=INK2, length=0)
        ax.set_xlabel("F1", color=INK2)

    h1 = plt.Rectangle((0, 0), 1, 1, color=BLUE); h2 = plt.Rectangle((0, 0), 1, 1, color=GRAY)
    fig.legend([h1, h2], ["진짜 라벨", "라벨을 무작위로 섞은 뒤 (평균, 선 = 95% 범위)"], loc="upper left",
               ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.005, 0.95))
    fig.suptitle("정답을 무작위로 섞어도 F1 0.98 — 분할 전 SMOTE 절차의 점수는 라벨 정보를 거의 반영하지 않는다", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.005, "분할 전 SMOTE 절차의 F1은 합성 표본이 포함된 균형 데이터·임계 0.5 기준, 나머지는 원 데이터에서 점수 상위 5% 행을 불량 판정한 F1. "
             "섞기 횟수: RF 20회, 이탈도 1,000회. 공식 라벨 파일(CN7 1,211행·불량 17, RG3 1,182행·불량 25).",
             fontsize=7.8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(EXP / "fig_A_label_shuffle.png", dpi=200, facecolor=SURF)
    plt.close(fig)
    print("  saved fig_A_label_shuffle.png")


def fig_b():
    setup(10)
    E = pd.read_pickle(EXP / "prospective_shots.pkl"); R = json.load(open(EXP / "prospective.json"))
    BLUE, RED, GRAY, INK, INK2, SURF, GRID = "#2a78d6", "#d03b3b", "#b9b8b2", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"

    C = E[E["prod"] == "CN7"].reset_index(drop=True); tau = R["CN7"]["tau"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.6), facecolor=SURF, gridspec_kw=dict(height_ratios=[1.15, 1]))


    def style(ax):
        ax.set_facecolor(SURF); ax.set_yscale("log"); ax.set_ylim(0.15, 30)
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]: ax.spines[s].set_color("#c9c8c2")
        ax.tick_params(colors=INK2, length=0)
        ax.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20]); ax.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10", "20"])


    # 위: 전체 1,389쇼트 (쇼트 순번)
    x = np.arange(len(C)); good = C.y == 0
    a1.scatter(x[good], C.s[good], s=6, color=GRAY, lw=0, label="양품 쇼트")
    bad = C[C.y == 1]
    a1.scatter(bad.index, bad.s, s=60, color=RED, edgecolor=SURF, lw=1.5, zorder=3, label="불량 쇼트")
    a1.axhline(tau, color=BLUE, lw=1.5, ls="--")
    a1.text(len(C) - 5, tau * 1.12, f"경보 임계 τ = {tau:.2f} (공식 기간 상위 5%로 고정)", ha="right", color=BLUE, fontsize=9)
    for dday, g in C.groupby(C.t.dt.date):
        a1.axvline(g.index.min() - 0.5, color="#c9c8c2", lw=0.8)
        a1.text(g.index.min() + 4, 24, pd.Timestamp(dday).strftime("%m/%d"), color=INK2, fontsize=9, va="center")
    a1.annotate("초기허용불량 5쇼트\n(4쇼트 경보)", (bad.index[0], bad.s.iloc[0]), xytext=(90, 9), textcoords="data",
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    mi = bad[bad.reason == "미성형"]
    a1.annotate("11/03 미성형 (LH 한쪽)\n놓침", (mi.index[0], mi.s.iloc[0]), xytext=(mi.index[0] - 330, 0.2),
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    style(a1); a1.set_xlim(-10, len(C) + 10); a1.set_xlabel("추가 기간 쇼트 순번 (CN7, 전수검사)", color=INK2)
    a1.set_ylabel("이탈도 점수 (로그)", color=INK2)
    a1.legend(loc="upper right", frameon=False, fontsize=9, bbox_to_anchor=(1.0, 0.93))

    # 아래: 10/27 00:30~01:25 확대 (실제 시각)
    Z = C[(C.t >= "2020-10-27 00:30") & (C.t <= "2020-10-27 01:25")]
    gz = Z[Z.y == 0]; bz = Z[Z.y == 1]
    a2.plot(Z.t, Z.s, color=GRAY, lw=1, zorder=1)
    a2.scatter(gz.t, gz.s, s=18, color=GRAY, lw=0, zorder=2)
    a2.scatter(bz.t, bz.s, s=70, color=RED, edgecolor=SURF, lw=1.5, zorder=3)
    a2.axhline(tau, color=BLUE, lw=1.5, ls="--")
    t0, t1 = pd.Timestamp("2020-10-27 00:46:23"), pd.Timestamp("2020-10-27 00:56:21")
    a2.axvspan(t0, t1, color="#f0efec", zorder=0)
    a2.text(t0 + (t1 - t0) / 2, 0.22, "약 10분 정지", ha="center", color=INK2, fontsize=9)
    for i, (_, r) in enumerate(bz.iterrows()):
        if i < 2:
            a2.text(r.t + pd.Timedelta(seconds=40), r.s * 1.08, r.top3.split(", ")[0], fontsize=8.5, color=INK, va="bottom")
    b3 = bz.iloc[2:]
    a2.annotate("나머지 3쇼트: z 3~5 (사이클·계량시간)\n2쇼트 경보, 1쇼트 임계 바로 아래", (b3.t.iloc[1], b3.s.iloc[1]),
                xytext=(pd.Timestamp("2020-10-27 01:09"), 4.2), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    style(a2); a2.set_xlabel("10/27 시각 (확대)", color=INK2); a2.set_ylabel("이탈도 점수 (로그)", color=INK2)
    a2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))

    c = R["CN7"]
    fig.suptitle(f"본 적 없는 미래 기간에서 CN7 불량 6쇼트 중 4쇼트 경보 — 경보율 {c['alarm_rate']*100:.1f}% (재현율 67%, 무작위 대비 약 8배)",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "설정·임계값은 공식 기간(10/16~10/23)에서 고정하고 10/27~11/03에 그대로 적용. 예측은 실행 전 문서로 기록(10/4 18:56). "
             "점수 옆 글자 = 가장 크게 벗어난 변수(z). Fisher 정확검정 p = 0.0005.", fontsize=7.8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(EXP / "fig_B_prospective_cn7.png", dpi=200, facecolor=SURF)
    plt.close(fig)
    print("  saved fig_B_prospective_cn7.png")


def main():
    fig_a()
    fig_b()


if __name__ == "__main__":
    main()
