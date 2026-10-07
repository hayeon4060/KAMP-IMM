"""검증 실험 B. 사전등록 전향(미래) 검증 — 과거기준선 이탈도

공식 기간에서 고정한 이탈도·임계값(공식 기간 행 점수 상위 5%)을 추가 기간(CN7 10/27~11/03, RG3 11/04)에
그대로 적용. 예측은 실행 전에 문서로 기록해 두었다(사전등록).
출력: outputs/experiments/prospective.json, pred_prospective_baseline.csv, prospective_shots.pkl(그림용)
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from paths import EXP
from score_z import past_baseline_z, rms, to_rows
from shots_data import load_all, stream, PRODS, VARS

RESTART_GAP = 1800  # 30분 넘는 공백 뒤 = 재기동
SHORT = {"Injection_Time": "사출시간", "Filling_Time": "충전시간", "Plasticizing_Time": "계량시간", "Cycle_Time": "사이클",
         "Clamp_Close_Time": "형체시간", "Cushion_Position": "쿠션", "Plasticizing_Position": "계량위치",
         "Clamp_Open_Position": "형개위치", "Max_Injection_Speed": "사출속도", "Max_Screw_RPM": "최대RPM",
         "Average_Screw_RPM": "평균RPM", "Max_Injection_Pressure": "사출압력", "Max_Switch_Over_Pressure": "전환압력",
         "Max_Back_Pressure": "최대배압", "Average_Back_Pressure": "평균배압", "Hopper_Temperature": "호퍼온도",
         "Mold_Temperature_3": "금형온도3", "Mold_Temperature_4": "금형온도4"}
SHORT.update({f"Barrel_Temperature_{i}": f"배럴온도{i}" for i in range(1, 7)})


def main():
    d = load_all(); L, S, U = d["L"], d["S"], d["U"]
    res, shot_tabs, row_tabs = {}, [], []
    for p in PRODS:
        st = stream(S, U, p)
        Z = past_baseline_z(st, 10); st["s"] = rms(Z)
        top = np.argsort(-np.nan_to_num(Z), 1)[:, :3]
        st["top3"] = [", ".join(f"{SHORT[VARS[j]]}({Z[i, j]:.0f})" for j in top[i]) for i in range(len(st))]
        st["gap"] = st.t.diff().dt.total_seconds()
        st["restart"] = st.gap > RESTART_GAP
        st["since_restart"] = st.groupby(st.restart.cumsum()).cumcount()
        Lo = L[(L["prod"] == p) & (L.period == "official")]
        r = to_rows(Lo, st, "s"); k = int(np.ceil(0.05 * len(r)))
        tau = np.sort(r)[::-1][k - 1]

        E = st[(st.src == "labeled") & (st.period == "extra")].copy()
        E["alarm"] = (E.s >= tau).astype(int)
        E["rank_in_period"] = E.s.rank(ascending=False, method="first").astype(int)
        E["prod"] = p
        n = len(E); kk = int(np.ceil(0.05 * n))
        good = E[E.y == 0]; bad = E[E.y == 1]
        tab = [[int(bad.alarm.sum()), int((1 - bad.alarm).sum())], [int(good.alarm.sum()), int((1 - good.alarm).sum())]]
        out = dict(tau=float(tau), shots=n, defects=int(E.y.sum()), alarms=int(E.alarm.sum()),
                   alarm_rate=float(E.alarm.mean()), good_alarm_rate=float(good.alarm.mean()),
                   defect_alarmed=int(bad.alarm.sum()), fisher_p=float(fisher_exact(tab, "greater")[1]),
                   top5pct_k=kk, defect_in_top5pct=int((bad.rank_in_period <= kk).sum()),
                   alarms_within10_after_restart=int((E.alarm & (E.since_restart < 10)).sum()),
                   restarts=int(E.restart.sum()))
        rr = {}   # 재기동 규칙 비교: 재기동 후 첫 N쇼트 검사
        for N in [3, 5, 10]:
            m = E.since_restart < N
            rr[N] = dict(inspected=int(m.sum()), caught=int((m & (E.y == 1)).sum()))
        out["restart_rule"] = rr
        res[p] = out
        print(f"  {p}: τ={tau:.4f} 쇼트 {n} 불량 {out['defects']} 경보 {out['alarms']} ({out['alarm_rate']*100:.1f}%) "
              f"불량 중 경보 {out['defect_alarmed']} Fisher p={out['fisher_p']:.4g}")
        shot_tabs.append(E)
        Le = L[(L["prod"] == p) & (L.period == "extra")].copy()
        m = E.set_index("TimeStamp")
        for c in ["s", "alarm", "rank_in_period", "since_restart", "top3"]:
            Le[c] = Le.TimeStamp.map(m[c])
        row_tabs.append(Le)
    json.dump(res, open(EXP / "prospective.json", "w"), ensure_ascii=False, indent=1, default=int)
    R = pd.concat(row_tabs)
    R = R.rename(columns={"_id": "원본_id", "prod": "제품", "TimeStamp": "쇼트시각", "side": "좌우", "s": "이탈도점수",
                          "alarm": "검사대상", "rank_in_period": "검사우선순위", "since_restart": "재기동후쇼트",
                          "top3": "이상변수Top3(z)", "y": "실제불량", "reason": "불량사유"})
    R[["원본_id", "제품", "쇼트시각", "좌우", "이탈도점수", "검사대상", "검사우선순위", "재기동후쇼트", "이상변수Top3(z)",
       "실제불량", "불량사유"]].to_csv(EXP / "pred_prospective_baseline.csv", index=False, encoding="utf-8-sig")
    pd.concat(shot_tabs).to_pickle(EXP / "prospective_shots.pkl")


if __name__ == "__main__":
    main()
