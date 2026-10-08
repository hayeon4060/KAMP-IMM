"""전처리 ③ 원본 라벨 행·쇼트 표·비라벨 정상 쇼트 (확률 층·비교 기준선 공용)

- 공식 라벨 행 = moldset_labeled.csv 행 0~2392 (= 공식 2차 파일의 Unnamed: 0), _id 로 원본과 연결
- 이중기록(시각·부품명이 같은 행) 제거
- 쇼트 = (제품, 시각) 묶음. LH/RH 두 행의 공정값은 같음
- 비라벨 정상 모드(0 모드 제외) 쇼트를 과거 기준선 흐름에 사용
"""
import numpy as np
import pandas as pd

from paths import DATA, KAMP_CACHE

VARS = ["Injection_Time", "Filling_Time", "Plasticizing_Time", "Cycle_Time", "Clamp_Close_Time",
        "Cushion_Position", "Plasticizing_Position", "Clamp_Open_Position", "Max_Injection_Speed",
        "Max_Screw_RPM", "Average_Screw_RPM", "Max_Injection_Pressure", "Max_Switch_Over_Pressure",
        "Max_Back_Pressure", "Average_Back_Pressure", "Barrel_Temperature_1", "Barrel_Temperature_2",
        "Barrel_Temperature_3", "Barrel_Temperature_4", "Barrel_Temperature_5", "Barrel_Temperature_6",
        "Hopper_Temperature", "Mold_Temperature_3", "Mold_Temperature_4"]
PRODS = ["CN7", "RG3"]


def official(prod):
    """공식 2차 라벨 파일 (표준화값)."""
    return pd.read_csv(DATA / f"moldset_labeled_{prod.lower()}.csv")


def raw_labeled():
    """원본 라벨 행 (이중기록 제거, 우진2호기 CN7/RG3). 열: prod, side, t, y, reason, period, oid(공식 행번호)"""
    L = pd.read_csv(DATA / "labeled_data.csv")
    L = L[L.EQUIP_NAME.str.contains("우진2호기") & L.PART_NAME.str[:3].isin(PRODS)].copy()
    M = pd.read_csv(DATA / "moldset_labeled.csv", usecols=["Unnamed: 0", "_id"])
    off = M[M["Unnamed: 0"] <= 2392].set_index("_id")["Unnamed: 0"]
    L["oid"] = L["_id"].map(off)
    L["t"] = pd.to_datetime(L.TimeStamp)
    L["prod"] = L.PART_NAME.str[:3]
    L["side"] = np.where(L.PART_NAME.str.contains("LH"), "LH", "RH")
    # 이중기록: 같은 시각·부품명 → 공식 행이 있으면 그것을 남김
    L = L.sort_values(["oid"], na_position="last").drop_duplicates(["TimeStamp", "PART_NAME"], keep="first")
    L["y"] = (L.PassOrFail == "N").astype(int)
    L["reason"] = L.Reason.where(L.y == 1, "")
    L["period"] = np.where(L.oid.notna(), "official", "extra")
    return L.sort_values(["prod", "t", "side"]).reset_index(drop=True)


def shots(L):
    """쇼트 단위 표: 제품·시각마다 1행. y=어느 쪽이든 불량, LH_fail/RH_fail, 사유."""
    g = L.groupby(["prod", "TimeStamp"], sort=False)
    S = g[VARS].first()
    S["t"] = g.t.first()
    S["period"] = g.period.first()
    S["n_rows"] = g.size()
    S["y"] = g.y.max()
    S["LH_fail"] = g.apply(lambda d: int(((d.side == "LH") & (d.y == 1)).any()))
    S["RH_fail"] = g.apply(lambda d: int(((d.side == "RH") & (d.y == 1)).any()))
    S["reason"] = g.reason.agg(lambda r: "+".join(sorted(set(x for x in r if x))))
    S = S.reset_index().sort_values(["prod", "t"]).reset_index(drop=True)
    S["both"] = S.LH_fail & S.RH_fail
    S["src"] = "labeled"
    return S


def unlabeled_normal():
    """비라벨 우진2호기 CN7/RG3 정상 모드 쇼트 (0 모드: 배럴온도1=0 제외), 시각당 1행."""
    U = pd.read_csv(DATA / "unlabeled_data.csv")
    U = U[U.EQUIP_NAME.str.contains("우진2호기") & U.PART_NAME.str[:3].isin(PRODS)].copy()
    U = U[U.Barrel_Temperature_1 != 0]
    U["prod"] = U.PART_NAME.str[:3]
    U["t"] = pd.to_datetime(U.TimeStamp)
    U = U.drop_duplicates(["prod", "TimeStamp"])
    U = U[["prod", "TimeStamp", "t"] + VARS].copy()
    U["src"] = "unlabeled"
    return U


def stream(S, U, prod):
    """과거 기준선용 쇼트 흐름: 라벨·비라벨 쇼트를 시각순으로 합침 (같은 시각은 라벨 쪽 값)."""
    a = S[S["prod"] == prod]
    b = U[(U["prod"] == prod) & ~U.TimeStamp.isin(a.TimeStamp)]
    return pd.concat([a, b], ignore_index=True).sort_values("t").reset_index(drop=True)


def build():
    """원본에서 L·S·U를 만들어 캐시에 저장 (run_all 1단계에서 항상 새로 생성)."""
    L = raw_labeled()
    S = shots(L)
    U = unlabeled_normal()
    out = dict(L=L, S=S, U=U)
    pd.to_pickle(out, KAMP_CACHE)
    print(f'  라벨 행 {len(L):,} / 쇼트 {len(S):,} / 비라벨 정상 쇼트 {len(U):,}')
    return out


def load_all():
    if not KAMP_CACHE.exists():
        return build()
    return pd.read_pickle(KAMP_CACHE)
