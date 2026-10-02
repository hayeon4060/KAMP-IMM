"""데이터 로딩 공용 함수.

노트북에서 이렇게 씁니다:

    import sys; sys.path.append("..")
    from src.load import load_labeled, PROC_COLS
    df = load_labeled("CN7")
"""
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# 공정 변수 24개. 2차 가공본(제출 모델 입력)에 들어 있는 것과 같습니다.
PROC_COLS = [
    "Injection_Time", "Filling_Time", "Plasticizing_Time", "Cycle_Time",
    "Clamp_Close_Time", "Cushion_Position", "Plasticizing_Position",
    "Clamp_Open_Position", "Max_Injection_Speed", "Max_Screw_RPM",
    "Average_Screw_RPM", "Max_Injection_Pressure", "Max_Switch_Over_Pressure",
    "Max_Back_Pressure", "Average_Back_Pressure",
    "Barrel_Temperature_1", "Barrel_Temperature_2", "Barrel_Temperature_3",
    "Barrel_Temperature_4", "Barrel_Temperature_5", "Barrel_Temperature_6",
    "Hopper_Temperature", "Mold_Temperature_3", "Mold_Temperature_4",
]

# 단계별로 묶은 것. 그래프 그릴 때 이 단위로 그리면 읽기 편합니다.
GROUPS = {
    "시간": ["Injection_Time", "Filling_Time", "Plasticizing_Time",
             "Clamp_Close_Time", "Cycle_Time"],
    "위치": ["Cushion_Position", "Plasticizing_Position", "Clamp_Open_Position"],
    "속도": ["Max_Injection_Speed", "Max_Screw_RPM", "Average_Screw_RPM"],
    "압력": ["Max_Injection_Pressure", "Max_Switch_Over_Pressure",
             "Max_Back_Pressure", "Average_Back_Pressure"],
    "배럴온도": [f"Barrel_Temperature_{i}" for i in range(1, 7)],
    "금형온도": ["Mold_Temperature_3", "Mold_Temperature_4"],
}


def load_labeled(product=None, drop_dup=True):
    """1차 가공본을 읽습니다. 시각·LH/RH·불량유형이 살아 있어 진단용으로 씁니다.

    product  : "CN7" | "RG3" | None(둘 다)
    drop_dup : 완전중복 행 제거. CN7은 41%가 중복이라 기본값 True.

    추가되는 컬럼:
      ts       datetime 으로 바꾼 시각
      prod     CN7 / RG3
      side     LH / RH  (같은 쇼트에서 나온 좌우 부품)
      y        불량 1, 양품 0
      shot     같은 시각 = 같은 쇼트 번호
      date     날짜
      shot_idx 그날 그 side 의 몇 번째 쇼트인지 (0부터)
    """
    df = pd.read_csv(RAW / "labeled_data.csv")
    df["ts"] = pd.to_datetime(df["TimeStamp"])
    df["prod"] = df["PART_NAME"].str.extract(r"^(\w+)")[0]
    df["side"] = df["PART_NAME"].str.extract(r"(LH|RH)")[0]
    df["y"] = (df["PassOrFail"] == "N").astype(int)
    df = df[df["prod"].isin(["CN7", "RG3"])]

    if drop_dup:
        cols = [c for c in df.columns if c != "_id"]
        df = df.drop_duplicates(subset=cols)

    if product:
        df = df[df["prod"] == product]

    df = df.sort_values("ts").reset_index(drop=True)
    df["shot"] = df.groupby(["prod", "ts"]).ngroup()
    df["date"] = df["ts"].dt.date
    df["shot_idx"] = df.groupby(["prod", "date", "side"]).cumcount()
    return df


def load_moldset(product, labeled=True):
    """2차 가공본을 읽습니다. 표준화되어 있고 시각·LH/RH·불량유형이 없습니다.
    제출용 모델은 이 형식으로 학습해야 합니다."""
    kind = "labeled" if labeled else "unlabeled"
    df = pd.read_csv(RAW / f"moldset_{kind}_{product.lower()}.csv")
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def day_slice(df, date_str):
    """하루치만 잘라냅니다. date_str 예: '2020-10-16'"""
    return df[df["ts"].dt.strftime("%Y-%m-%d") == date_str].copy()
