"""전처리 ④ 측면(LH/RH) 배분 Beta 사전 — 외부 데이터(unlabeled_data.csv의 ERR_FACT_QTY) 사용

평가 이전·마감 로트(계획일 CN7 ≤ 2020-10-15, RG3 ≤ 2020-10-20), 650톤-우진2호기,
로트 키 = (계획일, PART_FACT_SERIAL, PART_NO, 측면)별 ERR_FACT_QTY 최댓값 → 측면별 합 → 비율.
비율은 소수 4자리로 고정해 저장 (docs/PREREG_BETA_W.md 사전등록과 같은 값).
"""
import json
import numpy as np
import pandas as pd

from paths import DATA, W_PRIOR

CUT = {"CN7": "2020-10-15", "RG3": "2020-10-20"}


def main():
    R = pd.read_csv(DATA / "unlabeled_data.csv",
                    usecols=["TimeStamp", "PART_FACT_PLAN_DATE", "PART_FACT_SERIAL", "PART_NO", "PART_NAME",
                             "EQUIP_NAME", "ERR_FACT_QTY"])
    R = R[R.EQUIP_NAME.astype(str).str.contains("우진2호기", na=False)].copy()
    R["prod"] = R.PART_NAME.astype(str).str[:3]
    R = R[R["prod"].isin(["CN7", "RG3"])]
    R["side"] = np.where(R.PART_NAME.str.contains("LH"), "LH", "RH")
    R["pd_"] = R.PART_FACT_PLAN_DATE.astype(str).str[:10]
    prior = {}
    for prod, cut in CUT.items():
        lots = (R[(R["prod"] == prod) & (R.pd_ <= cut)]
                .groupby(["pd_", "PART_FACT_SERIAL", "PART_NO", "side"]).ERR_FACT_QTY.max().reset_index())
        s = lots.groupby("side").ERR_FACT_QTY.sum()
        tot = s.sum()
        prior[prod.lower()] = dict(ERR_LH=int(s.get("LH", 0)), ERR_RH=int(s.get("RH", 0)),
                                   share_LH=round(float(s.get("LH", 0) / tot), 4),
                                   share_RH=round(float(s.get("RH", 0) / tot), 4))
        print(f'  {prod}', prior[prod.lower()])
    json.dump(prior, open(W_PRIOR, "w"), indent=1)
    return prior


if __name__ == '__main__':
    main()
