"""전처리 ① 1차(원본) ↔ 2차(대회 제공) 데이터 연결

linked/linked_labeled.csv
    1차 moldset_labeled.csv에서 2차 파일(moldset_labeled_{cn7,rg3}.csv)에 해당하는 행을 골라
    가이드북과 같은 표준화(파일별, ddof=0)로 2차 값과 1e-6 이내 일치·라벨 100% 일치를 assert로 확인.
    규칙: CN7은 생산계획일 2020-10-27을 제외한 1,211행, RG3는 1,182행 전부.
    열: product, in_secondary, TimeStamp, cavity, PassOrFail, Reason (2차 범위 행만, 제품·시각·캐비티 순)
linked/linked_unlabeled_{cn7,rg3}.csv
    1차 unlabeled_data.csv에서 제품명이 CN7/RG3로 시작하는 행 (파일 순서 그대로)
    열: TimeStamp, EQUIP_NAME, PART_NAME, 공정변수 21개(*_raw)
"""
import numpy as np
import pandas as pd

from paths import DATA, LINKED

PRODUCTS = ['cn7', 'rg3']
EXCLUDE_DATES = {'cn7': ['2020-10-27'], 'rg3': []}
ALL_RAW = ['Injection_Time', 'Filling_Time', 'Plasticizing_Time', 'Cycle_Time', 'Clamp_Close_Time',
           'Cushion_Position', 'Plasticizing_Position', 'Max_Injection_Speed', 'Max_Screw_RPM',
           'Max_Injection_Pressure', 'Max_Switch_Over_Pressure', 'Average_Back_Pressure',
           'Barrel_Temperature_1', 'Barrel_Temperature_2', 'Barrel_Temperature_3',
           'Barrel_Temperature_4', 'Barrel_Temperature_5', 'Barrel_Temperature_6',
           'Hopper_Temperature', 'Mold_Temperature_3', 'Mold_Temperature_4']


def _features():
    """2차 공정변수 24개 (2차 파일 열 순서)"""
    return [c for c in pd.read_csv(DATA / 'moldset_labeled_cn7.csv', nrows=1).columns
            if c not in ('Unnamed: 0', 'PassOrFail')]


def _zscore(df, feats):
    x = df[feats].astype(float)
    sd = x.std(ddof=0).replace(0, 1)
    return (x - x.mean()) / sd


def link_labeled():
    feats = _features()
    M = pd.read_csv(DATA / 'moldset_labeled.csv', low_memory=False)
    if M['PassOrFail'].dtype == object:
        M['PassOrFail'] = M['PassOrFail'].map({'Y': 0, 'N': 1}).fillna(M['PassOrFail']).astype(int)
    M['plan_date'] = pd.to_datetime(M['PART_FACT_PLAN_DATE'].str[:10])
    M['TimeStamp'] = pd.to_datetime(M['TimeStamp'])
    M['product'] = M['PART_NAME'].str[:3].str.lower()
    M['cavity'] = M['PART_NAME'].str[-2:]
    M['row_1st'] = np.arange(len(M))

    parts = []
    for p in PRODUCTS:
        sec = pd.read_csv(DATA / f'moldset_labeled_{p}.csv')
        prim = M[M['product'] == p].copy()
        prim['in_secondary'] = ~prim['plan_date'].isin(pd.to_datetime(EXCLUDE_DATES[p]))
        kept = prim[prim['in_secondary']]
        diff = float(np.nanmax(np.abs(_zscore(kept, feats).reset_index(drop=True).values - sec[feats].values)))
        label_ok = (kept['PassOrFail'].values == sec['PassOrFail'].values).mean()
        assert len(kept) == len(sec) and diff < 1e-6 and label_ok == 1.0, f'{p} 연결 실패'
        print(f'  [{p}] labeled 2차 {len(sec):,}행 연결 (값 최대 차이 {diff:.1e}, 라벨 일치 {label_ok:.0%})')
        parts.append(prim)
    L = pd.concat(parts)
    out = (L[L['in_secondary']]
           .sort_values(['product', 'TimeStamp', 'cavity'], kind='stable')
           [['product', 'in_secondary', 'TimeStamp', 'cavity', 'PassOrFail', 'Reason']])
    out.to_csv(LINKED / 'linked_labeled.csv', index=False)
    return out


def link_unlabeled():
    use = ['TimeStamp', 'PART_NAME', 'EQUIP_NAME'] + ALL_RAW
    U = pd.read_csv(DATA / 'unlabeled_data.csv', usecols=use, low_memory=False)
    for p in PRODUCTS:
        u = U[U['PART_NAME'].str.upper().str.startswith(p.upper())].copy()
        u['TimeStamp'] = pd.to_datetime(u['TimeStamp'])
        u = u[['TimeStamp', 'EQUIP_NAME', 'PART_NAME'] + ALL_RAW].rename(columns={c: f'{c}_raw' for c in ALL_RAW})
        u.to_csv(LINKED / f'linked_unlabeled_{p}.csv', index=False)
        print(f'  [{p}] unlabeled {len(u):,}행')


def main():
    link_labeled()
    link_unlabeled()


if __name__ == '__main__':
    main()
