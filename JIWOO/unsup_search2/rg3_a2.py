"""PREREG 'RG3' 항목: CN7 W1 공식 AP가 가장 높은 A 후보(A2: dev38 + LW-Mahalanobis)를 RG3에 그대로 적용 → AP만 (탐색)"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import run as R

C = R.C
F = C.build_cache()
ws = (F['equip'] == C.EQUIP_W2) & (F['prod'] == 'RG3')
U = R.add_new_relations(R.for_product(F[ws], 'rg3'))
lab = R.add_new_relations(R.for_product(C.build_labeled_feats()['rg3'], 'rg3'))
ref = U[U['TimeStamp'] < '2020-10-21']                     # RG3 라벨 시작(10/21) 이전
s_lab, _ = R.lw(ref, lab, R.DEV38)
l = pd.read_csv(C.FINAL_INTER / 'linked' / 'linked_labeled.csv', parse_dates=['TimeStamp'])
e = l[(l['product'] == 'rg3') & l['in_secondary']]
s = e['TimeStamp'].map(pd.Series(s_lab, index=lab['TimeStamp'].values)).values
y = e['PassOrFail'].values
print(f'RG3 A2 학습 {len(ref):,}샷 | 공식 {len(y)}행 불량 {int(y.sum())} | AP {average_precision_score(y, s):.3f} '
      f'(무작위 {y.mean():.3f})')
