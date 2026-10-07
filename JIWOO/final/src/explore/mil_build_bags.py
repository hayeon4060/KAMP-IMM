"""[탐색, 선택 실행] MIL 가방 조립 — docs/PREREG_MIL_1006.md 그대로. 순서: mil_build_bags.py → mil_run.py (torch 필요). 라벨 미사용(라벨 시각은 제거에만)."""
import sys, pickle
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LEDGER, FINAL, INTER, EXP, DATA
OUTX = EXP / 'explore'; OUTX.mkdir(parents=True, exist_ok=True)
import final_model as FM

BASE = str(INTER)
CUT = {'cn7': '2020-10-15', 'rg3': '2020-10-20'}
R = pd.read_csv(DATA / 'unlabeled_data.csv',
                usecols=['TimeStamp', 'PART_FACT_PLAN_DATE', 'PART_FACT_SERIAL', 'PART_NO', 'PART_NAME', 'EQUIP_NAME', 'ERR_FACT_QTY'])
R = R[R.EQUIP_NAME.astype(str).str.contains('우진2호기', na=False)].copy()
R['prod'] = R.PART_NAME.astype(str).str[:3].str.lower()
R['side'] = np.where(R.PART_NAME.str.contains('LH'), 'LH', 'RH')
R['pd_'] = R.PART_FACT_PLAN_DATE.astype(str).str[:10]
R['TimeStamp'] = pd.to_datetime(R.TimeStamp)
R['lot'] = R.pd_ + '|' + R.PART_FACT_SERIAL.astype(str) + '|' + R.PART_NO.astype(str) + '|' + R.side
out = {}
for p in ('cn7', 'rg3'):
    shots = FM.load_labeled_shots(p, BASE)
    U, log = FM.build_reference(p, shots, BASE, period=None)
    m = FM.FinalModel.load(str(FINAL / f'{p}_final_model.pkl'))
    F = FM.make_features(U, m.steps, m.fr_step)
    X = m._prep(F); sc = m.score(F)
    inst = pd.DataFrame(X, columns=m.cols); inst['TimeStamp'] = F.TimeStamp.values
    inst['run_id'] = F.run_id.values; inst['maha'] = sc
    r = R[R['prod'] == p].drop_duplicates(['TimeStamp', 'side'])[['TimeStamp', 'side', 'lot', 'pd_', 'ERR_FACT_QTY']]
    I = inst.merge(r, on='TimeStamp', how='inner')
    lotk = R[R['prod'] == p].groupby('lot').agg(k=('ERR_FACT_QTY', 'max'), pd_=('pd_', 'first'), side=('side', 'first'))
    n = I.groupby('lot').size().rename('n')
    B = lotk.join(n, how='inner')
    B['keep'] = (B.pd_ <= CUT[p]) & (B.n >= 300) & (B.k <= B.n)
    print(p, log, '| inst rows', len(I), '| lots', len(B), 'kept', int(B.keep.sum()),
          '| LH/RH kept', B[B.keep].side.value_counts().to_dict(),
          '| pd range', B[B.keep].pd_.min(), B[B.keep].pd_.max(),
          '| sum k', int(B[B.keep].k.sum()), 'sum n', int(B[B.keep].n.sum()))
    print('  dropped reasons: late', int((B.pd_ > CUT[p]).sum()), 'small n', int(((B.pd_ <= CUT[p]) & (B.n < 300)).sum()),
          'k>n', int(((B.pd_ <= CUT[p]) & (B.n >= 300) & (B.k > B.n)).sum()))
    out[p] = dict(I=I, B=B, cols=m.cols)
pickle.dump(out, open(OUTX / 'mil_bags.pkl', 'wb'))
