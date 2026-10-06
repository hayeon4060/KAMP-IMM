"""
rg3_two_type.py — RG3: 하연님 가스 모델(흔들림 std4)에 '미성형용 변수'를 더해 가스·미성형을 함께 잡기

데이터·검증은 하연님 22번과 동일
  - 하연님 레포 reports/tables의 샷 단위 표 (RG3 591샷: 정상 566, 가스 17, 미성형 8)
    original_labeled_shot_features.csv가 있으면 그것, 없으면 run_shot_feature_candidates.csv + 관측 구간표로 구성
  - 검증: 관측 구간(comparison_block) 하나씩 빼기 / 과거 구간만으로 학습
  - 랜덤포레스트 설정도 동일 (트리 200, depth 8, leaf 3, balanced_subsample, seed 42)
  - 판정 임계값: 학습 구간 '안쪽' 교차검증 점수에서 정상의 99% 분위수 (평가 라벨 미사용)

변수 묶음 (결과 보기 전에 고정)
  GAS_STD4  : 하연님 가스 신호 — 배럴2·호퍼·쿠션·배럴5의 직전 10샷 표준편차
  SHORT6    : 미성형 신호 — RG3 미성형 = 계량 지연(11·12번) + 재료 부족·충전 이상
              계량시간 편차·직전 대비 변화, 배압 편차, 계량위치 편차, 쿠션 편차, 충전시간 편차 (모두 직전 10샷 기준)
  ⚠ SHORT6은 labeled 분석(11·12번)에서 나온 가설로 고름 → 완전히 사전 독립은 아님

모델 (사전 고정)
  A_gas_std4      : GAS_STD4, 가스 vs 정상으로 학습 (하연님 std4 재현 — 미성형 샷은 학습에서 제외, 평가엔 포함)
  B_short_short6  : SHORT6, 미성형 vs 정상으로 학습 (가스 샷은 학습에서 제외)
  C_union_max     : max(A, B) — 둘 중 하나라도 의심되면 (하연님 max_union과 같은 방식)
  D_all_in_one    : GAS_STD4 + SHORT6, 불량 전체(가스+미성형) vs 정상으로 한 모델

평가: 불량 전체 AP / 가스 AP(가스 vs 정상) / 미성형 AP(미성형 vs 정상) / 구간별 상위 10% 검사 시 잡은 수(유형별) /
      학습 정상 99% 임계값의 TP·FP(유형별)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score

GAS_STD4 = ['Barrel_Temperature_2__past10_std', 'Hopper_Temperature__past10_std',
            'Cushion_Position__past10_std', 'Barrel_Temperature_5__past10_std']
SHORT6 = ['Plasticizing_Time__dev10', 'Plasticizing_Time__delta1', 'Average_Back_Pressure__dev10',
          'Plasticizing_Position__dev10', 'Cushion_Position__dev10', 'Filling_Time__dev10']
MODELS = {
    'A_gas_std4':     dict(cols=GAS_STD4, target='gas'),
    'B_short_short6': dict(cols=SHORT6, target='short'),
    'D_all_in_one':   dict(cols=GAS_STD4 + SHORT6, target='any'),
}
UNION = ('C_union_max', 'A_gas_std4', 'B_short_short6')


# ====================================================================== 데이터
def load(tables='../../KAMP-IMM_hayeon/reports/tables'):
    tables = Path(tables)
    f = tables / 'original_labeled_shot_features.csv'
    if f.exists():
        d = pd.read_csv(f, parse_dates=['TimeStamp'])
        src = f.name
    else:
        d = pd.read_csv(tables / 'run_shot_feature_candidates.csv', parse_dates=['TimeStamp'])
        blk = pd.read_csv(tables / 'run_label_observation_blocks.csv', parse_dates=['start', 'end'])
        d['comparison_block'] = np.nan
        for b in blk.itertuples():
            m = d['product'].eq(b.product) & d['TimeStamp'].between(b.start, b.end)
            d.loc[m, 'comparison_block'] = b.label_block
        src = 'run_shot_feature_candidates.csv + run_label_observation_blocks.csv'
    d = d[d['product'].eq('RG3')].sort_values('TimeStamp').reset_index(drop=True)
    r = d['Reason'].fillna('정상').astype(str)
    d['type'] = np.select([r.str.contains('가스'), r.str.contains('미성형')], ['가스', '미성형'], '정상')
    d.loc[d['PassOrFail'].fillna(0).eq(0), 'type'] = '정상'
    missing = [c for c in GAS_STD4 + SHORT6 if c not in d.columns]
    assert not missing, f'열 없음: {missing}'
    assert d['comparison_block'].notna().all(), '관측 구간이 안 붙은 샷이 있음'
    print(f'입력: {src} | RG3 {len(d)}샷 —', d['type'].value_counts().to_dict(),
          '| 구간:', d.groupby('comparison_block')['type'].apply(lambda s: dict(s.value_counts())).to_dict())
    return d


def _target(d, kind):
    t = d['type'].values
    if kind == 'gas':
        return (t == '가스').astype(int), t != '미성형'          # (y, 학습에 쓸 샷)
    if kind == 'short':
        return (t == '미성형').astype(int), t != '가스'
    return (t != '정상').astype(int), np.ones(len(d), bool)


def _rf():
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                         RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3,
                                                class_weight='balanced_subsample', random_state=42, n_jobs=-1))


def fit_predict(d, name, train, test):
    spec = MODELS[name]
    y, use = _target(d, spec['target'])
    tr = train[use[train]]
    if len(np.unique(y[tr])) < 2:
        return None
    m = _rf().fit(d.iloc[tr][spec['cols']], y[tr])
    return m.predict_proba(d.iloc[test][spec['cols']])[:, 1]


# ====================================================================== 검증
def run(d, schemes=('leave_one_block_out', 'past_blocks_only'), q=0.99):
    groups = d['comparison_block'].values
    blocks = list(pd.unique(groups))                                    # 시간 순
    preds, logs = [], []
    for scheme in schemes:
        for j, b in enumerate(blocks):
            test = np.flatnonzero(groups == b)
            train = (np.flatnonzero(groups != b) if scheme == 'leave_one_block_out'
                     else np.flatnonzero(np.isin(groups, blocks[:j])))
            if len(train) == 0:
                continue
            outer, thr = {}, {}
            for name in MODELS:
                s = fit_predict(d, name, train, test)
                if s is None:
                    continue
                # 안쪽: 학습 구간을 하나씩 빼서 OOF 점수 → 정상 99% 분위수 = 판정 임계값
                inner = np.full(len(train), np.nan)
                for ib in pd.unique(groups[train]):
                    m = groups[train] == ib
                    si = fit_predict(d, name, train[~m], train[m])
                    if si is not None:
                        inner[m] = si
                outer[name] = s
                thr[name] = inner
            if UNION[1] in outer and UNION[2] in outer:
                outer[UNION[0]] = np.maximum(outer[UNION[1]], outer[UNION[2]])
                thr[UNION[0]] = np.fmax(thr[UNION[1]], thr[UNION[2]])
            tnorm = d['type'].values[train] == '정상'
            for name, s in outer.items():
                cal = thr[name][tnorm]
                cal = cal[np.isfinite(cal)]
                th = float(np.quantile(cal, q, method='higher')) if len(cal) else 1.0
                for k, i in enumerate(test):
                    preds.append({'scheme': scheme, 'model': name, 'block': b, 'TimeStamp': d['TimeStamp'].iloc[i],
                                  'type': d['type'].iloc[i], 'score': float(s[k]), 'threshold': th,
                                  'alarm': bool(s[k] > th)})
            logs.append({'scheme': scheme, 'block': b, 'train': len(train), 'test': len(test),
                         'train 가스': int((d['type'].values[train] == '가스').sum()),
                         'train 미성형': int((d['type'].values[train] == '미성형').sum()),
                         'test 가스': int((d['type'].values[test] == '가스').sum()),
                         'test 미성형': int((d['type'].values[test] == '미성형').sum()),
                         '평가된 모델': ', '.join(outer)})
            print('완료', scheme, b, flush=True)
    return pd.DataFrame(preds), pd.DataFrame(logs)


def _ap(y, s):
    return average_precision_score(y, s) if 0 < y.sum() < len(y) else np.nan


def metrics(P):
    rows = []
    for (scheme, name), g in P.groupby(['scheme', 'model'], sort=False):
        g = g.reset_index(drop=True)
        t = g['type'].values
        s = g['score'].values + np.random.RandomState(42).rand(len(g)) * 1e-9
        top = np.zeros(len(g), bool)
        for _, part in g.groupby('block', sort=False):
            ids = part.index.to_numpy()
            top[ids[np.argsort(-s[ids])[:int(np.ceil(len(ids) * 0.1))]]] = True
        a = g['alarm'].values
        gas, short, nrm = t == '가스', t == '미성형', t == '정상'
        rows.append({'scheme': scheme, 'model': name, '샷': len(g), '가스': int(gas.sum()), '미성형': int(short.sum()),
                     'AP 전체': _ap((~nrm).astype(int), s), '기준선 전체': (~nrm).mean(),
                     'AP 가스': _ap(gas[~short].astype(int), s[~short]), '기준선 가스': gas[~short].mean(),
                     'AP 미성형': _ap(short[~gas].astype(int), s[~gas]), '기준선 미성형': short[~gas].mean() if (~gas).any() else np.nan,
                     '상위10% 가스': f'{int((top & gas).sum())}/{int(gas.sum())}',
                     '상위10% 미성형': f'{int((top & short).sum())}/{int(short.sum())}',
                     '상위10% 선택': int(top.sum()),
                     '경보 TP 가스': int((a & gas).sum()), '경보 TP 미성형': int((a & short).sum()),
                     '경보 FP': int((a & nrm).sum()), '정상 중 경보 비율': float((a & nrm).sum() / max(1, nrm.sum()))})
    return pd.DataFrame(rows)
