"""선택 실험: TabPFN(사전학습 표형 분류기) vs RF 지도 기준선 — 같은 입력·같은 분할·같은 채점

실행 (패키지 루트에서, run_all.py를 먼저 한 번 실행해 outputs/가 있어야 함)
    pip install tabpfn
    python src/exp_tabpfn.py            # TabPFN (첫 실행 때 모델 가중치를 인터넷에서 내려받음)
    MODEL=rf python src/exp_tabpfn.py   # 점검용: RF로 바꿔 기존 비교표 수치(공식 0.014 / 미래 0.161)가 나오는지 확인

프로토콜 (compare.py의 RF 기준선과 동일)
    입력  : 24개 공정변수(쇼트 단위, 1차 원값), 결측 0
    공식  : 쇼트 순서대로 5개 연속 블록 fold, 시험 블록 앞뒤 ±10쇼트는 학습에서 제외 (OOF 점수)
    미래  : 공식 기간 전체로 학습한 모델을 추가(미래) 기간에 적용 = '알려진 사건(10/16)으로 학습 → 새 사건' 시험
    채점  : 원장 v2 행(LH/RH 행 단위)에 쇼트 점수를 붙여 AP, 상위 5% TP. 최종 모델(score)·RF·z 를 같은 행에서 나란히
결과 : outputs/experiments/tabpfn_compare.csv
지위 : 탐색적(사전등록 외). 최종 모델·원장은 바꾸지 않는다. 외부 사전학습 모델 → 보고서에 출처·버전 명기.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from paths import LEDGER, EXP
from shots_data import VARS, load_all

MODEL = os.environ.get('MODEL', 'tabpfn').lower()


def make_clf():
    if MODEL == 'rf':
        return RandomForestClassifier(200, random_state=0, n_jobs=4)
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(random_state=0, device=os.environ.get('DEVICE', 'cpu'))


def p1(X_tr, y_tr, X_te):
    """학습 fold에 불량이 없으면(한 클래스) 상수 0 — RF 기준선의 처리와 같은 효과"""
    if len(np.unique(y_tr)) < 2:
        return np.zeros(len(X_te))
    pr = make_clf().fit(X_tr, y_tr).predict_proba(X_te)
    return pr[:, 1]


def shot_scores(S):
    m = {}
    for prod in ['CN7', 'RG3']:
        sh = S[S['prod'] == prod].sort_values('t').reset_index(drop=True)
        off = sh[sh.period == 'official'].reset_index(drop=True)
        ext = sh[sh.period == 'extra'].reset_index(drop=True)
        X, y = off[VARS].fillna(0).to_numpy(float), off.y.to_numpy(int)
        n = len(off); oof = np.full(n, np.nan)
        for k, f in enumerate(np.array_split(np.arange(n), 5)):
            test = np.zeros(n, bool); test[f] = True
            near = np.zeros(n, bool); near[max(0, f[0] - 10):f[-1] + 11] = True
            oof[test] = p1(X[~near], y[~near], X[test])
            print(f'  [{prod}] 공식 fold {k + 1}/5 (학습 불량 {int(y[~near].sum())})', flush=True)
        for ts, v in zip(off.TimeStamp, oof): m[(prod, ts)] = v
        if len(ext):
            pe = p1(X, y, ext[VARS].fillna(0).to_numpy(float))
            for ts, v in zip(ext.TimeStamp, pe): m[(prod, ts)] = v
            print(f'  [{prod}] 미래 {len(ext)}쇼트 예측 (공식 전체 학습, 불량 {int(y.sum())})', flush=True)
    return m


def evaluate(L, col):
    rows = []
    for (prod, per), g in L.groupby(['product', 'period']):
        s = g[col].to_numpy(float); ok = ~np.isnan(s); y = g.y.to_numpy(int)[ok]; s = s[ok]
        k = int(np.ceil(0.05 * len(y)))
        o = np.lexsort((np.random.RandomState(0).rand(len(s)), -s))   # 동점은 무작위
        rows.append(dict(제품=prod, 기간=per, 모델=col, 행=len(y), 불량=int(y.sum()),
                         AP=round(float(average_precision_score(y, s)), 3) if y.sum() else np.nan,
                         TP_상위5pct=int(y[o[:k]].sum())))
    return rows


def main():
    d = load_all()
    L = pd.read_csv(LEDGER / 'ledger_v1_plus.csv')
    name = 'TabPFN' if MODEL == 'tabpfn' else 'RF(점검)'
    print(f'{name}: 쇼트 점수 계산')
    m = shot_scores(d['S'])
    L[name] = [m.get((p, t), np.nan) for p, t in zip(L['product'], L.TimeStamp)]
    L = L.rename(columns={'score': '최종(Mahalanobis)', 'rf': 'RF 지도', 'z': '강건 z'})
    R = []
    for col in ['최종(Mahalanobis)', 'RF 지도', '강건 z', name]:
        R += evaluate(L, col)
    R = pd.DataFrame(R)
    EXP.mkdir(parents=True, exist_ok=True)
    out = EXP / ('tabpfn_compare.csv' if MODEL == 'tabpfn' else 'tabpfn_compare_rfcheck.csv')
    R.to_csv(out, index=False, encoding='utf-8-sig')
    print(R.pivot_table(index=['제품', '기간'], columns='모델', values='AP').round(3).to_string())
    print(R.pivot_table(index=['제품', '기간'], columns='모델', values='TP_상위5pct').to_string())
    print(f'→ {out}')


if __name__ == '__main__':
    main()
