import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # notebooks 폴더에서 실행: python 53_sanngrok/exp53a_sanngrok_frame.py
"""
53-A. 우리 최종 모델(M_maha_free)을 sanngrok 평가 틀로 재평가

sanngrok 틀 (molding_pipeline.py 그대로 사용)
  - labeled_data.csv 원본, 완전 중복행만 제거, S14, CN7/RG3, LH/RH '행' 단위
  - 타깃: PassOrFail N=1 (초기허용불량도 불량)
  - 날짜 확장창: 과거 날짜 전체 → 다음 날짜 예측. 과거에 불량이 없으면 그 날은 평가 안 함
  - 임계값: 과거 날짜 점수에서 F1 최대(동률이면 높은 값) → 다음 날짜에 적용
우리 모델은 비지도(unlabeled 정상만 학습)라 날짜별 재학습이 필요 없음 → 점수는 고정, 임계값만 과거 날짜로 정함
  (sanngrok은 과거 날짜를 3-fold OOF로 점수 내서 임계값을 고름. 우리 점수는 과거 날짜 라벨을 본 적이 없으니 그대로가 OOF와 같음)
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sanngrok_pipeline import prepare, threshold_f1, summarize

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
HERE = _o.path.dirname(_o.path.abspath(__file__))
DATA = _o.path.join(HERE, 'labeled_data.csv')            # KAMP labeled_data.csv 원본 (sanngrok 레포와 md5 동일)
if not _o.path.exists(DATA):                               # 없으면 팀 데이터셋 폴더에서
    DATA = '../../1. 사출성형기 AI 데이터셋/dataset/labeled_data.csv'
SG_PRED = _o.path.join(HERE, 'sanngrok_predictions.csv')  # sanngrok 레포 outputs/predictions.csv
MD = './outputs/final_model'


def our_scores(p):
    s = pd.read_csv(f'{MD}/{p}_scores.csv')
    return s.set_index('TimeStamp')[['score', 'pct', 'run_id']]


def forward_ours(df, fam):
    """sanngrok forward_eval과 같은 분할, 점수만 우리 모델"""
    d = df[df.family == fam]
    dates = sorted(d.date.unique())
    out = []
    for i in range(1, len(dates)):
        tr = d[d.date < dates[i]]; te = d[d.date == dates[i]]
        ytr = tr.y.values
        if not (0 < ytr.sum() < len(ytr)):
            continue
        thr = threshold_f1(ytr, tr['our_score'].values)
        for name, pred in [('OURS_maha_free (과거날짜 F1최대 임계값)', (te['our_score'].values >= thr)),
                           ('OURS_maha_free (q0.99, 라벨 없이)', (te['our_pct'].values >= 0.99)),
                           ('OURS_maha_free (q0.999, 라벨 없이)', (te['our_pct'].values >= 0.999))]:
            out.append(pd.DataFrame(dict(model=name, family=fam, date=dates[i], _id=te['_id'].values,
                                         ts=te.TimeStamp.values, side=te.side.values, reason=te.Reason.values,
                                         y=te.y.values, score=te['our_score'].values, pred=pred.astype(int))))
    return pd.concat(out, ignore_index=True)


def main():
    df, n_raw = prepare(DATA)
    cov = []
    parts = []
    for fam in ('CN7', 'RG3'):
        sc = our_scores(fam.lower())
        d = df[df.family == fam].copy()
        d['our_score'] = d['TimeStamp'].map(sc['score'])
        d['our_pct'] = d['TimeStamp'].map(sc['pct'])
        cov.append(dict(family=fam, 행=len(d), 점수있는행=int(d.our_score.notna().sum()),
                        불량행=int(d.y.sum()), 점수있는불량행=int((d.our_score.notna() & (d.y == 1)).sum())))
        # 점수 없는 행(우리 base에서 빠진 샷) = 최저점 (검사 안 함)
        d['our_score'] = d['our_score'].fillna(d['our_score'].min() - 1)
        d['our_pct'] = d['our_pct'].fillna(0)
        parts.append(d)
    df = pd.concat(parts)
    print('원본', n_raw, '행 / 점수 매칭'); print(pd.DataFrame(cov).to_string(index=False))

    P = pd.concat([forward_ours(df, f) for f in ('CN7', 'RG3')], ignore_index=True)
    SG = pd.read_csv(SG_PRED)
    keep = ['M0_LR_Std(P1,베이스라인)', 'M3_LR_Robust(P5)', 'M4_LR_Robust+Resid(P5+P7)', 'M6_Hybrid(M4+M5)',
            'M5_DAE(정상only)', 'M1_RF(가이드북최고)']
    SG = SG[SG.model.isin(keep)][P.columns]
    ALL = pd.concat([SG, P], ignore_index=True)

    cols = ['family', 'model', 'n', 'pos', 'TP', 'FP', 'FN', 'precision', 'recall', 'F1', 'AP_pooled', 'top20_capture']
    print('\n=== A. sanngrok 틀 그대로 (초기허용불량 = 불량) ===')
    A = summarize(ALL)[cols]; print(A.round(3).to_string(index=False))

    print('\n=== B. 같은 틀에서 초기허용불량 행만 평가에서 뺌 (우리 팀 정의) ===')
    B = summarize(ALL[ALL.reason.fillna('') != '초기허용불량'])[cols]; print(B.round(3).to_string(index=False))

    print('\n=== C. 테스트 날짜의 불량 행 구성 ===')
    t = P[(P.model == P.model.iloc[0]) & (P.y == 1)]
    print(t.groupby(['family', 'date', 'reason']).size().to_string())

    print('\n=== D. 불량 행별 순위 (그 날짜 안 상위 %; 낮을수록 위) — M4 vs 우리 ===')
    rk = ALL[ALL.model.isin(['M4_LR_Robust+Resid(P5+P7)', 'OURS_maha_free (과거날짜 F1최대 임계값)'])].copy()
    rk['rank_pct'] = rk.groupby(['model', 'family', 'date'])['score'].rank(ascending=False, pct=True)
    w = rk[rk.y == 1].pivot_table(index=['family', 'date', 'ts', 'side', 'reason'], columns='model',
                                  values=['rank_pct', 'pred'], aggfunc='first')
    print(w.round(3).to_string())

    print('\n=== E. 같은 행에서 M4 점수와 우리 점수의 순위 상관 (Spearman, 날짜별) ===')
    m4 = ALL[ALL.model == 'M4_LR_Robust+Resid(P5+P7)'][['_id', 'score']].rename(columns={'score': 'm4'})
    ou = P[P.model == P.model.iloc[0]][['_id', 'family', 'date', 'score']].merge(m4, on='_id')
    print(ou.groupby(['family', 'date']).apply(lambda g: g['score'].corr(g['m4'], method='spearman')).round(3).to_string())

    A.to_csv(_o.path.join(HERE, 'exp53a_A_sanngrok_frame.csv'), index=False, encoding='utf-8-sig')
    B.to_csv(_o.path.join(HERE, 'exp53a_B_no_initial.csv'), index=False, encoding='utf-8-sig')
    w.to_csv(_o.path.join(HERE, 'exp53a_D_defect_ranks.csv'), encoding='utf-8-sig')


if __name__ == '__main__':
    main()
