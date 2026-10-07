"""[탐색, 선택 실행] 공개 분석 사례(Lee, 2023, 정적 Mahalanobis) 방법 재구현 — 보고서 2-2절.
원 저장소의 코드·데이터는 쓰지 않고, 공개 설명대로 방법만 다시 구현한다.
  방법: 원본 labeled_data.csv 중복 행 제거 → 제품별(CN7/RG3) float 공정변수(상수 열, Switch_Over_Position,
        Clamp_Open_Position 제외) → 정상 행 90% 무작위 학습, StandardScaler, 표본 공분산 Mahalanobis,
        임계 = 카이제곱(0.9, df = 변수 수 − 1). 시험셋 = 정상 10% + 불량 전부 (원 사례와 같은 구성).
  출력: 시험셋 F1(원 사례 방식)과 실제 불량 비율로 환산한 F1, 날짜·사유별 검출률, 날짜별 정상 경보율.
실행: python src/explore/public_md_check.py  (약 10초) → outputs/experiments/explore/public_md_check.txt
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, scipy as sp
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, EXP

OUTX = EXP / 'explore'; OUTX.mkdir(parents=True, exist_ok=True)
lines = []
def say(s=''):
    print(s); lines.append(str(s))

df = pd.read_csv(DATA / 'labeled_data.csv', low_memory=False).drop_duplicates()
df = df.drop(columns=[f'Mold_Temperature_{i}' for i in (1, 2, 5, 6, 7, 8, 9, 10, 11, 12)], errors='ignore')
df['y'] = (df.PassOrFail == 'N').astype(int)
num = [c for c in df.columns if df[c].dtype == 'float64' and c not in ('Switch_Over_Position', 'Clamp_Open_Position')]
say(f'중복 제거 후 {len(df)}행, 불량 {df.y.sum()}행')


def fit_md(A_raw):
    sc = StandardScaler().fit(A_raw); A = sc.transform(A_raw)
    mu, inv = A.mean(0), sp.linalg.inv(np.cov(A.T)); crit = chi2.ppf(0.9, df=A.shape[1] - 1)
    return lambda X: (lambda Z: np.einsum('ij,jk,ik->i', Z, inv, Z))(sc.transform(X) - mu) > crit


for prod in ['CN7', 'RG3']:
    g = df[df.PART_NAME.str.contains(prod)].reset_index(drop=True)
    cols = [c for c in num if g[c].nunique() > 1]
    Y, N = g[g.y == 0].reset_index(drop=True), g[g.y == 1].reset_index(drop=True)
    det = np.zeros(len(N)); fpr, f1_test, f1_true = [], [], []
    for s in range(50):
        tr, te = train_test_split(Y[cols], test_size=0.1, random_state=s)
        alarm = fit_md(tr)
        aY, aN = alarm(te), alarm(N[cols]); det += aN
        fp, tp = aY.mean(), aN.mean(); fpr.append(fp)
        p1 = aN.sum() / max(aN.sum() + aY.sum(), 1); f1_test.append(2 * p1 * tp / (p1 + tp) if p1 + tp else 0)
        p2 = aN.sum() / max(aN.sum() + fp * len(Y), 1e-9); f1_true.append(2 * p2 * tp / (p2 + tp) if p2 + tp else 0)
    N['검출률'] = det / 50; N['date'] = N.TimeStamp.str[:10]
    say(f'\n== {prod}: 정상 {len(Y)}, 불량 {len(N)}, 변수 {len(cols)} | 시험셋 불량 비율 {len(N) / (len(N) + round(0.1 * len(Y))):.3f} '
        f'(실제 {len(N) / len(g):.3f})')
    say(f'   정상 경보율 {np.mean(fpr):.3f} | 불량 검출률 {N["검출률"].mean():.3f} | '
        f'F1 시험셋 {np.mean(f1_test):.2f} | F1 실제 비율 환산 {np.mean(f1_true):.2f}')
    say(N.groupby(['date', 'Reason'])['검출률'].agg(['size', 'mean']).round(2).to_string())
    al = np.zeros(len(Y), bool)
    for tr, te in KFold(10, shuffle=True, random_state=0).split(Y):
        al[te] = fit_md(Y.loc[tr, cols])(Y.loc[te, cols])
    Y['al'] = al; Y['date'] = Y.TimeStamp.str[:10]
    say('   날짜별 정상 경보율 (10-fold, 각 정상 행은 학습 밖에서 채점)')
    say(Y.groupby('date').al.agg(['size', 'mean']).round(3).T.to_string())

(OUTX / 'public_md_check.txt').write_text('\n'.join(lines), encoding='utf-8')
