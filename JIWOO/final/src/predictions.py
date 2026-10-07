"""제출용 예측 파일 (원장에서 생성)

pred_official.csv    : 공식 2차 라벨 행 전부 (2,393행). 확률은 그 행의 Run을 보지 않은 보정기로 예측(LORO OOF).
                       원행번호 = 대회 2차 파일의 Unnamed: 0
pred_prospective.csv : 공식 기간으로 모든 설정을 고정한 뒤 추가 기간(CN7 10/27~11/03, RG3 11/04)에 그대로 적용
열: 제품, 쇼트시각, 좌우, 불량확률, 확률구간_하, 확률구간_상, 검사대상(경보), 검사우선순위, 이상변수Top3, 깨진관계, 실제불량
  - 검사대상(경보) = 점수 백분위 pct ≥ 0.999 (정상 기준 분포 99.9 백분위, 라벨 무사용)
  - 검사우선순위 = 제품·기간 안에서 불량확률 내림차순 순위 (동률은 점수로)
  - 이상변수Top3 = 같은 Run 직전 기준선 대비 편차(robust z) 절대값 상위 3개 변수
  - 깨진관계 = |충전시간 편차 − 사출속도 편차| ≥ 3, |흐름저항(사출압/속도) 편차| ≥ 3
"""
import numpy as np
import pandas as pd

from paths import LEDGER, PRED

COLS = ['제품', '쇼트시각', '좌우', '불량확률', '확률구간_하', '확률구간_상', '검사대상(경보)', '검사우선순위',
        '이상변수Top3', '깨진관계', '실제불량']


def _top3(s):
    if not isinstance(s, str) or s.count('nan') == 3:
        return '기준선 형성 중(Run 첫 5쇼트)'
    return s


def _make(g):
    g = g.copy()
    pr = g['p'].values + 1e-9 * np.nan_to_num(g['score'].values)
    g['검사우선순위'] = pd.Series(-pr, index=g.index).groupby(g['product']).rank(method='first').astype(int)
    out = pd.DataFrame({
        '제품': g['product'], '쇼트시각': g['TimeStamp'], '좌우': g['cavity'],
        '불량확률': g['p'].round(5), '확률구간_하': g['p_lo'].round(5), '확률구간_상': g['p_hi'].round(5),
        '검사대상(경보)': g['alarm'].astype(int), '검사우선순위': g['검사우선순위'],
        '이상변수Top3': g['top3'].map(_top3), '깨진관계': g['relation'].fillna(''), '실제불량': g['y'].astype(int)})
    return out


def main():
    L = pd.read_csv(LEDGER / 'ledger_v1.csv')
    off = L[L.period == 'official']
    T = _make(off)
    T.insert(0, '원행번호', off['oid'].astype(int).values)
    T = T.sort_values(['제품', '원행번호'])
    T.to_csv(PRED / 'pred_official.csv', index=False, encoding='utf-8-sig')
    ext = L[L.period == 'extra']
    E = _make(ext).sort_values(['제품', '쇼트시각', '좌우'])
    E.to_csv(PRED / 'pred_prospective.csv', index=False, encoding='utf-8-sig')
    for nm, d in (('pred_official.csv', T), ('pred_prospective.csv', E)):
        print(f'  {nm}: {len(d):,}행, 경보 {int(d["검사대상(경보)"].sum())}행, 실제불량 {int(d["실제불량"].sum())}행')


if __name__ == '__main__':
    main()
