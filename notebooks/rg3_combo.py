"""
rg3_combo.py — 43번: RG3 최종 후보 = 가스 RF(기존, labeled만) + 미성형 N1(unlabeled를 정상으로 보강) → 검사 예산 나누기

  가스 모델  : RF 1000트리, 원값 21 + 새 변수 3개의 직전 10샷 std (24개), 가스 vs 정상, 클래스 가중치   (37번 반복 19 / 41번 M0)
  미성형 모델: 같은 변수 + unlabeled 3,000샷을 '정상'으로 추가(가중치 0.3), 미성형 vs 정상            (41번 N1)
  결합       : 검사 샷 K 중 g×K는 가스 모델 상위, 나머지는 미성형 모델 순위 (g는 미리 정함: 0.68 = 불량 수 비율 17:8, 0.5)

평가: Run 하나씩 빼고 검증(LORO), 시드 3개. 검사 비율별 Precision·Recall·F1·유형별 개수(42번과 같은 표)
검증: permutation_test — 라벨을 Run 안에서 원형 이동(불량 몰림 구조 유지, 위치만 무작위) → 두 모델 재학습·결합까지 반복
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import semi_normal as SN
import inspect_curve as IC

OUT = './outputs/rg3_combo'
SHARES = (1.0, 0.68, 0.5, 0.0)          # 1.0 = 가스 모델만, 0.0 = 미성형 모델만


def scores(D, seed=0):
    gas = SN.loro(D, 'M0_supervised', '가스', seed)
    short = SN.loro(D, 'N1_normal_augment', '미성형', seed)
    return gas, short


def model_ap(D, s, target):
    lab = D['lab']
    m = lab['eval'].values
    t = lab['type'].values[m]
    y = (lab['PassOrFail'].values[m] == 1).astype(int)
    v = np.asarray(s, float)[m]
    v = np.where(np.isfinite(v), v, np.nanmin(v[np.isfinite(v)]) - 1)
    mm = (t == target) | (y == 0)
    return average_precision_score((t[mm] == target).astype(int), v[mm])


def run(seeds=(0, 1, 2), base_dir='.', verbose=True):
    os.makedirs(OUT, exist_ok=True)
    D = SN.load_std('rg3', base_dir)
    lab = D['lab']
    if verbose:
        print(f'RG3 평가 {int(lab["eval"].sum())}샷 {lab.loc[lab["eval"], "type"].value_counts().to_dict()} | 변수 {len(D["names"])}개',
              flush=True)
    curves, aps, S = [], [], {}
    for sd in seeds:
        gas, short = scores(D, sd)
        S[sd] = (gas, short)
        aps.append({'seed': sd, '가스 모델 AP_가스': model_ap(D, gas, '가스'), '가스 모델 AP_미성형': model_ap(D, gas, '미성형'),
                    '미성형 모델 AP_미성형': model_ap(D, short, '미성형'), '미성형 모델 AP_가스': model_ap(D, short, '가스')})
        for g in SHARES:
            nm = {1.0: '가스 모델만', 0.0: '미성형 모델만'}.get(g, f'결합 g={g}')
            T = IC.split_curve(lab, gas, short, g, name=nm)
            T['seed'] = sd
            curves.append(T)
        if verbose:
            print(f'  seed {sd} 완료', flush=True)
    C = pd.concat(curves, ignore_index=True)
    for col in ('불량', '가스', '미성형'):
        C[f'{col} 수'] = C[col].str.split('/').str[0].astype(int)
    tot = {c: C[c].iloc[0].split('/')[1] for c in ('불량', '가스', '미성형')}
    summ = C.groupby(['모델', '검사 비율'], sort=False).agg(
        검사_샷=('검사 샷', 'first'),
        불량_평균=('불량 수', 'mean'), 가스_평균=('가스 수', 'mean'), 가스_범위=('가스 수', lambda x: f'{x.min()}~{x.max()}'),
        미성형_평균=('미성형 수', 'mean'), 미성형_범위=('미성형 수', lambda x: f'{x.min()}~{x.max()}'),
        Precision=('Precision', 'mean'), Precision_최대=('Precision 최대(불량수/검사샷)', 'first'),
        정규화_Precision=('정규화 Precision', 'mean'), Recall=('Recall', 'mean'), F1=('F1', 'mean'),
        무작위대비=('무작위 대비(배)', 'mean')).reset_index()
    summ = summ.rename(columns={'불량_평균': f'불량(/{tot["불량"]})', '가스_평균': f'가스(/{tot["가스"]})',
                                '미성형_평균': f'미성형(/{tot["미성형"]})'})
    summ['목표(가스≥7 & 미성형≥4)'] = np.where((summ[f'가스(/{tot["가스"]})'] >= 7) & (summ[f'미성형(/{tot["미성형"]})'] >= 4), 'O', '-')
    A = pd.DataFrame(aps)
    C.to_csv(f'{OUT}/curves_by_seed.csv', index=False, encoding='utf-8-sig')
    summ.to_csv(f'{OUT}/summary.csv', index=False, encoding='utf-8-sig')
    A.to_csv(f'{OUT}/model_ap.csv', index=False, encoding='utf-8-sig')
    return {'summary': summ, 'curves': C, 'ap': A, 'scores': S, 'D': D}


def at_rate(R, rate='10%'):
    """한 검사 비율에서 모델별 요약"""
    return R['summary'][R['summary']['검사 비율'] == rate].reset_index(drop=True)


def plot(R, seed=0, save=None):
    lab = R['D']['lab']
    gas, short = R['scores'][seed]
    T = {nm: IC.split_curve(lab, gas, short, g, name=nm)
         for g, nm in ((1.0, '가스 모델만'), (0.68, '결합 g=0.68'), (0.5, '결합 g=0.5'), (0.0, '미성형 모델만'))}
    IC.plot_curves(T, f'RG3 가스 RF + 미성형 N1 결합 — 검사 비율별 (seed {seed})', save=save)


def permutation_test(n=50, gas_share=0.68, rate=0.10, seed=0, base_dir='.', verbose=True):
    """라벨 원형 이동 후 두 모델 재학습 + 결합 → (가스, 미성형, 합계) 포착 수가 우연을 넘는지. n=50이면 수십 분"""
    D = SN.load_std('rg3', base_dir)
    lab0 = D['lab']

    def stat(Dx):
        g, s = scores(Dx, seed)
        T = IC.split_curve(Dx['lab'], g, s, gas_share, rates=(rate,))
        gg, ss = int(T['가스'].iloc[0].split('/')[0]), int(T['미성형'].iloc[0].split('/')[0])
        # 미성형 모델 단독 포착도 따로 (N1 자체의 신호 확인)
        Ts = IC.split_curve(Dx['lab'], s, s, 1.0, rates=(rate,))
        return gg, ss, gg + ss, int(Ts['미성형'].iloc[0].split('/')[0])

    obs = stat(D)
    rng = np.random.default_rng(seed + 500)
    ev = np.flatnonzero(lab0['eval'].values)
    runs = lab0['run_id'].values
    null = []
    for b in range(n):
        L = lab0.copy()
        tcol = L['type'].values.copy()
        for r in pd.unique(runs[ev]):
            ii = ev[runs[ev] == r]
            tcol[ii] = np.roll(lab0['type'].values[ii], rng.integers(0, len(ii)))
        L['type'] = tcol
        L['PassOrFail'] = np.where(np.isin(tcol, ['가스', '미성형', '기타']), 1, 0)
        null.append(stat(dict(D, lab=L)))
        if verbose and b % 10 == 0:
            print('순열', b, flush=True)
    null = np.asarray(null)
    names = ('결합-가스', '결합-미성형', '결합-합계', '미성형 모델 단독-미성형')
    rows = [{'지표': nm, '관측': obs[i], '무작위 평균': round(float(null[:, i].mean()), 2),
             '무작위 95%': float(np.quantile(null[:, i], 0.95)), 'p': float((1 + (null[:, i] >= obs[i]).sum()) / (n + 1))}
            for i, nm in enumerate(names)]
    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f'{OUT}/permutation.csv', index=False, encoding='utf-8-sig')
    return out


def rank_table(R):
    """검사 비율과 무관한 분류 지표 (R-Precision, Precision@Recall, ROC-AUC, AP) — 시드별·모델별"""
    lab = R['D']['lab']
    rows = []
    for sd, (gas, short) in R['scores'].items():
        for nm, s in (('가스 모델', gas), ('미성형 모델', short)):
            rows.append({'seed': sd, '모델': nm, **IC.rank_metrics(lab, s)})
    return pd.DataFrame(rows)
