"""
rg3_active.py — 49b번: RG3에 '가스 RF + 주변 확장' 운영 루프 적용 (CN7에서 본 주변 확장 이득이 RG3에서도 재현되나)

배경: 49번에서 RG3는 라벨 없는 기본 점수(최종 모델·std4)만 써서 첫 불량을 거의 못 찾음 → 주변 확장도 효과 없음.
      RG3 가스는 지도 RF(직전 10샷 std 24개)가 8/17로 검증된 모델 → 이걸 기본 점수로 넣어 다시 시뮬레이션.

기본 점수 (RG3)
  gasrf_loro    : Run 하나씩 빼고 학습한 가스 RF 점수 (37·41·43번과 같음). ⚠ 다른 Run에는 '나중 Run'도 포함 → 운영보다 약간 유리
  gasrf_forward : **시간 순 학습** — 각 Run은 그보다 먼저 시작한 Run들의 라벨로만 학습한 RF로 점수 (현장과 같은 조건)
                  먼저 시작한 Run에 가스가 2개 미만이면 그 Run은 combo 점수 순위로 대체
  combo         : 49번과 같음 (최종 경보 먼저 + std4) — 비교용
전략: 49번과 같음 (S0 무작위 / S1 고정 / S2 고정 + 주변 확장 / S3 온라인 / S4 온라인 + 확장), 묶음 100샷, 예산 10%
판단: S2 − S1 > 0 이 RG3에서도 나오면 '불량 발견 → 주변 확장'은 두 제품에서 확인된 운영 규칙
      (주의: 확장의 이득은 불량이 몰려 나오는 성질에서 옴. RG3 불량이 흩어져 있으면 이득이 작은 게 정상)
추가: 불량 간격 표 — 같은 Run에서 다음 불량까지 몇 샷인지 (주변 ±5샷 확장이 닿는 비율)
"""
import os
import numpy as np
import pandas as pd

import active_learning as AL
import semi_normal as SN
import semi_sup as SS

OUT = './outputs/active_learning'


def _map(labD, lab, s):
    """semi_normal 데이터 순서 → active_learning lab 순서 (TimeStamp 기준)"""
    m = pd.Series(np.asarray(s, float), index=labD['TimeStamp'].values)
    m = m[~m.index.duplicated()]
    return m.reindex(lab['TimeStamp'].values).values


def forward_gasrf(D, seed=0):
    """Run 시작 시각 순으로, 각 Run = 더 먼저 시작한 Run들의 라벨로만 학습한 가스 RF 점수"""
    lab, Zl = D['lab'], D['Zl']
    y, use = SS.target_y(lab, '가스')
    runs = lab['run_id'].values
    order = lab.groupby('run_id')['TimeStamp'].min().sort_values().index
    s = np.full(len(lab), np.nan)
    info = []
    for k, r in enumerate(order):
        te = runs == r
        tr = np.isin(runs, list(order[:k])) & use
        n_pos = int(y[tr].sum())
        info.append({'Run': r, '학습에 쓴 이전 Run': list(order[:k]), '이전 가스 수': n_pos})
        if n_pos < 2:
            continue
        s[te] = SN.fold_predict('M0_supervised', Zl[tr], y[tr], Zl[te], D['Zr'], '가스', seed)
    return s, pd.DataFrame(info)


def prepare(seed=0, base_dir='.'):
    Dal = AL.prepare('rg3')
    lab = Dal['lab']
    D = SN.load_std('rg3', base_dir)
    loro = _map(D['lab'], lab, SN.loro(D, 'M0_supervised', '가스', seed))
    fwd, info = forward_gasrf(D, seed)
    fwd = _map(D['lab'], lab, fwd)
    combo_rank = pd.Series(Dal['base']['combo']).rank(pct=True).values
    # 점수가 없는 Run(학습 불가)은 combo 순위로 대체 (Run 단위라 섞여도 순위 비교에 문제 없음)
    Dal['base']['gasrf_loro'] = np.where(np.isfinite(loro), loro, combo_rank)
    Dal['base']['gasrf_forward'] = np.where(np.isfinite(fwd), fwd, combo_rank)
    Dal['forward_info'] = info
    Dal['fallback_runs'] = {'gasrf_loro': sorted(pd.unique(lab.loc[~np.isfinite(loro) & lab['eval'].values, 'run_id'])),
                            'gasrf_forward': sorted(pd.unique(lab.loc[~np.isfinite(fwd) & lab['eval'].values, 'run_id']))}
    return Dal


def run(bases=('gasrf_loro', 'gasrf_forward', 'combo'), seeds=(0, 1, 2, 3, 4), block=100, nbr=5, verbose=True):
    os.makedirs(OUT, exist_ok=True)
    D = prepare()
    S, E, LOGS = [], [], {}
    for base in bases:
        for st in AL.STRATS:
            sds = seeds if st.startswith(('S0', 'S3', 'S4')) else seeds[:1]
            for sd in sds:
                L = AL.simulate(D, st, base, block=block, nbr=nbr, seed=sd)
                LOGS[('rg3', base, st, sd)] = L
                S.append(AL.summarize_log(D, L, st, base, sd))
                E += AL.episode_view(D, L, st, base, sd)
        if verbose:
            print(f'  [rg3] {base} 완료', flush=True)
    S, E = pd.DataFrame(S), pd.DataFrame(E)
    agg = S.groupby(['기본 점수', '전략'], sort=False).agg(
        검사_샷=('검사 샷', 'first'), 찾은_불량=('찾은 불량', 'mean'), 범위=('찾은 불량', lambda x: f'{x.min()}~{x.max()}'),
        전체_불량=('전체 불량', 'first'), 가스=('가스', 'mean'), 가스_전체=('가스 전체', 'first'),
        미성형=('미성형', 'mean'), 미성형_전체=('미성형 전체', 'first'),
        Precision=('Precision', 'mean'), Recall=('Recall', 'mean'), F1=('F1', 'mean')).reset_index()
    agg.to_csv(f'{OUT}/rg3_gasrf_summary_block{block}.csv', index=False, encoding='utf-8-sig')
    E.to_csv(f'{OUT}/rg3_gasrf_episodes_block{block}.csv', index=False, encoding='utf-8-sig')
    return {'summary': agg, 'by_seed': S, 'episodes': E, 'logs': LOGS, 'D': D, 'gaps': defect_gaps(D, nbr)}


def defect_gaps(D, nbr=5):
    """같은 Run 안에서 불량 → 다음 불량까지 샷 간격. 간격 ≤ nbr 비율 = 주변 확장이 닿을 수 있는 비율"""
    lab = D['lab']
    y = (lab['PassOrFail'].values == 1) & lab['eval'].values
    rows = []
    for r in pd.unique(lab.loc[y, 'run_id']):
        s = np.sort(lab['seq'].values[y & (lab['run_id'] == r).values])
        g = np.diff(s)
        rows.append({'Run': r, '불량 수': len(s), '불량 seq': ', '.join(map(str, s)),
                     '간격 중앙값': float(np.median(g)) if len(g) else np.nan,
                     f'간격 ≤{nbr} 비율': float(np.mean(g <= nbr)) if len(g) else np.nan})
    return pd.DataFrame(rows)


def gain_table(R):
    """기본 점수별 S2 − S1 (주변 확장 이득), S3 − S1 (온라인 학습 이득)"""
    s = R['summary'].set_index(['기본 점수', '전략'])['찾은_불량']
    out = []
    for b in s.index.get_level_values(0).unique():
        g = s.loc[b]
        out.append({'기본 점수': b, 'S1 고정': g.get('S1 고정 순위'), 'S2 − S1 (주변 확장)': g.get('S2 고정 + 주변 확장') - g.get('S1 고정 순위'),
                    'S3 − S1 (온라인 학습)': g.get('S3 온라인 학습') - g.get('S1 고정 순위'),
                    'S4 − S1': g.get('S4 온라인 + 주변 확장') - g.get('S1 고정 순위'), '무작위': g.get('S0 무작위')})
    return pd.DataFrame(out)
